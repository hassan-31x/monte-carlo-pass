#!/usr/bin/env python3
"""Causal space-time Transformer for soccer trajectory generation."""

from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_2PI = math.log(2.0 * math.pi)
LOG_SIGMA_MIN = -3.0
LOG_SIGMA_MAX = 3.0
U_MAX_ABS = 8.0
SIGMA2_FLOOR = 1e-3

STOP_CONTINUE = 0
STOP_UNKNOWN_BREAK = 9
STOP_PAD = 10
STOP_TOKEN_VOCAB_SIZE = STOP_PAD + 1


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    return torch.stack((-x_odd, x_even), dim=-1).flatten(-2)


class MultiheadSelfAttention(nn.Module):
    """Self-attention with optional RoPE and grouped-query attention."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        use_rope: bool = False,
        use_gqa: bool = False,
        gqa_groups: int = 1,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads}).")
        if use_gqa:
            if gqa_groups <= 0 or gqa_groups > n_heads:
                raise ValueError(f"gqa_groups ({gqa_groups}) must be in [1, n_heads={n_heads}].")
            if n_heads % gqa_groups != 0:
                raise ValueError(f"n_heads ({n_heads}) must be divisible by gqa_groups ({gqa_groups}).")

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        if self.head_dim % 2 != 0:
            raise ValueError(
                f"head_dim must be even for RoPE. got d_model={d_model}, n_heads={n_heads}, head_dim={self.head_dim}"
            )
        self.use_rope = bool(use_rope)
        self.dropout = float(dropout)
        self.kv_heads = int(gqa_groups) if use_gqa else int(n_heads)

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, self.kv_heads * self.head_dim)
        self.v_proj = nn.Linear(d_model, self.kv_heads * self.head_dim)
        self.out_proj = nn.Linear(d_model, d_model)

    def _rope_cos_sin(
        self,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pos = torch.arange(seq_len, device=device, dtype=torch.float32)
        inv_freq = 1.0 / (
            10000.0
            ** (
                torch.arange(0, self.head_dim, 2, device=device, dtype=torch.float32)
                / float(self.head_dim)
            )
        )
        freqs = torch.outer(pos, inv_freq)  # [L, Hd/2]
        emb = torch.repeat_interleave(freqs, 2, dim=-1)  # [L, Hd]
        cos = emb.cos().to(dtype=dtype).view(1, 1, seq_len, self.head_dim)
        sin = emb.sin().to(dtype=dtype).view(1, 1, seq_len, self.head_dim)
        return cos, sin

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz, seq_len, _ = x.shape

        q = self.q_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)  # [B,H,L,Hd]
        k = self.k_proj(x).view(bsz, seq_len, self.kv_heads, self.head_dim).transpose(1, 2)  # [B,Hkv,L,Hd]
        v = self.v_proj(x).view(bsz, seq_len, self.kv_heads, self.head_dim).transpose(1, 2)  # [B,Hkv,L,Hd]

        if self.use_rope:
            cos, sin = self._rope_cos_sin(seq_len=seq_len, device=x.device, dtype=q.dtype)
            q = q * cos + _rotate_half(q) * sin
            k = k * cos + _rotate_half(k) * sin

        if self.n_heads != self.kv_heads:
            repeat_factor = self.n_heads // self.kv_heads
            k = k.repeat_interleave(repeat_factor, dim=1)
            v = v.repeat_interleave(repeat_factor, dim=1)

        add_mask: torch.Tensor | None = None
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                add_mask = torch.zeros((1, 1, seq_len, seq_len), device=x.device, dtype=q.dtype)
                add_mask = add_mask.masked_fill(attn_mask.view(1, 1, seq_len, seq_len), float("-inf"))
            else:
                add_mask = attn_mask.to(device=x.device, dtype=q.dtype).view(1, 1, seq_len, seq_len)

        if key_padding_mask is not None:
            pad_mask = key_padding_mask.view(bsz, 1, 1, seq_len)  # True = disallow
            if add_mask is None:
                add_mask = torch.zeros((bsz, 1, seq_len, seq_len), device=x.device, dtype=q.dtype)
            elif add_mask.shape[0] == 1:
                add_mask = add_mask.expand(bsz, -1, -1, -1).clone()
            add_mask = add_mask.masked_fill(pad_mask, float("-inf"))

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=add_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)
        return self.out_proj(out)


class SpaceTimeBlock(nn.Module):
    """Transformer block with causal temporal attention + spatial attention."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        use_rope: bool = False,
        use_gqa: bool = False,
        gqa_groups: int = 1,
    ) -> None:
        super().__init__()
        d_ff = int(d_model * mlp_ratio)
        self.use_custom_attn = bool(use_rope or use_gqa)

        self.temporal_ln = nn.LayerNorm(d_model)
        self.spatial_ln = nn.LayerNorm(d_model)
        self.mlp_ln = nn.LayerNorm(d_model)

        if self.use_custom_attn:
            self.temporal_attn_custom = MultiheadSelfAttention(
                d_model=d_model,
                n_heads=n_heads,
                dropout=dropout,
                use_rope=bool(use_rope),
                use_gqa=bool(use_gqa),
                gqa_groups=int(gqa_groups),
            )
            self.spatial_attn_custom = MultiheadSelfAttention(
                d_model=d_model,
                n_heads=n_heads,
                dropout=dropout,
                use_rope=False,
                use_gqa=bool(use_gqa),
                gqa_groups=int(gqa_groups),
            )
            self.temporal_attn = None
            self.spatial_attn = None
        else:
            self.temporal_attn = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=n_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.spatial_attn = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=n_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.temporal_attn_custom = None
            self.spatial_attn_custom = None

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        obs_mask: torch.Tensor,
        causal_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward.

        Args:
            x: [B, H, N, D]
            obs_mask: [B, H, N] bool (True = observed)
            causal_mask: [H, H] bool (True = disallow)
        """
        bsz, hist, n_ent, d_model = x.shape

        # Temporal causal attention per entity.
        xt = x.permute(0, 2, 1, 3).reshape(bsz * n_ent, hist, d_model)
        tm = ~obs_mask.permute(0, 2, 1).reshape(bsz * n_ent, hist)
        all_missing_t = tm.all(dim=1)
        if all_missing_t.any():
            tm = tm.clone()
            tm[all_missing_t] = False

        xt_in = self.temporal_ln(xt)
        if self.use_custom_attn:
            xt_out = self.temporal_attn_custom(
                xt_in,
                attn_mask=causal_mask,
                key_padding_mask=tm,
            )
        else:
            xt_out, _ = self.temporal_attn(
                xt_in,
                xt_in,
                xt_in,
                attn_mask=causal_mask,
                key_padding_mask=tm,
                need_weights=False,
            )
        xt = xt + self.drop(xt_out)
        x = xt.reshape(bsz, n_ent, hist, d_model).permute(0, 2, 1, 3)

        # Spatial attention per timestep.
        xs = x.reshape(bsz * hist, n_ent, d_model)
        sm = ~obs_mask.reshape(bsz * hist, n_ent)
        all_missing_s = sm.all(dim=1)
        if all_missing_s.any():
            sm = sm.clone()
            sm[all_missing_s] = False

        xs_in = self.spatial_ln(xs)
        if self.use_custom_attn:
            xs_out = self.spatial_attn_custom(
                xs_in,
                attn_mask=None,
                key_padding_mask=sm,
            )
        else:
            xs_out, _ = self.spatial_attn(
                xs_in,
                xs_in,
                xs_in,
                key_padding_mask=sm,
                need_weights=False,
            )
        xs = xs + self.drop(xs_out)
        x = xs.reshape(bsz, hist, n_ent, d_model)

        x = x + self.mlp(self.mlp_ln(x))
        return x


class CausalSequenceBlock(nn.Module):
    """Simple causal self-attention block over a 1D token sequence."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        use_gqa: bool = False,
        gqa_groups: int = 1,
    ) -> None:
        super().__init__()
        d_ff = int(d_model * mlp_ratio)
        self.use_custom_attn = bool(use_gqa)

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        if self.use_custom_attn:
            self.attn_custom = MultiheadSelfAttention(
                d_model=d_model,
                n_heads=n_heads,
                dropout=dropout,
                use_rope=False,
                use_gqa=bool(use_gqa),
                gqa_groups=int(gqa_groups),
            )
            self.attn = None
        else:
            self.attn = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=n_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.attn_custom = None

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        x_in = self.ln1(x)
        if self.use_custom_attn:
            x_attn = self.attn_custom(
                x_in,
                attn_mask=causal_mask,
                key_padding_mask=None,
            )
        else:
            x_attn, _ = self.attn(
                x_in,
                x_in,
                x_in,
                attn_mask=causal_mask,
                need_weights=False,
            )
        x = x + self.drop(x_attn)
        x = x + self.mlp(self.ln2(x))
        return x


class TrajectoryTransformer(nn.Module):
    """Joint/single next-step trajectory model with strict temporal causality."""

    def __init__(
        self,
        d_in: int = 4,
        n_entities: int = 23,
        delta_dim: int = 3,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        max_history: int = 256,
        joint_rank: int = 8,
        use_rope: bool = False,
        use_gqa: bool = False,
        gqa_groups: int = 1,
        use_oneshot_head: bool = False,
        max_oneshot_steps: int = 1024,
        use_diffusion_head: bool = False,
        diffusion_steps: int = 1000,
    ) -> None:
        super().__init__()
        self.n_entities = n_entities
        self.n_targets = n_entities
        self.delta_dim = int(delta_dim)
        self.d_model = d_model
        self.max_history = max_history
        self.joint_rank = joint_rank
        self.joint_dim = self.n_targets * self.delta_dim
        self.use_rope = bool(use_rope)
        self.use_gqa = bool(use_gqa)
        self.gqa_groups = int(gqa_groups)
        self.use_oneshot_head = bool(use_oneshot_head)
        self.max_oneshot_steps = int(max_oneshot_steps)
        self.use_diffusion_head = bool(use_diffusion_head)
        self.diffusion_steps = int(diffusion_steps)
        self.oneshot_chunk_size = 64
        if self.use_oneshot_head and self.max_oneshot_steps < 1:
            raise ValueError("max_oneshot_steps must be >= 1 when use_oneshot_head is enabled.")
        if self.use_diffusion_head and not self.use_oneshot_head:
            raise ValueError("use_diffusion_head requires use_oneshot_head=True.")
        if self.use_diffusion_head and self.diffusion_steps < 2:
            raise ValueError("diffusion_steps must be >= 2 when use_diffusion_head is enabled.")

        self.in_proj = nn.Linear(d_in, d_model)
        self.time_emb = nn.Embedding(max_history, d_model)
        self.entity_emb = nn.Embedding(n_entities, d_model)
        self.type_emb = nn.Embedding(4, d_model)  # 0 home, 1 away, 2 ball, 3 pad
        self.missing_emb = nn.Parameter(torch.zeros(d_model))

        self.blocks = nn.ModuleList(
            [
                SpaceTimeBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    use_rope=self.use_rope,
                    use_gqa=self.use_gqa,
                    gqa_groups=self.gqa_groups,
                )
                for _ in range(n_layers)
            ]
        )
        self.final_ln = nn.LayerNorm(d_model)

        # Multi-agent joint heads.
        self.mu_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, self.delta_dim),
        )
        self.u_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, self.delta_dim * joint_rank),
        )
        self.sigma_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, self.joint_dim),
        )

        # Single-agent head.
        self.single_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 2 * self.delta_dim),  # mu + log_sigma
        )

        # Optional one-shot horizon conditioning head (kept optional for checkpoint compatibility).
        if self.use_oneshot_head:
            self.oneshot_step_emb = nn.Embedding(self.max_oneshot_steps, d_model)
        else:
            self.oneshot_step_emb = None

        # Optional diffusion denoiser heads for one-shot training.
        if self.use_diffusion_head:
            self.diffusion_timestep_emb = nn.Embedding(self.diffusion_steps, d_model)
            self.diffusion_multi_in = nn.Linear(self.delta_dim, d_model)
            self.diffusion_multi_out = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, self.delta_dim),
            )
            self.diffusion_single_in = nn.Linear(self.delta_dim, d_model)
            self.diffusion_single_out = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, self.delta_dim),
            )

            # Causal discrete stop-token head (joint with one-shot trajectory denoising).
            # Predictable classes exclude unknown_break; unknown timing is supervised via
            # a masked likelihood in train.py.
            self.diffusion_stop_vocab_size = STOP_TOKEN_VOCAB_SIZE
            self.diffusion_stop_predictable_size = 10  # [continue, 8 restart types, PAD]
            self.diffusion_stop_traj_in = nn.Linear(self.n_targets * self.delta_dim, d_model)
            self.diffusion_stop_ctx_in = nn.Linear(d_model, d_model)
            self.diffusion_stop_hist_token_emb = nn.Embedding(self.diffusion_stop_vocab_size, d_model)
            self.diffusion_stop_blocks = nn.ModuleList(
                [
                    CausalSequenceBlock(
                        d_model=d_model,
                        n_heads=n_heads,
                        mlp_ratio=mlp_ratio,
                        dropout=dropout,
                        use_gqa=bool(use_gqa),
                        gqa_groups=int(gqa_groups),
                    )
                    for _ in range(2)
                ]
            )
            self.diffusion_stop_ln = nn.LayerNorm(d_model)
            self.diffusion_stop_out = nn.Linear(d_model, self.diffusion_stop_vocab_size)
        else:
            self.diffusion_timestep_emb = None
            self.diffusion_multi_in = None
            self.diffusion_multi_out = None
            self.diffusion_single_in = None
            self.diffusion_single_out = None
            self.diffusion_stop_vocab_size = None
            self.diffusion_stop_predictable_size = None
            self.diffusion_stop_traj_in = None
            self.diffusion_stop_ctx_in = None
            self.diffusion_stop_hist_token_emb = None
            self.diffusion_stop_blocks = None
            self.diffusion_stop_ln = None
            self.diffusion_stop_out = None

    def _causal_mask(self, hist: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(hist, hist, dtype=torch.bool, device=device), diagonal=1)

    def encode(
        self,
        x: torch.Tensor,
        obs_mask: torch.Tensor,
        entity_type: torch.Tensor,
    ) -> torch.Tensor:
        """Encode history.

        Args:
            x: [B, H, N, 4]
            obs_mask: [B, H, N] bool
            entity_type: [B, N] long or [N] long
        """
        bsz, hist, n_ent, _ = x.shape
        if hist > self.max_history:
            raise ValueError(f"history={hist} exceeds max_history={self.max_history}")
        if n_ent != self.n_entities:
            raise ValueError(f"Expected n_entities={self.n_entities}, got {n_ent}")

        if entity_type.dim() == 1:
            entity_type = entity_type.unsqueeze(0).expand(bsz, -1)

        t_ids = torch.arange(hist, device=x.device)
        n_ids = torch.arange(n_ent, device=x.device)

        x = self.in_proj(x)
        x = x + self.time_emb(t_ids)[None, :, None, :]
        x = x + self.entity_emb(n_ids)[None, None, :, :]
        x = x + self.type_emb(entity_type.clamp(min=0, max=3))[:, None, :, :]

        missing = (~obs_mask).float().unsqueeze(-1)
        x = x + missing * self.missing_emb.view(1, 1, 1, -1)

        causal_mask = self._causal_mask(hist=hist, device=x.device)
        for blk in self.blocks:
            x = blk(x, obs_mask=obs_mask, causal_mask=causal_mask)

        return self.final_ln(x)

    def forward_multi(
        self,
        x: torch.Tensor,
        obs_mask: torch.Tensor,
        entity_type: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        enc = self.encode(x=x, obs_mask=obs_mask, entity_type=entity_type)
        h_last = enc[:, -1, :, :]            # [B, N, D]
        h_entities = h_last[:, : self.n_targets, :]  # [B, 23, D]

        mu = self.mu_head(h_entities).reshape(h_entities.shape[0], self.joint_dim)  # [B, 46]
        u = self.u_head(h_entities).reshape(h_entities.shape[0], self.joint_dim, self.joint_rank)
        u = torch.clamp(u, min=-U_MAX_ABS, max=U_MAX_ABS)

        entity_obs = obs_mask[:, -1, : self.n_targets].float()  # [B, 23]
        denom = entity_obs.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = (h_entities * entity_obs.unsqueeze(-1)).sum(dim=1) / denom
        log_sigma = self.sigma_head(pooled).clamp(min=LOG_SIGMA_MIN, max=LOG_SIGMA_MAX)

        return {
            "mu": mu,
            "log_sigma": log_sigma,
            "U": u,
        }

    def forward_single(
        self,
        x: torch.Tensor,
        obs_mask: torch.Tensor,
        entity_type: torch.Tensor,
        target_entity_idx: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        enc = self.encode(x=x, obs_mask=obs_mask, entity_type=entity_type)
        h_last = enc[:, -1, : self.n_targets, :]  # [B, 23, D]

        batch_idx = torch.arange(h_last.shape[0], device=h_last.device)
        target_entity_idx = target_entity_idx.long().clamp(min=0, max=self.n_targets - 1)
        h_target = h_last[batch_idx, target_entity_idx, :]

        out = self.single_head(h_target)
        mu = out[:, : self.delta_dim]
        log_sigma = out[:, self.delta_dim :].clamp(min=LOG_SIGMA_MIN, max=LOG_SIGMA_MAX)

        return {
            "mu": mu,
            "log_sigma": log_sigma,
        }

    def decode_multi_delta(
        self,
        out: Dict[str, torch.Tensor],
        sample: bool = False,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Return next-step normalized entity deltas [B, 23, D_out]."""
        if sample:
            delta = sample_multi(
                mu=out["mu"],
                log_sigma=out["log_sigma"],
                u=out["U"],
                temperature=temperature,
            )
        else:
            delta = out["mu"]
        return delta.view(delta.shape[0], self.n_targets, self.delta_dim)

    def decode_single_delta(
        self,
        out: Dict[str, torch.Tensor],
        sample: bool = False,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Return next-step normalized single-entity deltas [B, D_out]."""
        if sample:
            return sample_single(
                mu=out["mu"],
                log_sigma=out["log_sigma"],
                temperature=temperature,
            )
        return out["mu"]

    def _step_embeddings(self, steps: int, device: torch.device) -> torch.Tensor:
        if not self.use_oneshot_head or self.oneshot_step_emb is None:
            raise RuntimeError("One-shot head is disabled. Construct model with use_oneshot_head=True.")
        if steps < 1:
            raise ValueError("steps must be >= 1 for one-shot forward.")
        if steps > self.max_oneshot_steps:
            raise ValueError(
                f"Requested steps={steps} exceeds max_oneshot_steps={self.max_oneshot_steps}. "
                "Increase --one-shot-frames."
            )
        step_ids = torch.arange(steps, device=device, dtype=torch.long)
        return self.oneshot_step_emb(step_ids)  # [S, D]

    def _diffusion_timestep_embeddings(self, diffusion_timestep: torch.Tensor) -> torch.Tensor:
        if not self.use_diffusion_head or self.diffusion_timestep_emb is None:
            raise RuntimeError("Diffusion head is disabled. Construct model with use_diffusion_head=True.")
        if diffusion_timestep.dim() != 1:
            raise ValueError(f"Expected diffusion_timestep shape [B], got {tuple(diffusion_timestep.shape)}")
        if diffusion_timestep.numel() == 0:
            raise ValueError("diffusion_timestep cannot be empty.")
        t = diffusion_timestep.long().clamp(min=0, max=self.diffusion_steps - 1)
        return self.diffusion_timestep_emb(t)  # [B, D]

    def _future_causal_mask(self, steps: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(steps, steps, dtype=torch.bool, device=device), diagonal=1)

    def forward_stop_tokens_from_context(
        self,
        *,
        context_h_entities: torch.Tensor,
        traj_norm: torch.Tensor,
        history_stop_tokens: torch.Tensor | None,
        steps: int,
    ) -> torch.Tensor:
        """Causally predict stop tokens from denoised one-shot trajectory proposals.

        Returns logits over 10 classes:
          [continue, 8 concrete restart types, PAD]
        """
        if (
            not self.use_diffusion_head
            or self.diffusion_stop_traj_in is None
            or self.diffusion_stop_ctx_in is None
            or self.diffusion_stop_hist_token_emb is None
            or self.diffusion_stop_blocks is None
            or self.diffusion_stop_ln is None
            or self.diffusion_stop_out is None
        ):
            raise RuntimeError("Diffusion stop head is disabled.")

        if context_h_entities.dim() != 3 or context_h_entities.shape[1] != self.n_targets:
            raise ValueError(
                f"context_h_entities must have shape [B,{self.n_targets},D], got {tuple(context_h_entities.shape)}"
            )
        if traj_norm.shape[1] != steps or traj_norm.shape[2] != self.n_targets or traj_norm.shape[3] != self.delta_dim:
            raise ValueError(
                f"Expected traj_norm [B,{steps},{self.n_targets},{self.delta_dim}], got {tuple(traj_norm.shape)}"
            )

        bsz = traj_norm.shape[0]
        step_emb = self._step_embeddings(steps=steps, device=traj_norm.device)  # [S,D]
        step_emb = step_emb.view(1, steps, self.d_model)

        context_summary = context_h_entities.mean(dim=1)  # [B,D]
        context_proj = self.diffusion_stop_ctx_in(context_summary).unsqueeze(1)  # [B,1,D]

        if history_stop_tokens is None or history_stop_tokens.numel() == 0:
            hist_last = torch.full((bsz,), STOP_CONTINUE, device=traj_norm.device, dtype=torch.long)
        else:
            hist_last = history_stop_tokens[:, -1].long().clamp(min=0, max=STOP_PAD)
        hist_tok = self.diffusion_stop_hist_token_emb(hist_last).unsqueeze(1)  # [B,1,D]

        traj_flat = traj_norm.reshape(bsz, steps, self.n_targets * self.delta_dim)
        x = self.diffusion_stop_traj_in(traj_flat) + step_emb + context_proj + hist_tok

        causal_mask = self._future_causal_mask(steps=steps, device=traj_norm.device)
        for blk in self.diffusion_stop_blocks:
            x = blk(x, causal_mask=causal_mask)
        x = self.diffusion_stop_ln(x)

        # Full logits are over raw token ids [0..10], including unknown_break (9).
        # Expose only predictable classes by dropping unknown_break:
        # [0..8, 10] -> 10 classes.
        logits_full = self.diffusion_stop_out(x)  # [B,S,11]
        logits_pred = torch.cat(
            [logits_full[..., :STOP_UNKNOWN_BREAK], logits_full[..., STOP_PAD : STOP_PAD + 1]],
            dim=-1,
        )
        return logits_pred

    @staticmethod
    def decode_stop_tokens_from_logits(stop_logits: torch.Tensor) -> torch.Tensor:
        """Map 10-way predicted stop classes to raw token ids with PAD=10.

        Pred classes:
          0 continue
          1..8 concrete restart types
          9 PAD
        """
        pred = torch.argmax(stop_logits, dim=-1)
        mapping = torch.tensor(
            [STOP_CONTINUE, 1, 2, 3, 4, 5, 6, 7, 8, STOP_PAD],
            device=pred.device,
            dtype=torch.long,
        )
        return mapping[pred]

    def forward_multi_oneshot(
        self,
        x: torch.Tensor,
        obs_mask: torch.Tensor,
        entity_type: torch.Tensor,
        steps: int,
    ) -> Dict[str, torch.Tensor]:
        """Predict a full horizon in one encoder pass for multi-agent mode."""
        enc = self.encode(x=x, obs_mask=obs_mask, entity_type=entity_type)
        h_last = enc[:, -1, :, :]  # [B, N, D]
        h_entities = h_last[:, : self.n_targets, :]  # [B, 23, D]
        bsz = h_entities.shape[0]

        step_emb = self._step_embeddings(steps=steps, device=x.device)  # [S, D]

        entity_obs = obs_mask[:, -1, : self.n_targets].float()  # [B, 23]
        denom = entity_obs.sum(dim=1, keepdim=True).clamp_min(1.0)

        mu_chunks = []
        log_sigma_chunks = []
        u_chunks = []
        for start in range(0, steps, self.oneshot_chunk_size):
            end = min(steps, start + self.oneshot_chunk_size)
            s = end - start

            s_emb = step_emb[start:end].view(1, s, 1, self.d_model)  # [1, s, 1, D]
            h_step = h_entities.unsqueeze(1) + s_emb  # [B, s, 23, D]

            mu_chunk = self.mu_head(h_step).reshape(bsz, s, self.joint_dim)
            u_chunk = self.u_head(h_step).reshape(bsz, s, self.joint_dim, self.joint_rank)
            u_chunk = torch.clamp(u_chunk, min=-U_MAX_ABS, max=U_MAX_ABS)

            pooled = (h_step * entity_obs[:, None, :, None]).sum(dim=2) / denom[:, None, :]
            log_sigma_chunk = self.sigma_head(pooled).clamp(min=LOG_SIGMA_MIN, max=LOG_SIGMA_MAX)

            mu_chunks.append(mu_chunk)
            u_chunks.append(u_chunk)
            log_sigma_chunks.append(log_sigma_chunk)

        return {
            "mu": torch.cat(mu_chunks, dim=1),  # [B, S, 44]
            "log_sigma": torch.cat(log_sigma_chunks, dim=1),  # [B, S, 44]
            "U": torch.cat(u_chunks, dim=1),  # [B, S, 44, R]
        }

    def forward_single_oneshot(
        self,
        x: torch.Tensor,
        obs_mask: torch.Tensor,
        entity_type: torch.Tensor,
        target_entity_idx: torch.Tensor,
        steps: int,
    ) -> Dict[str, torch.Tensor]:
        """Predict a full horizon in one encoder pass for single-agent mode."""
        enc = self.encode(x=x, obs_mask=obs_mask, entity_type=entity_type)
        h_last = enc[:, -1, : self.n_targets, :]  # [B, 23, D]
        bsz = h_last.shape[0]

        batch_idx = torch.arange(bsz, device=h_last.device)
        target_entity_idx = target_entity_idx.long().clamp(min=0, max=self.n_targets - 1)
        h_target = h_last[batch_idx, target_entity_idx, :]  # [B, D]

        step_emb = self._step_embeddings(steps=steps, device=x.device)  # [S, D]

        mu_chunks = []
        log_sigma_chunks = []
        for start in range(0, steps, self.oneshot_chunk_size):
            end = min(steps, start + self.oneshot_chunk_size)
            s = end - start

            s_emb = step_emb[start:end].view(1, s, self.d_model)  # [1, s, D]
            h_step = h_target.unsqueeze(1) + s_emb  # [B, s, D]
            out = self.single_head(h_step)  # [B, s, 2*D_out]

            mu_chunks.append(out[..., : self.delta_dim])
            log_sigma_chunks.append(out[..., self.delta_dim :].clamp(min=LOG_SIGMA_MIN, max=LOG_SIGMA_MAX))

        return {
            "mu": torch.cat(mu_chunks, dim=1),  # [B, S, D_out]
            "log_sigma": torch.cat(log_sigma_chunks, dim=1),  # [B, S, D_out]
        }

    def forward_multi_oneshot_diffusion(
        self,
        x: torch.Tensor,
        obs_mask: torch.Tensor,
        entity_type: torch.Tensor,
        noisy_target: torch.Tensor,
        diffusion_timestep: torch.Tensor,
        steps: int,
    ) -> Dict[str, torch.Tensor]:
        """Predict diffusion noise for one-shot multi-agent targets."""
        if (
            not self.use_diffusion_head
            or self.diffusion_multi_in is None
            or self.diffusion_multi_out is None
        ):
            raise RuntimeError("Diffusion head is disabled. Construct model with use_diffusion_head=True.")
        if noisy_target.shape[1] != steps:
            raise ValueError(f"noisy_target steps={noisy_target.shape[1]} does not match steps={steps}")
        if noisy_target.shape[2] != self.n_targets or noisy_target.shape[3] != self.delta_dim:
            raise ValueError(
                f"Expected noisy_target shape [B,{steps},{self.n_targets},{self.delta_dim}], got {tuple(noisy_target.shape)}"
            )

        enc = self.encode(x=x, obs_mask=obs_mask, entity_type=entity_type)
        h_last = enc[:, -1, :, :]  # [B, N, D]
        h_entities = h_last[:, : self.n_targets, :]  # [B, 23, D]
        bsz = h_entities.shape[0]

        step_emb = self._step_embeddings(steps=steps, device=x.device)  # [S, D]
        diff_emb = self._diffusion_timestep_embeddings(diffusion_timestep=diffusion_timestep)  # [B, D]

        pred_chunks = []
        for start in range(0, steps, self.oneshot_chunk_size):
            end = min(steps, start + self.oneshot_chunk_size)
            s = end - start

            s_emb = step_emb[start:end].view(1, s, 1, self.d_model)  # [1, s, 1, D]
            t_emb = diff_emb.view(bsz, 1, 1, self.d_model)  # [B, 1, 1, D]
            noisy_chunk = noisy_target[:, start:end]  # [B, s, 23, D_out]
            noisy_proj = self.diffusion_multi_in(noisy_chunk)  # [B, s, 23, D]

            h_step = h_entities.unsqueeze(1) + s_emb + t_emb + noisy_proj
            pred_noise = self.diffusion_multi_out(h_step)  # [B, s, 23, D_out]
            pred_chunks.append(pred_noise)

        return {
            "pred_noise": torch.cat(pred_chunks, dim=1),
            "context_h_entities": h_entities,
        }

    def forward_single_oneshot_diffusion(
        self,
        x: torch.Tensor,
        obs_mask: torch.Tensor,
        entity_type: torch.Tensor,
        target_entity_idx: torch.Tensor,
        noisy_target: torch.Tensor,
        diffusion_timestep: torch.Tensor,
        steps: int,
    ) -> Dict[str, torch.Tensor]:
        """Predict diffusion noise for one-shot single-agent targets."""
        if (
            not self.use_diffusion_head
            or self.diffusion_single_in is None
            or self.diffusion_single_out is None
        ):
            raise RuntimeError("Diffusion head is disabled. Construct model with use_diffusion_head=True.")
        if noisy_target.shape[1] != steps or noisy_target.shape[2] != self.delta_dim:
            raise ValueError(
                f"Expected noisy_target shape [B,{steps},{self.delta_dim}], got {tuple(noisy_target.shape)}"
            )

        enc = self.encode(x=x, obs_mask=obs_mask, entity_type=entity_type)
        h_last = enc[:, -1, : self.n_targets, :]  # [B, 23, D]
        bsz = h_last.shape[0]

        batch_idx = torch.arange(bsz, device=h_last.device)
        target_entity_idx = target_entity_idx.long().clamp(min=0, max=self.n_targets - 1)
        h_target = h_last[batch_idx, target_entity_idx, :]  # [B, D]

        step_emb = self._step_embeddings(steps=steps, device=x.device)  # [S, D]
        diff_emb = self._diffusion_timestep_embeddings(diffusion_timestep=diffusion_timestep)  # [B, D]

        pred_chunks = []
        for start in range(0, steps, self.oneshot_chunk_size):
            end = min(steps, start + self.oneshot_chunk_size)
            s = end - start

            s_emb = step_emb[start:end].view(1, s, self.d_model)  # [1, s, D]
            t_emb = diff_emb.view(bsz, 1, self.d_model)  # [B, 1, D]
            noisy_chunk = noisy_target[:, start:end]  # [B, s, D_out]
            noisy_proj = self.diffusion_single_in(noisy_chunk)  # [B, s, D]

            h_step = h_target.unsqueeze(1) + s_emb + t_emb + noisy_proj
            pred_noise = self.diffusion_single_out(h_step)  # [B, s, D_out]
            pred_chunks.append(pred_noise)

        return {"pred_noise": torch.cat(pred_chunks, dim=1)}

    def decode_multi_delta_seq(
        self,
        out: Dict[str, torch.Tensor],
        sample: bool = False,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Return one-shot normalized entity deltas [B, S, 23, D_out]."""
        if sample:
            bsz, steps, _ = out["mu"].shape
            sampled = sample_multi(
                mu=out["mu"].reshape(bsz * steps, self.joint_dim),
                log_sigma=out["log_sigma"].reshape(bsz * steps, self.joint_dim),
                u=out["U"].reshape(bsz * steps, self.joint_dim, self.joint_rank),
                temperature=temperature,
            )
            return sampled.reshape(bsz, steps, self.n_targets, self.delta_dim)
        return out["mu"].reshape(out["mu"].shape[0], out["mu"].shape[1], self.n_targets, self.delta_dim)

    def decode_single_delta_seq(
        self,
        out: Dict[str, torch.Tensor],
        sample: bool = False,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Return one-shot normalized single-entity deltas [B, S, D_out]."""
        if sample:
            bsz, steps, _ = out["mu"].shape
            sampled = sample_single(
                mu=out["mu"].reshape(bsz * steps, self.delta_dim),
                log_sigma=out["log_sigma"].reshape(bsz * steps, self.delta_dim),
                temperature=temperature,
            )
            return sampled.reshape(bsz, steps, self.delta_dim)
        return out["mu"]


def lowrank_gaussian_nll(
    target: torch.Tensor,
    mu: torch.Tensor,
    log_sigma: torch.Tensor,
    u: torch.Tensor,
) -> torch.Tensor:
    """Negative log-likelihood for Sigma = diag(sigma^2) + U U^T.

    Args:
        target: [B, D]
        mu: [B, D]
        log_sigma: [B, D]
        u: [B, D, R]
    """
    if target.numel() == 0:
        return torch.tensor(0.0, device=mu.device, dtype=torch.float32)

    device_type = target.device.type
    with torch.autocast(device_type=device_type, enabled=False):
        # Keep linear algebra in fp32 for numerical stability and AMP dtype consistency.
        target = target.double()
        mu = mu.double()
        log_sigma = log_sigma.double()
        u = u.double()

        bsz, dim = target.shape
        rank = u.shape[-1]

        sigma2 = torch.exp(2.0 * log_sigma).clamp_min(SIGMA2_FLOOR)  # [B, D]
        sigma2_inv = 1.0 / sigma2
        diff = target - mu  # [B, D]

        a_inv_u = u * sigma2_inv.unsqueeze(-1)  # [B, D, R]
        eye = torch.eye(rank, device=target.device, dtype=torch.float64).unsqueeze(0).expand(bsz, -1, -1)
        m = eye + torch.matmul(u.transpose(1, 2), a_inv_u)  # [B, R, R]
        m = 0.5 * (m + m.transpose(-1, -2))

        jitter = 1e-5
        chol = None
        for _ in range(8):
            try:
                chol = torch.linalg.cholesky(m + jitter * eye)
                break
            except RuntimeError:
                jitter *= 10.0
        if chol is None:
            # Robust fallback for numerically bad batches.
            evals, evecs = torch.linalg.eigh(m)
            evals = torch.clamp(evals, min=1e-6)
            logdet_m = torch.log(evals).sum(dim=1)

            a_inv_diff = diff * sigma2_inv
            b = torch.matmul(u.transpose(1, 2), a_inv_diff.unsqueeze(-1)).squeeze(-1)  # [B, R]
            qt_b = torch.matmul(evecs.transpose(1, 2), b.unsqueeze(-1)).squeeze(-1)
            solve = torch.matmul(evecs, (qt_b / evals).unsqueeze(-1)).squeeze(-1)

            logdet_a = torch.log(sigma2).sum(dim=1)
            quad_diag = (diff * a_inv_diff).sum(dim=1)
            quad_corr = (b * solve).sum(dim=1)
            quad = quad_diag - quad_corr
            nll = 0.5 * (dim * LOG_2PI + logdet_a + logdet_m + quad)
            return nll.float().mean()

        # log |Sigma| = sum log(sigma^2) + log |I + U^T A^{-1} U|
        logdet_a = torch.log(sigma2).sum(dim=1)
        logdet_m = 2.0 * torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)).sum(dim=1)

        # quadratic form via Woodbury
        a_inv_diff = diff * sigma2_inv
        b = torch.matmul(u.transpose(1, 2), a_inv_diff.unsqueeze(-1)).squeeze(-1)  # [B, R]
        solve = torch.cholesky_solve(b.unsqueeze(-1), chol).squeeze(-1)  # [B, R]

        quad_diag = (diff * a_inv_diff).sum(dim=1)
        quad_corr = (b * solve).sum(dim=1)
        quad = quad_diag - quad_corr

        nll = 0.5 * (dim * LOG_2PI + logdet_a + logdet_m + quad)
    return nll.float().mean()


def diagonal_gaussian_nll(
    target: torch.Tensor,
    mu: torch.Tensor,
    log_sigma: torch.Tensor,
) -> torch.Tensor:
    """Diagonal Gaussian NLL for single-agent deltas."""
    with torch.autocast(device_type=target.device.type, enabled=False):
        target = target.double()
        mu = mu.double()
        log_sigma = log_sigma.double()
        sigma2 = torch.exp(2.0 * log_sigma).clamp_min(SIGMA2_FLOOR)
        nll = 0.5 * (2.0 * log_sigma + ((target - mu) ** 2) / sigma2 + LOG_2PI)
    return nll.sum(dim=-1).float().mean()


def sample_multi(
    mu: torch.Tensor,
    log_sigma: torch.Tensor,
    u: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Sample from joint low-rank Gaussian.

    Returns:
        samples [B, D]
    """
    eps_diag = torch.randn_like(mu) * temperature
    eps_rank = torch.randn(mu.shape[0], u.shape[-1], device=mu.device, dtype=mu.dtype) * temperature

    diag_part = torch.exp(log_sigma) * eps_diag
    lowrank_part = torch.bmm(u, eps_rank.unsqueeze(-1)).squeeze(-1)
    return mu + diag_part + lowrank_part


def sample_single(
    mu: torch.Tensor,
    log_sigma: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    eps = torch.randn_like(mu) * temperature
    return mu + torch.exp(log_sigma) * eps
