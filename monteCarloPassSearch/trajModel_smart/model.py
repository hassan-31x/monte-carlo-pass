#!/usr/bin/env python3
"""SMART trajectory transformer: discrete motion tokens with cross-entropy classification.

Architecture: 6-layer space-time transformer (~7.4M params).
- Token embedding (separate for player/ball) + Fourier position encoding + entity type embedding
- Causal temporal attention with RoPE + bidirectional spatial attention
- Output: classification heads for player (2048) and ball (1024) vocab

No Gaussian regression, no sigma collapse, no free-running degradation.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- RoPE ----------

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    return torch.stack((-x_odd, x_even), dim=-1).flatten(-2)


class MultiheadSelfAttention(nn.Module):
    """Self-attention with optional RoPE."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, use_rope: bool = False) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        assert self.head_dim % 2 == 0, "head_dim must be even for RoPE"
        self.use_rope = use_rope
        self.dropout = dropout

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def _rope_cos_sin(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        pos = torch.arange(seq_len, device=device, dtype=torch.float32)
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, self.head_dim, 2, device=device, dtype=torch.float32) / float(self.head_dim)))
        freqs = torch.outer(pos, inv_freq)
        emb = torch.repeat_interleave(freqs, 2, dim=-1)
        cos = emb.cos().to(dtype=dtype).view(1, 1, seq_len, self.head_dim)
        sin = emb.sin().to(dtype=dtype).view(1, 1, seq_len, self.head_dim)
        return cos, sin

    def forward(self, x: torch.Tensor, attn_mask=None, key_padding_mask=None) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        q = self.q_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        if self.use_rope:
            cos, sin = self._rope_cos_sin(seq_len, x.device, q.dtype)
            q = q * cos + _rotate_half(q) * sin
            k = k * cos + _rotate_half(k) * sin

        add_mask = None
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                add_mask = torch.zeros((1, 1, seq_len, seq_len), device=x.device, dtype=q.dtype)
                add_mask = add_mask.masked_fill(attn_mask.view(1, 1, seq_len, seq_len), float("-inf"))
            else:
                add_mask = attn_mask.to(device=x.device, dtype=q.dtype).view(1, 1, seq_len, seq_len)

        if key_padding_mask is not None:
            pad_mask = key_padding_mask.view(bsz, 1, 1, seq_len)
            if add_mask is None:
                add_mask = torch.zeros((bsz, 1, seq_len, seq_len), device=x.device, dtype=q.dtype)
            elif add_mask.shape[0] == 1:
                add_mask = add_mask.expand(bsz, -1, -1, -1).clone()
            add_mask = add_mask.masked_fill(pad_mask, float("-inf"))

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=add_mask,
            dropout_p=self.dropout if self.training else 0.0, is_causal=False,
        )
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)
        return self.out_proj(out)


class SpaceTimeBlock(nn.Module):
    """Transformer block: causal temporal attention + bidirectional spatial attention + MLP."""

    def __init__(self, d_model: int, n_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1,
                 use_rope: bool = True) -> None:
        super().__init__()
        d_ff = int(d_model * mlp_ratio)

        self.temporal_ln = nn.LayerNorm(d_model)
        self.spatial_ln = nn.LayerNorm(d_model)
        self.mlp_ln = nn.LayerNorm(d_model)

        self.temporal_attn = MultiheadSelfAttention(d_model, n_heads, dropout, use_rope=use_rope)
        self.spatial_attn = MultiheadSelfAttention(d_model, n_heads, dropout, use_rope=False)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, obs_mask: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, N, D] entity representations
            obs_mask: [B, T, N] bool mask (True = observed)
            causal_mask: [T, T] bool upper-triangular (True = blocked)
        """
        bsz, T, N, D = x.shape

        # Temporal attention: per entity across time
        xt = x.permute(0, 2, 1, 3).reshape(bsz * N, T, D)
        tm = ~obs_mask.permute(0, 2, 1).reshape(bsz * N, T)
        all_miss = tm.all(dim=1)
        if all_miss.any():
            tm = tm.clone()
            tm[all_miss] = False
        xt_in = self.temporal_ln(xt)
        xt_out = self.temporal_attn(xt_in, attn_mask=causal_mask, key_padding_mask=tm)
        xt = xt + self.drop(xt_out)
        x = xt.reshape(bsz, N, T, D).permute(0, 2, 1, 3)

        # Spatial attention: all entities within each timestep (bidirectional)
        xs = x.reshape(bsz * T, N, D)
        sm = ~obs_mask.reshape(bsz * T, N)
        all_miss_s = sm.all(dim=1)
        if all_miss_s.any():
            sm = sm.clone()
            sm[all_miss_s] = False
        xs_in = self.spatial_ln(xs)
        xs_out = self.spatial_attn(xs_in, attn_mask=None, key_padding_mask=sm)
        xs = xs + self.drop(xs_out)
        x = xs.reshape(bsz, T, N, D)

        # MLP
        return x + self.mlp(self.mlp_ln(x))


# ---------- Fourier position encoding ----------

class FourierPositionEncoder(nn.Module):
    """Encode absolute (x, y) position using sinusoidal Fourier features -> d_model."""

    def __init__(self, d_model: int, n_freq: int = 64) -> None:
        super().__init__()
        self.n_freq = n_freq
        # Input: 2D (x, y) -> n_freq*2*2 = n_freq*4 Fourier features -> d_model
        self.proj = nn.Linear(n_freq * 4, d_model)

        # Fixed frequency bands (not learned)
        freqs = torch.exp(torch.linspace(math.log(1.0), math.log(100.0), n_freq))
        self.register_buffer("freqs", freqs)  # [n_freq]

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        """
        Args:
            xy: [..., 2] positions in metres

        Returns:
            [..., d_model] Fourier position embeddings
        """
        # Scale positions by frequency bands: [..., 2] x [n_freq] -> [..., 2, n_freq]
        scaled = xy.unsqueeze(-1) * self.freqs  # [..., 2, n_freq]
        # Concat sin and cos: [..., 2, n_freq*2]
        fourier = torch.cat([scaled.sin(), scaled.cos()], dim=-1)
        # Flatten last two dims: [..., 4*n_freq]
        fourier = fourier.flatten(-2)
        return self.proj(fourier)


# ---------- Main Model ----------

class SMARTTransformer(nn.Module):
    """SMART decoder-only space-time transformer for soccer trajectory prediction.

    Discretizes trajectories into motion tokens and uses cross-entropy classification
    instead of Gaussian regression.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        vocab_player: int = 2048,
        vocab_ball: int = 1024,
        n_entity_types: int = 3,  # 0=home, 1=away, 2=ball
        max_seq_len: int = 64,
        use_rope: bool = True,
        n_fourier_freq: int = 64,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.vocab_player = vocab_player
        self.vocab_ball = vocab_ball
        self.max_seq_len = max_seq_len

        # Token embeddings (separate for player and ball)
        self.player_token_emb = nn.Embedding(vocab_player, d_model)
        self.ball_token_emb = nn.Embedding(vocab_ball, d_model)

        # Positional encodings
        self.pos_encoder = FourierPositionEncoder(d_model, n_fourier_freq)
        self.type_emb = nn.Embedding(n_entity_types, d_model)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            SpaceTimeBlock(d_model, n_heads, mlp_ratio, dropout, use_rope=use_rope)
            for _ in range(n_layers)
        ])
        self.final_ln = nn.LayerNorm(d_model)

        # Classification output heads
        self.player_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Linear(d_model, vocab_player),
        )
        self.ball_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Linear(d_model, vocab_ball),
        )

    def _causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        """Upper-triangular bool mask: True = blocked position."""
        return torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1)

    def forward(
        self,
        player_tokens: torch.Tensor,   # [B, T, 22] int64
        ball_tokens: torch.Tensor,      # [B, T] int64
        player_pos: torch.Tensor,       # [B, T, 22, 2] float
        ball_pos: torch.Tensor,         # [B, T, 2] float
        entity_types: torch.Tensor,     # [B, 23] int64 (or [23])
        obs_mask: torch.Tensor,         # [B, T, 23] bool
        perm: Optional[torch.Tensor] = None,  # [B, T, 22] int64 permutation indices
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass producing next-token logits for all entities.

        The sequence layout per timestep is [BALL, P1, P2, ..., P22] where
        player order follows the permutation.

        Args:
            player_tokens: token IDs for 22 players
            ball_tokens: token IDs for ball
            player_pos: (x,y) positions for players at token boundaries
            ball_pos: (x,y) positions for ball at token boundaries
            entity_types: entity type per slot [B, 23]
            obs_mask: observation mask
            perm: player permutation indices per timestep. If None, identity order.

        Returns:
            player_logits: [B, T, 22, vocab_player]
            ball_logits: [B, T, vocab_ball]
        """
        B, T, N_players = player_tokens.shape
        device = player_tokens.device

        if entity_types.dim() == 1:
            entity_types = entity_types.unsqueeze(0).expand(B, -1)

        # Apply permutation to player tokens and positions
        if perm is not None:
            # perm: [B, T, 22] -> gather along player dim
            player_tokens = torch.gather(player_tokens, 2, perm)
            player_pos = torch.gather(player_pos, 2, perm.unsqueeze(-1).expand(-1, -1, -1, 2))
            # Also permute player obs_mask (columns 1..22)
            player_obs = obs_mask[:, :, 1:]  # [B, T, 22]
            player_obs = torch.gather(player_obs, 2, perm)
            obs_mask = torch.cat([obs_mask[:, :, :1], player_obs], dim=2)

        # Embed tokens
        player_emb = self.player_token_emb(player_tokens)  # [B, T, 22, D]
        ball_emb = self.ball_token_emb(ball_tokens)         # [B, T, D]

        # Fourier position encoding
        player_pos_emb = self.pos_encoder(player_pos)       # [B, T, 22, D]
        ball_pos_emb = self.pos_encoder(ball_pos)            # [B, T, D]

        # Entity type embedding: ball is slot 0, players are slots 1..22
        # entity_types: [B, 23] where 0=ball type at index 0, player types at 1..22
        ball_type_emb = self.type_emb(entity_types[:, 0])               # [B, D]
        player_type_emb = self.type_emb(entity_types[:, 1:])            # [B, 22, D]

        # Combine embeddings: token + position + type
        ball_h = ball_emb + ball_pos_emb + ball_type_emb[:, None, :]                      # [B, T, D]
        player_h = player_emb + player_pos_emb + player_type_emb[:, None, :, :]            # [B, T, 22, D]

        # Assemble sequence: [BALL, P1, ..., P22] per timestep -> [B, T, 23, D]
        h = torch.cat([ball_h.unsqueeze(2), player_h], dim=2)  # [B, T, 23, D]

        # Transformer blocks
        causal_mask = self._causal_mask(T, device)
        for block in self.blocks:
            h = block(h, obs_mask, causal_mask)
        h = self.final_ln(h)

        # Split back into ball and player representations
        ball_repr = h[:, :, 0, :]       # [B, T, D]
        player_repr = h[:, :, 1:, :]    # [B, T, 22, D]

        # Classification heads
        player_logits = self.player_head(player_repr)   # [B, T, 22, vocab_player]
        ball_logits = self.ball_head(ball_repr)          # [B, T, vocab_ball]

        # Un-permute player logits back to canonical order
        if perm is not None:
            inv_perm = torch.zeros_like(perm)
            inv_perm.scatter_(2, perm, torch.arange(N_players, device=device).view(1, 1, -1).expand_as(perm))
            player_logits = torch.gather(
                player_logits, 2,
                inv_perm.unsqueeze(-1).expand(-1, -1, -1, self.vocab_player),
            )

        return player_logits, ball_logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
