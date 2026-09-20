#!/usr/bin/env python3
"""ballAtTouch full-trajectory model."""

from __future__ import annotations

import torch
import torch.nn as nn


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + h
        x = x + self.ffn(self.norm2(x))
        return x


class FullTrajectoryBallAtTouchModel(nn.Module):
    """Predicts future ball positions until the next observed touch."""

    def __init__(
        self,
        *,
        ball_dim: int = 8,
        self_dim: int = 32,
        other_dim: int = 32,
        k_other: int = 8,
        max_traj_len: int = 160,
        d_model: int = 192,
        n_heads: int = 4,
        n_layers: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.k_other = int(k_other)
        self.max_traj_len = int(max_traj_len)

        self.ball_embed = nn.Linear(ball_dim, d_model)
        self.self_embed = nn.Linear(self_dim, d_model)
        self.other_embed = nn.Linear(other_dim, d_model)

        n_tokens = self.k_other + 2
        self.pos_embed = nn.Parameter(torch.randn(1, n_tokens, d_model) * 0.02)

        self.blocks = nn.Sequential(
            *[TransformerBlock(d_model, n_heads, dropout) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

        self.trunk = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.traj_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, self.max_traj_len * 3),
        )
        self.stop_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, self.max_traj_len),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        ball_feats: torch.Tensor,
        self_feats: torch.Tensor,
        other_feats: torch.Tensor,
    ) -> dict:
        ball_tok = self.ball_embed(ball_feats).unsqueeze(1)
        self_tok = self.self_embed(self_feats).unsqueeze(1)
        other_tok = self.other_embed(other_feats)

        tokens = torch.cat([ball_tok, self_tok, other_tok], dim=1)
        tokens = tokens + self.pos_embed
        tokens = self.blocks(tokens)
        tokens = self.norm(tokens)

        h = self.trunk(tokens[:, 1, :])
        traj = self.traj_head(h).view(ball_feats.shape[0], self.max_traj_len, 3)
        stop_logits = self.stop_head(h)
        return {"traj": traj, "stop_logits": stop_logits}

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
