#!/usr/bin/env python3
"""ballAtTouch: transformer regressor predicting outgoing ball velocity at touch.

Architecture:
  - Ball token:   Linear(BALL_DIM, d_model)  → [B, 1, d_model]
  - Self token:   Linear(SELF_DIM,  d_model)  → [B, 1, d_model]
  - Other tokens: Linear(OTHER_DIM, d_model)  → [B, K_other, d_model]
  - Total: K_other + 2 tokens
  - N transformer encoder layers (full attention)
  - Output: regression head on "self" token → 3D velocity [vx, vy, vz]

Loss: SmoothL1 (Huber) on normalised velocity components.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn  = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                            batch_first=True)
        self.ffn   = nn.Sequential(
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


class BallAtTouchModel(nn.Module):
    """Predicts 3D outgoing ball velocity at moment of touch.

    Args:
        ball_dim:   dimension of ball feature vector  (default 6)
        self_dim:   dimension of self (touching player) feature vector (default 32)
        other_dim:  per-player feature dimension for other players (default 32)
        k_other:    number of 'other' player tokens (default 8)
        d_model:    transformer hidden dim
        n_heads:    attention heads
        n_layers:   transformer block count
        dropout:    dropout rate
    """

    def __init__(
        self,
        ball_dim:  int = 8,
        self_dim:  int = 32,
        other_dim: int = 32,
        k_other:   int = 8,
        d_model:   int = 128,
        n_heads:   int = 4,
        n_layers:  int = 4,
        dropout:   float = 0.1,
    ) -> None:
        super().__init__()
        self.k_other = k_other

        self.ball_embed  = nn.Linear(ball_dim,  d_model)
        self.self_embed  = nn.Linear(self_dim,  d_model)
        self.other_embed = nn.Linear(other_dim, d_model)

        # Positional embeddings: [ball | self | other0 .. other_{k-1}]
        n_tokens = k_other + 2
        self.pos_embed = nn.Parameter(torch.randn(1, n_tokens, d_model) * 0.02)

        self.blocks = nn.Sequential(*[
            TransformerBlock(d_model, n_heads, dropout) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        # Regression head on self token (index 1)
        self.out_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 3),  # [vx, vy, vz]
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
        ball_feats:  torch.Tensor,   # [B, ball_dim]
        self_feats:  torch.Tensor,   # [B, self_dim]
        other_feats: torch.Tensor,   # [B, k_other, other_dim]
    ) -> torch.Tensor:
        """Returns predicted velocity [B, 3] (normalised units)."""
        ball_tok  = self.ball_embed(ball_feats).unsqueeze(1)   # [B, 1, D]
        self_tok  = self.self_embed(self_feats).unsqueeze(1)   # [B, 1, D]
        other_tok = self.other_embed(other_feats)              # [B, K, D]

        tokens = torch.cat([ball_tok, self_tok, other_tok], dim=1)  # [B, K+2, D]
        tokens = tokens + self.pos_embed

        tokens = self.blocks(tokens)
        tokens = self.norm(tokens)

        # Predict from self token (index 1)
        vel_pred = self.out_head(tokens[:, 1, :])  # [B, 3]
        return vel_pred

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class GaussianBallAtTouchModel(nn.Module):
    """Predicts 3D outgoing ball velocity as a diagonal Gaussian.

    Same transformer architecture as BallAtTouchModel, but the head outputs
    mu (3) and log_sigma (3) so we can train with Gaussian NLL and sample
    at inference time.

    Forward returns {"mu": [B,3], "log_sigma": [B,3]}.
    """

    def __init__(
        self,
        ball_dim:  int = 8,
        self_dim:  int = 32,
        other_dim: int = 32,
        k_other:   int = 8,
        d_model:   int = 128,
        n_heads:   int = 4,
        n_layers:  int = 4,
        dropout:   float = 0.1,
    ) -> None:
        super().__init__()
        self.k_other = k_other

        self.ball_embed  = nn.Linear(ball_dim,  d_model)
        self.self_embed  = nn.Linear(self_dim,  d_model)
        self.other_embed = nn.Linear(other_dim, d_model)

        n_tokens = k_other + 2
        self.pos_embed = nn.Parameter(torch.randn(1, n_tokens, d_model) * 0.02)

        self.blocks = nn.Sequential(*[
            TransformerBlock(d_model, n_heads, dropout) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

        # Shared trunk, separate mu / log_sigma heads
        self.trunk = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
        )
        self.mu_head       = nn.Linear(d_model // 2, 3)
        self.log_sigma_head = nn.Linear(d_model // 2, 3)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Init log_sigma bias to log(0.3) ≈ -1.2 so initial sigma ≈ 0.3
        # (matches rough std of normalised velocity)
        nn.init.constant_(self.log_sigma_head.bias, -1.2)

    def forward(
        self,
        ball_feats:  torch.Tensor,   # [B, ball_dim]
        self_feats:  torch.Tensor,   # [B, self_dim]
        other_feats: torch.Tensor,   # [B, k_other, other_dim]
    ) -> dict:
        """Returns {"mu": [B,3], "log_sigma": [B,3]}."""
        ball_tok  = self.ball_embed(ball_feats).unsqueeze(1)
        self_tok  = self.self_embed(self_feats).unsqueeze(1)
        other_tok = self.other_embed(other_feats)

        tokens = torch.cat([ball_tok, self_tok, other_tok], dim=1)
        tokens = tokens + self.pos_embed

        tokens = self.blocks(tokens)
        tokens = self.norm(tokens)

        h = self.trunk(tokens[:, 1, :])              # [B, d_model//2]
        mu        = self.mu_head(h)                   # [B, 3]
        log_sigma = self.log_sigma_head(h)            # [B, 3]
        # Clamp for numerical stability: sigma in [exp(-4), exp(2)] ≈ [0.018, 7.4]
        log_sigma = log_sigma.clamp(-4.0, 2.0)

        return {"mu": mu, "log_sigma": log_sigma}

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
