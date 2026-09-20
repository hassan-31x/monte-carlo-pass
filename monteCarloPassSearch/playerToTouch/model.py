#!/usr/bin/env python3
"""playerToTouch: transformer classifier predicting who touches ball next.

Architecture:
  - Ball token:    Linear(BALL_DIM, d_model)         → [B, 1, d_model]
  - Player tokens: Linear(PLAYER_HIST_DIM, d_model)  → [B, K, d_model]
  - No-touch token: learned embedding                 → [B, 1, d_model]
  - Total tokens: K+2  (ball + K players + no-touch)
  - N transformer encoder layers
  - Output: Linear(d_model, 1) on K player tokens and no-touch token
            → logits [B, K+1] (K player probs + no-touch)

K+1 logits → CrossEntropyLoss (label in [0..K], K = no touch)
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerBlock(nn.Module):
    """Pre-norm transformer block with multi-head self-attention + FFN."""

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
        x2, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + x2
        x = x + self.ffn(self.norm2(x))
        return x


class TouchPredictorModel(nn.Module):
    """Predicts P(player_k touches ball next) for k in [0..K] (K = no touch).

    Args:
        ball_dim:       dimension of ball feature vector (default 6)
        player_hist_dim: dimension of flattened player history (default H*4=32)
        k:              number of nearest players to consider (default 8)
        d_model:        transformer hidden dim
        n_heads:        attention heads
        n_layers:       number of transformer blocks
        dropout:        dropout rate
    """

    def __init__(
        self,
        ball_dim:        int = 8,
        player_hist_dim: int = 32,
        k:               int = 8,
        d_model:         int = 128,
        n_heads:         int = 4,
        n_layers:        int = 4,
        dropout:         float = 0.1,
    ) -> None:
        super().__init__()
        self.k = k

        self.ball_embed   = nn.Linear(ball_dim, d_model)
        self.player_embed = nn.Linear(player_hist_dim, d_model)
        # Learnable no-touch token
        self.no_touch_tok = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        # Positional embeddings: 1 ball + K players + 1 no-touch = K+2 positions
        self.pos_embed = nn.Parameter(torch.randn(1, k + 2, d_model) * 0.02)

        self.blocks = nn.Sequential(*[
            TransformerBlock(d_model, n_heads, dropout) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        # Head produces 1 logit per token; we use K player tokens + no-touch token
        self.out_head = nn.Linear(d_model, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        ball_feats:   torch.Tensor,  # [B, ball_dim]
        player_feats: torch.Tensor,  # [B, K, player_hist_dim]
    ) -> torch.Tensor:
        """Returns logits [B, K+1]. Caller applies CrossEntropyLoss."""
        B = ball_feats.size(0)

        ball_tok    = self.ball_embed(ball_feats).unsqueeze(1)    # [B, 1, D]
        player_tok  = self.player_embed(player_feats)              # [B, K, D]
        no_touch_tok = self.no_touch_tok.expand(B, -1, -1)        # [B, 1, D]

        # Sequence: [ball | p0 | p1 | ... | p_{K-1} | no-touch]
        tokens = torch.cat([ball_tok, player_tok, no_touch_tok], dim=1)  # [B, K+2, D]
        tokens = tokens + self.pos_embed

        tokens = self.blocks(tokens)
        tokens = self.norm(tokens)

        # Logits for player tokens [1..K] and no-touch token [K+1]
        player_logits   = self.out_head(tokens[:, 1:self.k + 1, :]).squeeze(-1)  # [B, K]
        no_touch_logit  = self.out_head(tokens[:, self.k + 1, :])                # [B, 1]
        logits = torch.cat([player_logits, no_touch_logit], dim=1)           # [B, K+1]
        return logits

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SurvivalTouchModel(nn.Module):
    """Discrete-time survival model for ball touch prediction.

    Outputs:
      - hazard h(t) = P(touch at t | no touch before t)  via ball token → sigmoid
      - player_logits = unnormalised log-probs over K players  (who touches)

    Token sequence: [ball | p0 | ... | p_{K-1}]  (K+1 tokens, no no-touch token)
    """

    def __init__(
        self,
        ball_dim:        int = 8,
        player_hist_dim: int = 32,
        k:               int = 8,
        d_model:         int = 128,
        n_heads:         int = 4,
        n_layers:        int = 4,
        dropout:         float = 0.1,
    ) -> None:
        super().__init__()
        self.k = k

        self.ball_embed   = nn.Linear(ball_dim, d_model)
        self.player_embed = nn.Linear(player_hist_dim, d_model)
        # K+1 positions: 1 ball + K players (no no-touch token)
        self.pos_embed = nn.Parameter(torch.randn(1, k + 1, d_model) * 0.02)

        self.blocks = nn.Sequential(*[
            TransformerBlock(d_model, n_heads, dropout) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

        # Hazard head: ball token → scalar logit → sigmoid gives h(t)
        self.hazard_head = nn.Linear(d_model, 1)
        # Player head: each player token → scalar logit
        self.player_head = nn.Linear(d_model, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Init hazard bias to log(1/50) ≈ -3.9 so baseline hazard ≈ 2%
        nn.init.constant_(self.hazard_head.bias, -3.9)

    def forward(
        self,
        ball_feats:   torch.Tensor,  # [B, ball_dim]
        player_feats: torch.Tensor,  # [B, K, player_hist_dim]
    ) -> dict:
        """Returns {"hazard": [B,1], "player_logits": [B,K]}."""
        B = ball_feats.size(0)

        ball_tok   = self.ball_embed(ball_feats).unsqueeze(1)    # [B, 1, D]
        player_tok = self.player_embed(player_feats)              # [B, K, D]

        # Sequence: [ball | p0 | p1 | ... | p_{K-1}]
        tokens = torch.cat([ball_tok, player_tok], dim=1)  # [B, K+1, D]
        tokens = tokens + self.pos_embed

        tokens = self.blocks(tokens)
        tokens = self.norm(tokens)

        # Hazard from ball token (index 0)
        hazard_logit = self.hazard_head(tokens[:, 0, :])           # [B, 1]
        # Player logits from player tokens [1..K]
        player_logits = self.player_head(tokens[:, 1:, :]).squeeze(-1)  # [B, K]

        return {"hazard": hazard_logit, "player_logits": player_logits}

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
