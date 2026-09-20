#!/usr/bin/env python3
"""xG Transformer Model - predicts P(goal) from shot features."""

import math
import torch
import torch.nn as nn


class FeatureTokenizer(nn.Module):
    """Convert a fixed-size feature vector into a sequence of tokens."""

    def __init__(self, n_features: int, d_model: int):
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model
        # Each feature gets its own projection and type embedding
        self.feat_proj = nn.Linear(1, d_model)
        self.feat_type_emb = nn.Embedding(n_features, d_model)
        # Learnable mask embedding for unknown features
        self.mask_emb = nn.Parameter(torch.randn(d_model) * 0.02)
        # CLS token
        self.cls_token = nn.Parameter(torch.randn(d_model) * 0.02)

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        """
        x: [B, n_features] - normalized feature values
        mask: [B, n_features] - 1 for known, 0 for unknown
        Returns: [B, n_features+1, d_model] - tokens with CLS prepended
        """
        B = x.shape[0]
        # Project each feature to d_model
        # x: [B, F] -> [B, F, 1] -> feat_proj -> [B, F, d_model]
        tokens = self.feat_proj(x.unsqueeze(-1))  # [B, F, d_model]

        # Add feature type embeddings
        feat_ids = torch.arange(self.n_features, device=x.device)
        tokens = tokens + self.feat_type_emb(feat_ids).unsqueeze(0)

        # Replace unknown features with mask embedding
        mask_expanded = mask.unsqueeze(-1)  # [B, F, 1]
        tokens = tokens * mask_expanded + self.mask_emb.unsqueeze(0).unsqueeze(0) * (1 - mask_expanded)

        # Prepend CLS token
        cls = self.cls_token.unsqueeze(0).unsqueeze(0).expand(B, 1, -1)
        tokens = torch.cat([cls, tokens], dim=1)  # [B, F+1, d_model]

        return tokens


class TransformerBlock(nn.Module):
    """Standard pre-norm transformer encoder block."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor):
        h = self.norm1(x)
        h, _ = self.attn(h, h, h)
        x = x + h
        x = x + self.ffn(self.norm2(x))
        return x


class xGTransformer(nn.Module):
    """
    Transformer-based xG model.
    Tokenizes shot features, processes with transformer, predicts P(goal).
    """

    def __init__(
        self,
        n_features: int = 10,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.tokenizer = FeatureTokenizer(n_features, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, dropout) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        """
        x: [B, n_features] - normalized feature values
        mask: [B, n_features] - feature known mask
        Returns: [B, 1] logits (apply sigmoid for probability)
        """
        tokens = self.tokenizer(x, mask)  # [B, F+1, d_model]
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)
        cls_out = tokens[:, 0]  # CLS token
        return self.head(cls_out)  # [B, 1]

    def predict_xg(self, x: torch.Tensor, mask: torch.Tensor):
        """Returns P(goal) in [0, 1]."""
        return torch.sigmoid(self.forward(x, mask)).squeeze(-1)
