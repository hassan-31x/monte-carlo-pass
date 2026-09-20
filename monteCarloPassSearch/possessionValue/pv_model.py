#!/usr/bin/env python3
"""Possession Value (PV) Transformer Model.

Takes past k frames of tracking data (23 entities x 6 features) and predicts
possession value for each team:
  PV = P(goal within 10s) ≈ P(shot within 10s) × E[xG | shot]
  pv_home - pv_away gives net possession value.
"""

import torch
import torch.nn as nn


class EntityEncoder(nn.Module):
    def __init__(self, feat_dim: int, d_model: int, n_entity_types: int = 4):
        super().__init__()
        self.feat_proj = nn.Linear(feat_dim, d_model)
        self.entity_type_emb = nn.Embedding(n_entity_types, d_model)
        self.missing_emb = nn.Parameter(torch.randn(d_model) * 0.02)

    def forward(self, x, mask, entity_type):
        tokens = self.feat_proj(x)
        if entity_type.dim() == 1:
            entity_type = entity_type.unsqueeze(0).expand(x.shape[0], -1)
        type_emb = self.entity_type_emb(entity_type)
        tokens = tokens + type_emb.unsqueeze(1)
        mask_expanded = mask.unsqueeze(-1)
        tokens = tokens * mask_expanded + self.missing_emb * (1 - mask_expanded)
        return tokens


class SpatialAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

    def forward(self, x):
        h = self.norm(x)
        h, _ = self.attn(h, h, h)
        return x + h


class TemporalAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

    def forward(self, x, causal_mask=None):
        h = self.norm(x)
        h, _ = self.attn(h, h, h, attn_mask=causal_mask)
        return x + h


class SpaceTimeBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.spatial = SpatialAttentionBlock(d_model, n_heads, dropout)
        self.temporal = TemporalAttentionBlock(d_model, n_heads, dropout)
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, entity_tokens, frame_tokens, causal_mask=None):
        B, T, N, D = entity_tokens.shape
        spatial_in = entity_tokens.reshape(B * T, N, D)
        spatial_out = self.spatial(spatial_in)
        entity_tokens = spatial_out.reshape(B, T, N, D)
        frame_tokens = entity_tokens.mean(dim=2)
        frame_tokens = self.temporal(frame_tokens, causal_mask=causal_mask)
        frame_tokens = frame_tokens + self.ffn(self.norm_ffn(frame_tokens))
        return entity_tokens, frame_tokens


class PossessionValueTransformer(nn.Module):
    """
    Possession Value transformer.
    Input: tracking data windows [B, T, 23, 6]
    Output: pv_home, pv_away ∈ [0, 1] — P(goal within horizon) per team.
    """

    def __init__(
        self,
        feat_dim: int = 6,
        n_entities: int = 23,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        dropout: float = 0.15,
        max_seq_len: int = 128,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_entities = n_entities

        self.entity_encoder = EntityEncoder(feat_dim, d_model)
        self.time_emb = nn.Embedding(max_seq_len, d_model)

        self.blocks = nn.ModuleList([
            SpaceTimeBlock(d_model, n_heads, dropout) for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

        # Single head per team: predicts P(goal within horizon)
        self.pv_home_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )
        self.pv_away_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        # Bias heads toward low values (goals are rare ~2% of windows)
        for head in [self.pv_home_head, self.pv_away_head]:
            nn.init.constant_(head[-1].bias, -4.0)

    def _causal_mask(self, T: int, device: torch.device):
        mask = torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()
        return mask.float().masked_fill(mask, float("-inf"))

    def forward(self, features, mask, entity_type):
        """
        features: [B, T, N, feat_dim]
        mask: [B, T, N] (float)
        entity_type: [B, N] or [N]
        Returns dict with: pv_home, pv_away, pv (net), logit_home, logit_away
        """
        B, T, N, F = features.shape

        entity_tokens = self.entity_encoder(features, mask, entity_type)

        time_ids = torch.arange(T, device=features.device)
        time_emb = self.time_emb(time_ids)
        entity_tokens = entity_tokens + time_emb.unsqueeze(0).unsqueeze(2)

        frame_tokens = entity_tokens.mean(dim=2)
        causal_mask = self._causal_mask(T, features.device)

        for block in self.blocks:
            entity_tokens, frame_tokens = block(entity_tokens, frame_tokens, causal_mask)

        frame_tokens = self.norm(frame_tokens)
        last_frame = frame_tokens[:, -1]  # [B, D]

        logit_home = self.pv_home_head(last_frame).squeeze(-1)
        logit_away = self.pv_away_head(last_frame).squeeze(-1)

        pv_home = torch.sigmoid(logit_home)
        pv_away = torch.sigmoid(logit_away)

        return {
            "pv_home": pv_home,
            "pv_away": pv_away,
            "pv": pv_home - pv_away,
            "logit_home": logit_home,
            "logit_away": logit_away,
        }
