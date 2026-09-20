#!/usr/bin/env python3
"""Expected Threat (xT) Transformer Model.

Takes past k frames of tracking data (23 entities x 6 features) and predicts
threat values for each team:
  xT = home_threat - away_threat
  where threat = P(shot) * E[xG | shot]
"""

import math
import torch
import torch.nn as nn


class EntityEncoder(nn.Module):
    """Encode per-entity features into d_model dimension."""

    def __init__(self, feat_dim: int, d_model: int, n_entity_types: int = 4):
        super().__init__()
        self.feat_proj = nn.Linear(feat_dim, d_model)
        self.entity_type_emb = nn.Embedding(n_entity_types, d_model)
        # Learnable embedding for unobserved entities
        self.missing_emb = nn.Parameter(torch.randn(d_model) * 0.02)

    def forward(self, x, mask, entity_type):
        """
        x: [B, T, N, feat_dim] - tracking features
        mask: [B, T, N] - observation mask
        entity_type: [B, N] or [N] - entity type ids (0=home, 1=away, 2=ball, 3=pad)
        Returns: [B, T, N, d_model]
        """
        tokens = self.feat_proj(x)  # [B, T, N, d_model]

        # Add entity type embeddings
        if entity_type.dim() == 1:
            entity_type = entity_type.unsqueeze(0).expand(x.shape[0], -1)
        type_emb = self.entity_type_emb(entity_type)  # [B, N, d_model]
        tokens = tokens + type_emb.unsqueeze(1)  # broadcast over T

        # Replace unobserved with missing embedding
        mask_expanded = mask.unsqueeze(-1)  # [B, T, N, 1]
        tokens = tokens * mask_expanded + self.missing_emb * (1 - mask_expanded)

        return tokens


class SpatialAttentionBlock(nn.Module):
    """Attention across entities within each frame."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

    def forward(self, x):
        """x: [B*T, N, d_model]"""
        h = self.norm(x)
        h, _ = self.attn(h, h, h)
        return x + h


class TemporalAttentionBlock(nn.Module):
    """Causal attention across time steps."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

    def forward(self, x, causal_mask=None):
        """x: [B, T, d_model]"""
        h = self.norm(x)
        h, _ = self.attn(h, h, h, attn_mask=causal_mask)
        return x + h


class SpaceTimeBlock(nn.Module):
    """Combined spatial + temporal attention + FFN."""

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
        """
        entity_tokens: [B, T, N, d_model] - per-entity tokens
        frame_tokens: [B, T, d_model] - per-frame tokens (pooled from entities)
        Returns: updated entity_tokens, frame_tokens
        """
        B, T, N, D = entity_tokens.shape

        # Spatial attention: attend across entities per frame
        spatial_in = entity_tokens.reshape(B * T, N, D)
        spatial_out = self.spatial(spatial_in)
        entity_tokens = spatial_out.reshape(B, T, N, D)

        # Pool entities → frame tokens (mean of observed)
        frame_tokens = entity_tokens.mean(dim=2)  # [B, T, D]

        # Temporal attention: attend across time
        frame_tokens = self.temporal(frame_tokens, causal_mask=causal_mask)

        # FFN
        frame_tokens = frame_tokens + self.ffn(self.norm_ffn(frame_tokens))

        return entity_tokens, frame_tokens


class xTTransformer(nn.Module):
    """
    Expected Threat transformer.
    Takes tracking data windows and predicts threat values for home and away teams.

    Output:
        home_threat: P(home scores) ≈ P(home shot) × E[xG]
        away_threat: P(away scores) ≈ P(away shot) × E[xG]
        xT = home_threat - away_threat
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

        # Temporal position embeddings
        self.time_emb = nn.Embedding(max_seq_len, d_model)

        # Space-time transformer blocks
        self.blocks = nn.ModuleList([
            SpaceTimeBlock(d_model, n_heads, dropout) for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

        # Prediction heads
        # Shot probability heads (per team)
        self.shot_home_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )
        self.shot_away_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

        # xG prediction heads (per team, conditioned on shot)
        self.xg_home_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid(),
        )
        self.xg_away_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _causal_mask(self, T: int, device: torch.device):
        """Create causal attention mask."""
        mask = torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()
        return mask.float().masked_fill(mask, float("-inf"))

    def forward(self, features, mask, entity_type):
        """
        features: [B, T, N, feat_dim] - tracking features
        mask: [B, T, N] - observation mask (float)
        entity_type: [B, N] or [N] - entity type ids
        Returns dict with: home_threat, away_threat, xt_value,
                          shot_home_logit, shot_away_logit, xg_home, xg_away
        """
        B, T, N, F = features.shape

        # Encode entities
        entity_tokens = self.entity_encoder(features, mask, entity_type)  # [B, T, N, D]

        # Add temporal position embeddings
        time_ids = torch.arange(T, device=features.device)
        time_emb = self.time_emb(time_ids)  # [T, D]
        entity_tokens = entity_tokens + time_emb.unsqueeze(0).unsqueeze(2)

        # Initial frame tokens (mean pool)
        frame_tokens = entity_tokens.mean(dim=2)  # [B, T, D]

        # Causal mask for temporal attention
        causal_mask = self._causal_mask(T, features.device)

        # Process through space-time blocks
        for block in self.blocks:
            entity_tokens, frame_tokens = block(entity_tokens, frame_tokens, causal_mask)

        frame_tokens = self.norm(frame_tokens)

        # Use last frame's representation for prediction
        last_frame = frame_tokens[:, -1]  # [B, D]

        # Predictions
        shot_home_logit = self.shot_home_head(last_frame).squeeze(-1)  # [B]
        shot_away_logit = self.shot_away_head(last_frame).squeeze(-1)  # [B]
        xg_home = self.xg_home_head(last_frame).squeeze(-1)  # [B]
        xg_away = self.xg_away_head(last_frame).squeeze(-1)  # [B]

        p_shot_home = torch.sigmoid(shot_home_logit)
        p_shot_away = torch.sigmoid(shot_away_logit)

        home_threat = p_shot_home * xg_home
        away_threat = p_shot_away * xg_away
        xt_value = home_threat - away_threat

        return {
            "xt_value": xt_value,
            "home_threat": home_threat,
            "away_threat": away_threat,
            "shot_home_logit": shot_home_logit,
            "shot_away_logit": shot_away_logit,
            "xg_home": xg_home,
            "xg_away": xg_away,
            "p_shot_home": p_shot_home,
            "p_shot_away": p_shot_away,
        }
