#!/usr/bin/env python3
"""SMART tokenizer: discretize player/ball displacement sequences into motion tokens.

Player tokens: 5-step displacement [(dx1,dy1),...,(dx5,dy5)] = 10D -> 2048 vocab
Ball tokens:   5-step displacement [(dx1,dy1,dz1),...,(dx5,dy5,dz5)] = 15D -> 1024 vocab

Matching: L2 distance in standardized space to nearest centroid.
During training, sample uniformly from top-3 nearest (noise injection).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np


STEPS_PER_TOKEN = 5          # 5 frames per token at 12.5Hz = 0.4s
PLAYER_CLAMP = 0.8           # max per-step displacement in metres (~10 m/s)
BALL_CLAMP = 3.2             # max per-step displacement (~40 m/s)
PLAYER_VOCAB_SIZE = 2048
BALL_VOCAB_SIZE = 1024
PLAYER_DIM = 10              # 5 steps * 2 (dx, dy)
BALL_DIM = 15                # 5 steps * 3 (dx, dy, dz)


class MotionTokenizer:
    """Tokenizes displacement sequences using pre-built K-means vocabularies."""

    def __init__(
        self,
        player_vocab_path: str | Path,
        ball_vocab_path: str | Path,
    ) -> None:
        pv = np.load(player_vocab_path)
        bv = np.load(ball_vocab_path)

        self.player_centroids = pv["centroids"].astype(np.float32)  # [2048, 10]
        self.player_std = pv["std"].astype(np.float32)              # [10]
        self.ball_centroids = bv["centroids"].astype(np.float32)    # [1024, 15]
        self.ball_std = bv["std"].astype(np.float32)                # [15]

        # Pre-standardize centroids for fast matching
        self._player_centroids_std = self.player_centroids / np.maximum(self.player_std, 1e-6)
        self._ball_centroids_std = self.ball_centroids / np.maximum(self.ball_std, 1e-6)

    def encode_player(
        self,
        displacements: np.ndarray,
        noise_top_k: int = 0,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """Encode player displacement sequences to token IDs.

        Args:
            displacements: [T, 10] or [T, N, 10] array of 5-step displacement sequences.
            noise_top_k: if >1, sample uniformly from top-K nearest centroids.
            rng: random generator for noise sampling.

        Returns:
            Token IDs with same leading shape as input.
        """
        return self._encode(
            displacements, self._player_centroids_std, self.player_std,
            noise_top_k, rng,
        )

    def encode_ball(
        self,
        displacements: np.ndarray,
        noise_top_k: int = 0,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """Encode ball displacement sequences to token IDs."""
        return self._encode(
            displacements, self._ball_centroids_std, self.ball_std,
            noise_top_k, rng,
        )

    def decode_player(self, token_ids: np.ndarray) -> np.ndarray:
        """Decode player token IDs back to displacement sequences.

        Args:
            token_ids: integer array of any shape.

        Returns:
            Displacement array with shape (*token_ids.shape, 10).
        """
        return self.player_centroids[token_ids.ravel()].reshape(*token_ids.shape, PLAYER_DIM)

    def decode_ball(self, token_ids: np.ndarray) -> np.ndarray:
        """Decode ball token IDs back to displacement sequences.

        Returns:
            Displacement array with shape (*token_ids.shape, 15).
        """
        return self.ball_centroids[token_ids.ravel()].reshape(*token_ids.shape, BALL_DIM)

    @staticmethod
    def _encode(
        displacements: np.ndarray,
        centroids_std: np.ndarray,
        std: np.ndarray,
        noise_top_k: int,
        rng: Optional[np.random.Generator],
    ) -> np.ndarray:
        orig_shape = displacements.shape[:-1]
        D = displacements.shape[-1]
        flat = displacements.reshape(-1, D).astype(np.float32)

        # Standardize
        flat_std = flat / np.maximum(std, 1e-6)

        # L2 distance to each centroid: ||a - b||^2 = ||a||^2 - 2*a.b + ||b||^2
        a_sq = (flat_std ** 2).sum(axis=1, keepdims=True)     # [M, 1]
        b_sq = (centroids_std ** 2).sum(axis=1, keepdims=True) # [K, 1]
        dists = a_sq - 2.0 * flat_std @ centroids_std.T + b_sq.T  # [M, K]

        if noise_top_k > 1 and rng is not None:
            # Get top-K indices and sample uniformly
            top_k_idx = np.argpartition(dists, noise_top_k, axis=1)[:, :noise_top_k]
            choices = rng.integers(0, noise_top_k, size=flat.shape[0])
            ids = top_k_idx[np.arange(flat.shape[0]), choices]
        else:
            ids = np.argmin(dists, axis=1)

        return ids.reshape(orig_shape).astype(np.int64)


def extract_player_displacements(
    positions: np.ndarray,
    clamp: float = PLAYER_CLAMP,
) -> np.ndarray:
    """Extract 5-step displacement sequences from player position timeseries.

    Args:
        positions: [T, 2] array of (x, y) positions at 12.5Hz.
        clamp: max per-step displacement to clamp.

    Returns:
        [T//5, 10] array of displacement sequences. Incomplete trailing steps are dropped.
    """
    T = positions.shape[0]
    n_tokens = T // STEPS_PER_TOKEN
    if n_tokens == 0:
        return np.zeros((0, PLAYER_DIM), dtype=np.float32)

    # Reshape into tokens
    pos = positions[:n_tokens * STEPS_PER_TOKEN].reshape(n_tokens, STEPS_PER_TOKEN, 2)

    # Compute per-step displacements within each token
    # prev positions: [first pos of token, then frames 0..3]
    prev = np.concatenate([pos[:, :1, :], pos[:, :-1, :]], axis=1)
    deltas = pos - prev  # [n_tokens, 5, 2]

    # Clamp
    deltas = np.clip(deltas, -clamp, clamp)

    return deltas.reshape(n_tokens, PLAYER_DIM).astype(np.float32)


def extract_ball_displacements(
    positions: np.ndarray,
    clamp: float = BALL_CLAMP,
) -> np.ndarray:
    """Extract 5-step displacement sequences from ball position timeseries.

    Args:
        positions: [T, 3] array of (x, y, z) positions at 12.5Hz.
        clamp: max per-step displacement to clamp.

    Returns:
        [T//5, 15] array of displacement sequences.
    """
    T = positions.shape[0]
    n_tokens = T // STEPS_PER_TOKEN
    if n_tokens == 0:
        return np.zeros((0, BALL_DIM), dtype=np.float32)

    pos = positions[:n_tokens * STEPS_PER_TOKEN].reshape(n_tokens, STEPS_PER_TOKEN, 3)
    prev = np.concatenate([pos[:, :1, :], pos[:, :-1, :]], axis=1)
    deltas = pos - prev
    deltas = np.clip(deltas, -clamp, clamp)

    return deltas.reshape(n_tokens, BALL_DIM).astype(np.float32)


def extract_absolute_positions(
    positions: np.ndarray,
    steps_per_token: int = STEPS_PER_TOKEN,
) -> np.ndarray:
    """Extract the position at the START of each token step.

    Args:
        positions: [T, D] position array at 12.5Hz.

    Returns:
        [T//steps_per_token, D] positions at token boundaries.
    """
    T = positions.shape[0]
    n_tokens = T // steps_per_token
    if n_tokens == 0:
        return np.zeros((0, positions.shape[-1]), dtype=np.float32)
    indices = np.arange(n_tokens) * steps_per_token
    return positions[indices].astype(np.float32)
