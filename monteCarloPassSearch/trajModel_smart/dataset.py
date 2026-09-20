#!/usr/bin/env python3
"""SMART dataset: sliding windows of tokenized motion sequences.

Loads preprocessed Sportec clips, downsamples 25Hz->12.5Hz, extracts displacement
sequences, tokenizes them, and returns windows of (history + rollout) token steps.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from tokenizer import (
    BALL_CLAMP,
    BALL_DIM,
    PLAYER_CLAMP,
    PLAYER_DIM,
    STEPS_PER_TOKEN,
    MotionTokenizer,
    extract_absolute_positions,
    extract_ball_displacements,
    extract_player_displacements,
)

DOWNSAMPLE_FACTOR = 2  # 25Hz -> 12.5Hz
N_PLAYERS = 22
BALL_ENTITY_TYPE = 2

# Stop event constants
STOP_CONTINUE = 0
STOP_UNKNOWN_BREAK = 9


class SMARTDataset(Dataset):
    """Sliding window dataset for SMART motion tokens.

    Each sample is a window of T = history + rollout token steps.
    Each token step spans 5 frames at 12.5Hz = 0.4s.
    """

    def __init__(
        self,
        entries: List[Dict[str, Any]],
        tokenizer: MotionTokenizer,
        history: int = 8,
        rollout: int = 24,
        window_stride: int = 8,
        noise_top_k: int = 0,
        augment_flip: bool = False,
        clip_cache_size: int = 8,
        seed: int = 42,
    ) -> None:
        self.tokenizer = tokenizer
        self.history = history
        self.rollout = rollout
        self.total_steps = history + rollout
        self.noise_top_k = noise_top_k
        self.augment_flip = augment_flip
        self._entries = entries
        self._cache: OrderedDict = OrderedDict()
        self._cache_size = max(1, clip_cache_size)
        self._rng = np.random.default_rng(seed)

        # Build window index: (clip_idx, token_start)
        ei_list, t_list = [], []
        frames_per_window = self.total_steps * STEPS_PER_TOKEN * DOWNSAMPLE_FACTOR

        for ei, entry in enumerate(entries):
            clip_len = int(entry["length"])  # frames at 25Hz
            ds_len = clip_len // DOWNSAMPLE_FACTOR
            n_token_steps = ds_len // STEPS_PER_TOKEN

            if n_token_steps < self.total_steps + 1:
                continue

            # Check stop events
            stop_ids = self._get_stop_ids(entry)
            n_starts = n_token_steps - self.total_steps

            for t in range(0, n_starts, window_stride):
                if stop_ids is not None:
                    # Check if window contains restarts
                    frame_start = t * STEPS_PER_TOKEN * DOWNSAMPLE_FACTOR
                    frame_end = (t + self.total_steps) * STEPS_PER_TOKEN * DOWNSAMPLE_FACTOR
                    frame_end = min(frame_end, len(stop_ids))
                    window_stops = stop_ids[frame_start:frame_end]
                    if np.any((window_stops > STOP_CONTINUE) & (window_stops < STOP_UNKNOWN_BREAK)):
                        continue
                ei_list.append(ei)
                t_list.append(t)

        if ei_list:
            self._window_index = np.column_stack([
                np.array(ei_list, dtype=np.int32),
                np.array(t_list, dtype=np.int32),
            ])
        else:
            self._window_index = np.zeros((0, 2), dtype=np.int32)

        self.num_windows = len(self._window_index)

    def _get_stop_ids(self, entry: Dict) -> Optional[np.ndarray]:
        try:
            clip = np.load(entry["clip_path"])
            if "stop_event_id" in clip:
                return np.asarray(clip["stop_event_id"])
        except Exception:
            pass
        return None

    def _load_clip(self, entry: Dict) -> Dict[str, np.ndarray]:
        key = entry["clip_path"]
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        clip = np.load(entry["clip_path"])
        data = {k: np.asarray(clip[k]) for k in clip.files}
        if len(self._cache) >= self._cache_size:
            self._cache.popitem(last=False)
        self._cache[key] = data
        return data

    def __len__(self) -> int:
        return self.num_windows

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ei, t0 = int(self._window_index[idx, 0]), int(self._window_index[idx, 1])
        entry = self._entries[ei]
        clip = self._load_clip(entry)

        features = clip["features"]        # [T_raw, N, F] at 25Hz
        entity_type = clip["entity_type"]   # [N]
        mask = clip.get("mask", np.ones(features.shape[:2], dtype=bool))

        T_raw, N, F = features.shape

        # Downsample 25Hz -> 12.5Hz
        features = features[::DOWNSAMPLE_FACTOR].astype(np.float32)
        mask = mask[::DOWNSAMPLE_FACTOR]
        T_ds = features.shape[0]

        # Identify ball and player indices
        ball_idx = np.where(entity_type == BALL_ENTITY_TYPE)[0]
        player_idxs = np.where((entity_type == 0) | (entity_type == 1))[0]

        # Ensure exactly 22 players and 1 ball
        assert len(ball_idx) > 0, f"No ball entity in clip"
        bi = int(ball_idx[0])

        # Pad/trim to exactly 22 players
        if len(player_idxs) > N_PLAYERS:
            player_idxs = player_idxs[:N_PLAYERS]

        # Slice window in downsampled frames
        frame_start = t0 * STEPS_PER_TOKEN
        frame_end = (t0 + self.total_steps) * STEPS_PER_TOKEN
        win_feat = features[frame_start:frame_end]        # [W, N, F]
        win_mask = mask[frame_start:frame_end]             # [W, N]

        # Data augmentation: random flips (before displacement extraction)
        flip_x = self.augment_flip and (self._rng.random() < 0.5)
        flip_y = self.augment_flip and (self._rng.random() < 0.5)
        if flip_x:
            win_feat[..., 0] *= -1  # x
            win_feat[..., 2] *= -1  # vx
        if flip_y:
            win_feat[..., 1] *= -1  # y
            win_feat[..., 3] *= -1  # vy

        # Extract ball positions and displacements
        ball_xy = win_feat[:, bi, :2]       # [W, 2]
        if F >= 6:
            ball_z = win_feat[:, bi, 4:5]   # [W, 1]
        else:
            ball_z = np.zeros((win_feat.shape[0], 1), dtype=np.float32)
        ball_pos_3d = np.concatenate([ball_xy, ball_z], axis=1)  # [W, 3]

        ball_disps = extract_ball_displacements(ball_pos_3d, clamp=BALL_CLAMP)  # [total_steps, 15]
        ball_abs_pos = extract_absolute_positions(ball_xy, STEPS_PER_TOKEN)     # [total_steps, 2]

        # Tokenize ball
        ball_tokens = self.tokenizer.encode_ball(
            ball_disps,
            noise_top_k=self.noise_top_k if self.augment_flip else 0,
            rng=self._rng,
        )  # [total_steps]

        # Extract player positions and displacements
        player_token_ids = np.zeros((self.total_steps, N_PLAYERS), dtype=np.int64)
        player_abs_pos = np.zeros((self.total_steps, N_PLAYERS, 2), dtype=np.float32)
        player_obs = np.zeros((self.total_steps, N_PLAYERS), dtype=bool)

        for j, pi in enumerate(player_idxs):
            p_xy = win_feat[:, pi, :2]  # [W, 2]
            p_mask = win_mask[:, pi]

            p_disps = extract_player_displacements(p_xy, clamp=PLAYER_CLAMP)  # [total_steps, 10]
            p_abs = extract_absolute_positions(p_xy, STEPS_PER_TOKEN)          # [total_steps, 2]

            p_tokens = self.tokenizer.encode_player(
                p_disps,
                noise_top_k=self.noise_top_k if self.augment_flip else 0,
                rng=self._rng,
            )

            n_tok = min(p_tokens.shape[0], self.total_steps)
            player_token_ids[:n_tok, j] = p_tokens[:n_tok]
            player_abs_pos[:n_tok, j] = p_abs[:n_tok]

            # Obs mask: check if player is observed at each token boundary
            obs_indices = np.arange(self.total_steps) * STEPS_PER_TOKEN
            obs_indices = np.minimum(obs_indices, len(p_mask) - 1)
            player_obs[:, j] = p_mask[obs_indices]

        # Pad missing players (if fewer than 22)
        for j in range(len(player_idxs), N_PLAYERS):
            player_obs[:, j] = False

        # Ball obs mask
        ball_obs_indices = np.arange(self.total_steps) * STEPS_PER_TOKEN
        ball_obs_indices = np.minimum(ball_obs_indices, win_mask.shape[0] - 1)
        ball_obs = win_mask[ball_obs_indices, bi]

        # Combined obs mask: [T, 23] with ball at index 0
        obs_mask_full = np.zeros((self.total_steps, N_PLAYERS + 1), dtype=bool)
        obs_mask_full[:, 0] = ball_obs
        obs_mask_full[:, 1:] = player_obs

        # Entity types: [23] ball=2 at index 0, then player types
        entity_types_ordered = np.zeros(N_PLAYERS + 1, dtype=np.int64)
        entity_types_ordered[0] = BALL_ENTITY_TYPE
        for j, pi in enumerate(player_idxs):
            entity_types_ordered[j + 1] = int(entity_type[pi])
        # Pad entities get type 0 (home) but obs_mask=False so they're ignored

        # Random permutation of player order per timestep
        perm = np.zeros((self.total_steps, N_PLAYERS), dtype=np.int64)
        for t in range(self.total_steps):
            perm[t] = self._rng.permutation(N_PLAYERS)

        return {
            "player_tokens": torch.from_numpy(player_token_ids),    # [T, 22]
            "ball_tokens": torch.from_numpy(ball_tokens),            # [T]
            "player_pos": torch.from_numpy(player_abs_pos),          # [T, 22, 2]
            "ball_pos": torch.from_numpy(ball_abs_pos),              # [T, 2]
            "entity_types": torch.from_numpy(entity_types_ordered),  # [23]
            "obs_mask": torch.from_numpy(obs_mask_full),             # [T, 23]
            "perm": torch.from_numpy(perm),                          # [T, 22]
        }


def load_manifest(path: Path) -> List[Dict[str, Any]]:
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def create_datasets(
    preprocessed_dir: str | Path,
    tokenizer: MotionTokenizer,
    history: int = 8,
    rollout: int = 24,
    window_stride: int = 8,
    noise_top_k: int = 3,
    augment_flip: bool = False,
    clip_cache_size: int = 8,
    seed: int = 42,
) -> Tuple[SMARTDataset, Optional[SMARTDataset]]:
    """Create train and val datasets from preprocessed directory."""
    preprocessed_dir = Path(preprocessed_dir)

    train_manifest = preprocessed_dir / "train" / "manifest.jsonl"
    val_manifest = preprocessed_dir / "val" / "manifest.jsonl"

    if not train_manifest.exists():
        raise FileNotFoundError(f"No train manifest at {train_manifest}")

    train_entries = load_manifest(train_manifest)
    val_entries = load_manifest(val_manifest) if val_manifest.exists() else []

    print(f"Train clips: {len(train_entries)}, Val clips: {len(val_entries)}")

    train_ds = SMARTDataset(
        entries=train_entries, tokenizer=tokenizer,
        history=history, rollout=rollout, window_stride=window_stride,
        noise_top_k=noise_top_k, augment_flip=augment_flip,
        clip_cache_size=clip_cache_size, seed=seed,
    )

    val_ds = None
    if val_entries:
        val_ds = SMARTDataset(
            entries=val_entries, tokenizer=tokenizer,
            history=history, rollout=rollout, window_stride=window_stride,
            noise_top_k=0, augment_flip=False,
            clip_cache_size=clip_cache_size, seed=seed + 1,
        )

    print(f"Train windows: {train_ds.num_windows:,}, Val windows: {val_ds.num_windows if val_ds else 0:,}")
    return train_ds, val_ds
