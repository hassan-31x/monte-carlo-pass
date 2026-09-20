#!/usr/bin/env python3
"""PyTorch dataset for playerToTouch model.

Loads preprocessed .npz files (output of preprocess.py).
Each sample: (ball_feats [6], player_feats [K, H*4], label int[0..K])
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

# Constants must match preprocess.py
K               = 8
H               = 8
BALL_DIM        = 8
PLAYER_FEAT_DIM = 4
PLAYER_HIST_DIM = H * PLAYER_FEAT_DIM   # 32


class TouchDataset(Dataset):
    """Loads all .npz files from a preprocessed split directory."""

    def __init__(self, npz_dir: str, k: int = K, augment: bool = False) -> None:
        self.k = k
        self.augment = augment
        self.ball_feats:   List[np.ndarray] = []
        self.player_feats: List[np.ndarray] = []
        self.labels:       List[np.ndarray] = []

        npz_dir = Path(npz_dir)
        for path in sorted(npz_dir.glob("*.npz")):
            data = np.load(str(path))
            self.ball_feats.append(data["ball_feats"].astype(np.float32))
            self.player_feats.append(data["player_feats"].astype(np.float32))
            self.labels.append(data["labels"].astype(np.int64))

        if not self.ball_feats:
            raise RuntimeError(f"No .npz files found in {npz_dir}")

        self._bf = np.concatenate(self.ball_feats)      # [N, 6]
        self._pf = np.concatenate(self.player_feats)    # [N, K, H*4]
        self._lb = np.concatenate(self.labels)          # [N]

        # Precompute flip index masks for augmentation
        # Ball feats: [x, y, vx, vy, z, vz] -> flip x: negate idx 0,2; flip y: negate idx 1,3
        # Player feats: [K, H*4] where each 4-group is [dx, dy, vx, vy]
        #   flip x: negate idx 0,2 in each group; flip y: negate idx 1,3 in each group
        if augment:
            pf_x_idx = []
            pf_y_idx = []
            for h in range(H):
                pf_x_idx.extend([h * PLAYER_FEAT_DIM + 0, h * PLAYER_FEAT_DIM + 2])
                pf_y_idx.extend([h * PLAYER_FEAT_DIM + 1, h * PLAYER_FEAT_DIM + 3])
            self._pf_x_idx = np.array(pf_x_idx)
            self._pf_y_idx = np.array(pf_y_idx)

    def __len__(self) -> int:
        return len(self._lb)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bf = self._bf[idx].copy()          # [6]
        pf = self._pf[idx].copy()          # [K, H*4]
        lb = self._lb[idx]

        if self.augment:
            if np.random.random() < 0.5:  # horizontal flip
                bf[0] *= -1; bf[2] *= -1
                pf[:, self._pf_x_idx] *= -1
            if np.random.random() < 0.5:  # vertical flip
                bf[1] *= -1; bf[3] *= -1
                pf[:, self._pf_y_idx] *= -1

        return (torch.from_numpy(bf), torch.from_numpy(pf),
                torch.tensor(lb, dtype=torch.long))

    @property
    def n_classes(self) -> int:
        return self.k + 1   # K players + no-touch


class SurvivalTouchDataset(Dataset):
    """Loads survival-format .npz files (is_event, player_label per frame)."""

    def __init__(self, npz_dir: str, k: int = K, augment: bool = False) -> None:
        self.k = k
        self.augment = augment
        bf_parts: List[np.ndarray] = []
        pf_parts: List[np.ndarray] = []
        ie_parts: List[np.ndarray] = []
        pl_parts: List[np.ndarray] = []

        npz_dir = Path(npz_dir)
        for path in sorted(npz_dir.glob("*.npz")):
            data = np.load(str(path))
            bf_parts.append(data["ball_feats"].astype(np.float32))
            pf_parts.append(data["player_feats"].astype(np.float32))
            ie_parts.append(data["is_event"].astype(np.float32))
            pl_parts.append(data["player_label"].astype(np.int64))

        if not bf_parts:
            raise RuntimeError(f"No .npz files found in {npz_dir}")

        self._bf = np.concatenate(bf_parts)      # [N, 6]
        self._pf = np.concatenate(pf_parts)      # [N, K, H*4]
        self._ie = np.concatenate(ie_parts)       # [N] float32 (0 or 1)
        self._pl = np.concatenate(pl_parts)       # [N] int64 (0..K-1 or -1)

        if augment:
            pf_x_idx = []
            pf_y_idx = []
            for h in range(H):
                pf_x_idx.extend([h * PLAYER_FEAT_DIM + 0, h * PLAYER_FEAT_DIM + 2])
                pf_y_idx.extend([h * PLAYER_FEAT_DIM + 1, h * PLAYER_FEAT_DIM + 3])
            self._pf_x_idx = np.array(pf_x_idx)
            self._pf_y_idx = np.array(pf_y_idx)

    def __len__(self) -> int:
        return len(self._ie)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        bf = self._bf[idx].copy()
        pf = self._pf[idx].copy()
        ie = self._ie[idx]
        pl = self._pl[idx]

        if self.augment:
            if np.random.random() < 0.5:
                bf[0] *= -1; bf[2] *= -1
                pf[:, self._pf_x_idx] *= -1
            if np.random.random() < 0.5:
                bf[1] *= -1; bf[3] *= -1
                pf[:, self._pf_y_idx] *= -1

        return (torch.from_numpy(bf), torch.from_numpy(pf),
                torch.tensor(ie, dtype=torch.float32),
                torch.tensor(pl, dtype=torch.long))

    @property
    def n_events(self) -> int:
        return int(self._ie.sum())

    @property
    def event_rate(self) -> float:
        return float(self._ie.mean())


def create_datasets(preprocessed_dir: str, k: int = K, augment: bool = False):
    train_ds = TouchDataset(str(Path(preprocessed_dir) / "train"), k=k, augment=augment)
    val_ds   = TouchDataset(str(Path(preprocessed_dir) / "val"),   k=k, augment=False)
    return train_ds, val_ds


def create_survival_datasets(preprocessed_dir: str, k: int = K, augment: bool = False):
    train_ds = SurvivalTouchDataset(str(Path(preprocessed_dir) / "train"), k=k, augment=augment)
    val_ds   = SurvivalTouchDataset(str(Path(preprocessed_dir) / "val"),   k=k, augment=False)
    return train_ds, val_ds


def load_meta(preprocessed_dir: str) -> dict:
    meta_path = Path(preprocessed_dir) / "meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            return json.load(f)
    return {}
