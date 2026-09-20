#!/usr/bin/env python3
"""PyTorch dataset for ballAtTouch model.

Loads preprocessed .npz files (output of preprocess.py).
Each sample: (ball_feats [6], self_feats [H*4], other_feats [K_other, H*4], label_vel [3])
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

# Constants must match preprocess.py
K_OTHER  = 8
H        = 8
BALL_DIM = 8
FEAT_DIM = 4
SELF_DIM = H * FEAT_DIM   # 32
OTHER_DIM = H * FEAT_DIM  # 32


class BallAtTouchDataset(Dataset):
    """Loads all .npz files from a preprocessed split directory."""

    def __init__(self, npz_dir: str, augment: bool = False) -> None:
        self.augment = augment
        bf_list: List[np.ndarray]  = []
        sf_list: List[np.ndarray]  = []
        of_list: List[np.ndarray]  = []
        lv_list: List[np.ndarray]  = []
        vm_list: List[np.ndarray]  = []

        npz_dir = Path(npz_dir)
        for path in sorted(npz_dir.glob("*.npz")):
            data = np.load(str(path))
            n = len(data["label_vel"])
            bf_list.append(data["ball_feats"].astype(np.float32))
            sf_list.append(data["self_feats"].astype(np.float32))
            of_list.append(data["other_feats"].astype(np.float32))
            lv_list.append(data["label_vel"].astype(np.float32))
            # vel_obs_mask may be absent in files produced before this change;
            # default to [True, True, False] (OPTA 2D: vz unobserved).
            if "vel_obs_mask" in data:
                vm_list.append(data["vel_obs_mask"].astype(bool))
            else:
                vm_list.append(
                    np.tile(np.array([True, True, False], dtype=bool), (n, 1))
                )

        if not bf_list:
            raise RuntimeError(f"No .npz files found in {npz_dir}")

        self._bf = np.concatenate(bf_list)   # [N, 6]
        self._sf = np.concatenate(sf_list)   # [N, H*4]
        self._of = np.concatenate(of_list)   # [N, K_other, H*4]
        self._lv = np.concatenate(lv_list)   # [N, 3]
        self._vm = np.concatenate(vm_list)   # [N, 3] bool

    def __len__(self) -> int:
        return len(self._lv)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        bf = self._bf[idx].copy()   # [8]: x, y, vx, vy, adx, ady, z, vz
        sf = self._sf[idx].copy()   # [H*4]
        of = self._of[idx].copy()   # [K_other, H*4]
        lv = self._lv[idx].copy()   # [3]: vx, vy, vz
        vm = self._vm[idx]          # [3] bool (no mutation needed)

        if self.augment:
            # Each frame in player features: [rel_px, rel_py, vx, vy]
            # x-indices within each 4-dim frame: 0, 2
            # y-indices within each 4-dim frame: 1, 3
            pf_x_idx = np.array([h * FEAT_DIM + d for h in range(H) for d in (0, 2)])
            pf_y_idx = np.array([h * FEAT_DIM + d for h in range(H) for d in (1, 3)])

            if np.random.random() < 0.5:  # x-flip
                bf[0] *= -1; bf[2] *= -1; bf[4] *= -1   # x, vx, anchor_dx
                sf[pf_x_idx] *= -1
                of[:, pf_x_idx] *= -1
                lv[0] *= -1                               # label vx

            if np.random.random() < 0.5:  # y-flip
                bf[1] *= -1; bf[3] *= -1; bf[5] *= -1   # y, vy, anchor_dy
                sf[pf_y_idx] *= -1
                of[:, pf_y_idx] *= -1
                lv[1] *= -1                               # label vy

        return (
            torch.from_numpy(bf),
            torch.from_numpy(sf),
            torch.from_numpy(of),
            torch.from_numpy(lv),
            torch.from_numpy(vm),
        )


def create_datasets(preprocessed_dir: str, augment: bool = False):
    train_ds = BallAtTouchDataset(str(Path(preprocessed_dir) / "train"), augment=augment)
    val_ds   = BallAtTouchDataset(str(Path(preprocessed_dir) / "val"), augment=False)
    return train_ds, val_ds


def load_meta(preprocessed_dir: str) -> dict:
    meta_path = Path(preprocessed_dir) / "meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            return json.load(f)
    return {}
