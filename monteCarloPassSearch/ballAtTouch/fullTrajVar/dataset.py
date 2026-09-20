#!/usr/bin/env python3
"""PyTorch dataset for ballAtTouch full-trajectory prediction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

K_OTHER = 8
H = 8
BALL_DIM = 8
FEAT_DIM = 4
SELF_DIM = H * FEAT_DIM
OTHER_DIM = H * FEAT_DIM


class BallAtTouchTrajectoryDataset(Dataset):
    """Loads .npz files containing touch context and future ball trajectories."""

    def __init__(self, npz_dir: str, augment: bool = False) -> None:
        self.augment = augment
        bf_list: List[np.ndarray] = []
        sf_list: List[np.ndarray] = []
        of_list: List[np.ndarray] = []
        traj_list: List[np.ndarray] = []
        mask_list: List[np.ndarray] = []
        stop_list: List[np.ndarray] = []

        for path in sorted(Path(npz_dir).glob("*.npz")):
            data = np.load(str(path))
            bf_list.append(data["ball_feats"].astype(np.float32))
            sf_list.append(data["self_feats"].astype(np.float32))
            of_list.append(data["other_feats"].astype(np.float32))
            traj_list.append(data["traj_pos"].astype(np.float32))
            mask_list.append(data["traj_mask"].astype(bool))
            stop_list.append(data["stop_idx"].astype(np.int64))

        if not bf_list:
            raise RuntimeError(f"No .npz files found in {npz_dir}")

        self._bf = np.concatenate(bf_list)
        self._sf = np.concatenate(sf_list)
        self._of = np.concatenate(of_list)
        self._traj = np.concatenate(traj_list)
        self._mask = np.concatenate(mask_list)
        self._stop = np.concatenate(stop_list)

    def __len__(self) -> int:
        return int(self._bf.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        bf = self._bf[idx].copy()
        sf = self._sf[idx].copy()
        of = self._of[idx].copy()
        traj = self._traj[idx].copy()
        mask = self._mask[idx].copy()
        stop_idx = self._stop[idx].copy()

        if self.augment:
            pf_x_idx = np.array([h * FEAT_DIM + d for h in range(H) for d in (0, 2)])
            pf_y_idx = np.array([h * FEAT_DIM + d for h in range(H) for d in (1, 3)])

            if np.random.random() < 0.5:
                bf[0] *= -1.0
                bf[2] *= -1.0
                bf[4] *= -1.0
                sf[pf_x_idx] *= -1.0
                of[:, pf_x_idx] *= -1.0
                traj[:, 0] *= -1.0

            if np.random.random() < 0.5:
                bf[1] *= -1.0
                bf[3] *= -1.0
                bf[5] *= -1.0
                sf[pf_y_idx] *= -1.0
                of[:, pf_y_idx] *= -1.0
                traj[:, 1] *= -1.0

        return (
            torch.from_numpy(bf),
            torch.from_numpy(sf),
            torch.from_numpy(of),
            torch.from_numpy(traj),
            torch.from_numpy(mask),
            torch.tensor(stop_idx, dtype=torch.long),
        )


def create_datasets(preprocessed_dir: str, augment: bool = False):
    train_ds = BallAtTouchTrajectoryDataset(str(Path(preprocessed_dir) / "train"), augment=augment)
    val_ds = BallAtTouchTrajectoryDataset(str(Path(preprocessed_dir) / "val"), augment=False)
    return train_ds, val_ds


def load_meta(preprocessed_dir: str) -> dict:
    meta_path = Path(preprocessed_dir) / "meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            return json.load(f)
    return {}
