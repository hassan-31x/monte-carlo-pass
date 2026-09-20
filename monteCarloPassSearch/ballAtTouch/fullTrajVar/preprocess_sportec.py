#!/usr/bin/env python3
"""Extract BAT fixed-horizon trajectory samples from converted Sportec clips."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from tqdm import tqdm

HALF_X = 52.5
HALF_Y = 34.0
VEL_SCALE = 10.0
Z_SCALE = 5.0
K_OTHER = 8
H_HIST = 8
FEAT_DIM = 4
TOUCH_SKIP = 5
RECV_VEL_DELTA = 2.0
RECV_PROXIMITY = 3.0
RECV_MAX_SEARCH = 200
MAX_OUTGOING_SPEED = 40.0
FPS = 25
TRAJ_HORIZON_SEC = 3.0
MAX_TRAJ_LEN = int(FPS * TRAJ_HORIZON_SEC)

SPORTEC_BASE = Path(
    "/mnt/data/remains/opta2026/ballPlayerTrajModel/publicData/preprocessed/sportec_xyz_metres"
)
OUT_DIR = Path("/mnt/data/remains/opta2026/ballAtTouch/fullTrajVar/preprocessed_sportec")


def detect_all_touches(
    features: np.ndarray,
    mask: np.ndarray,
    ball_idx: int,
    player_indices: np.ndarray,
    *,
    vel_delta: float = RECV_VEL_DELTA,
    proximity: float = RECV_PROXIMITY,
    min_gap: int = TOUCH_SKIP,
) -> List[Tuple[int, int]]:
    touches: List[Tuple[int, int]] = []
    last_touch_frame = -min_gap - 1

    for t in range(1, features.shape[0]):
        if not mask[t, ball_idx] or not mask[t - 1, ball_idx]:
            continue
        if t - last_touch_frame < min_gap:
            continue
        bv_curr = features[t, ball_idx, 2:4]
        bv_prev = features[t - 1, ball_idx, 2:4]
        if float(np.linalg.norm(bv_curr - bv_prev)) < float(vel_delta):
            continue
        bp = features[t, ball_idx, :2]
        vis_players = player_indices[mask[t, player_indices].astype(bool)]
        if len(vis_players) == 0:
            continue
        dists = np.linalg.norm(features[t, vis_players, :2] - bp[None, :], axis=1)
        if float(dists.min()) <= float(proximity):
            toucher = int(vis_players[int(np.argmin(dists))])
            touches.append((t, toucher))
            last_touch_frame = t

    return touches


def detect_reception(
    features: np.ndarray,
    mask: np.ndarray,
    ball_idx: int,
    player_indices: np.ndarray,
    kick_frame: int,
    passer_entity: int,
) -> Optional[Tuple[int, int]]:
    for t in range(int(kick_frame) + TOUCH_SKIP, min(int(kick_frame) + RECV_MAX_SEARCH, features.shape[0])):
        if not mask[t, ball_idx] or not mask[t - 1, ball_idx]:
            continue
        bv_curr = features[t, ball_idx, 2:4]
        bv_prev = features[t - 1, ball_idx, 2:4]
        if float(np.linalg.norm(bv_curr - bv_prev)) < RECV_VEL_DELTA:
            continue
        bp = features[t, ball_idx, :2]
        vis_players = player_indices[mask[t, player_indices].astype(bool)]
        vis_players = vis_players[vis_players != int(passer_entity)]
        if len(vis_players) == 0:
            continue
        dists = np.linalg.norm(features[t, vis_players, :2] - bp[None, :], axis=1)
        if float(dists.min()) <= RECV_PROXIMITY:
            return t, int(vis_players[int(np.argmin(dists))])
    return None


def _player_history(
    features: np.ndarray,
    mask: np.ndarray,
    frame: int,
    player_idx: int,
    ball_pos: np.ndarray,
) -> np.ndarray:
    feats = np.zeros(H_HIST * FEAT_DIM, dtype=np.float32)
    last_valid = np.zeros(FEAT_DIM, dtype=np.float32)

    for h in range(H_HIST):
        t = int(frame) - H_HIST + 1 + h
        if t < 0 or not mask[t, player_idx]:
            feats[h * FEAT_DIM : (h + 1) * FEAT_DIM] = last_valid
            continue
        pp = features[t, player_idx, :2]
        pv = features[t, player_idx, 2:4]
        feat = np.array(
            [
                (pp[0] - ball_pos[0]) / HALF_X,
                (pp[1] - ball_pos[1]) / HALF_Y,
                pv[0] / VEL_SCALE,
                pv[1] / VEL_SCALE,
            ],
            dtype=np.float32,
        )
        feats[h * FEAT_DIM : (h + 1) * FEAT_DIM] = feat
        last_valid = feat

    return feats


def _build_sample(
    *,
    features: np.ndarray,
    mask: np.ndarray,
    ball_idx: int,
    player_indices: np.ndarray,
    touch_frame: int,
    toucher_entity: int,
    max_traj_len: int,
) -> Optional[dict]:
    if touch_frame < H_HIST - 1:
        return None
    if not bool(mask[touch_frame, ball_idx]):
        return None

    ball_pos = features[touch_frame, ball_idx, :2]
    ball_vel = features[touch_frame, ball_idx, 2:4]
    ball_z = float(features[touch_frame, ball_idx, 4])
    ball_vz = float(features[touch_frame, ball_idx, 5])

    anchor_frame = touch_frame - 1
    anchor_pos = features[anchor_frame, ball_idx, :2] if anchor_frame >= 0 and bool(mask[anchor_frame, ball_idx]) else ball_pos
    ball_feats = np.array(
        [
            ball_pos[0] / HALF_X,
            ball_pos[1] / HALF_Y,
            ball_vel[0] / VEL_SCALE,
            ball_vel[1] / VEL_SCALE,
            (anchor_pos[0] - ball_pos[0]) / HALF_X,
            (anchor_pos[1] - ball_pos[1]) / HALF_Y,
            ball_z / Z_SCALE,
            ball_vz / VEL_SCALE,
        ],
        dtype=np.float32,
    )

    self_feats = _player_history(features, mask, touch_frame, toucher_entity, ball_pos)

    valid_players = player_indices[mask[touch_frame, player_indices].astype(bool)]
    others_valid = valid_players[valid_players != int(toucher_entity)]
    if len(others_valid) == 0:
        other_indices = np.full(K_OTHER, toucher_entity, dtype=int)
    else:
        other_pos = features[touch_frame, others_valid, :2]
        other_dists = np.linalg.norm(other_pos - ball_pos[None, :], axis=1)
        other_indices = others_valid[np.argsort(other_dists)[:K_OTHER]]
        if len(other_indices) < K_OTHER:
            pad = np.full(K_OTHER - len(other_indices), other_indices[-1], dtype=int)
            other_indices = np.concatenate([other_indices, pad])

    other_feats = np.stack(
        [_player_history(features, mask, touch_frame, int(pi), ball_pos) for pi in other_indices]
    )

    future_end = min(int(features.shape[0] - 1), int(touch_frame + max_traj_len))
    if future_end <= int(touch_frame):
        return None
    future_mask = mask[touch_frame + 1 : future_end + 1, ball_idx].astype(bool)
    if future_mask.size == 0:
        return None
    invalid_idx = np.where(~future_mask)[0]
    valid_len = int(invalid_idx[0]) if invalid_idx.size > 0 else int(future_mask.size)
    if valid_len <= 0:
        return None
    future_ball = features[touch_frame + 1 : touch_frame + 1 + valid_len, ball_idx, :].astype(np.float32)
    speed_xy = np.linalg.norm(future_ball[:, 2:4], axis=1)
    if float(np.nanmax(speed_xy)) > MAX_OUTGOING_SPEED or float(np.nanmax(np.abs(future_ball[:, 5]))) > MAX_OUTGOING_SPEED:
        return None

    traj = np.zeros((int(max_traj_len), 3), dtype=np.float32)
    traj_mask = np.zeros((int(max_traj_len),), dtype=bool)
    rel = np.zeros((future_ball.shape[0], 3), dtype=np.float32)
    rel[:, 0] = (future_ball[:, 0] - ball_pos[0]) / HALF_X
    rel[:, 1] = (future_ball[:, 1] - ball_pos[1]) / HALF_Y
    rel[:, 2] = future_ball[:, 4] / Z_SCALE
    traj[: rel.shape[0]] = rel
    traj_mask[: rel.shape[0]] = True
    stop_idx = np.int64(rel.shape[0] - 1)

    return {
        "ball_feats": ball_feats,
        "self_feats": self_feats,
        "other_feats": other_feats,
        "traj_pos": traj,
        "traj_mask": traj_mask,
        "stop_idx": stop_idx,
    }


def _save_split(out_split: Path, filename: str, samples: List[dict]) -> int:
    if not samples:
        return 0
    out_split.mkdir(parents=True, exist_ok=True)

    arrs = {
        "ball_feats": np.stack([s["ball_feats"] for s in samples]),
        "self_feats": np.stack([s["self_feats"] for s in samples]),
        "other_feats": np.stack([s["other_feats"] for s in samples]),
        "traj_pos": np.stack([s["traj_pos"] for s in samples]),
        "traj_mask": np.stack([s["traj_mask"] for s in samples]),
        "stop_idx": np.asarray([s["stop_idx"] for s in samples], dtype=np.int64),
    }
    np.savez_compressed(str(out_split / filename), **arrs)
    print(f"    {filename}: {arrs['ball_feats'].shape[0]} samples")
    print(f"      traj_pos: {arrs['traj_pos'].shape}")
    return int(arrs["ball_feats"].shape[0])


def _extract_touch_samples(entry: dict, *, pass_only: bool, max_traj_len: int) -> List[dict]:
    d = np.load(entry["clip_path"])
    features = d["features"].astype(np.float32)
    entity_type = d["entity_type"].astype(np.int64)
    mask = d["mask"].astype(bool)
    kf = int(d["kick_frame_local"])

    ball_idx_arr = np.where(entity_type == 2)[0]
    if len(ball_idx_arr) == 0:
        return []
    ball_idx = int(ball_idx_arr[0])
    player_indices = np.where(entity_type != 2)[0]
    touches = detect_all_touches(features, mask, ball_idx, player_indices)
    if len(touches) < 1:
        return []

    samples: List[dict] = []

    if pass_only:
        vis_players = player_indices[mask[kf, player_indices].astype(bool)]
        if len(vis_players) == 0 or not bool(mask[kf, ball_idx]):
            return []
        bp_kf = features[kf, ball_idx, :2]
        dists = np.linalg.norm(features[kf, vis_players, :2] - bp_kf[None, :], axis=1)
        passer_entity = int(vis_players[int(np.argmin(dists))])
        reception = detect_reception(features, mask, ball_idx, player_indices, kf, passer_entity)
        if reception is None:
            return []
        recv_frame, recv_entity = reception
        sample = _build_sample(
            features=features,
            mask=mask,
            ball_idx=ball_idx,
            player_indices=player_indices,
            touch_frame=int(recv_frame),
            toucher_entity=int(recv_entity),
            max_traj_len=max_traj_len,
        )
        return [sample] if sample is not None else []

    for touch_frame, toucher_entity in touches:
        sample = _build_sample(
            features=features,
            mask=mask,
            ball_idx=ball_idx,
            player_indices=player_indices,
            touch_frame=int(touch_frame),
            toucher_entity=int(toucher_entity),
            max_traj_len=max_traj_len,
        )
        if sample is not None:
            samples.append(sample)

    return samples


def process_split(split: str, *, all_touches: bool, max_traj_len: int) -> int:
    manifest_path = SPORTEC_BASE / split / "manifest.jsonl"
    if not manifest_path.exists():
        print(f"  [skip] missing manifest: {manifest_path}")
        return 0
    with open(manifest_path) as f:
        entries = [json.loads(line) for line in f]

    out_split = OUT_DIR / split
    pass_samples: List[dict] = []
    for entry in tqdm(entries, desc=f"{split}-pass"):
        pass_samples.extend(_extract_touch_samples(entry, pass_only=True, max_traj_len=max_traj_len))
    total = _save_split(out_split, "sportec_pass.npz", pass_samples)

    if all_touches and split == "train":
        all_touch_samples: List[dict] = []
        for entry in tqdm(entries, desc=f"{split}-alltouch"):
            all_touch_samples.extend(_extract_touch_samples(entry, pass_only=False, max_traj_len=max_traj_len))
        total += _save_split(out_split, "sportec_alltouch.npz", all_touch_samples)

    return total


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--all-touches", action="store_true", help="Include every detected touch in the training split.")
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--max-traj-len", type=int, default=MAX_TRAJ_LEN)
    args = p.parse_args()

    global OUT_DIR
    if args.out_dir:
        OUT_DIR = Path(args.out_dir)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    total_train = process_split("train", all_touches=bool(args.all_touches), max_traj_len=int(args.max_traj_len))
    total_val = process_split("val", all_touches=False, max_traj_len=int(args.max_traj_len))

    meta = {
        "k_other": K_OTHER,
        "history": H_HIST,
        "ball_dim": 8,
        "feat_dim": FEAT_DIM,
        "pitch_half_x": HALF_X,
        "pitch_half_y": HALF_Y,
        "vel_scale": VEL_SCALE,
        "z_scale": Z_SCALE,
        "traj_horizon_sec": float(TRAJ_HORIZON_SEC),
        "max_traj_len": int(args.max_traj_len),
        "total_train": int(total_train),
        "total_val": int(total_val),
    }
    with open(OUT_DIR / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nDone. train={total_train} val={total_val} out={OUT_DIR}")


if __name__ == "__main__":
    main()
