#!/usr/bin/env python3
"""Preprocess raw tracking parquets into ballAtTouch training samples.

At each detected ball touch, extracts:
  - ball_feats:  [BALL_DIM]       normalised ball state just before touch
  - self_feats:  [H * FEAT_DIM]   past H frames of touching player
  - other_feats: [K_OTHER, H * FEAT_DIM]  past H frames of K nearest others
  - label_vel:   [3]              outgoing ball velocity (vx, vy, vz≈0)

Touch detection: same velocity-change heuristic as playerToTouch.
The label is ball_vel at touch frame (outgoing velocity after contact).

Output: per-game .npz files in <out-dir>/train/ and <out-dir>/val/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

# ── constants (must match dataset.py) ────────────────────────────────────────
BALL_TEAM_PREFIX = "b"
PITCH_HALF_X    = 52.5
PITCH_HALF_Y    = 34.0
VEL_SCALE       = 10.0

K_OTHER         = 8    # nearest others (not self) per touch
H               = 8    # history frames
BALL_DIM        = 6    # [px, py, vx, vy, anc_x, anc_y]
FEAT_DIM        = 4    # [rel_px, rel_py, vx, vy] per frame

MAX_FRAME_GAP   = 5
TOUCH_VEL_DELTA = 3.0
TOUCH_PROXIMITY = 2.5
REFRACTORY      = 8


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--tracking-dir", default="/mnt/data/remains")
    p.add_argument("--out-dir",
                   default="/mnt/data/remains/opta2026/ballAtTouch/preprocessed")
    p.add_argument("--k-other", type=int, default=K_OTHER)
    p.add_argument("--history", type=int, default=H)
    p.add_argument("--touch-vel-delta", type=float, default=TOUCH_VEL_DELTA)
    p.add_argument("--touch-proximity", type=float, default=TOUCH_PROXIMITY)
    p.add_argument("--train-frac", type=float, default=0.9)
    p.add_argument("--max-games", type=int, default=None)
    return p.parse_args()


def load_game_arrays(path: Path):
    try:
        df = pd.read_parquet(path)
    except Exception as e:
        print(f"[warn] {path}: {e}")
        return None

    is_ball = df["team_id_opta"].str.startswith(BALL_TEAM_PREFIX)
    ball_df = (
        df[is_ball].drop_duplicates("frame_count").sort_values("frame_count")
    )
    player_df = df[~is_ball].copy()

    all_frames = np.sort(df["frame_count"].unique()).astype(np.int64)
    T = len(all_frames)
    frame_to_t = {int(f): i for i, f in enumerate(all_frames)}

    ball_pos = np.full((T, 2), np.nan, dtype=np.float32)
    ball_vel = np.full((T, 2), np.nan, dtype=np.float32)
    bt = np.array([frame_to_t[int(f)] for f in ball_df["frame_count"]])
    ball_pos[bt] = ball_df[["pos_x", "pos_y"]].values.astype(np.float32)
    ball_vel[bt] = ball_df[["speed_x", "speed_y"]].values.astype(np.float32)

    all_pids = sorted(player_df["player_id"].unique())
    pid_to_n = {pid: n for n, pid in enumerate(all_pids)}
    N = len(all_pids)

    player_pos = np.full((T, N, 2), np.nan, dtype=np.float32)
    player_vel = np.full((T, N, 2), np.nan, dtype=np.float32)

    pdf = player_df.copy()
    pdf["t_idx"] = pdf["frame_count"].map(frame_to_t)
    pdf["n_idx"] = pdf["player_id"].map(pid_to_n)
    pdf = pdf.dropna(subset=["pos_x", "pos_y", "t_idx", "n_idx"])
    t_arr = pdf["t_idx"].astype(int).values
    n_arr = pdf["n_idx"].astype(int).values
    player_pos[t_arr, n_arr] = pdf[["pos_x", "pos_y"]].values.astype(np.float32)
    player_vel[t_arr, n_arr] = (
        pdf[["speed_x", "speed_y"]].fillna(0.0).values.astype(np.float32)
    )

    return all_frames, ball_pos, ball_vel, all_pids, player_pos, player_vel


def get_segments(all_frames: np.ndarray, max_gap: int = MAX_FRAME_GAP) -> List[np.ndarray]:
    if len(all_frames) == 0:
        return []
    cuts = np.where(np.diff(all_frames) > max_gap)[0] + 1
    return [s for s in np.split(np.arange(len(all_frames)), cuts) if len(s) > 1]


def detect_touches(
    seg_tidx, ball_pos, ball_vel, player_pos,
    vel_delta, proximity, refractory,
) -> List[Tuple[int, int]]:
    touches = []
    last_t = -refractory
    for i in range(1, len(seg_tidx)):
        t      = int(seg_tidx[i])
        t_prev = int(seg_tidx[i - 1])
        if t - last_t < refractory:
            continue
        bv_c = ball_vel[t]
        bv_p = ball_vel[t_prev]
        if np.any(np.isnan(bv_c)) or np.any(np.isnan(bv_p)):
            continue
        if np.linalg.norm(bv_c - bv_p) < vel_delta:
            continue

        best_n, best_dist = -1, np.inf
        for ct in (t_prev, t):
            bp = ball_pos[ct]
            if np.any(np.isnan(bp)):
                continue
            ppos = player_pos[ct]
            valid = ~np.any(np.isnan(ppos), axis=1)
            vidx = np.where(valid)[0]
            if not len(vidx):
                continue
            dists = np.linalg.norm(ppos[vidx] - bp, axis=1)
            mi = int(np.argmin(dists))
            if dists[mi] < best_dist:
                best_dist, best_n = dists[mi], vidx[mi]

        if best_dist <= proximity and best_n >= 0:
            touches.append((t, int(best_n)))
            last_t = t

    return touches


def player_history(
    t_curr: int,
    seg_tidx: np.ndarray,
    seg_i: int,
    n_idx: int,
    ball_pos_t: np.ndarray,
    player_pos: np.ndarray,
    player_vel: np.ndarray,
    H: int,
) -> np.ndarray:
    """Build [H*4] feature vector for player n_idx, history up to seg_i."""
    feats = np.zeros(H * FEAT_DIM, dtype=np.float32)
    last_valid = np.zeros(FEAT_DIM, dtype=np.float32)
    bp = ball_pos_t  # current ball pos for relative coords

    for h in range(H):
        seg_back = seg_i - H + 1 + h
        if seg_back < 0:
            feats[h * FEAT_DIM:(h + 1) * FEAT_DIM] = last_valid
            continue
        t_h = int(seg_tidx[seg_back])
        if np.any(np.isnan(player_pos[t_h, n_idx])):
            feats[h * FEAT_DIM:(h + 1) * FEAT_DIM] = last_valid
            continue
        pp = player_pos[t_h, n_idx]
        pv = player_vel[t_h, n_idx]
        feat = np.array([
            (pp[0] - bp[0]) / PITCH_HALF_X,
            (pp[1] - bp[1]) / PITCH_HALF_Y,
            pv[0] / VEL_SCALE,
            pv[1] / VEL_SCALE,
        ], dtype=np.float32)
        feats[h * FEAT_DIM:(h + 1) * FEAT_DIM] = feat
        last_valid = feat

    return feats


def build_touch_samples(
    seg_tidx: np.ndarray,
    ball_pos: np.ndarray,
    ball_vel: np.ndarray,
    player_pos: np.ndarray,
    player_vel: np.ndarray,
    touches: List[Tuple[int, int]],
    K_other: int,
    H: int,
) -> Optional[dict]:
    """Build ballAtTouch samples: one per touch event in segment."""
    # Build a mapping t → seg_i for quick lookup
    t_to_segi = {int(t): i for i, t in enumerate(seg_tidx)}

    ball_feats_list: List[np.ndarray] = []
    self_feats_list: List[np.ndarray] = []
    other_feats_list: List[np.ndarray] = []
    labels_list: List[np.ndarray] = []
    masks_list: List[np.ndarray] = []

    anchor = np.zeros(2, dtype=np.float32)
    touch_set_t = {t for t, _ in touches}

    for touch_t, self_n in touches:
        seg_i = t_to_segi.get(touch_t, None)
        if seg_i is None or seg_i < H - 1:
            continue
        # Need one frame after touch for label (outgoing vel = ball_vel[touch_t])
        # Label: ball velocity AT touch (outgoing)
        bp = ball_pos[touch_t]
        bv_out = ball_vel[touch_t]
        if np.any(np.isnan(bp)) or np.any(np.isnan(bv_out)):
            continue

        # Ball input state: ball just before touch (t_prev)
        t_prev = int(seg_tidx[seg_i - 1])
        bp_prev = ball_pos[t_prev]
        bv_prev = ball_vel[t_prev]
        if np.any(np.isnan(bp_prev)) or np.any(np.isnan(bv_prev)):
            continue

        # Ball features (state before touch)
        ball_f = np.array([
            bp_prev[0] / PITCH_HALF_X,
            bp_prev[1] / PITCH_HALF_Y,
            bv_prev[0] / VEL_SCALE,
            bv_prev[1] / VEL_SCALE,
            (anchor[0] - bp_prev[0]) / PITCH_HALF_X,
            (anchor[1] - bp_prev[1]) / PITCH_HALF_Y,
        ], dtype=np.float32)

        # Self history
        self_f = player_history(
            touch_t, seg_tidx, seg_i, self_n, bp, player_pos, player_vel, H
        )

        # K_other nearest others
        ppos_t = player_pos[touch_t]
        valid = ~np.any(np.isnan(ppos_t), axis=1)
        valid[self_n] = False  # exclude self
        vidx = np.where(valid)[0]
        if len(vidx) == 0:
            other_nn = np.full(K_other, self_n, dtype=int)
        else:
            dists = np.linalg.norm(ppos_t[vidx] - bp, axis=1)
            other_nn = vidx[np.argsort(dists)[:K_other]]
            if len(other_nn) < K_other:
                pad = np.full(K_other - len(other_nn), other_nn[-1])
                other_nn = np.concatenate([other_nn, pad])

        other_f = np.stack([
            player_history(touch_t, seg_tidx, seg_i, int(n), bp, player_pos, player_vel, H)
            for n in other_nn
        ])  # [K_other, H*4]

        # Label: outgoing ball velocity [vx, vy, vz=0]
        label_v = np.array([bv_out[0] / VEL_SCALE, bv_out[1] / VEL_SCALE, 0.0],
                            dtype=np.float32)

        # Observation mask: vz is unobserved for OPTA (2D tracking only)
        vel_obs_mask = np.array([True, True, False], dtype=bool)

        ball_feats_list.append(ball_f)
        self_feats_list.append(self_f)
        other_feats_list.append(other_f)
        labels_list.append(label_v)
        masks_list.append(vel_obs_mask)

        # Update anchor
        if not np.any(np.isnan(bp)):
            anchor = bp.copy()

    if not labels_list:
        return None
    return {
        "ball_feats":    np.stack(ball_feats_list),   # [N, 6]
        "self_feats":    np.stack(self_feats_list),   # [N, H*4]
        "other_feats":   np.stack(other_feats_list),  # [N, K_other, H*4]
        "label_vel":     np.stack(labels_list),        # [N, 3]
        "vel_obs_mask":  np.stack(masks_list),         # [N, 3] bool
    }


def process_game(path: Path, args: argparse.Namespace):
    res = load_game_arrays(path)
    if res is None:
        return None
    all_frames, ball_pos, ball_vel, all_pids, player_pos, player_vel = res

    segments = get_segments(all_frames)
    all_samples = []

    for seg_tidx in segments:
        if len(seg_tidx) < args.history + 2:
            continue
        touches = detect_touches(
            seg_tidx, ball_pos, ball_vel, player_pos,
            args.touch_vel_delta, args.touch_proximity, REFRACTORY,
        )
        if not touches:
            continue
        samp = build_touch_samples(
            seg_tidx, ball_pos, ball_vel, player_pos, player_vel,
            touches, args.k_other, args.history,
        )
        if samp:
            all_samples.append(samp)

    if not all_samples:
        return None
    return {
        "ball_feats":   np.concatenate([s["ball_feats"]   for s in all_samples]),
        "self_feats":   np.concatenate([s["self_feats"]   for s in all_samples]),
        "other_feats":  np.concatenate([s["other_feats"]  for s in all_samples]),
        "label_vel":    np.concatenate([s["label_vel"]    for s in all_samples]),
        "vel_obs_mask": np.concatenate([s["vel_obs_mask"] for s in all_samples]),
    }


def main():
    args = parse_args()
    out = Path(args.out_dir)
    (out / "train").mkdir(parents=True, exist_ok=True)
    (out / "val").mkdir(parents=True, exist_ok=True)

    parquets = sorted(Path(args.tracking_dir).glob("*_tracking.parquet"))
    if args.max_games:
        parquets = parquets[: args.max_games]
    n_train = int(len(parquets) * args.train_frac)

    meta = {
        "games": [], "k_other": args.k_other, "history": args.history,
        "ball_dim": BALL_DIM, "feat_dim": FEAT_DIM,
        "pitch_half_x": PITCH_HALF_X, "pitch_half_y": PITCH_HALF_Y,
        "vel_scale": VEL_SCALE,
    }
    total_train = total_val = 0

    for idx, path in enumerate(tqdm(parquets, desc="preprocessing")):
        split = "train" if idx < n_train else "val"
        game_id = path.stem.replace("_tracking", "")
        result = process_game(path, args)
        if result is None:
            print(f"  [skip] {game_id}")
            continue
        n = len(result["label_vel"])
        out_path = out / split / f"{game_id}.npz"
        np.savez_compressed(str(out_path), **result)
        meta["games"].append({"game_id": game_id, "split": split, "n_touches": n})
        if split == "train":
            total_train += n
        else:
            total_val += n
        print(f"  {game_id} [{split}]: {n} touches")

    meta["total_train"] = total_train
    meta["total_val"]   = total_val
    with open(out / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nDone. train_touches={total_train}  val_touches={total_val}")
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
