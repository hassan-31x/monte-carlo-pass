#!/usr/bin/env python3
"""Preprocess raw tracking parquets into playerToTouch training samples.

For each frame in a continuous play segment (no large frame gaps), extracts:
  - ball state: normalized [pos_x, pos_y, vel_x, vel_y, anc_x, anc_y]
  - K nearest players: past H frames of [rel_pos_x, rel_pos_y, vel_x, vel_y]
  - label: index (0..K-1) of the K-nearest player who touches ball at *next* frame,
           or K for no-touch

Ball identification: team_id_opta starts with 'b' (OPTA convention)
Touch detection: ball velocity-change >= threshold AND player within proximity
No-touch frames are heavily downsampled (--downsample-no-touch) for class balance.

Output: per-game .npz files in <out-dir>/train/ and <out-dir>/val/ with keys:
  ball_feats    [N, BALL_DIM]
  player_feats  [N, K, H*PLAYER_FEAT_DIM]
  labels        [N]  int in [0..K]
  player_n_idx  [N, K]  entity-column indices (for debugging)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

# ── pitch geometry ────────────────────────────────────────────────────────────
BALL_TEAM_PREFIX = "b"      # team_id_opta starts with this for ball entity
FPS             = 25
PITCH_HALF_X    = 52.5      # metres; pitch centered at (0,0)
PITCH_HALF_Y    = 34.0
VEL_SCALE       = 10.0      # m/s normalisation for velocities

# ── sample config ─────────────────────────────────────────────────────────────
K               = 8         # nearest players to consider
H               = 8         # history frames per player
BALL_DIM        = 6         # [px, py, vx, vy, anc_x, anc_y]
PLAYER_FEAT_DIM = 4         # [rel_px, rel_py, vx, vy] per frame

MAX_FRAME_GAP   = 5         # frames; larger gap → segment boundary
TOUCH_VEL_DELTA = 3.0       # m/s velocity-change threshold for touch
TOUCH_PROXIMITY = 2.5       # m player-ball distance at touch frame
REFRACTORY      = 8         # min frames between successive touches


# ── helpers ───────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--tracking-dir", default="/mnt/data/remains",
                   help="Directory containing *_tracking.parquet files.")
    p.add_argument("--out-dir",
                   default="/mnt/data/remains/opta2026/playerToTouch/preprocessed")
    p.add_argument("--k", type=int, default=K)
    p.add_argument("--history", type=int, default=H)
    p.add_argument("--touch-vel-delta", type=float, default=TOUCH_VEL_DELTA)
    p.add_argument("--touch-proximity", type=float, default=TOUCH_PROXIMITY)
    p.add_argument("--train-frac", type=float, default=0.9)
    p.add_argument("--downsample-no-touch", type=float, default=1.0,
                   help="Keep this fraction of no-touch frames (class rebalancing).")
    p.add_argument("--max-games", type=int, default=None)
    return p.parse_args()


def load_game_arrays(
    path: Path,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], np.ndarray, np.ndarray]]:
    """Load parquet → (all_frames, ball_pos[T,2], ball_vel[T,2],
                        all_pids, player_pos[T,N,2], player_vel[T,N,2])"""
    try:
        df = pd.read_parquet(path)
    except Exception as e:
        print(f"[warn] {path}: {e}")
        return None

    required = {"frame_count", "pos_x", "pos_y", "speed_x", "speed_y", "team_id_opta", "player_id"}
    if not required.issubset(df.columns):
        print(f"[warn] missing cols in {path}")
        return None

    is_ball = df["team_id_opta"].str.startswith(BALL_TEAM_PREFIX)
    ball_df = (
        df[is_ball]
        .drop_duplicates("frame_count")
        .sort_values("frame_count")
    )
    player_df = df[~is_ball].copy()

    all_frames = np.sort(df["frame_count"].unique()).astype(np.int64)
    T = len(all_frames)
    frame_to_t: dict = {int(f): i for i, f in enumerate(all_frames)}

    # Ball arrays [T, 2]
    ball_pos = np.full((T, 2), np.nan, dtype=np.float32)
    ball_vel = np.full((T, 2), np.nan, dtype=np.float32)
    bt = np.array([frame_to_t[int(f)] for f in ball_df["frame_count"]])
    ball_pos[bt] = ball_df[["pos_x", "pos_y"]].values.astype(np.float32)
    ball_vel_raw = ball_df[["speed_x", "speed_y"]].values.astype(np.float32)
    ball_vel[bt] = ball_vel_raw

    # Player arrays [T, N, 2]
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
    vel_vals = pdf[["speed_x", "speed_y"]].fillna(0.0).values.astype(np.float32)
    player_vel[t_arr, n_arr] = vel_vals

    return all_frames, ball_pos, ball_vel, all_pids, player_pos, player_vel


def get_segments(all_frames: np.ndarray, max_gap: int = MAX_FRAME_GAP) -> List[np.ndarray]:
    """Return list of index arrays into all_frames, one per contiguous segment."""
    if len(all_frames) == 0:
        return []
    cuts = np.where(np.diff(all_frames) > max_gap)[0] + 1
    return [seg for seg in np.split(np.arange(len(all_frames)), cuts) if len(seg) > 1]


def detect_touches(
    seg_tidx: np.ndarray,
    ball_pos: np.ndarray,
    ball_vel: np.ndarray,
    player_pos: np.ndarray,
    vel_delta: float,
    proximity: float,
    refractory: int,
) -> List[Tuple[int, int]]:
    """Return list of (t_idx, player_n_idx) for detected touches in segment."""
    touches: List[Tuple[int, int]] = []
    last_t = -refractory

    for i in range(1, len(seg_tidx)):
        t = int(seg_tidx[i])
        t_prev = int(seg_tidx[i - 1])
        if t - last_t < refractory:
            continue
        bv_c = ball_vel[t]
        bv_p = ball_vel[t_prev]
        if np.any(np.isnan(bv_c)) or np.any(np.isnan(bv_p)):
            continue
        if np.linalg.norm(bv_c - bv_p) < vel_delta:
            continue

        # Nearest player at t_prev or t
        best_n, best_dist = -1, np.inf
        for ct in (t_prev, t):
            bp = ball_pos[ct]
            if np.any(np.isnan(bp)):
                continue
            ppos = player_pos[ct]          # [N, 2]
            valid = ~np.any(np.isnan(ppos), axis=1)
            if not valid.any():
                continue
            vidx = np.where(valid)[0]
            dists = np.linalg.norm(ppos[vidx] - bp, axis=1)
            mi = int(np.argmin(dists))
            if dists[mi] < best_dist:
                best_dist = dists[mi]
                best_n = vidx[mi]

        if best_dist <= proximity and best_n >= 0:
            touches.append((t, best_n))
            last_t = t

    return touches


def build_samples(
    seg_tidx: np.ndarray,
    ball_pos: np.ndarray,
    ball_vel: np.ndarray,
    player_pos: np.ndarray,
    player_vel: np.ndarray,
    touches: List[Tuple[int, int]],
    K: int,
    H: int,
    downsample_no_touch: float,
    rng: np.random.Generator,
) -> Optional[dict]:
    """Build playerToTouch classification samples for one segment."""
    # touch_at_t[t_next] = n_idx means player n touches AT frame t_next
    touch_at_t = {t: n for t, n in touches}
    touch_set_t = {t for t, _ in touches}

    ball_feats_list: List[np.ndarray] = []
    player_feats_list: List[np.ndarray] = []
    labels: List[int] = []
    pn_list: List[np.ndarray] = []

    anchor = np.zeros(2, dtype=np.float32)

    # Iterate over frames with enough history and at least 1 future frame
    for seg_i in range(H - 1, len(seg_tidx) - 1):
        t = int(seg_tidx[seg_i])

        # Update anchor at touch frames
        if t in touch_set_t and not np.any(np.isnan(ball_pos[t])):
            anchor = ball_pos[t].copy()

        bp = ball_pos[t]
        bv = ball_vel[t]
        if np.any(np.isnan(bp)) or np.any(np.isnan(bv)):
            continue

        t_next = int(seg_tidx[seg_i + 1])
        touch_n = touch_at_t.get(t_next, None)

        # K-NN at current frame
        ppos_t = player_pos[t]  # [N, 2]
        valid = ~np.any(np.isnan(ppos_t), axis=1)
        vidx = np.where(valid)[0]
        if len(vidx) == 0:
            continue
        dists = np.linalg.norm(ppos_t[vidx] - bp, axis=1)
        knn = vidx[np.argsort(dists)[:K]]
        if len(knn) < K:
            knn = np.concatenate([knn, np.full(K - len(knn), knn[-1])])

        # Label
        if touch_n is not None:
            where = np.where(knn == touch_n)[0]
            if len(where) == 0:
                continue  # touching player outside K-NN → skip
            label = int(where[0])
        else:
            label = K  # no touch

        # Ball features (normalised)
        bf = np.array([
            bp[0] / PITCH_HALF_X,
            bp[1] / PITCH_HALF_Y,
            bv[0] / VEL_SCALE,
            bv[1] / VEL_SCALE,
            (anchor[0] - bp[0]) / PITCH_HALF_X,
            (anchor[1] - bp[1]) / PITCH_HALF_Y,
        ], dtype=np.float32)

        # Player history features [K, H*4]
        pf = np.zeros((K, H * PLAYER_FEAT_DIM), dtype=np.float32)
        for k_i, n_idx in enumerate(knn):
            last_valid = np.zeros(PLAYER_FEAT_DIM, dtype=np.float32)
            for h_i in range(H):
                # h_i=0 is oldest; h_i=H-1 is current frame t
                seg_back = seg_i - H + 1 + h_i
                t_h = int(seg_tidx[seg_back]) if seg_back >= 0 else -1
                offset = h_i * PLAYER_FEAT_DIM
                if t_h < 0 or np.any(np.isnan(player_pos[t_h, n_idx])):
                    pf[k_i, offset:offset + PLAYER_FEAT_DIM] = last_valid
                    continue
                pp = player_pos[t_h, n_idx]
                pv = player_vel[t_h, n_idx]
                feat = np.array([
                    (pp[0] - bp[0]) / PITCH_HALF_X,
                    (pp[1] - bp[1]) / PITCH_HALF_Y,
                    pv[0] / VEL_SCALE,
                    pv[1] / VEL_SCALE,
                ], dtype=np.float32)
                pf[k_i, offset:offset + PLAYER_FEAT_DIM] = feat
                last_valid = feat

        ball_feats_list.append(bf)
        player_feats_list.append(pf)
        labels.append(label)
        pn_list.append(knn.copy())

    if not labels:
        return None
    return {
        "ball_feats":   np.stack(ball_feats_list),       # [N, 6]
        "player_feats": np.stack(player_feats_list),     # [N, K, H*4]
        "labels":       np.array(labels, dtype=np.int64),
        "player_n_idx": np.stack(pn_list),               # [N, K]
    }


def process_game(path: Path, args: argparse.Namespace, rng: np.random.Generator):
    res = load_game_arrays(path)
    if res is None:
        return None
    all_frames, ball_pos, ball_vel, all_pids, player_pos, player_vel = res

    segments = get_segments(all_frames)
    all_samples: List[dict] = []

    for seg_tidx in segments:
        if len(seg_tidx) < args.history + 2:
            continue
        touches = detect_touches(
            seg_tidx, ball_pos, ball_vel, player_pos,
            args.touch_vel_delta, args.touch_proximity, REFRACTORY,
        )
        samp = build_samples(
            seg_tidx, ball_pos, ball_vel, player_pos, player_vel,
            touches, args.k, args.history, args.downsample_no_touch, rng,
        )
        if samp is not None:
            all_samples.append(samp)

    if not all_samples:
        return None

    return {
        "ball_feats":   np.concatenate([s["ball_feats"]   for s in all_samples]),
        "player_feats": np.concatenate([s["player_feats"] for s in all_samples]),
        "labels":       np.concatenate([s["labels"]       for s in all_samples]),
        "player_n_idx": np.concatenate([s["player_n_idx"] for s in all_samples]),
    }


def main():
    args = parse_args()
    out = Path(args.out_dir)
    (out / "train").mkdir(parents=True, exist_ok=True)
    (out / "val").mkdir(parents=True, exist_ok=True)

    tracking_dir = Path(args.tracking_dir)
    parquets = sorted(tracking_dir.glob("*_tracking.parquet"))
    if args.max_games:
        parquets = parquets[: args.max_games]

    n_train = int(len(parquets) * args.train_frac)
    rng = np.random.default_rng(42)

    meta = {
        "games": [],
        "k": args.k,
        "history": args.history,
        "ball_dim": BALL_DIM,
        "player_feat_dim": PLAYER_FEAT_DIM,
        "pitch_half_x": PITCH_HALF_X,
        "pitch_half_y": PITCH_HALF_Y,
        "vel_scale": VEL_SCALE,
    }

    total_train, total_val = 0, 0
    for idx, path in enumerate(tqdm(parquets, desc="preprocessing")):
        split = "train" if idx < n_train else "val"
        game_id = path.stem.replace("_tracking", "")
        result = process_game(path, args, rng)
        if result is None:
            print(f"  [skip] {game_id}: no samples")
            continue
        n = len(result["labels"])
        touch_count = int((result["labels"] < args.k).sum())
        out_path = out / split / f"{game_id}.npz"
        np.savez_compressed(str(out_path), **result)
        meta["games"].append({
            "game_id": game_id, "split": split,
            "n_samples": n, "n_touch_samples": touch_count,
        })
        if split == "train":
            total_train += n
        else:
            total_val += n
        print(f"  {game_id} [{split}]: {n} samples ({touch_count} touch)")

    meta["total_train"] = total_train
    meta["total_val"] = total_val
    with open(out / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nDone. train={total_train}  val={total_val}")
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
