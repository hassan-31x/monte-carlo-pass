#!/usr/bin/env python3
"""Extract playerToTouch training samples from converted Sportec clips.

Reads Sportec clips (already in metres) and produces .npz files matching the
format of the OPTA-derived PTT preprocessed data (ball_feats, player_feats, labels).

Each Sportec clip has:
  - features[T, N, 6]: (x_m, y_m, vx_ms, vy_ms, z_m, vz_ms)
  - entity_type[N]: 0=home, 1=away, 2=ball
  - kick_frame_local: int (the frame where the kick/touch event occurs)
  - mask[T, N]: visibility mask

For each clip we produce:
  - 1 POSITIVE sample at kick_frame_local (label = k-NN index of nearest player)
  - Up to N_NEG random non-kick frames as NEGATIVE samples (label = K = no-touch)

Output .npz files have keys: ball_feats[N_samples,6], player_feats[N_samples,K,32], labels[N_samples]
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from tqdm import tqdm


# ── constants (must match OPTA preprocess.py) ────────────────────────────────
K               = 8        # nearest players to consider
H               = 8        # history frames per player
BALL_DIM        = 8
PLAYER_FEAT_DIM = 4        # [rel_px, rel_py, vx, vy]
PLAYER_HIST_DIM = H * PLAYER_FEAT_DIM   # 32

PITCH_HALF_X    = 52.5     # metres
PITCH_HALF_Y    = 34.0
VEL_SCALE       = 10.0     # m/s normalisation
Z_SCALE         = 5.0      # metres normalisation for ball height
FPS             = 25

N_NEG           = 8        # negative samples per clip

# ── reception detection ──────────────────────────────────────────────────────
TOUCH_SKIP      = 5        # skip first N frames after kick (ball still with passer)
RECV_VEL_DELTA  = 2.0      # m/s ball velocity-change threshold for reception
RECV_PROXIMITY  = 3.0      # m player-ball distance at reception frame
RECV_MAX_SEARCH = 200      # max frames after kick to search for reception
POS_WINDOW      = 5        # create positive samples in last N frames before reception


# ── helpers ──────────────────────────────────────────────────────────────────

MAX_SEARCH      = RECV_MAX_SEARCH  # alias for survival mode


def detect_all_touches(
    features: np.ndarray,       # [T, N, 6]
    mask: np.ndarray,           # [T, N]
    ball_idx: int,
    player_indices: np.ndarray,
    vel_delta: float = RECV_VEL_DELTA,
    proximity: float = RECV_PROXIMITY,
    min_gap: int = TOUCH_SKIP,
) -> List[Tuple[int, int]]:
    """Detect ALL ball touch events in a clip.

    Returns list of (frame, toucher_entity_idx) sorted by frame.
    """
    T = features.shape[0]
    touches: List[Tuple[int, int]] = []
    last_touch_frame = -min_gap - 1

    for t in range(1, T):
        if not mask[t, ball_idx] or not mask[t - 1, ball_idx]:
            continue
        if t - last_touch_frame < min_gap:
            continue
        bv_curr = features[t, ball_idx, 2:4]
        bv_prev = features[t - 1, ball_idx, 2:4]
        vel_change = np.linalg.norm(bv_curr - bv_prev)
        if vel_change < vel_delta:
            continue
        bp = features[t, ball_idx, :2]
        vis = mask[t, player_indices].astype(bool)
        vis_players = player_indices[vis]
        if len(vis_players) == 0:
            continue
        dists = np.sqrt(np.sum((features[t, vis_players, :2] - bp) ** 2, axis=1))
        if dists.min() <= proximity:
            toucher = int(vis_players[np.argmin(dists)])
            touches.append((t, toucher))
            last_touch_frame = t

    return touches


def process_clip_survival_alltouch(clip_path: str) -> Optional[dict]:
    """Process one clip for survival training using ALL consecutive touch pairs.

    For each pair (touch_i, touch_i+1): touch_i is "kick", touch_i+1 is "reception".
    The toucher at touch_i is the passer (excluded from K-NN).
    Emits frames from touch_i + TOUCH_SKIP to touch_i+1.
    """
    data = np.load(clip_path)
    features = data["features"]
    mask_arr = data["mask"]
    entity_type = data["entity_type"]
    T = features.shape[0]

    ball_idx, player_indices = get_ball_and_player_indices(entity_type)
    if ball_idx is None or len(player_indices) == 0:
        return None

    touches = detect_all_touches(features, mask_arr, ball_idx, player_indices)
    if len(touches) < 2:
        return None

    all_bf, all_pf, all_ie, all_pl = [], [], [], []

    for i in range(len(touches) - 1):
        kick_frame, passer_entity = touches[i]
        recv_frame, receiver_entity = touches[i + 1]

        frame_start = kick_frame + TOUCH_SKIP
        frame_end = recv_frame  # inclusive

        if frame_start > frame_end or frame_start >= T:
            continue

        for t in range(max(frame_start, H), frame_end + 1):
            if t >= T:
                break

            is_event = 1 if t == recv_frame else 0

            if is_event:
                sample = build_sample(
                    features, mask_arr, ball_idx, player_indices,
                    frame=t, kick_frame=kick_frame,
                    exclude_entity=passer_entity,
                    target_entity=receiver_entity,
                )
            else:
                sample = build_sample(
                    features, mask_arr, ball_idx, player_indices,
                    frame=t, kick_frame=kick_frame,
                    label=0,
                    exclude_entity=passer_entity,
                )

            if sample is None:
                continue

            player_label = sample["label"] if is_event else -1
            all_bf.append(sample["ball_feats"])
            all_pf.append(sample["player_feats"])
            all_ie.append(is_event)
            all_pl.append(player_label)

    if not all_bf:
        return None

    return {
        "ball_feats":   np.stack(all_bf),
        "player_feats": np.stack(all_pf),
        "is_event":     np.array(all_ie, dtype=np.int64),
        "player_label": np.array(all_pl, dtype=np.int64),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract PTT samples from Sportec clips")
    p.add_argument("--sportec-dir",
                   default="/mnt/data/remains/opta2026/ballPlayerTrajModel/publicData/preprocessed/sportec_xyz_metres",
                   help="Directory with {train,val}/manifest.jsonl")
    p.add_argument("--out-dir",
                   default="/mnt/data/remains/opta2026/playerToTouch/preprocessed_sportec_v2",
                   help="Output directory for .npz files")
    p.add_argument("--n-neg", type=int, default=N_NEG,
                   help="Number of negative (no-touch) samples per clip")
    p.add_argument("--mode", choices=["classifier", "survival"], default="classifier",
                   help="'classifier' = old pos/neg sampling, 'survival' = emit every frame")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--all-touches", action="store_true",
                   help="Extract from all detected touches (train only)")
    return p.parse_args()


def get_ball_and_player_indices(entity_type: np.ndarray):
    """Return (ball_idx, player_indices) from entity_type array."""
    ball_idx = np.where(entity_type == 2)[0]
    if len(ball_idx) == 0:
        return None, None
    ball_idx = int(ball_idx[0])
    player_indices = np.where(entity_type < 2)[0]
    return ball_idx, player_indices


def detect_reception(
    features: np.ndarray,       # [T, N, 6]
    mask: np.ndarray,           # [T, N]
    ball_idx: int,
    player_indices: np.ndarray,
    kick_frame: int,
    passer_entity: int,
    vel_delta: float = RECV_VEL_DELTA,
    proximity: float = RECV_PROXIMITY,
    max_search: int = RECV_MAX_SEARCH,
    min_gap: int = TOUCH_SKIP,
) -> Optional[Tuple[int, int]]:
    """Detect first reception event after kick_frame (excluding passer).

    Returns (frame, receiver_entity_idx) or None.
    """
    T = features.shape[0]
    for t in range(kick_frame + min_gap, min(kick_frame + max_search, T)):
        if not mask[t, ball_idx] or not mask[t - 1, ball_idx]:
            continue
        bv_curr = features[t, ball_idx, 2:4]
        bv_prev = features[t - 1, ball_idx, 2:4]
        vel_change = np.linalg.norm(bv_curr - bv_prev)
        if vel_change < vel_delta:
            continue
        # Nearest non-passer player within proximity
        bp = features[t, ball_idx, :2]
        vis = mask[t, player_indices].astype(bool)
        vis_players = player_indices[vis]
        vis_players = vis_players[vis_players != passer_entity]
        if len(vis_players) == 0:
            continue
        dists = np.sqrt(np.sum((features[t, vis_players, :2] - bp) ** 2, axis=1))
        if dists.min() <= proximity:
            return (t, int(vis_players[np.argmin(dists)]))
    return None


def build_sample(
    features: np.ndarray,     # [T, N, 6]
    mask: np.ndarray,         # [T, N]
    ball_idx: int,
    player_indices: np.ndarray,
    frame: int,               # the frame to build sample for
    kick_frame: int,          # the kick frame (for anchor)
    label: int = -1,          # 0..K-1 for touch, K for no-touch (-1 = use target_entity)
    exclude_entity: Optional[int] = None,   # entity index to exclude from K-NN (passer)
    target_entity: Optional[int] = None,    # entity index of the receiver (for positive samples)
) -> Optional[dict]:
    """Build a single PTT sample at the given frame.

    If target_entity is set, finds it in K-NN and uses its index as label.
    Returns dict with ball_feats[6], player_feats[K,32], label or None on failure.
    """
    T = features.shape[0]

    # Ball state at this frame
    if not mask[frame, ball_idx]:
        return None
    bx = features[frame, ball_idx, 0]
    by = features[frame, ball_idx, 1]
    bvx = features[frame, ball_idx, 2]
    bvy = features[frame, ball_idx, 3]
    bz  = features[frame, ball_idx, 4]
    bvz = features[frame, ball_idx, 5]

    # Anchor: ball position at kick_frame
    if mask[kick_frame, ball_idx]:
        anc_x = features[kick_frame, ball_idx, 0]
        anc_y = features[kick_frame, ball_idx, 1]
    else:
        anc_x, anc_y = bx, by  # fallback

    # Ball features (normalised)
    ball_feats = np.array([
        bx / PITCH_HALF_X,
        by / PITCH_HALF_Y,
        bvx / VEL_SCALE,
        bvy / VEL_SCALE,
        (anc_x - bx) / PITCH_HALF_X,
        (anc_y - by) / PITCH_HALF_Y,
        bz / Z_SCALE,
        bvz / VEL_SCALE,
    ], dtype=np.float32)

    # Find K nearest visible players at this frame (excluding passer)
    vis_mask = mask[frame, player_indices].astype(bool)
    vis_players = player_indices[vis_mask]
    if exclude_entity is not None:
        vis_players = vis_players[vis_players != exclude_entity]
    if len(vis_players) == 0:
        return None

    px = features[frame, vis_players, 0]
    py = features[frame, vis_players, 1]
    dists = np.sqrt((px - bx)**2 + (py - by)**2)
    sorted_order = np.argsort(dists)
    knn = vis_players[sorted_order[:K]]

    # Pad if fewer than K players visible
    if len(knn) < K:
        knn = np.concatenate([knn, np.full(K - len(knn), knn[-1], dtype=knn.dtype)])

    # If target_entity is specified, find it in K-NN
    if target_entity is not None:
        target_in_knn = np.where(knn == target_entity)[0]
        if len(target_in_knn) == 0:
            return None  # target not in K-NN → skip
        label = int(target_in_knn[0])

    # Player history features [K, H*4]
    player_feats = np.zeros((K, H * PLAYER_FEAT_DIM), dtype=np.float32)
    for k_i in range(K):
        n_idx = knn[k_i]
        last_valid = np.zeros(PLAYER_FEAT_DIM, dtype=np.float32)
        for h_i in range(H):
            # h_i=0 is oldest, h_i=H-1 is current frame
            t_h = frame - H + 1 + h_i
            offset = h_i * PLAYER_FEAT_DIM

            if t_h < 0 or not mask[t_h, n_idx]:
                player_feats[k_i, offset:offset + PLAYER_FEAT_DIM] = last_valid
                continue

            pp_x = features[t_h, n_idx, 0]
            pp_y = features[t_h, n_idx, 1]
            pv_x = features[t_h, n_idx, 2]
            pv_y = features[t_h, n_idx, 3]

            feat = np.array([
                (pp_x - bx) / PITCH_HALF_X,
                (pp_y - by) / PITCH_HALF_Y,
                pv_x / VEL_SCALE,
                pv_y / VEL_SCALE,
            ], dtype=np.float32)
            player_feats[k_i, offset:offset + PLAYER_FEAT_DIM] = feat
            last_valid = feat

    return {
        "ball_feats": ball_feats,
        "player_feats": player_feats,
        "label": label,
    }


def process_clip(
    clip_path: str,
    kick_frame_local: Optional[int],
    n_neg: int,
    rng: np.random.Generator,
) -> Optional[dict]:
    """Process one Sportec clip and return samples.

    Identifies the passer (kicker) at kick_frame_local, detects the reception
    event, and builds positive samples near reception and negative samples
    during ball flight – all with the passer excluded from K-NN candidates.
    """
    data = np.load(clip_path)
    features = data["features"]       # [T, N, 6]
    mask_arr = data["mask"]           # [T, N]
    entity_type = data["entity_type"] # [N]

    T = features.shape[0]

    # Load kick_frame_local from npz if not provided
    if kick_frame_local is None:
        if "kick_frame_local" in data:
            kick_frame_local = int(data["kick_frame_local"])
        else:
            return None

    # Bounds check
    if kick_frame_local < 0 or kick_frame_local >= T:
        return None

    ball_idx, player_indices = get_ball_and_player_indices(entity_type)
    if ball_idx is None or len(player_indices) == 0:
        return None

    # ── Identify passer at kick frame (nearest player to ball) ──
    if not mask_arr[kick_frame_local, ball_idx]:
        return None
    vis_mask = mask_arr[kick_frame_local, player_indices].astype(bool)
    vis_players = player_indices[vis_mask]
    if len(vis_players) == 0:
        return None
    bx_kf = features[kick_frame_local, ball_idx, 0]
    by_kf = features[kick_frame_local, ball_idx, 1]
    px_kf = features[kick_frame_local, vis_players, 0]
    py_kf = features[kick_frame_local, vis_players, 1]
    dists_kf = np.sqrt((px_kf - bx_kf) ** 2 + (py_kf - by_kf) ** 2)
    passer_entity = int(vis_players[np.argmin(dists_kf)])

    # ── Detect reception event after kick ──
    reception = detect_reception(
        features, mask_arr, ball_idx, player_indices,
        kick_frame_local, passer_entity,
    )

    samples_bf: List[np.ndarray] = []
    samples_pf: List[np.ndarray] = []
    samples_lb: List[int] = []

    # ── Positive samples: near reception frame ──
    if reception is not None:
        recv_frame, receiver_entity = reception
        pos_start = max(recv_frame - POS_WINDOW + 1, kick_frame_local + TOUCH_SKIP)
        for t in range(pos_start, recv_frame + 1):
            if t < H or t >= T:
                continue
            sample = build_sample(
                features, mask_arr, ball_idx, player_indices,
                frame=t,
                kick_frame=kick_frame_local,
                exclude_entity=passer_entity,
                target_entity=receiver_entity,
            )
            if sample is not None:
                samples_bf.append(sample["ball_feats"])
                samples_pf.append(sample["player_feats"])
                samples_lb.append(sample["label"])

    # ── Negative samples: during ball flight (passer excluded) ──
    neg_start = kick_frame_local + TOUCH_SKIP
    neg_end = (reception[0] - POS_WINDOW) if reception else min(
        kick_frame_local + RECV_MAX_SEARCH, T - 1
    )
    eligible = [
        t for t in range(max(neg_start, H), neg_end + 1)
        if mask_arr[t, ball_idx]
    ]

    if eligible:
        n_pick = min(n_neg, len(eligible))
        neg_frames = rng.choice(eligible, size=n_pick, replace=False)

        for t in neg_frames:
            sample = build_sample(
                features, mask_arr, ball_idx, player_indices,
                frame=t,
                kick_frame=kick_frame_local,
                label=K,  # no-touch
                exclude_entity=passer_entity,
            )
            if sample is not None:
                samples_bf.append(sample["ball_feats"])
                samples_pf.append(sample["player_feats"])
                samples_lb.append(sample["label"])

    if not samples_bf:
        return None

    return {
        "ball_feats":   np.stack(samples_bf),                          # [N, 6]
        "player_feats": np.stack(samples_pf),                          # [N, K, 32]
        "labels":       np.array(samples_lb, dtype=np.int64),          # [N]
    }


def process_clip_survival(
    clip_path: str,
    kick_frame_local: Optional[int],
) -> Optional[dict]:
    """Process one clip for survival training: emit EVERY frame from kick+TOUCH_SKIP
    to min(recv_frame, kick+MAX_SEARCH).

    Returns dict with:
      ball_feats[N_frames, 6], player_feats[N_frames, K, 32],
      is_event[N_frames] (0/1), player_label[N_frames] (0..K-1 or -1)
    """
    data = np.load(clip_path)
    features = data["features"]       # [T, N, 6]
    mask_arr = data["mask"]           # [T, N]
    entity_type = data["entity_type"] # [N]

    T = features.shape[0]

    if kick_frame_local is None:
        if "kick_frame_local" in data:
            kick_frame_local = int(data["kick_frame_local"])
        else:
            return None
    if kick_frame_local < 0 or kick_frame_local >= T:
        return None

    ball_idx, player_indices = get_ball_and_player_indices(entity_type)
    if ball_idx is None or len(player_indices) == 0:
        return None

    # Identify passer at kick frame
    if not mask_arr[kick_frame_local, ball_idx]:
        return None
    vis_mask = mask_arr[kick_frame_local, player_indices].astype(bool)
    vis_players = player_indices[vis_mask]
    if len(vis_players) == 0:
        return None
    bx_kf = features[kick_frame_local, ball_idx, 0]
    by_kf = features[kick_frame_local, ball_idx, 1]
    px_kf = features[kick_frame_local, vis_players, 0]
    py_kf = features[kick_frame_local, vis_players, 1]
    dists_kf = np.sqrt((px_kf - bx_kf) ** 2 + (py_kf - by_kf) ** 2)
    passer_entity = int(vis_players[np.argmin(dists_kf)])

    # Detect reception
    reception = detect_reception(
        features, mask_arr, ball_idx, player_indices,
        kick_frame_local, passer_entity,
    )

    recv_frame = reception[0] if reception else None
    receiver_entity = reception[1] if reception else None

    # Frame range: kick+TOUCH_SKIP to min(recv_frame, kick+MAX_SEARCH)
    frame_start = kick_frame_local + TOUCH_SKIP
    if recv_frame is not None:
        frame_end = recv_frame  # inclusive
    else:
        frame_end = min(kick_frame_local + MAX_SEARCH, T - 1)

    if frame_start > frame_end or frame_start >= T:
        return None

    all_bf, all_pf, all_ie, all_pl = [], [], [], []

    for t in range(max(frame_start, H), frame_end + 1):
        if t >= T:
            break

        # is_event: 1 only at the reception frame
        is_event = 1 if (recv_frame is not None and t == recv_frame) else 0

        # Build sample (passer excluded from K-NN)
        if is_event and receiver_entity is not None:
            sample = build_sample(
                features, mask_arr, ball_idx, player_indices,
                frame=t, kick_frame=kick_frame_local,
                exclude_entity=passer_entity,
                target_entity=receiver_entity,
            )
        else:
            sample = build_sample(
                features, mask_arr, ball_idx, player_indices,
                frame=t, kick_frame=kick_frame_local,
                label=0,  # placeholder, won't be used for CE loss
                exclude_entity=passer_entity,
            )

        if sample is None:
            continue

        player_label = sample["label"] if is_event else -1

        all_bf.append(sample["ball_feats"])
        all_pf.append(sample["player_feats"])
        all_ie.append(is_event)
        all_pl.append(player_label)

    if not all_bf:
        return None

    return {
        "ball_feats":   np.stack(all_bf),                              # [N, 6]
        "player_feats": np.stack(all_pf),                              # [N, K, 32]
        "is_event":     np.array(all_ie, dtype=np.int64),              # [N]
        "player_label": np.array(all_pl, dtype=np.int64),              # [N]
    }


def process_split(
    sportec_dir: Path,
    out_dir: Path,
    split: str,
    n_neg: int,
    rng: np.random.Generator,
) -> dict:
    """Process one split (train or val). Returns stats dict."""
    manifest_path = sportec_dir / split / "manifest.jsonl"
    if not manifest_path.exists():
        print(f"  [warn] No manifest at {manifest_path}")
        return {"n_clips": 0, "n_samples": 0, "n_touch": 0, "n_no_touch": 0}

    out_split = out_dir / split
    out_split.mkdir(parents=True, exist_ok=True)

    # Read manifest
    entries = []
    with open(manifest_path) as f:
        for line in f:
            entries.append(json.loads(line))

    n_clips = 0
    total_samples = 0
    total_touch = 0
    total_no_touch = 0

    # Process in batches and save per-batch .npz files
    BATCH = 500  # clips per output .npz file
    batch_bf, batch_pf, batch_lb = [], [], []
    file_idx = 0

    for entry in tqdm(entries, desc=f"  {split}"):
        clip_path = entry["clip_path"]

        # kick_frame_local must be read from the npz -- the manifest's
        # kick_frame - start_frame is unreliable (frame numbering mismatch
        # for some matches).  process_clip loads the npz anyway, so we pass
        # kick_frame_local=None and let it read from the npz.
        result = process_clip(clip_path, None, n_neg, rng)
        if result is None:
            continue

        n_clips += 1
        n_samp = len(result["labels"])
        n_t = int((result["labels"] < K).sum())
        total_samples += n_samp
        total_touch += n_t
        total_no_touch += (n_samp - n_t)

        batch_bf.append(result["ball_feats"])
        batch_pf.append(result["player_feats"])
        batch_lb.append(result["labels"])

        if len(batch_bf) >= BATCH:
            np.savez_compressed(
                str(out_split / f"sportec_{file_idx:04d}.npz"),
                ball_feats=np.concatenate(batch_bf),
                player_feats=np.concatenate(batch_pf),
                labels=np.concatenate(batch_lb),
            )
            file_idx += 1
            batch_bf, batch_pf, batch_lb = [], [], []

    # Flush remaining
    if batch_bf:
        np.savez_compressed(
            str(out_split / f"sportec_{file_idx:04d}.npz"),
            ball_feats=np.concatenate(batch_bf),
            player_feats=np.concatenate(batch_pf),
            labels=np.concatenate(batch_lb),
        )

    stats = {
        "n_clips": n_clips,
        "n_samples": total_samples,
        "n_touch": total_touch,
        "n_no_touch": total_no_touch,
    }
    print(f"  {split}: {n_clips} clips → {total_samples} samples "
          f"({total_touch} touch, {total_no_touch} no-touch)")
    return stats


def process_split_survival(
    sportec_dir: Path,
    out_dir: Path,
    split: str,
    all_touches: bool = False,
) -> dict:
    """Process one split for survival mode. Returns stats dict."""
    manifest_path = sportec_dir / split / "manifest.jsonl"
    if not manifest_path.exists():
        print(f"  [warn] No manifest at {manifest_path}")
        return {"n_clips": 0, "n_frames": 0, "n_events": 0, "n_nonevents": 0}

    out_split = out_dir / split
    out_split.mkdir(parents=True, exist_ok=True)

    entries = []
    with open(manifest_path) as f:
        for line in f:
            entries.append(json.loads(line))

    n_clips = 0
    total_frames = 0
    total_events = 0
    total_nonevents = 0

    BATCH = 500
    batch_bf, batch_pf, batch_ie, batch_pl = [], [], [], []
    file_idx = 0

    def _flush():
        nonlocal file_idx, batch_bf, batch_pf, batch_ie, batch_pl
        if not batch_bf:
            return
        np.savez_compressed(
            str(out_split / f"survival_{file_idx:04d}.npz"),
            ball_feats=np.concatenate(batch_bf),
            player_feats=np.concatenate(batch_pf),
            is_event=np.concatenate(batch_ie),
            player_label=np.concatenate(batch_pl),
        )
        file_idx += 1
        batch_bf, batch_pf, batch_ie, batch_pl = [], [], [], []

    # Pass-only (labeled kick → reception)
    for entry in tqdm(entries, desc=f"  {split} (pass)"):
        clip_path = entry["clip_path"]
        result = process_clip_survival(clip_path, None)
        if result is None:
            continue
        n_clips += 1
        n_f = len(result["is_event"])
        n_e = int(result["is_event"].sum())
        total_frames += n_f
        total_events += n_e
        total_nonevents += (n_f - n_e)
        batch_bf.append(result["ball_feats"])
        batch_pf.append(result["player_feats"])
        batch_ie.append(result["is_event"])
        batch_pl.append(result["player_label"])
        if len(batch_bf) >= BATCH:
            _flush()

    _flush()
    pass_frames = total_frames
    pass_events = total_events
    print(f"  {split} pass-only: {n_clips} clips → {pass_frames} frames "
          f"({pass_events} events)")

    # All-touches (every consecutive touch pair)
    if all_touches:
        at_clips = 0
        at_frames = 0
        at_events = 0
        for entry in tqdm(entries, desc=f"  {split} (alltouch)"):
            clip_path = entry["clip_path"]
            result = process_clip_survival_alltouch(clip_path)
            if result is None:
                continue
            at_clips += 1
            n_f = len(result["is_event"])
            n_e = int(result["is_event"].sum())
            at_frames += n_f
            at_events += n_e
            total_frames += n_f
            total_events += n_e
            total_nonevents += (n_f - n_e)
            batch_bf.append(result["ball_feats"])
            batch_pf.append(result["player_feats"])
            batch_ie.append(result["is_event"])
            batch_pl.append(result["player_label"])
            if len(batch_bf) >= BATCH:
                _flush()
        _flush()
        print(f"  {split} all-touch: {at_clips} clips → {at_frames} frames "
              f"({at_events} events)")

    event_rate = total_events / max(total_frames, 1) * 100
    stats = {
        "n_clips": n_clips,
        "n_frames": total_frames,
        "n_events": total_events,
        "n_nonevents": total_nonevents,
        "event_rate_pct": round(event_rate, 2),
    }
    print(f"  {split} TOTAL: {total_frames} frames "
          f"({total_events} events, rate={event_rate:.2f}%)")
    return stats


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    sportec_dir = Path(args.sportec_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Sportec dir: {sportec_dir}")
    print(f"Output dir:  {out_dir}")
    print(f"Mode: {args.mode}")
    print()

    if args.mode == "survival":
        stats = {}
        for split in ("train", "val"):
            use_at = args.all_touches and split == "train"
            stats[split] = process_split_survival(sportec_dir, out_dir, split,
                                                   all_touches=use_at)

        meta = {
            "source": "sportec_xyz_metres",
            "mode": "survival",
            "k": K,
            "history": H,
            "ball_dim": BALL_DIM,
            "player_feat_dim": PLAYER_FEAT_DIM,
            "pitch_half_x": PITCH_HALF_X,
            "pitch_half_y": PITCH_HALF_Y,
            "vel_scale": VEL_SCALE,
            "touch_skip": TOUCH_SKIP,
            "max_search": MAX_SEARCH,
            "passer_excluded": True,
            "recv_vel_delta": RECV_VEL_DELTA,
            "recv_proximity": RECV_PROXIMITY,
            **stats,
        }
        with open(out_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        total = stats["train"]["n_frames"] + stats["val"]["n_frames"]
        print(f"\nDone. Total frames: {total}")
        print(f"  train: {stats['train']['n_frames']} ({stats['train']['n_events']} events)")
        print(f"  val:   {stats['val']['n_frames']} ({stats['val']['n_events']} events)")
    else:
        print(f"Negatives per clip: {args.n_neg}")
        stats = {}
        for split in ("train", "val"):
            stats[split] = process_split(sportec_dir, out_dir, split, args.n_neg, rng)

        meta = {
            "source": "sportec_xyz_metres",
            "mode": "classifier",
            "k": K,
            "history": H,
            "ball_dim": BALL_DIM,
            "player_feat_dim": PLAYER_FEAT_DIM,
            "pitch_half_x": PITCH_HALF_X,
            "pitch_half_y": PITCH_HALF_Y,
            "vel_scale": VEL_SCALE,
            "n_neg_per_clip": args.n_neg,
            "passer_excluded": True,
            "recv_vel_delta": RECV_VEL_DELTA,
            "recv_proximity": RECV_PROXIMITY,
            "pos_window": POS_WINDOW,
            **stats,
        }
        with open(out_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        total = stats["train"]["n_samples"] + stats["val"]["n_samples"]
        print(f"\nDone. Total samples: {total}")
        print(f"  train: {stats['train']['n_samples']}")
        print(f"  val:   {stats['val']['n_samples']}")


if __name__ == "__main__":
    main()
