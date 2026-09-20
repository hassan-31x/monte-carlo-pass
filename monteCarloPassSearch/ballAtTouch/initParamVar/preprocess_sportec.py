#!/usr/bin/env python3
"""Extract ballAtTouch training samples from converted Sportec clips.

Each Sportec clip has a single kick event at kick_frame_local.
We detect the RECEPTION frame (where a non-passer player touches the ball)
and sample there: the input context is at the reception frame, and the label
is the outgoing ball velocity right after the receiver touches it.

Features are [x_m, y_m, vx_ms, vy_ms, z_m, vz_ms] per entity per frame.
Entity types: 0=home, 1=away, 2=ball.

Output: per-split .npz files with keys:
  ball_feats[N,6], self_feats[N,32], other_feats[N,K_OTHER,32],
  label_vel[N,3], vel_obs_mask[N,3]
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from tqdm import tqdm

# Constants - must match OPTA BAT preprocessing
HALF_X = 52.5
HALF_Y = 34.0
VEL_SCALE = 10.0
Z_SCALE = 5.0
K_OTHER = 8
H_HIST = 8
FEAT_DIM = 4  # [rel_px, rel_py, vx, vy] per frame

# Reception detection parameters (same as PTT)
TOUCH_SKIP = 5       # skip first N frames after kick
RECV_VEL_DELTA = 2.0  # m/s ball velocity-change threshold
RECV_PROXIMITY = 3.0  # m player-ball distance at reception
RECV_MAX_SEARCH = 200  # max frames after kick to search
MAX_OUTGOING_SPEED = 40.0  # m/s — filter tracking errors

SPORTEC_BASE = Path(
    "/mnt/data/remains/opta2026/ballPlayerTrajModel/publicData/preprocessed/sportec_xyz_metres"
)
OUT_DIR = Path("/mnt/data/remains/opta2026/ballAtTouch/preprocessed_sportec_recv")


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


def _build_sample_at_touch(
    features: np.ndarray,       # [T, N, 6]
    mask: np.ndarray,           # [T, N]
    ball_idx: int,
    player_indices: np.ndarray,
    touch_frame: int,
    toucher_entity: int,
) -> Optional[dict]:
    """Build a BAT sample at a given touch frame.

    The toucher is "self", label is outgoing ball velocity at touch_frame+1.
    """
    T = features.shape[0]

    if touch_frame + 1 >= T:
        return None
    if not mask[touch_frame, ball_idx] or not mask[touch_frame + 1, ball_idx]:
        return None
    if touch_frame < H_HIST - 1:
        return None

    ball_pos = features[touch_frame, ball_idx, :2]
    ball_vel = features[touch_frame, ball_idx, 2:4]
    ball_z   = features[touch_frame, ball_idx, 4]
    ball_vz  = features[touch_frame, ball_idx, 5]

    # Anchor: ball position one frame before touch
    anchor_frame = touch_frame - 1
    if anchor_frame < 0 or not mask[anchor_frame, ball_idx]:
        anchor_pos = ball_pos.copy()
    else:
        anchor_pos = features[anchor_frame, ball_idx, :2]

    ball_feats = np.array([
        ball_pos[0] / HALF_X,
        ball_pos[1] / HALF_Y,
        ball_vel[0] / VEL_SCALE,
        ball_vel[1] / VEL_SCALE,
        (anchor_pos[0] - ball_pos[0]) / HALF_X,
        (anchor_pos[1] - ball_pos[1]) / HALF_Y,
        ball_z / Z_SCALE,
        ball_vz / VEL_SCALE,
    ], dtype=np.float32)

    # Self = toucher
    self_feats = _player_history(features, mask, touch_frame, toucher_entity, ball_pos)

    # K_OTHER nearest players excluding toucher
    vis = mask[touch_frame, player_indices].astype(bool)
    valid_players = player_indices[vis]
    others_valid = valid_players[valid_players != toucher_entity]
    if len(others_valid) == 0:
        other_indices = np.full(K_OTHER, toucher_entity, dtype=int)
    else:
        other_positions = features[touch_frame, others_valid, :2]
        other_dists = np.linalg.norm(other_positions - ball_pos[None, :], axis=1)
        sorted_idx = np.argsort(other_dists)[:K_OTHER]
        other_indices = others_valid[sorted_idx]
        if len(other_indices) < K_OTHER:
            pad = np.full(K_OTHER - len(other_indices), other_indices[-1], dtype=int)
            other_indices = np.concatenate([other_indices, pad])

    other_feats = np.stack([
        _player_history(features, mask, touch_frame, int(pi), ball_pos)
        for pi in other_indices
    ])

    # Label: outgoing ball velocity at touch_frame+1
    vx_out = features[touch_frame + 1, ball_idx, 2]
    vy_out = features[touch_frame + 1, ball_idx, 3]
    vz_out = features[touch_frame + 1, ball_idx, 5]

    speed_2d = np.sqrt(vx_out**2 + vy_out**2)
    if speed_2d > MAX_OUTGOING_SPEED or abs(vz_out) > MAX_OUTGOING_SPEED:
        return None

    label_vel = np.array([
        vx_out / VEL_SCALE,
        vy_out / VEL_SCALE,
        vz_out / VEL_SCALE,
    ], dtype=np.float32)

    vel_obs_mask = np.array([True, True, True], dtype=bool)

    return {
        "ball_feats": ball_feats,
        "self_feats": self_feats,
        "other_feats": other_feats,
        "label_vel": label_vel,
        "vel_obs_mask": vel_obs_mask,
    }


def extract_alltouch_samples(clip_path: str) -> List[dict]:
    """Extract BAT samples from ALL detected touches in a clip."""
    d = np.load(clip_path)
    features = d["features"]
    entity_type = d["entity_type"]
    mask = d["mask"]

    ball_indices = np.where(entity_type == 2)[0]
    if len(ball_indices) == 0:
        return []
    ball_idx = ball_indices[0]
    player_indices = np.where(entity_type != 2)[0]

    touches = detect_all_touches(features, mask, ball_idx, player_indices)

    samples = []
    for touch_frame, toucher_entity in touches:
        sample = _build_sample_at_touch(
            features, mask, ball_idx, player_indices,
            touch_frame, toucher_entity,
        )
        if sample is not None:
            samples.append(sample)

    return samples


def detect_reception(
    features: np.ndarray,       # [T, N, 6]
    mask: np.ndarray,           # [T, N]
    ball_idx: int,
    player_indices: np.ndarray,
    kick_frame: int,
    passer_entity: int,
) -> Optional[Tuple[int, int]]:
    """Detect first reception event after kick_frame (excluding passer).

    Returns (frame, receiver_entity_idx) or None.
    """
    T = features.shape[0]
    for t in range(kick_frame + TOUCH_SKIP, min(kick_frame + RECV_MAX_SEARCH, T)):
        if not mask[t, ball_idx] or not mask[t - 1, ball_idx]:
            continue
        bv_curr = features[t, ball_idx, 2:4]
        bv_prev = features[t - 1, ball_idx, 2:4]
        vel_change = np.linalg.norm(bv_curr - bv_prev)
        if vel_change < RECV_VEL_DELTA:
            continue
        bp = features[t, ball_idx, :2]
        vis = mask[t, player_indices].astype(bool)
        vis_players = player_indices[vis]
        vis_players = vis_players[vis_players != passer_entity]
        if len(vis_players) == 0:
            continue
        dists = np.sqrt(np.sum((features[t, vis_players, :2] - bp) ** 2, axis=1))
        if dists.min() <= RECV_PROXIMITY:
            return (t, int(vis_players[np.argmin(dists)]))
    return None


def _player_history(
    features: np.ndarray,  # [T, N, 6]
    mask: np.ndarray,       # [T, N]
    frame: int,
    player_idx: int,
    ball_pos: np.ndarray,   # [2] ball position at frame
) -> np.ndarray:
    """Build [H_HIST * FEAT_DIM] feature vector for a player's recent history."""
    feats = np.zeros(H_HIST * FEAT_DIM, dtype=np.float32)
    last_valid = np.zeros(FEAT_DIM, dtype=np.float32)

    for h in range(H_HIST):
        t = frame - H_HIST + 1 + h
        if t < 0 or mask[t, player_idx] == 0:
            feats[h * FEAT_DIM:(h + 1) * FEAT_DIM] = last_valid
            continue

        pp = features[t, player_idx, :2]
        pv = features[t, player_idx, 2:4]

        feat = np.array([
            (pp[0] - ball_pos[0]) / HALF_X,
            (pp[1] - ball_pos[1]) / HALF_Y,
            pv[0] / VEL_SCALE,
            pv[1] / VEL_SCALE,
        ], dtype=np.float32)
        feats[h * FEAT_DIM:(h + 1) * FEAT_DIM] = feat
        last_valid = feat

    return feats


def extract_sample(clip_path: str) -> Optional[dict]:
    """Extract one BAT sample at the RECEPTION frame of a Sportec clip."""
    d = np.load(clip_path)
    features = d["features"]       # [T, N, 6]
    entity_type = d["entity_type"]  # [N]
    mask = d["mask"]                # [T, N]
    kf = int(d["kick_frame_local"])

    T, N, _ = features.shape

    ball_indices = np.where(entity_type == 2)[0]
    if len(ball_indices) == 0:
        return None
    ball_idx = ball_indices[0]

    player_indices = np.where(entity_type != 2)[0]

    # Identify passer at kick frame (nearest player to ball)
    if kf < 0 or kf >= T or not mask[kf, ball_idx]:
        return None
    vis_kf = mask[kf, player_indices].astype(bool)
    vis_players_kf = player_indices[vis_kf]
    if len(vis_players_kf) == 0:
        return None
    bp_kf = features[kf, ball_idx, :2]
    dists_kf = np.linalg.norm(features[kf, vis_players_kf, :2] - bp_kf, axis=1)
    passer_entity = int(vis_players_kf[np.argmin(dists_kf)])

    # Detect reception
    reception = detect_reception(features, mask, ball_idx, player_indices,
                                 kf, passer_entity)
    if reception is None:
        return None

    recv_frame, receiver_entity = reception

    # Need recv_frame+1 for outgoing velocity label
    if recv_frame + 1 >= T:
        return None
    if not mask[recv_frame, ball_idx] or not mask[recv_frame + 1, ball_idx]:
        return None
    # Need H_HIST frames of history before reception
    if recv_frame < H_HIST - 1:
        return None

    # --- Build features at RECEPTION frame ---
    ball_pos = features[recv_frame, ball_idx, :2]
    ball_vel = features[recv_frame, ball_idx, 2:4]
    ball_z   = features[recv_frame, ball_idx, 4]
    ball_vz  = features[recv_frame, ball_idx, 5]

    # Anchor: ball position one frame before reception
    anchor_frame = recv_frame - 1
    if anchor_frame < 0 or not mask[anchor_frame, ball_idx]:
        anchor_pos = ball_pos.copy()
    else:
        anchor_pos = features[anchor_frame, ball_idx, :2]

    ball_feats = np.array([
        ball_pos[0] / HALF_X,
        ball_pos[1] / HALF_Y,
        ball_vel[0] / VEL_SCALE,
        ball_vel[1] / VEL_SCALE,
        (anchor_pos[0] - ball_pos[0]) / HALF_X,
        (anchor_pos[1] - ball_pos[1]) / HALF_Y,
        ball_z / Z_SCALE,
        ball_vz / VEL_SCALE,
    ], dtype=np.float32)

    # Self = receiver
    self_feats = _player_history(features, mask, recv_frame, receiver_entity, ball_pos)

    # K_OTHER nearest players excluding receiver
    vis_recv = mask[recv_frame, player_indices].astype(bool)
    valid_players = player_indices[vis_recv]
    others_valid = valid_players[valid_players != receiver_entity]
    if len(others_valid) == 0:
        other_indices = np.full(K_OTHER, receiver_entity, dtype=int)
    else:
        other_positions = features[recv_frame, others_valid, :2]
        other_dists = np.linalg.norm(other_positions - ball_pos[None, :], axis=1)
        sorted_idx = np.argsort(other_dists)[:K_OTHER]
        other_indices = others_valid[sorted_idx]
        if len(other_indices) < K_OTHER:
            pad = np.full(K_OTHER - len(other_indices), other_indices[-1], dtype=int)
            other_indices = np.concatenate([other_indices, pad])

    other_feats = np.stack([
        _player_history(features, mask, recv_frame, int(pi), ball_pos)
        for pi in other_indices
    ])  # [K_OTHER, H_HIST*FEAT_DIM]

    # Label: outgoing ball velocity at recv_frame+1
    vx_out = features[recv_frame + 1, ball_idx, 2]
    vy_out = features[recv_frame + 1, ball_idx, 3]
    vz_out = features[recv_frame + 1, ball_idx, 5]

    # Filter tracking errors (unrealistic speeds)
    speed_2d = np.sqrt(vx_out**2 + vy_out**2)
    if speed_2d > MAX_OUTGOING_SPEED or abs(vz_out) > MAX_OUTGOING_SPEED:
        return None

    label_vel = np.array([
        vx_out / VEL_SCALE,
        vy_out / VEL_SCALE,
        vz_out / VEL_SCALE,
    ], dtype=np.float32)

    vel_obs_mask = np.array([True, True, True], dtype=bool)

    return {
        "ball_feats": ball_feats,
        "self_feats": self_feats,
        "other_feats": other_feats,
        "label_vel": label_vel,
        "vel_obs_mask": vel_obs_mask,
    }


def _save_split(out_split: Path, filename: str, samples: List[dict]) -> int:
    """Stack samples and save as a single .npz file."""
    if not samples:
        return 0
    out_split.mkdir(parents=True, exist_ok=True)

    ball_feats = np.stack([s["ball_feats"] for s in samples])
    self_feats = np.stack([s["self_feats"] for s in samples])
    other_feats = np.stack([s["other_feats"] for s in samples])
    label_vel = np.stack([s["label_vel"] for s in samples])
    vel_obs_mask = np.stack([s["vel_obs_mask"] for s in samples])

    np.savez_compressed(
        str(out_split / filename),
        ball_feats=ball_feats,
        self_feats=self_feats,
        other_feats=other_feats,
        label_vel=label_vel,
        vel_obs_mask=vel_obs_mask,
    )

    n = len(label_vel)
    print(f"    {filename}: {n} samples")
    print(f"      ball_feats: {ball_feats.shape}")
    print(f"      label_vel mean_abs={np.abs(label_vel).mean(axis=0)}")
    return n


def process_split(split: str, all_touches: bool = False) -> int:
    """Process all clips in one split."""
    manifest_path = SPORTEC_BASE / split / "manifest.jsonl"
    if not manifest_path.exists():
        print(f"  [skip] No manifest at {manifest_path}")
        return 0

    entries: List[dict] = []
    with open(manifest_path) as f:
        for line in f:
            entries.append(json.loads(line))

    # Always collect pass-only (labeled kick → reception) samples
    pass_samples: List[dict] = []
    skipped = 0
    for entry in tqdm(entries, desc=f"sportec-{split} (pass)"):
        sample = extract_sample(entry["clip_path"])
        if sample is None:
            skipped += 1
            continue
        pass_samples.append(sample)

    print(f"  {split} pass-only: {len(pass_samples)} samples ({skipped} clips skipped)")

    out_split = OUT_DIR / split

    if not all_touches:
        return _save_split(out_split, "sportec_all.npz", pass_samples)

    # All-touches: extract from every detected touch in every clip
    touch_samples: List[dict] = []
    # Collect pass-reception frames to deduplicate
    pass_frames = set()  # (clip_path, touch_frame) already covered by pass samples
    for entry in tqdm(entries, desc=f"sportec-{split} (alltouch)"):
        clip_path = entry["clip_path"]
        samples = extract_alltouch_samples(clip_path)
        touch_samples.extend(samples)

    print(f"  {split} all-touch: {len(touch_samples)} samples from all clips")

    # Save pass samples and all-touch samples as separate files
    # (dataset loader concatenates all .npz in the directory)
    n1 = _save_split(out_split, "sportec_pass.npz", pass_samples)
    n2 = _save_split(out_split, "sportec_alltouch.npz", touch_samples)
    return n1 + n2


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--all-touches", action="store_true",
                   help="Extract from all detected touches (train only)")
    p.add_argument("--out-dir", type=str, default=None,
                   help="Override output directory")
    p.add_argument("--sportec-dir", type=str, default=None,
                   help="Override source directory containing split manifests")
    args = p.parse_args()

    global OUT_DIR, SPORTEC_BASE
    if args.out_dir:
        OUT_DIR = Path(args.out_dir)
    if args.sportec_dir:
        SPORTEC_BASE = Path(args.sportec_dir)

    print("=" * 60)
    print("Extracting ballAtTouch samples from Sportec clips")
    print(f"  Source: {SPORTEC_BASE}")
    print(f"  Output: {OUT_DIR}")
    print(f"  All-touches (train): {args.all_touches}")
    print("=" * 60)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    total = 0
    # Train: optionally use all-touches; Val: always pass-only
    total += process_split("train", all_touches=args.all_touches)
    total += process_split("val", all_touches=False)

    print(f"\nDone. Total Sportec BAT samples: {total}")


if __name__ == "__main__":
    main()
