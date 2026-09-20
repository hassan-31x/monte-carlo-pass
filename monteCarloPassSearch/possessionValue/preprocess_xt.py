#!/usr/bin/env python3
"""Create xT (Expected Threat) dataset from Sportec tracking clips + shot events.

Links shot events (with CalculatedFrame) to preprocessed tracking clips.
For each window of k frames, labels are based on future shot events within a horizon.
"""

import argparse
import json
import glob
import os
import numpy as np
from pathlib import Path
from collections import defaultdict


def load_sportec_shots(shots_path: str):
    """Load Sportec shot events with CalculatedFrame for linking to tracking."""
    with open(shots_path) as f:
        shots = json.load(f)

    by_match = defaultdict(list)
    for s in shots:
        match_id = s["match_id"]
        frame = s.get("CalculatedFrame")
        if frame is None:
            continue
        frame = int(frame)
        xg = float(s.get("xG") or 0.05)
        is_goal = "GOAL" in str(s.get("result", "")).upper()
        side = s.get("side", "unknown")
        by_match[match_id].append({
            "frame": frame,
            "xG": xg,
            "is_goal": is_goal,
            "side": side,
        })

    # Sort by frame within each match
    for mid in by_match:
        by_match[mid].sort(key=lambda s: s["frame"])

    return dict(by_match)


def load_clip_manifests(preprocessed_dir: str, splits=("train", "val", "test")):
    """Load clip manifests and group by match."""
    clips_by_match = defaultdict(list)
    split_map = {}

    for split in splits:
        manifest_path = os.path.join(preprocessed_dir, split, "manifest.jsonl")
        if not os.path.exists(manifest_path):
            continue
        with open(manifest_path) as f:
            for line in f:
                entry = json.loads(line.strip())
                # source_file is like "sportec:J03WMX"
                match_id = entry["source_file"].split(":")[-1]
                entry["match_id"] = match_id
                entry["split"] = split
                clips_by_match[match_id].append(entry)
                split_map[entry["clip_path"]] = split

    return dict(clips_by_match), split_map


def find_shots_in_clip(clip_start, clip_end, shots):
    """Find shots whose CalculatedFrame falls within clip frame range."""
    result = []
    for s in shots:
        if clip_start <= s["frame"] <= clip_end:
            local_frame = s["frame"] - clip_start
            result.append({**s, "local_frame": local_frame})
    return result


def create_xt_windows(
    clip_path: str,
    clip_shots: list,
    window_size: int = 64,
    stride: int = 32,
    horizon_frames: int = 250,
    discount_per_frame: float = 0.998,
):
    """
    Create xT windows from a single clip.

    For each window of window_size frames, compute labels:
    - home_threat: discounted xG of next home shot within horizon
    - away_threat: discounted xG of next away shot within horizon
    - has_shot_home: binary indicator
    - has_shot_away: binary indicator

    Returns list of (features, mask, entity_type, labels) tuples.
    """
    data = np.load(clip_path)
    features = data["features"]  # [T, 23, 6]
    mask = data["mask"]  # [T, 23]
    entity_type = data["entity_type"]  # [23]
    T = features.shape[0]

    windows = []

    for start in range(0, T - window_size, stride):
        end = start + window_size

        # Look ahead: [end, end+horizon_frames)
        home_threat = 0.0
        away_threat = 0.0
        has_shot_home = 0.0
        has_shot_away = 0.0
        next_xg = 0.0
        next_shot_side = "none"

        for s in clip_shots:
            lf = s["local_frame"]
            if lf < end:
                continue
            if lf >= end + horizon_frames:
                break

            delta = lf - end
            discount = discount_per_frame ** delta
            xg_val = s["xG"] * discount

            if s["side"] == "home":
                if has_shot_home < 0.5:  # only first shot
                    home_threat = xg_val
                    has_shot_home = 1.0
                    if next_shot_side == "none":
                        next_xg = s["xG"]
                        next_shot_side = "home"
            elif s["side"] == "away":
                if has_shot_away < 0.5:
                    away_threat = xg_val
                    has_shot_away = 1.0
                    if next_shot_side == "none":
                        next_xg = s["xG"]
                        next_shot_side = "away"

        # xT = home_threat - away_threat
        xt_value = home_threat - away_threat

        window_features = features[start:end]  # [W, 23, 6]
        window_mask = mask[start:end]  # [W, 23]

        labels = np.array([
            xt_value,
            home_threat,
            away_threat,
            has_shot_home,
            has_shot_away,
            next_xg,
        ], dtype=np.float32)

        windows.append({
            "features": window_features.astype(np.float32),
            "mask": window_mask.astype(np.uint8),
            "entity_type": entity_type.astype(np.int64),
            "labels": labels,
        })

    return windows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preprocessed-dir",
        default="/mnt/data/remains/opta2026/ballPlayerTrajModel/publicData/preprocessed/sportec_xyz_metres",
    )
    parser.add_argument(
        "--sportec-shots",
        default="/mnt/data/remains/opta2026/expectedThreat/sportec_shots.json",
    )
    parser.add_argument(
        "--out-dir",
        default="/mnt/data/remains/opta2026/expectedThreat/xt_data",
    )
    parser.add_argument("--window-size", type=int, default=64)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--horizon-frames", type=int, default=250, help="Look-ahead frames for shot search (250=10s)")
    parser.add_argument("--discount", type=float, default=0.998)
    args = parser.parse_args()

    out = Path(args.out_dir)

    print("Loading Sportec shot events...")
    shots_by_match = load_sportec_shots(args.sportec_shots)
    total_shots = sum(len(v) for v in shots_by_match.values())
    print(f"  {total_shots} shots across {len(shots_by_match)} matches")

    print("Loading clip manifests...")
    clips_by_match, split_map = load_clip_manifests(args.preprocessed_dir)
    total_clips = sum(len(v) for v in clips_by_match.values())
    print(f"  {total_clips} clips across {len(clips_by_match)} matches")

    # Process clips and create windows
    split_windows = defaultdict(list)
    clips_with_shots = 0
    total_windows = 0
    total_positive = 0

    for match_id, clips in clips_by_match.items():
        match_shots = shots_by_match.get(match_id, [])
        if not match_shots:
            print(f"  Warning: no shots found for match {match_id}")

        for clip_entry in clips:
            clip_path = clip_entry["clip_path"]
            split = clip_entry["split"]
            start_frame = clip_entry["start_frame"]
            end_frame = clip_entry["end_frame"]

            clip_shots = find_shots_in_clip(start_frame, end_frame, match_shots)
            if clip_shots:
                clips_with_shots += 1

            windows = create_xt_windows(
                clip_path=clip_path,
                clip_shots=clip_shots,
                window_size=args.window_size,
                stride=args.stride,
                horizon_frames=args.horizon_frames,
                discount_per_frame=args.discount,
            )

            for w in windows:
                split_windows[split].append(w)
                total_windows += 1
                if abs(w["labels"][0]) > 1e-6:
                    total_positive += 1

    print(f"\nTotal windows: {total_windows}")
    print(f"Windows with non-zero xT: {total_positive} ({100*total_positive/max(total_windows,1):.1f}%)")
    print(f"Clips containing shots: {clips_with_shots}/{total_clips}")

    # Save per split
    for split, windows in split_windows.items():
        if not windows:
            continue

        split_dir = out / split
        split_dir.mkdir(parents=True, exist_ok=True)

        # Save as individual npz files in chunks for memory efficiency
        chunk_size = 2000
        n_chunks = (len(windows) + chunk_size - 1) // chunk_size

        manifest_entries = []
        for chunk_idx in range(n_chunks):
            chunk_start = chunk_idx * chunk_size
            chunk_end = min(chunk_start + chunk_size, len(windows))
            chunk = windows[chunk_start:chunk_end]

            all_features = np.stack([w["features"] for w in chunk])
            all_masks = np.stack([w["mask"] for w in chunk])
            all_entity_type = chunk[0]["entity_type"]  # same for all
            all_labels = np.stack([w["labels"] for w in chunk])

            chunk_path = split_dir / f"chunk_{chunk_idx:04d}.npz"
            np.savez_compressed(
                chunk_path,
                features=all_features,
                mask=all_masks,
                entity_type=all_entity_type,
                labels=all_labels,
            )
            manifest_entries.append({
                "path": str(chunk_path),
                "n_windows": len(chunk),
            })

        # Stats
        all_labels = np.stack([w["labels"] for w in windows])
        n_pos_home = (all_labels[:, 3] > 0.5).sum()
        n_pos_away = (all_labels[:, 4] > 0.5).sum()
        xt_vals = all_labels[:, 0]

        print(f"  {split}: {len(windows)} windows, "
              f"home_shots={n_pos_home}, away_shots={n_pos_away}, "
              f"xT range=[{xt_vals.min():.4f}, {xt_vals.max():.4f}], "
              f"xT mean={xt_vals.mean():.6f}")

        # Save manifest
        with open(split_dir / "manifest.json", "w") as f:
            json.dump(manifest_entries, f, indent=2)

    # Save metadata
    meta = {
        "window_size": args.window_size,
        "stride": args.stride,
        "horizon_frames": args.horizon_frames,
        "discount": args.discount,
        "n_entities": 23,
        "feat_dim": 6,
        "label_names": ["xt_value", "home_threat", "away_threat",
                        "has_shot_home", "has_shot_away", "next_xg"],
        "total_windows": total_windows,
        "total_positive": total_positive,
    }
    with open(out / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
