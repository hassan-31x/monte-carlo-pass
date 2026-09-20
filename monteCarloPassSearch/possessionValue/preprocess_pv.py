#!/usr/bin/env python3
"""Create PV (Possession Value) dataset from Sportec tracking clips + shot events.

Labels are soft: for each window, if a shot happens within the horizon,
label = xG of that shot (from provider or our xG model). Otherwise label = 0.

PV ≈ P(goal within horizon) ≈ P(shot within horizon) × E[xG | shot]
Training with BCE on these soft labels naturally learns this product.
"""

import argparse
import json
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

    for mid in by_match:
        by_match[mid].sort(key=lambda s: s["frame"])

    return dict(by_match)


def load_clip_manifests(preprocessed_dir: str, splits=("train", "val", "test")):
    """Load clip manifests and group by match."""
    clips_by_match = defaultdict(list)

    for split in splits:
        manifest_path = os.path.join(preprocessed_dir, split, "manifest.jsonl")
        if not os.path.exists(manifest_path):
            continue
        with open(manifest_path) as f:
            for line in f:
                entry = json.loads(line.strip())
                match_id = entry["source_file"].split(":")[-1]
                entry["match_id"] = match_id
                entry["split"] = split
                clips_by_match[match_id].append(entry)

    return dict(clips_by_match)


def find_shots_in_clip(clip_start, clip_end, shots):
    result = []
    for s in shots:
        if clip_start <= s["frame"] <= clip_end:
            local_frame = s["frame"] - clip_start
            result.append({**s, "local_frame": local_frame})
    return result


def create_pv_windows(
    clip_path: str,
    clip_shots: list,
    window_size: int = 64,
    stride: int = 32,
    horizon_frames: int = 250,
):
    """
    Create PV windows from a single clip.

    For each window, look ahead horizon_frames for the NEXT shot by each team.
    Label = xG of that shot (soft target for BCE).
    No discounting — just raw xG of the first upcoming shot per team.
    """
    data = np.load(clip_path)
    features = data["features"]  # [T, 23, 6]
    mask = data["mask"]  # [T, 23]
    entity_type = data["entity_type"]  # [23]
    T = features.shape[0]

    windows = []

    for start in range(0, T - window_size, stride):
        end = start + window_size

        # Look ahead for first shot per team
        home_xg = 0.0
        away_xg = 0.0
        has_shot_home = False
        has_shot_away = False

        for s in clip_shots:
            lf = s["local_frame"]
            if lf < end:
                continue
            if lf >= end + horizon_frames:
                break

            if s["side"] == "home" and not has_shot_home:
                home_xg = s["xG"]
                has_shot_home = True
            elif s["side"] == "away" and not has_shot_away:
                away_xg = s["xG"]
                has_shot_away = True

        window_features = features[start:end]
        window_mask = mask[start:end]

        # Labels: [home_xg_target, away_xg_target, has_shot_home, has_shot_away]
        labels = np.array([
            home_xg,
            away_xg,
            float(has_shot_home),
            float(has_shot_away),
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
        default="/mnt/data/remains/opta2026/expectedThreat/pv_data",
    )
    parser.add_argument("--window-size", type=int, default=64)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--horizon-frames", type=int, default=250,
                        help="Look-ahead frames (250=10s at 25fps)")
    args = parser.parse_args()

    out = Path(args.out_dir)

    print("Loading Sportec shot events...")
    shots_by_match = load_sportec_shots(args.sportec_shots)
    total_shots = sum(len(v) for v in shots_by_match.values())
    print(f"  {total_shots} shots across {len(shots_by_match)} matches")

    print("Loading clip manifests...")
    clips_by_match = load_clip_manifests(args.preprocessed_dir)
    total_clips = sum(len(v) for v in clips_by_match.values())
    print(f"  {total_clips} clips across {len(clips_by_match)} matches")

    split_windows = defaultdict(list)
    clips_with_shots = 0
    total_windows = 0
    total_positive = 0

    for match_id, clips in clips_by_match.items():
        match_shots = shots_by_match.get(match_id, [])

        for clip_entry in clips:
            clip_path = clip_entry["clip_path"]
            split = clip_entry["split"]
            start_frame = clip_entry["start_frame"]
            end_frame = clip_entry["end_frame"]

            clip_shots = find_shots_in_clip(start_frame, end_frame, match_shots)
            if clip_shots:
                clips_with_shots += 1

            windows = create_pv_windows(
                clip_path=clip_path,
                clip_shots=clip_shots,
                window_size=args.window_size,
                stride=args.stride,
                horizon_frames=args.horizon_frames,
            )

            for w in windows:
                split_windows[split].append(w)
                total_windows += 1
                if w["labels"][2] > 0.5 or w["labels"][3] > 0.5:
                    total_positive += 1

    print(f"\nTotal windows: {total_windows}")
    print(f"Windows with shot within horizon: {total_positive} "
          f"({100*total_positive/max(total_windows,1):.1f}%)")
    print(f"Clips containing shots: {clips_with_shots}/{total_clips}")

    # Save per split
    for split, windows in split_windows.items():
        if not windows:
            continue

        split_dir = out / split
        split_dir.mkdir(parents=True, exist_ok=True)

        chunk_size = 2000
        n_chunks = (len(windows) + chunk_size - 1) // chunk_size

        manifest_entries = []
        for chunk_idx in range(n_chunks):
            chunk_start = chunk_idx * chunk_size
            chunk_end = min(chunk_start + chunk_size, len(windows))
            chunk = windows[chunk_start:chunk_end]

            all_features = np.stack([w["features"] for w in chunk])
            all_masks = np.stack([w["mask"] for w in chunk])
            all_entity_type = chunk[0]["entity_type"]
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

        all_labels = np.stack([w["labels"] for w in windows])
        n_home = (all_labels[:, 2] > 0.5).sum()
        n_away = (all_labels[:, 3] > 0.5).sum()
        home_xg = all_labels[:, 0]
        away_xg = all_labels[:, 1]

        print(f"  {split}: {len(windows)} windows, "
              f"home_shots={n_home} (mean_xG={home_xg[home_xg > 0].mean():.3f}), "
              f"away_shots={n_away} (mean_xG={away_xg[away_xg > 0].mean():.3f})")

        with open(split_dir / "manifest.json", "w") as f:
            json.dump(manifest_entries, f, indent=2)

    meta = {
        "window_size": args.window_size,
        "stride": args.stride,
        "horizon_frames": args.horizon_frames,
        "n_entities": 23,
        "feat_dim": 6,
        "label_names": ["home_xg_target", "away_xg_target",
                        "has_shot_home", "has_shot_away"],
        "total_windows": total_windows,
        "total_positive": total_positive,
    }
    with open(out / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
