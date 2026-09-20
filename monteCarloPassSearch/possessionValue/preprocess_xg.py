#!/usr/bin/env python3
"""Extract shot features from Sportec + StatsBomb open data for xG model training.

Features limited to those available in the Sportec dataset:
  distance, angle, is_header, is_right_foot, is_left_foot,
  inside_box, pressure, gk_distance, n_defenders, player_speed

StatsBomb shots use shot.statsbomb_xg as the soft label.
"""

import argparse
import glob
import json
import math
import os
import numpy as np
from pathlib import Path


# Standard pitch dimensions (metres)
PITCH_X = 105.0
PITCH_Y = 68.0
GOAL_X = 105.0
GOAL_Y = 34.0  # centre of goal

# StatsBomb pitch is 120 x 80
SB_PITCH_X = 120.0
SB_PITCH_Y = 80.0


def parse_sportec_shots(sportec_shots_path: str):
    """Parse pre-extracted Sportec shot data."""
    with open(sportec_shots_path) as f:
        raw = json.load(f)

    shots = []
    for s in raw:
        distance = float(s.get("DistanceToGoal") or 20.0)
        angle = float(s.get("AngleToGoal") or 20.0)

        shot_type = str(s.get("TypeOfShot", "")).lower()
        is_header = 1.0 if "head" in shot_type else 0.0
        is_right_foot = 1.0 if "right" in shot_type else 0.0
        is_left_foot = 1.0 if "left" in shot_type else 0.0

        inside_box = 1.0 if str(s.get("InsideBox", "")).lower() == "true" else 0.0
        pressure = float(s.get("Pressure") or 0.0)
        gk_distance = float(s.get("GoalDistanceGoalkeeper") or -1.0)
        n_defenders = float(s.get("AmountOfDefenders") or -1.0)
        player_speed = float(s.get("PlayerSpeed") or -1.0)
        provider_xg = float(s.get("xG") or -1.0)

        is_goal = 1.0 if "GOAL" in str(s.get("result", "")).upper() else 0.0

        shots.append({
            "source": "sportec",
            "distance": distance,
            "angle": angle,
            "is_header": is_header,
            "is_right_foot": is_right_foot,
            "is_left_foot": is_left_foot,
            "inside_box": inside_box,
            "pressure": pressure,
            "gk_distance": gk_distance,
            "n_defenders": n_defenders,
            "player_speed": player_speed,
            "is_goal": is_goal,
            "provider_xg": provider_xg,
        })
    return shots


def parse_statsbomb_shots(events_dir: str):
    """Parse StatsBomb open-data event files for shots.

    Extracts Sportec-compatible features + statsbomb_xg as soft label.
    """
    shots = []
    files = sorted(glob.glob(os.path.join(events_dir, "*.json")))
    for fp in files:
        with open(fp) as f:
            events = json.load(f)
        for e in events:
            if e.get("type", {}).get("name") != "Shot":
                continue
            shot = e.get("shot", {})

            # Skip penalties — trivial xG, not interesting
            if shot.get("type", {}).get("name") == "Penalty":
                continue

            loc = e.get("location")
            if not loc or len(loc) < 2:
                continue

            # Convert StatsBomb 120x80 → 105x68 metres
            x_m = loc[0] * PITCH_X / SB_PITCH_X
            y_m = loc[1] * PITCH_Y / SB_PITCH_Y

            distance = math.sqrt((GOAL_X - x_m) ** 2 + (GOAL_Y - y_m) ** 2)
            angle = math.degrees(math.atan2(abs(GOAL_Y - y_m), max(GOAL_X - x_m, 0.01)))

            inside_box = 1.0 if (GOAL_X - x_m <= 16.5 and abs(GOAL_Y - y_m) <= 20.16) else 0.0

            # Body part
            bp = shot.get("body_part", {}).get("name", "")
            is_header = 1.0 if bp == "Head" else 0.0
            is_right_foot = 1.0 if bp == "Right Foot" else 0.0
            is_left_foot = 1.0 if bp == "Left Foot" else 0.0

            # Pressure (binary in StatsBomb)
            pressure = 1.0 if e.get("under_pressure") else 0.0

            # From freeze frame: GK distance and defender count
            gk_distance = -1.0
            n_defenders = -1.0
            ff = shot.get("freeze_frame")
            if ff:
                n_def = 0
                for p in ff:
                    if p.get("teammate", False):
                        continue
                    ploc = p.get("location")
                    if not ploc or len(ploc) < 2:
                        continue
                    px = ploc[0] * PITCH_X / SB_PITCH_X
                    py = ploc[1] * PITCH_Y / SB_PITCH_Y
                    if p.get("position", {}).get("name") == "Goalkeeper":
                        gk_distance = math.sqrt((px - GOAL_X) ** 2 + (py - GOAL_Y) ** 2)
                    else:
                        # Defender closer to goal than the shooter
                        d_to_goal = math.sqrt((px - GOAL_X) ** 2 + (py - GOAL_Y) ** 2)
                        if d_to_goal < distance:
                            n_def += 1
                n_defenders = float(n_def)

            # Player speed not available in StatsBomb event data
            player_speed = -1.0

            # Outcome
            outcome = shot.get("outcome", {}).get("name", "")
            is_goal = 1.0 if outcome == "Goal" else 0.0

            # Soft label
            provider_xg = float(shot.get("statsbomb_xg", -1.0))

            shots.append({
                "source": "statsbomb",
                "distance": distance,
                "angle": angle,
                "is_header": is_header,
                "is_right_foot": is_right_foot,
                "is_left_foot": is_left_foot,
                "inside_box": inside_box,
                "pressure": pressure,
                "gk_distance": gk_distance,
                "n_defenders": n_defenders,
                "player_speed": player_speed,
                "is_goal": is_goal,
                "provider_xg": provider_xg,
            })
    return shots


FEATURE_NAMES = [
    "distance", "angle", "is_header", "is_right_foot", "is_left_foot",
    "inside_box", "pressure", "gk_distance", "n_defenders", "player_speed",
]


def shots_to_arrays(shots):
    """Convert list of shot dicts to numpy arrays."""
    n = len(shots)
    n_feat = len(FEATURE_NAMES)
    X = np.zeros((n, n_feat), dtype=np.float32)
    y = np.zeros((n,), dtype=np.float32)
    xg_soft = np.full((n,), -1.0, dtype=np.float32)
    mask = np.ones((n, n_feat), dtype=np.float32)

    for i, s in enumerate(shots):
        for j, fname in enumerate(FEATURE_NAMES):
            val = float(s[fname])
            if val < 0:
                X[i, j] = 0.0
                mask[i, j] = 0.0  # unknown feature
            else:
                X[i, j] = val
        y[i] = float(s["is_goal"])
        if float(s["provider_xg"]) >= 0:
            xg_soft[i] = float(s["provider_xg"])

    return X, y, xg_soft, mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sportec-shots", default="/mnt/data/remains/opta2026/expectedThreat/sportec_shots.json")
    parser.add_argument("--statsbomb-events", default="/mnt/data/mywork/soccerWork/hudlWork/open-data/data/events")
    parser.add_argument("--out-dir", default="/mnt/data/remains/opta2026/expectedThreat/xg_data")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Parsing Sportec shots...")
    sportec_shots = parse_sportec_shots(args.sportec_shots) if os.path.isfile(args.sportec_shots) else []
    print(f"  {len(sportec_shots)} shots from Sportec, {sum(1 for s in sportec_shots if s['is_goal'])} goals")

    print("Parsing StatsBomb shots...")
    sb_shots = parse_statsbomb_shots(args.statsbomb_events) if os.path.isdir(args.statsbomb_events) else []
    print(f"  {len(sb_shots)} shots from StatsBomb, {sum(1 for s in sb_shots if s['is_goal'])} goals")

    all_shots = sportec_shots + sb_shots
    print(f"\nTotal: {len(all_shots)} shots, {sum(1 for s in all_shots if s['is_goal'])} goals")

    X, y, xg_soft, feat_mask = shots_to_arrays(all_shots)

    # Compute normalization stats on all data (only observed features)
    valid_mask = feat_mask > 0.5
    feat_mean = np.zeros(X.shape[1], dtype=np.float32)
    feat_std = np.ones(X.shape[1], dtype=np.float32)
    for j in range(X.shape[1]):
        vals = X[:, j][valid_mask[:, j]]
        if len(vals) > 1:
            feat_mean[j] = vals.mean()
            feat_std[j] = max(vals.std(), 1e-6)

    # Normalize
    X_norm = (X - feat_mean) / feat_std
    X_norm[~valid_mask] = 0.0

    # Split
    rng = np.random.RandomState(args.seed)
    indices = rng.permutation(len(all_shots))
    n_test = int(len(all_shots) * args.test_ratio)
    n_val = int(len(all_shots) * args.val_ratio)
    test_idx = indices[:n_test]
    val_idx = indices[n_test:n_test + n_val]
    train_idx = indices[n_test + n_val:]

    for split, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        np.savez_compressed(
            out / f"{split}.npz",
            X=X_norm[idx],
            X_raw=X[idx],
            y=y[idx],
            xg_soft=xg_soft[idx],
            feat_mask=feat_mask[idx],
        )
        n_goals = int(y[idx].sum())
        n_xg = int((xg_soft[idx] >= 0).sum())
        print(f"  {split}: {len(idx)} shots, {n_goals} goals ({100*n_goals/max(len(idx),1):.1f}%), {n_xg} with soft xG")

    # Save metadata
    meta = {
        "feature_names": FEATURE_NAMES,
        "n_features": len(FEATURE_NAMES),
        "feat_mean": feat_mean.tolist(),
        "feat_std": feat_std.tolist(),
        "n_total": len(all_shots),
        "n_goals": int(y.sum()),
        "sources": {
            "sportec": len(sportec_shots),
            "statsbomb": len(sb_shots),
        },
    }
    with open(out / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
