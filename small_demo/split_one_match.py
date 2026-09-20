#!/usr/bin/env python3
"""Build leakage-safe chronological manifests from one-match clips."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import numpy as np


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def clip_interval(entry: dict) -> tuple[float, float]:
    with np.load(entry["clip_path"]) as clip:
        if "periods" in clip and "times" in clip:
            periods = np.asarray(clip["periods"], dtype=np.float64)
            times = np.asarray(clip["times"], dtype=np.float64)
            key = periods * 100_000.0 + times
            return float(key[0]), float(key[-1])
        frames = np.asarray(clip["frames"], dtype=np.float64)
        return float(frames[0]), float(frames[-1])


def estimated_smart_windows(entry: dict, history: int = 8, rollout: int = 24, stride: int = 8) -> int:
    token_steps = (int(entry["length"]) // 2) // 5
    return max(0, math.ceil(max(0, token_steps - history - rollout) / stride))


def cap_by_windows(rows: list[dict], cap: int, seed: int) -> list[dict]:
    if cap <= 0:
        return rows
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(rows)).tolist()
    chosen, total = [], 0
    for idx in order:
        chosen.append(rows[idx])
        total += estimated_smart_windows(rows[idx])
        if total >= cap:
            break
    return sorted(chosen, key=lambda row: row["clip_index"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--out-root", required=True, type=Path)
    parser.add_argument("--ratios", default="0.70,0.15,0.15")
    parser.add_argument("--train-smart-windows", type=int, default=512)
    parser.add_argument("--eval-smart-windows", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ratios = [float(value) for value in args.ratios.split(",")]
    if len(ratios) != 3 or not np.isclose(sum(ratios), 1.0):
        raise ValueError("--ratios must contain three values summing to one")

    source_manifest = args.source_root / "train" / "manifest.jsonl"
    rows = read_jsonl(source_manifest)
    if not rows:
        raise RuntimeError(f"No clips in {source_manifest}")
    annotated = [(row, *clip_interval(row)) for row in rows]
    lo = min(start for _, start, _ in annotated)
    hi = max(end for _, _, end in annotated)
    cut_train = lo + ratios[0] * (hi - lo)
    cut_val = lo + (ratios[0] + ratios[1]) * (hi - lo)

    split_rows = {"train": [], "val": [], "test": []}
    dropped = 0
    for row, start, end in annotated:
        if end <= cut_train:
            split_rows["train"].append(row)
        elif start >= cut_train and end <= cut_val:
            split_rows["val"].append(row)
        elif start >= cut_val:
            split_rows["test"].append(row)
        else:
            dropped += 1

    split_rows["train"] = cap_by_windows(
        split_rows["train"], args.train_smart_windows, args.seed
    )
    split_rows["val"] = cap_by_windows(
        split_rows["val"], args.eval_smart_windows, args.seed + 1
    )
    split_rows["test"] = cap_by_windows(
        split_rows["test"], args.eval_smart_windows, args.seed + 2
    )
    for split, values in split_rows.items():
        if not values:
            raise RuntimeError(f"Chronological split {split!r} has no complete clips")
        write_jsonl(args.out_root / split / "manifest.jsonl", values)

    for name in ("normalization_stats.json", "preprocessed_meta.json"):
        source = args.source_root / name
        if source.exists():
            args.out_root.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, args.out_root / name)

    report = {
        "source_manifest": str(source_manifest),
        "interval": [lo, hi],
        "cuts": [cut_train, cut_val],
        "dropped_boundary_clips": dropped,
        "splits": {
            key: {
                "clips": len(values),
                "estimated_smart_windows": sum(estimated_smart_windows(row) for row in values),
            }
            for key, values in split_rows.items()
        },
    }
    with (args.out_root / "split_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
