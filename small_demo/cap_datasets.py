#!/usr/bin/env python3
"""Deterministically cap generated PTT, BAT, and PV arrays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def cap_npz_directory(directory: Path, cap: int, seed: int, prefix: str) -> dict:
    files = sorted(directory.glob("*.npz"))
    if not files:
        return {"before": 0, "after": 0, "status": "missing"}
    arrays: dict[str, list[np.ndarray]] = {}
    for path in files:
        with np.load(path) as data:
            for key in data.files:
                arrays.setdefault(key, []).append(np.asarray(data[key]))
    merged = {key: np.concatenate(parts, axis=0) for key, parts in arrays.items()}
    count = len(next(iter(merged.values())))
    rng = np.random.default_rng(seed)
    if "is_event" in merged:
        positive = np.flatnonzero(merged["is_event"] > 0.5)
        negative = np.flatnonzero(merged["is_event"] <= 0.5)
        if len(positive) >= cap:
            selected = rng.choice(positive, size=min(cap, len(positive)), replace=False)
        else:
            selected = np.concatenate([
                positive,
                rng.choice(negative, size=min(len(negative), cap - len(positive)), replace=False),
            ])
    else:
        selected = rng.choice(count, size=min(cap, count), replace=False)
    selected = np.sort(selected)
    for path in files:
        path.unlink()
    out = directory / f"{prefix}_capped.npz"
    np.savez_compressed(out, **{key: value[selected] for key, value in merged.items()})
    report = {"before": count, "after": len(selected), "file": str(out)}
    if "is_event" in merged:
        report["events"] = int(np.sum(merged["is_event"][selected] > 0.5))
    return report


def cap_pv(directory: Path, cap: int, seed: int) -> dict:
    manifest_path = directory / "manifest.json"
    if not manifest_path.exists():
        return {"before": 0, "after": 0, "status": "missing"}
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    arrays: dict[str, list[np.ndarray]] = {}
    for entry in manifest:
        with np.load(entry["path"]) as data:
            for key in data.files:
                if key == "entity_type":
                    continue
                arrays.setdefault(key, []).append(np.asarray(data[key]))
            entity_type = np.asarray(data["entity_type"])
    merged = {key: np.concatenate(parts, axis=0) for key, parts in arrays.items()}
    count = len(merged["labels"])
    rng = np.random.default_rng(seed)
    # Preserve every positive window when possible, then sample negatives.
    positive = np.flatnonzero(merged["labels"][:, 2:4].max(axis=1) > 0.5)
    negative = np.flatnonzero(merged["labels"][:, 2:4].max(axis=1) <= 0.5)
    if len(positive) >= cap:
        selected = rng.choice(positive, size=cap, replace=False)
    else:
        needed = min(len(negative), cap - len(positive))
        selected = np.concatenate([positive, rng.choice(negative, size=needed, replace=False)])
    selected = np.sort(selected)
    for entry in manifest:
        Path(entry["path"]).unlink(missing_ok=True)
    out = directory / "chunk_capped.npz"
    np.savez_compressed(
        out,
        **{key: value[selected] for key, value in merged.items()},
        entity_type=entity_type,
    )
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump([{"path": str(out), "n_windows": len(selected)}], handle, indent=2)
    return {
        "before": count,
        "after": len(selected),
        "positives": int(len(np.intersect1d(selected, positive))),
        "file": str(out),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ptt-root", required=True, type=Path)
    parser.add_argument("--bat-root", required=True, type=Path)
    parser.add_argument("--pv-root", required=True, type=Path)
    parser.add_argument("--touch-train", type=int, default=8192)
    parser.add_argument("--touch-eval", type=int, default=1024)
    parser.add_argument("--bat-train", type=int, default=1024)
    parser.add_argument("--bat-eval", type=int, default=128)
    parser.add_argument("--pv-train", type=int, default=2048)
    parser.add_argument("--pv-eval", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()

    report: dict[str, dict] = {}
    for offset, split in enumerate(("train", "val")):
        report[f"ptt_{split}"] = cap_npz_directory(
            args.ptt_root / split,
            args.touch_train if split == "train" else args.touch_eval,
            args.seed + offset,
            "survival",
        )
        report[f"bat_{split}"] = cap_npz_directory(
            args.bat_root / split,
            args.bat_train if split == "train" else args.bat_eval,
            args.seed + 10 + offset,
            "bat",
        )
    for offset, split in enumerate(("train", "val", "test")):
        report[f"pv_{split}"] = cap_pv(
            args.pv_root / split,
            args.pv_train if split == "train" else args.pv_eval,
            args.seed + 20 + offset,
        )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
