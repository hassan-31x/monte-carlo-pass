#!/usr/bin/env python3
"""Build a merged preprocessed dataset for single-stage training.

Default behavior:
- Train on all available sources from the start.
- Validate/test only on Sportec XYZ splits.
- Auto-upconvert XY clips to XYZ-compatible feature dim by zero-padding channels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset import ClipEntry, compute_normalization_stats, save_stats  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build merged preprocessed dir: train=all sources, val/test=sportec xyz."
    )
    parser.add_argument(
        "--current-preprocessed",
        type=str,
        default="/mnt/data/remains/opta2026/ballPlayerTrajModel/preprocessed",
    )
    parser.add_argument(
        "--metrica-preprocessed",
        type=str,
        default=None,
        help="Metrica preprocessed dir (xy or xyz).",
    )
    # Backward-compatible alias.
    parser.add_argument(
        "--metrica-xy-preprocessed",
        type=str,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--sportec-xyz-preprocessed",
        type=str,
        required=True,
        help="Sportec preprocessed dir generated with --ball-mode xyz.",
    )
    parser.add_argument(
        "--sportec-xy-preprocessed",
        type=str,
        default=None,
        help="Optional Sportec XY preprocessed dir.",
    )
    parser.add_argument(
        "--include-sportec-xy-in-train",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include Sportec XY train clips in merged train split (upconverted to target feature dim).",
    )
    parser.add_argument(
        "--out-root",
        type=str,
        default="/mnt/data/remains/opta2026/ballPlayerTrajModel/publicData/stages",
    )
    parser.add_argument(
        "--out-name",
        type=str,
        default="all_from_start_xyz_eval",
    )
    parser.add_argument(
        "--target-feature-dim",
        type=int,
        default=None,
        help="Force output feature dim. If omitted, inferred from sportec xyz clips.",
    )
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Rebuild converted clips when destination exists.",
    )
    parser.add_argument(
        "--eager-convert",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "If true, materialize padded/truncated converted npz clips under _converted_clips. "
            "Default false: keep original clip paths and rely on dataset.py dynamic pad/truncate."
        ),
    )
    return parser.parse_args()


def _load_manifest(path: Path) -> List[ClipEntry]:
    if not path.exists():
        return []
    out: List[ClipEntry] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(ClipEntry(**json.loads(line)))
    return out


def _save_manifest(entries: Sequence[ClipEntry], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(asdict(e)) + "\n")


def _read_split(pre_dir: Path, split: str) -> List[ClipEntry]:
    return _load_manifest(pre_dir / split / "manifest.jsonl")


def _combine(*parts: Iterable[ClipEntry]) -> List[ClipEntry]:
    out: List[ClipEntry] = []
    for p in parts:
        out.extend(list(p))
    return out


def _feature_dim_from_entry(entry: ClipEntry) -> int:
    with np.load(entry.clip_path, allow_pickle=False) as data:
        return int(data["features"].shape[-1])


def _infer_feature_dim(entries: Sequence[ClipEntry]) -> int:
    for e in entries:
        try:
            return _feature_dim_from_entry(e)
        except Exception:
            continue
    raise RuntimeError("Could not infer feature dimension from entries.")


def _copy_or_pad_clip_features(
    src_path: Path,
    dst_path: Path,
    target_dim: int,
    overwrite: bool,
) -> None:
    if dst_path.exists() and not overwrite:
        return

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with np.load(src_path, allow_pickle=False) as data:
        arrays: Dict[str, np.ndarray] = {k: data[k] for k in data.files}

    if "features" not in arrays:
        raise RuntimeError(f"Clip missing 'features': {src_path}")

    features = arrays["features"].astype(np.float32, copy=False)
    src_dim = int(features.shape[-1])
    if src_dim == target_dim:
        if src_path != dst_path:
            np.savez_compressed(dst_path, **arrays)
        return

    new_features = np.zeros((*features.shape[:-1], target_dim), dtype=np.float32)
    keep = min(src_dim, target_dim)
    new_features[..., :keep] = features[..., :keep]
    arrays["features"] = new_features

    # Remove precomputed deltas so downstream code always recomputes from updated features.
    arrays.pop("entity_delta", None)
    arrays.pop("entity_delta_norm", None)
    arrays.pop("entity_delta_mask", None)
    arrays.pop("entity_delta_dim_mask", None)

    np.savez_compressed(dst_path, **arrays)


def _remap_entries_to_dim(
    entries: Sequence[ClipEntry],
    target_dim: int,
    converted_root: Path,
    overwrite: bool,
) -> List[ClipEntry]:
    out: List[ClipEntry] = []
    cache: Dict[str, str] = {}
    for entry in entries:
        src = Path(entry.clip_path)
        src_key = str(src.resolve())
        src_dim = _feature_dim_from_entry(entry)

        if src_dim == target_dim:
            out.append(entry)
            continue

        if src_key in cache:
            dst = Path(cache[src_key])
        else:
            h = hashlib.sha1(src_key.encode("utf-8")).hexdigest()[:10]
            dst_name = f"{src.stem}_fd{target_dim}_{h}.npz"
            dst = converted_root / dst_name
            _copy_or_pad_clip_features(
                src_path=src,
                dst_path=dst,
                target_dim=target_dim,
                overwrite=overwrite,
            )
            cache[src_key] = str(dst)

        out.append(
            ClipEntry(
                clip_path=str(dst),
                length=int(entry.length),
                source_file=str(entry.source_file),
                clip_index=int(entry.clip_index),
                start_frame=int(entry.start_frame),
                end_frame=int(entry.end_frame),
                kick_frame=int(entry.kick_frame),
                real_player_count=int(entry.real_player_count),
            )
        )
    return out


def _write_merged_dataset(
    out_dir: Path,
    train_entries: List[ClipEntry],
    val_entries: List[ClipEntry],
    test_entries: List[ClipEntry],
    meta_extra: Dict[str, Any],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    _save_manifest(train_entries, out_dir / "train" / "manifest.jsonl")
    _save_manifest(val_entries, out_dir / "val" / "manifest.jsonl")
    _save_manifest(test_entries, out_dir / "test" / "manifest.jsonl")

    stats = compute_normalization_stats(train_entries, use_tqdm=True, tqdm_desc=f"Stats {out_dir.name}")
    save_stats(stats, out_dir / "normalization_stats.json")

    meta = {
        "out_dir": str(out_dir),
        "num_clips_train": len(train_entries),
        "num_clips_val": len(val_entries),
        "num_clips_test": len(test_entries),
        "num_frames_train": int(sum(e.length for e in train_entries)),
        "num_frames_val": int(sum(e.length for e in val_entries)),
        "num_frames_test": int(sum(e.length for e in test_entries)),
    }
    meta.update(meta_extra)
    with (out_dir / "preprocessed_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def main() -> None:
    args = parse_args()

    metrica_dir_str = args.metrica_preprocessed or args.metrica_xy_preprocessed
    if not metrica_dir_str:
        raise RuntimeError(
            "Provide --metrica-preprocessed (or legacy --metrica-xy-preprocessed)."
        )

    current_dir = Path(args.current_preprocessed)
    metrica_dir = Path(metrica_dir_str)
    sportec_xyz_dir = Path(args.sportec_xyz_preprocessed)
    sportec_xy_dir = Path(args.sportec_xy_preprocessed) if args.sportec_xy_preprocessed else None
    out_dir = Path(args.out_root) / args.out_name

    current_train = _read_split(current_dir, "train")
    metrica_train = _read_split(metrica_dir, "train")
    sportec_xyz_train_full = _read_split(sportec_xyz_dir, "train")
    sportec_xyz_val = _read_split(sportec_xyz_dir, "val")
    sportec_xyz_test = _read_split(sportec_xyz_dir, "test")

    # Validation/testing are strictly from Sportec XYZ.
    val_entries = list(sportec_xyz_val)
    test_entries = list(sportec_xyz_test)
    sportec_xyz_train = list(sportec_xyz_train_full)
    carved_eval_from_train = False

    if not val_entries and test_entries:
        val_entries = list(test_entries)
    if not test_entries and val_entries:
        test_entries = list(val_entries)

    # Fallback for tiny Sportec XYZ sets with no explicit val/test split:
    # carve deterministic holdout from Sportec XYZ train so evaluation remains XYZ-only.
    if not val_entries and not test_entries:
        if not sportec_xyz_train:
            raise RuntimeError(
                "Sportec XYZ has no train/val/test entries; cannot build XYZ-only eval."
            )
        carved_eval_from_train = True
        n = len(sportec_xyz_train)
        n_eval = max(1, int(round(0.1 * n)))
        n_eval = min(n_eval, max(1, n - 1))
        val_entries = list(sportec_xyz_train[:n_eval])
        test_entries = list(sportec_xyz_train[-n_eval:])
        train_keep_start = n_eval
        train_keep_end = max(train_keep_start, n - n_eval)
        sportec_xyz_train = list(sportec_xyz_train[train_keep_start:train_keep_end])
        if not sportec_xyz_train:
            sportec_xyz_train = list(sportec_xyz_train_full[n_eval:])

    train_entries = _combine(current_train, metrica_train, sportec_xyz_train)
    if args.include_sportec_xy_in_train:
        if sportec_xy_dir is None:
            raise RuntimeError(
                "--include-sportec-xy-in-train requires --sportec-xy-preprocessed."
            )
        train_entries = _combine(train_entries, _read_split(sportec_xy_dir, "train"))

    if not train_entries:
        raise RuntimeError("Merged train split is empty.")

    if args.target_feature_dim is not None:
        target_dim = int(args.target_feature_dim)
    else:
        target_dim = _infer_feature_dim(_combine(val_entries, test_entries, sportec_xyz_train))
    if target_dim < 4:
        raise RuntimeError(f"Unexpected target feature dim {target_dim}.")

    if bool(args.eager_convert):
        converted_root = out_dir / "_converted_clips"
        train_entries = _remap_entries_to_dim(
            entries=train_entries,
            target_dim=target_dim,
            converted_root=converted_root,
            overwrite=bool(args.overwrite),
        )
        val_entries = _remap_entries_to_dim(
            entries=val_entries,
            target_dim=target_dim,
            converted_root=converted_root,
            overwrite=bool(args.overwrite),
        )
        test_entries = _remap_entries_to_dim(
            entries=test_entries,
            target_dim=target_dim,
            converted_root=converted_root,
            overwrite=bool(args.overwrite),
        )

    _write_merged_dataset(
        out_dir=out_dir,
        train_entries=train_entries,
        val_entries=val_entries,
        test_entries=test_entries,
        meta_extra={
            "dataset_style": "single_stage_all_from_start",
            "sources_train": {
                "current": str(current_dir),
                "metrica": str(metrica_dir),
                "sportec_xyz": str(sportec_xyz_dir),
                "sportec_xy": str(sportec_xy_dir) if sportec_xy_dir is not None else None,
            },
            "sources_eval": {
                "val": str(sportec_xyz_dir),
                "test": str(sportec_xyz_dir),
            },
            "include_sportec_xy_in_train": bool(args.include_sportec_xy_in_train),
            "carved_xyz_eval_from_train": bool(carved_eval_from_train),
            "eager_convert": bool(args.eager_convert),
            "target_feature_dim": int(target_dim),
            "notes": (
                "Train on all sources from the start; validate and test on Sportec XYZ only. "
                "Non-target feature dims are aligned at load-time (or eagerly converted when enabled)."
            ),
        },
    )

    print(
        json.dumps(
            {
                "merged_out": str(out_dir),
                "train_clips": len(train_entries),
                "val_clips_xyz": len(val_entries),
                "test_clips_xyz": len(test_entries),
                "target_feature_dim": int(target_dim),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
