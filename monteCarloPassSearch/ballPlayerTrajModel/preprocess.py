#!/usr/bin/env python3
"""One-time preprocessing for kick-anchored joint (players + ball) trajectory training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from dataset import (
    PreprocessConfig,
    build_cache_for_files,
    compute_normalization_stats,
    discover_tracking_files,
    load_match_events,
    STOP_CLASS_NAMES,
    save_stats,
    split_files,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess tracking parquets into kick-anchored cached training clips."
    )

    parser.add_argument("--data-dir", type=str, default="/mnt/data/remains")
    parser.add_argument(
        "--passes-dir",
        type=str,
        default="/mnt/data/remains/eventData/passes",
        help="Directory with per-match pass event json files (e.g. 0.json ... 99.json).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="/mnt/data/remains/opta2026/ballPlayerTrajModel/preprocessed",
    )

    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--file-start", type=int, default=0)
    parser.add_argument("--file-end", type=int, default=99)
    parser.add_argument("--max-files", type=int, default=None)

    parser.add_argument("--min-players", type=int, default=20)
    parser.add_argument("--min-segment-frames", type=int, default=80)
    parser.add_argument("--max-roster-diff", type=int, default=2)
    parser.add_argument("--max-player-step", type=float, default=6.0)
    parser.add_argument("--max-ball-step", type=float, default=18.0)
    parser.add_argument("--max-frames-per-file", type=int, default=None)
    parser.add_argument("--history-frames", type=int, default=512)
    parser.add_argument("--future-frames", type=int, default=1024)
    parser.add_argument("--kick-contact-radius", type=float, default=1.2)
    parser.add_argument("--kick-speed-min", type=float, default=6.0)
    parser.add_argument("--kick-dv-min", type=float, default=1.5)
    parser.add_argument("--kick-speed-ratio-min", type=float, default=1.15)
    parser.add_argument("--kick-refractory-frames", type=int, default=12)
    parser.add_argument(
        "--use-pass-event-anchors",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Anchor clip split points using pass events (typeId==1) instead of only heuristics.",
    )
    parser.add_argument("--pass-anchor-tolerance-sec", type=float, default=0.6)
    parser.add_argument("--pass-anchor-refractory-frames", type=int, default=4)
    parser.add_argument(
        "--fallback-to-heuristic-when-no-pass",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If pass anchor mapping fails for a segment, fallback to kick heuristic anchors.",
    )
    parser.add_argument(
        "--allow-noncontiguous-windows",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Allow frame gaps inside clips and encode them via stop_event_id markers "
            "instead of hard-dropping non-contiguous windows."
        ),
    )

    parser.add_argument("--rebuild", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    file_indices = list(range(int(args.file_start), int(args.file_end) + 1))
    files = discover_tracking_files(data_dir=args.data_dir, file_indices=file_indices)
    if args.max_files is not None:
        files = files[: max(0, int(args.max_files))]

    if not files:
        raise RuntimeError(f"No tracking parquet files found in {args.data_dir} for requested index range.")

    train_files, val_files = split_files(files=files, val_ratio=args.val_ratio, seed=args.seed)

    cfg = PreprocessConfig(
        max_players=22,
        min_players=args.min_players,
        min_segment_frames=args.min_segment_frames,
        max_roster_diff=args.max_roster_diff,
        max_player_step=args.max_player_step,
        max_ball_step=args.max_ball_step,
        max_frames_per_file=args.max_frames_per_file,
        history_frames=args.history_frames,
        future_frames=args.future_frames,
        kick_contact_radius=args.kick_contact_radius,
        kick_speed_min=args.kick_speed_min,
        kick_dv_min=args.kick_dv_min,
        kick_speed_ratio_min=args.kick_speed_ratio_min,
        kick_refractory_frames=args.kick_refractory_frames,
        use_pass_event_anchors=bool(args.use_pass_event_anchors),
        pass_anchor_tolerance_sec=float(args.pass_anchor_tolerance_sec),
        pass_anchor_refractory_frames=int(args.pass_anchor_refractory_frames),
        fallback_to_heuristic_when_no_pass=bool(args.fallback_to_heuristic_when_no_pass),
        allow_noncontiguous_windows=bool(args.allow_noncontiguous_windows),
    )

    match_events_by_file: Dict[str, Dict[int, list[dict[str, Any]]]] = {}
    if cfg.use_pass_event_anchors:
        passes_dir = Path(args.passes_dir)
        if not passes_dir.exists():
            raise RuntimeError(f"Passes directory not found: {passes_dir}")

        total_pass_events = 0
        total_restart_events = 0
        for fp in files:
            match_idx = fp.stem.split("_")[0]
            pass_json = passes_dir / f"{match_idx}.json"
            per_period = load_match_events(pass_json)
            match_events_by_file[fp.name] = per_period
            match_events_by_file[match_idx] = per_period
            total_pass_events += sum(
                sum(1 for e in rows if int(e.get("type_id", -1)) == 1)
                for rows in per_period.values()
            )
            total_restart_events += sum(
                sum(
                    1
                    for e in rows
                    if (
                        (int(e.get("type_id", -1)) == 1 and any(int(q) in {5, 6, 107, 124} for q in e.get("qualifier_ids", [])))
                        or int(e.get("type_id", -1)) == 6
                    )
                )
                for rows in per_period.values()
            )
        print(
            f"Loaded pass-event anchors from {passes_dir} for {len(files)} files. "
            f"Total pass timestamps: {total_pass_events} | restart-like events: {total_restart_events}"
        )

    out_dir = Path(args.out_dir)
    train_dir = out_dir / "train"
    val_dir = out_dir / "val"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_entries = build_cache_for_files(
        train_files,
        train_dir,
        cfg=cfg,
        match_events_by_file=match_events_by_file,
        rebuild=args.rebuild,
        use_tqdm=True,
        tqdm_desc="Preprocessing train files",
    )
    val_entries = (
        build_cache_for_files(
            val_files,
            val_dir,
            cfg=cfg,
            match_events_by_file=match_events_by_file,
            rebuild=args.rebuild,
            use_tqdm=True,
            tqdm_desc="Preprocessing val files",
        )
        if val_files
        else []
    )

    if not train_entries:
        raise RuntimeError("No usable training clips produced. Relax preprocessing thresholds.")

    stats = compute_normalization_stats(
        train_entries,
        use_tqdm=True,
        tqdm_desc="Computing train normalization",
    )
    save_stats(stats, out_dir / "normalization_stats.json")

    meta: Dict[str, Any] = {
        "data_dir": str(args.data_dir),
        "out_dir": str(out_dir),
        "val_ratio": float(args.val_ratio),
        "seed": int(args.seed),
        "file_start": int(args.file_start),
        "file_end": int(args.file_end),
        "max_files": None if args.max_files is None else int(args.max_files),
        "preprocess_config": {
            "max_players": 22,
            "min_players": int(args.min_players),
            "min_segment_frames": int(args.min_segment_frames),
            "max_roster_diff": int(args.max_roster_diff),
            "max_player_step": float(args.max_player_step),
            "max_ball_step": float(args.max_ball_step),
            "max_frames_per_file": None if args.max_frames_per_file is None else int(args.max_frames_per_file),
            "history_frames": int(args.history_frames),
            "future_frames": int(args.future_frames),
            "kick_contact_radius": float(args.kick_contact_radius),
            "kick_speed_min": float(args.kick_speed_min),
            "kick_dv_min": float(args.kick_dv_min),
            "kick_speed_ratio_min": float(args.kick_speed_ratio_min),
            "kick_refractory_frames": int(args.kick_refractory_frames),
            "use_pass_event_anchors": bool(args.use_pass_event_anchors),
            "pass_anchor_tolerance_sec": float(args.pass_anchor_tolerance_sec),
            "pass_anchor_refractory_frames": int(args.pass_anchor_refractory_frames),
            "fallback_to_heuristic_when_no_pass": bool(args.fallback_to_heuristic_when_no_pass),
            "allow_noncontiguous_windows": bool(args.allow_noncontiguous_windows),
        },
        "stop_event_classes": list(STOP_CLASS_NAMES),
        "num_files_total": len(files),
        "num_files_train": len(train_files),
        "num_files_val": len(val_files),
        "num_clips_train": len(train_entries),
        "num_clips_val": len(val_entries),
        "num_frames_train": int(sum(e.length for e in train_entries)),
        "num_frames_val": int(sum(e.length for e in val_entries)),
        "train_files": [str(p) for p in train_files],
        "val_files": [str(p) for p in val_files],
    }

    with (out_dir / "preprocessed_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("Preprocessing complete.")
    print(json.dumps({
        "out_dir": str(out_dir),
        "num_clips_train": len(train_entries),
        "num_clips_val": len(val_entries),
        "num_files_total": len(files),
    }, indent=2))


if __name__ == "__main__":
    main()
