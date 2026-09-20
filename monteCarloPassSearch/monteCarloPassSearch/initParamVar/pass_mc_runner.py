#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import torch

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from csv_writer import CsvStreamWriter, load_done_keys
from distributed import run_distributed
from passer_event_labels import build_manifest_passer_labels
from sim_core import init_worker, process_clip_task
from variant_sampling import estimate_caps_from_manifest


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Monte Carlo pass search runner.")
    p.add_argument(
        "--manifest",
        default="/mnt/data/remains/opta2026/ballPlayerTrajModel/publicData/preprocessed/sportec_xyz/test/manifest.jsonl",
    )
    p.add_argument("--output-csv", required=True)
    p.add_argument(
        "--observed-params-csv",
        type=str,
        default="",
        help=(
            "Optional CSV whose observed rows provide per-clip base release "
            "params (v0x/v0y/v0z/spin_scalar) for variant sampling."
        ),
    )
    p.add_argument("--resume", action="store_true", help="Skip rows already present in output CSV.")
    p.add_argument("--flush-every", type=int, default=2000)
    p.add_argument("--max-clips", type=int, default=None)
    p.add_argument(
        "--clip-indices",
        type=str,
        default="",
        help="Comma-separated clip indices or ranges to run (e.g. '12,45,100-109').",
    )
    p.add_argument(
        "--clip-indices-file",
        type=str,
        default="",
        help="Text file containing clip indices/ranges (comma and/or newline separated).",
    )
    p.add_argument("--cap-estimation-max-clips", type=int, default=None)
    p.add_argument("--min-speed-xy-max", type=float, default=None)
    p.add_argument("--min-vz-abs-max", type=float, default=None)
    p.add_argument("--min-spin-abs-max", type=float, default=None)

    p.add_argument("--variants-local", type=int, default=256)
    p.add_argument("--variants-global", type=int, default=256)
    p.add_argument("--include-observed", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--observed-use-gt-players", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument(
        "--require-near-zero-fit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reject passes that do not meet the paper's near-zero kick-fit criterion.",
    )
    p.add_argument("--context-len", type=int, default=32)
    p.add_argument("--rollout-len", type=int, default=128)
    p.add_argument("--physics-len", type=int, default=25)
    p.add_argument("--eval-offset-after-touch", type=int, default=1)
    p.add_argument("--touch-threshold", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument(
        "--event-passer-labels",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resolve passer identity from raw provider PASS events and attach per-clip labels.",
    )
    p.add_argument(
        "--pass-event-tolerance-sec",
        type=float,
        default=0.6,
        help="Maximum |tracking_time - pass_event_time| for clip->event anchoring.",
    )
    p.add_argument(
        "--pass-event-refractory-frames",
        type=int,
        default=8,
        help="Refractory frame deduplication when matching pass events to tracking frames.",
    )

    p.add_argument("--gpus", default="0,1,2,3")
    p.add_argument("--workers-per-gpu", type=int, default=2)

    p.add_argument(
        "--smart-checkpoint",
        default="/mnt/data/remains/opta2026/trajModel_smart/checkpoints/smart_v1/best.pt",
    )
    p.add_argument(
        "--vocab-dir",
        default="/mnt/data/remains/opta2026/trajModel_smart/vocabs",
    )
    p.add_argument(
        "--touch-checkpoint",
        default="/mnt/data/remains/opta2026/playerToTouch/checkpoints/sportec_survival_v2/best.pt",
    )
    p.add_argument(
        "--bat-checkpoint",
        default="/mnt/data/remains/opta2026/ballAtTouch/initParamVar/checkpoints/sportec_gaussian_recv_v5/best.pt",
    )
    p.add_argument(
        "--pv-checkpoint",
        default="/mnt/data/remains/opta2026/possessionValue/checkpoints/pv_v1/best.pt",
    )
    p.add_argument(
        "--set-piece-pv-model",
        default="/mnt/data/remains/opta2026/possessionValue/checkpoints/set_piece_pv_v1/model.json",
    )
    return p.parse_args()


def load_manifest(path: str) -> List[dict]:
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


def _safe_float(v: object) -> float:
    try:
        return float(v)
    except Exception:
        return float("nan")


def _safe_int(v: object, default: int = 0) -> int:
    try:
        return int(float(v))
    except Exception:
        return int(default)


def load_observed_param_overrides(path: str) -> Dict[int, Dict[str, object]]:
    """Load observed release-parameter overrides keyed by clip_idx."""
    p = Path(path)
    if not path or (not p.exists()):
        return {}

    out: Dict[int, Dict[str, object]] = {}
    with p.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            group = str(row.get("variant_group", "") or "").strip().lower()
            vid = str(row.get("variant_id", "") or "").strip().lower()
            if group != "observed" and vid != "observed":
                continue
            try:
                clip_idx = int(float(row.get("clip_idx", "nan")))
            except Exception:
                continue

            v0x = _safe_float(row.get("v0x"))
            v0y = _safe_float(row.get("v0y"))
            v0z = _safe_float(row.get("v0z"))
            spin = _safe_float(row.get("spin_scalar", 0.0))
            if not np.isfinite(v0x) or not np.isfinite(v0y) or not np.isfinite(v0z):
                continue
            if not np.isfinite(spin):
                spin = 0.0

            out[clip_idx] = {
                "variant_id": "observed",
                "variant_group": "observed",
                "v0x": float(v0x),
                "v0y": float(v0y),
                "v0z": float(v0z),
                "spin_scalar": float(spin),
                "fit_rmse_xy": _safe_float(row.get("observed_fit_rmse_xy")),
                "fit_rmse_z": _safe_float(row.get("observed_fit_rmse_z")),
                "fit_near_zero": _safe_int(row.get("observed_fit_near_zero"), default=0),
                "fit_endpoint_xy_err": _safe_float(row.get("observed_fit_endpoint_xy_err")),
                "fit_endpoint_z_err": _safe_float(row.get("observed_fit_endpoint_z_err")),
                "fit_endpoint_enforced": _safe_int(row.get("observed_fit_endpoint_enforced"), default=0),
            }
    return out


def expected_variant_ids(include_observed: bool, n_local: int, n_global: int) -> List[str]:
    out: List[str] = []
    if include_observed:
        out.append("observed")
    out.extend([f"local_{i:03d}" for i in range(int(n_local))])
    out.extend([f"global_{i:03d}" for i in range(int(n_global))])
    return out


def parse_gpu_ids(s: str) -> List[int]:
    out = []
    for tok in str(s).split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(int(tok))
    return out


def parse_clip_indices(s: str) -> List[int]:
    out: List[int] = []
    seen: Set[int] = set()
    for tok in str(s).split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "-" in tok:
            lo_s, hi_s = tok.split("-", 1)
            lo = int(lo_s.strip())
            hi = int(hi_s.strip())
            if hi < lo:
                lo, hi = hi, lo
            for v in range(lo, hi + 1):
                if v not in seen:
                    out.append(v)
                    seen.add(v)
            continue
        v = int(tok)
        if v not in seen:
            out.append(v)
            seen.add(v)
    return out


def parse_clip_indices_file(path: str) -> List[int]:
    p = Path(path)
    if not path or (not p.exists()):
        return []
    txt = p.read_text()
    txt = txt.replace("\n", ",").replace("\r", ",").replace("\t", ",")
    return parse_clip_indices(txt)


def main() -> None:
    args = parse_args()
    manifest_all = load_manifest(args.manifest)
    observed_param_overrides = load_observed_param_overrides(args.observed_params_csv)
    if observed_param_overrides:
        print(
            f"[runner] observed param overrides loaded: "
            f"{len(observed_param_overrides)} clips from {args.observed_params_csv}"
        )
    clip_filter = parse_clip_indices(args.clip_indices)
    if args.clip_indices_file:
        clip_filter.extend(parse_clip_indices_file(args.clip_indices_file))
    if clip_filter:
        # Preserve first occurrence order while removing duplicates.
        seen = set()
        uniq = []
        for v in clip_filter:
            iv = int(v)
            if iv in seen:
                continue
            uniq.append(iv)
            seen.add(iv)
        clip_filter = uniq

    manifest_items: List[Tuple[int, dict]]
    if clip_filter:
        kept: List[Tuple[int, dict]] = []
        dropped = 0
        for cidx in clip_filter:
            if 0 <= int(cidx) < len(manifest_all):
                kept.append((int(cidx), manifest_all[int(cidx)]))
            else:
                dropped += 1
        manifest_items = kept
        if dropped:
            print(f"[runner] warning: dropped {dropped} out-of-range clip indices")
    else:
        manifest_items = [(i, e) for i, e in enumerate(manifest_all)]

    if args.max_clips is not None:
        manifest_items = manifest_items[: int(args.max_clips)]

    manifest = [entry for _, entry in manifest_items]
    print(
        f"[runner] loaded manifest: total={len(manifest_all)} selected={len(manifest_items)} clips"
    )
    if observed_param_overrides:
        selected_clip_idxs = set(int(cidx) for cidx, _ in manifest_items)
        covered = int(sum(1 for cidx in selected_clip_idxs if cidx in observed_param_overrides))
        print(f"[runner] observed override coverage: {covered}/{len(selected_clip_idxs)} selected clips")

    passer_labels_by_clip: Dict[int, Dict[str, object]] = {}
    if bool(args.event_passer_labels):
        t_label0 = time.time()
        try:
            label_entries = manifest_all if clip_filter else manifest
            resolved_labels = build_manifest_passer_labels(
                manifest_entries=label_entries,
                tolerance_sec=float(args.pass_event_tolerance_sec),
                refractory_frames=int(args.pass_event_refractory_frames),
            )
            if clip_filter:
                passer_labels_by_clip = {
                    int(cidx): resolved_labels.get(int(cidx), {}) for cidx, _ in manifest_items
                }
            else:
                passer_labels_by_clip = resolved_labels
            n_resolved = int(sum(1 for cidx, _ in manifest_items if int(cidx) in passer_labels_by_clip))
            n_labeled = int(
                sum(
                    1
                    for cidx, _ in manifest_items
                    if str((passer_labels_by_clip.get(int(cidx), {}) or {}).get("passer_player_name", "")).strip()
                )
            )
            print(
                "[runner] event passer labels "
                f"resolved={n_resolved}/{len(manifest_items)} "
                f"named={n_labeled} elapsed={time.time() - t_label0:.1f}s"
            )
        except Exception as exc:
            print(f"[runner] warning: event passer-label resolution failed: {type(exc).__name__}: {exc}")
            passer_labels_by_clip = {}

    caps = estimate_caps_from_manifest(
        manifest,
        max_entries=args.cap_estimation_max_clips,
    )
    if args.min_speed_xy_max is not None:
        caps.speed_xy_max = max(float(caps.speed_xy_max), float(args.min_speed_xy_max))
    if args.min_vz_abs_max is not None:
        caps.vz_abs_max = max(float(caps.vz_abs_max), float(args.min_vz_abs_max))
    if args.min_spin_abs_max is not None:
        caps.spin_abs_max = max(float(caps.spin_abs_max), float(args.min_spin_abs_max))
    print(
        "[runner] caps "
        f"speed_xy_max={caps.speed_xy_max:.3f} "
        f"vz_abs_max={caps.vz_abs_max:.3f} "
        f"spin_abs_max={caps.spin_abs_max:.3f}"
    )

    done_keys: Set[Tuple[int, str]] = set()
    if args.resume and Path(args.output_csv).exists():
        done_keys = load_done_keys(args.output_csv)
        print(f"[runner] resume mode: loaded {len(done_keys)} done keys")

    expected_ids = expected_variant_ids(args.include_observed, args.variants_local, args.variants_global)
    expected_set = set(expected_ids)
    done_by_clip: Dict[int, Set[str]] = {}
    for cidx, vid in done_keys:
        done_by_clip.setdefault(int(cidx), set()).add(str(vid))

    tasks: List[dict] = []
    for clip_idx, entry in manifest_items:
        existing = done_by_clip.get(int(clip_idx), set())
        if expected_set.issubset(existing) or "rejected_fit" in existing:
            continue
        skip_ids = sorted(expected_set.intersection(existing))
        tasks.append(
            {
                "clip_idx": int(clip_idx),
                "entry": entry,
                "caps": asdict(caps),
                "skip_variant_ids": skip_ids,
                "passer_event_label": passer_labels_by_clip.get(int(clip_idx), {}),
                "observed_param_override": observed_param_overrides.get(int(clip_idx)),
            }
        )
    print(f"[runner] queued tasks: {len(tasks)} clips")
    if not tasks:
        print("[runner] nothing to do")
        return

    gpus = parse_gpu_ids(args.gpus)
    if not gpus:
        gpus = [0]
    if not torch.cuda.is_available():
        print("[runner] CUDA unavailable; running workers on CPU.")
        gpus = [0]

    model_paths = {
        "smart_checkpoint": args.smart_checkpoint,
        "vocab_dir": args.vocab_dir,
        "touch_checkpoint": args.touch_checkpoint,
        "bat_checkpoint": args.bat_checkpoint,
        "pv_checkpoint": args.pv_checkpoint,
        "set_piece_pv_model": args.set_piece_pv_model,
    }
    cfg = {
        "context_len": int(args.context_len),
        "rollout_len": int(args.rollout_len),
        "physics_len": int(args.physics_len),
        "eval_offset_after_touch": int(args.eval_offset_after_touch),
        "touch_threshold": float(args.touch_threshold),
        "local_variants": int(args.variants_local),
        "global_variants": int(args.variants_global),
        "include_observed": bool(args.include_observed),
        "observed_use_gt_players": bool(args.observed_use_gt_players),
        "require_near_zero_fit": bool(args.require_near_zero_fit),
        "seed": int(args.seed),
    }

    writer = CsvStreamWriter(args.output_csv)
    t0 = time.time()
    rows_written = 0
    clip_done = 0
    clip_fail = 0

    try:
        for result in run_distributed(
            tasks,
            gpu_ids=gpus,
            workers_per_gpu=int(args.workers_per_gpu),
            init_fn=init_worker,
            init_kwargs={"model_paths": model_paths, "config": cfg},
            task_fn=process_clip_task,
        ):
            clip_done += 1
            rows = result.get("rows", [])
            n = writer.write_rows(rows)
            rows_written += n
            if not bool(result.get("ok", False)):
                clip_fail += 1

            if rows_written % int(args.flush_every) < n:
                writer.flush()

            if clip_done % 10 == 0:
                dt = max(1e-6, time.time() - t0)
                print(
                    f"[runner] clips={clip_done}/{len(tasks)} "
                    f"rows={rows_written} "
                    f"rows_per_sec={rows_written / dt:.2f} "
                    f"fail_clips={clip_fail}"
                )
    finally:
        writer.flush()
        writer.close()

    dt = max(1e-6, time.time() - t0)
    print(
        f"[runner] done clips={clip_done} rows={rows_written} "
        f"fail_clips={clip_fail} elapsed={dt:.1f}s"
    )
    print(f"[runner] output: {args.output_csv}")


if __name__ == "__main__":
    main()
