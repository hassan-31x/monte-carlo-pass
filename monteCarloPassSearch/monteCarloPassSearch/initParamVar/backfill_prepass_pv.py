#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch

from sim_core import _clip_features_to_metres, _compute_pre_pass_pv
from sim_viz_runtime import try_load_pv_model


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Backfill pre-pass PV columns and pv_added_vs_prepass for an existing MCPS CSV."
        )
    )
    p.add_argument("--input-csv", type=Path, required=True)
    p.add_argument("--output-csv", type=Path, default=None)
    p.add_argument(
        "--pv-checkpoint",
        type=str,
        default="/mnt/data/remains/opta2026/possessionValue/checkpoints/pv_v1/best.pt",
    )
    p.add_argument("--device", type=str, default="cuda:0")
    return p.parse_args()


def _safe_int(v: object, default: int = 0) -> int:
    try:
        return int(float(v))
    except Exception:
        return int(default)


def _clip_prepass_pv(
    *,
    clip_path: str,
    kfl: int,
    pv_result: tuple | None,
    device: torch.device,
) -> Tuple[float, float, float]:
    try:
        data = np.load(clip_path)
        features = _clip_features_to_metres(data["features"])
        entity_type = data["entity_type"].astype(np.int64)
        ball_idx_arr = np.where(entity_type == 2)[0]
        if len(ball_idx_arr) == 0:
            return math.nan, math.nan, math.nan
        ball_idx = int(ball_idx_arr[0])
        return _compute_pre_pass_pv(
            pv_result=pv_result,
            features_m=features,
            entity_type=entity_type,
            ball_idx=ball_idx,
            kfl=int(kfl),
            device=device,
        )
    except Exception:
        return math.nan, math.nan, math.nan


def main() -> None:
    args = _parse_args()
    input_csv = args.input_csv
    output_csv = args.output_csv or args.input_csv

    df = pd.read_csv(input_csv)
    need_cols = {"clip_idx", "clip_path", "kick_frame_local_refined", "pv_net", "variant_group"}
    missing = sorted(need_cols - set(df.columns))
    if missing:
        raise RuntimeError(f"Missing required input columns: {missing}")

    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(str(args.device))

    pv_result = try_load_pv_model(args.pv_checkpoint, device)
    if pv_result is None:
        raise RuntimeError("Could not load PV model for pre-pass backfill.")

    # One representative row per clip.
    clip_meta = (
        df[["clip_idx", "clip_path", "kick_frame_local_refined"]]
        .drop_duplicates(subset=["clip_idx"], keep="first")
        .copy()
    )
    clip_meta["clip_idx"] = clip_meta["clip_idx"].astype(int)

    prepass_map: Dict[int, Tuple[float, float, float]] = {}
    total = int(len(clip_meta))
    for i, row in enumerate(clip_meta.itertuples(index=False), 1):
        clip_idx = int(row.clip_idx)
        clip_path = str(row.clip_path)
        kfl = _safe_int(row.kick_frame_local_refined, default=0)
        prepass_map[clip_idx] = _clip_prepass_pv(
            clip_path=clip_path,
            kfl=kfl,
            pv_result=pv_result,
            device=device,
        )
        if i % 50 == 0 or i == total:
            print(f"[backfill_prepass_pv] clips={i}/{total}")

    df["clip_idx"] = df["clip_idx"].astype(int)
    df["pv_net"] = pd.to_numeric(df["pv_net"], errors="coerce")

    df["pre_pass_pv_home"] = df["clip_idx"].map(lambda c: prepass_map.get(int(c), (math.nan, math.nan, math.nan))[0])
    df["pre_pass_pv_away"] = df["clip_idx"].map(lambda c: prepass_map.get(int(c), (math.nan, math.nan, math.nan))[1])
    df["pre_pass_pv_net"] = df["clip_idx"].map(lambda c: prepass_map.get(int(c), (math.nan, math.nan, math.nan))[2])

    df["pv_added_vs_prepass"] = df["pv_net"] - df["pre_pass_pv_net"]
    df.loc[~(np.isfinite(df["pv_net"]) & np.isfinite(df["pre_pass_pv_net"])), "pv_added_vs_prepass"] = np.nan

    obs_added = (
        df[df["variant_group"].astype(str) == "observed"][["clip_idx", "pv_added_vs_prepass"]]
        .drop_duplicates(subset=["clip_idx"], keep="first")
        .set_index("clip_idx")["pv_added_vs_prepass"]
        .to_dict()
    )
    df["observed_pv_added_vs_prepass_for_clip"] = df["clip_idx"].map(obs_added)
    df["delta_pv_added_vs_observed"] = df["pv_added_vs_prepass"] - df["observed_pv_added_vs_prepass_for_clip"]
    df.loc[
        ~(np.isfinite(df["pv_added_vs_prepass"]) & np.isfinite(df["observed_pv_added_vs_prepass_for_clip"])),
        "delta_pv_added_vs_observed",
    ] = np.nan

    tmp = output_csv.with_suffix(output_csv.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(output_csv)
    print(f"[backfill_prepass_pv] wrote {output_csv}")


if __name__ == "__main__":
    main()
