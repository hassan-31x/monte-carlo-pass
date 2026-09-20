#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd


def _parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Rank passers by observed-pass PV differential versus hypothetical "
            "local/global distributions (mean/median and percentiles)."
        )
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=here / "pass_mc_sportec_test.csv",
        help="Monte Carlo pass-search CSV (default: pass_mc_sportec_test.csv in this directory).",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=here / "passer_rankings.csv",
        help="Output passer ranking CSV.",
    )
    parser.add_argument(
        "--output-csv-local",
        type=Path,
        default=here / "passer_rankings_local.csv",
        help="Local-only passer ranking CSV (sorted by avg_obs_minus_hyp_mean_local).",
    )
    parser.add_argument(
        "--output-csv-global",
        type=Path,
        default=here / "passer_rankings_global.csv",
        help="Global-only passer ranking CSV (sorted by avg_obs_minus_hyp_mean_global).",
    )
    parser.add_argument(
        "--clip-output-csv",
        type=Path,
        default=here / "pass_clip_percentiles.csv",
        help="Optional per-clip observed percentile CSV (audit/debug).",
    )
    parser.add_argument(
        "--no-name-resolution",
        action="store_true",
        help="Disable Sportec/Kloppy name resolution and keep CSV/fallback passer labels.",
    )
    return parser.parse_args()


def _coerce_bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    txt = s.astype(str).str.lower().str.strip()
    return txt.isin({"1", "true", "t", "yes", "y"})


def _percentile_of_zero(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    lt = float(np.sum(finite < 0.0))
    eq = float(np.sum(np.isclose(finite, 0.0, atol=1e-12)))
    return 100.0 * (lt + 0.5 * eq) / float(finite.size)


def _safe_float(v: object) -> float:
    try:
        out = float(v)
    except Exception:
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _nonempty_str(v: object) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip()
    return s if s else None


def _finite(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return arr
    return arr[np.isfinite(arr)]


def _nanmean(arr: np.ndarray) -> float:
    x = _finite(arr)
    if x.size == 0:
        return float("nan")
    return float(np.mean(x))


def _nanmedian(arr: np.ndarray) -> float:
    x = _finite(arr)
    if x.size == 0:
        return float("nan")
    return float(np.median(x))


def _first_non_null(values: Iterable[object]) -> Optional[object]:
    for v in values:
        if v is None:
            continue
        if isinstance(v, float) and np.isnan(v):
            continue
        if isinstance(v, str) and not v.strip():
            continue
        return v
    return None


def _team_side_from_player(player: object) -> Optional[str]:
    team = getattr(player, "team", None)
    if team is None:
        return None
    ground = getattr(team, "ground", None)
    if ground is not None:
        g = str(ground).lower()
        if "home" in g:
            return "home"
        if "away" in g:
            return "away"
    tid = str(getattr(team, "team_id", "")).lower()
    if "home" in tid:
        return "home"
    if "away" in tid:
        return "away"
    return None


def _team_side_from_team(team: object) -> Optional[str]:
    ground = getattr(team, "ground", None)
    if ground is not None:
        g = str(ground).lower()
        if "home" in g:
            return "home"
        if "away" in g:
            return "away"
    tid = str(getattr(team, "team_id", "")).lower()
    if "home" in tid:
        return "home"
    if "away" in tid:
        return "away"
    return None


def _pick_player_name(player_obj: object) -> Optional[str]:
    for attr in ("full_name", "name"):
        v = getattr(player_obj, attr, None)
        if v is not None and str(v).strip():
            return str(v).strip()
    return None


def _pick_player_position(player_obj: object) -> Optional[str]:
    for attr in ("starting_position", "position"):
        v = getattr(player_obj, attr, None)
        if v is not None and str(v).strip():
            return str(v).strip()
    positions = getattr(player_obj, "positions", None)
    if positions:
        first = next(iter(positions), None)
        if first is not None and str(first).strip():
            return str(first).strip()
    return None


def _jersey_sort_value(v: object) -> float:
    try:
        x = float(v)
    except Exception:
        return 99.0
    return x if math.isfinite(x) else 99.0


def _load_sportec_slot_metadata(match_id: str) -> Dict[int, Dict[str, object]]:
    """Rebuild the exact slot mapping used by kloppy_to_preprocessed for Sportec."""
    k2p_dir = Path(__file__).resolve().parents[1] / "ballPlayerTrajModel" / "publicData"
    if str(k2p_dir) not in sys.path:
        sys.path.insert(0, str(k2p_dir))

    import kloppy_to_preprocessed as k2p  # type: ignore

    ds = k2p._load_tracking_dataset(provider="sportec", match_id=str(match_id), limit=None)

    side_counts = {"home": Counter(), "away": Counter()}
    side_jersey: Dict[str, Dict[str, float]] = {"home": {}, "away": {}}
    player_lookup: Dict[str, object] = {}

    for fr in ds.records:
        for player in getattr(fr, "players_data", {}).keys():
            side = _team_side_from_player(player)
            if side not in {"home", "away"}:
                continue
            pid = str(getattr(player, "player_id", str(player)))
            side_counts[side][pid] += 1
            if pid not in side_jersey[side]:
                side_jersey[side][pid] = _jersey_sort_value(getattr(player, "jersey_no", np.nan))
            if pid not in player_lookup:
                player_lookup[pid] = player

    side_to_team_name: Dict[str, str] = {}
    metadata_player: Dict[str, Dict[str, object]] = {}
    for team in getattr(getattr(ds, "metadata", None), "teams", []) or []:
        side = _team_side_from_team(team)
        if side in {"home", "away"}:
            tname = getattr(team, "name", None)
            if tname is not None and str(tname).strip():
                side_to_team_name[side] = str(tname).strip()
        for p in getattr(team, "players", []) or []:
            pid = str(getattr(p, "player_id", ""))
            if not pid:
                continue
            metadata_player[pid] = {
                "player_name": _pick_player_name(p),
                "player_position": _pick_player_position(p),
                "jersey_no": _jersey_sort_value(getattr(p, "jersey_no", np.nan)),
            }

    slots: Dict[int, Dict[str, object]] = {}
    for side in ("home", "away"):
        ranked = sorted(
            side_counts[side].items(),
            key=lambda kv: (-kv[1], _jersey_sort_value(side_jersey[side].get(kv[0], 99.0)), kv[0]),
        )
        top = [pid for pid, _ in ranked[:11]]
        while len(top) < 11:
            top.append("")
        offset = 0 if side == "home" else 11
        for i, pid in enumerate(top):
            idx = offset + i
            rec: Dict[str, object] = {
                "team_side": side,
                "team_name": side_to_team_name.get(side),
                "player_id": pid or None,
                "player_name": None,
                "player_position": None,
                "jersey_no": float("nan"),
            }
            if pid:
                meta = metadata_player.get(pid, {})
                rec["player_name"] = meta.get("player_name")
                rec["player_position"] = meta.get("player_position")
                rec["jersey_no"] = meta.get("jersey_no", float("nan"))
                if rec["player_name"] is None and pid in player_lookup:
                    rec["player_name"] = _pick_player_name(player_lookup[pid])
                if rec["player_position"] is None and pid in player_lookup:
                    rec["player_position"] = _pick_player_position(player_lookup[pid])
                if not math.isfinite(float(rec["jersey_no"])) and pid in player_lookup:
                    rec["jersey_no"] = _jersey_sort_value(getattr(player_lookup[pid], "jersey_no", np.nan))
            slots[idx] = rec

    return slots


def _load_sportec_player_metadata(match_id: str) -> Dict[str, Dict[str, object]]:
    """Load per-player metadata keyed by provider player_id."""
    k2p_dir = Path(__file__).resolve().parents[1] / "ballPlayerTrajModel" / "publicData"
    if str(k2p_dir) not in sys.path:
        sys.path.insert(0, str(k2p_dir))

    import kloppy_to_preprocessed as k2p  # type: ignore

    ds = k2p._load_tracking_dataset(provider="sportec", match_id=str(match_id), limit=None)

    out: Dict[str, Dict[str, object]] = {}
    side_to_team_name: Dict[str, str] = {}
    for team in getattr(getattr(ds, "metadata", None), "teams", []) or []:
        side = _team_side_from_team(team)
        team_name = _nonempty_str(getattr(team, "name", None))
        if side in {"home", "away"} and team_name:
            side_to_team_name[side] = team_name
        for p in getattr(team, "players", []) or []:
            pid = _nonempty_str(getattr(p, "player_id", None))
            if not pid:
                continue
            out[pid] = {
                "player_id": pid,
                "player_name": _pick_player_name(p),
                "player_position": _pick_player_position(p),
                "jersey_no": _jersey_sort_value(getattr(p, "jersey_no", np.nan)),
                "team_side": side,
                "team_name": team_name or (side_to_team_name.get(side) if side else None),
            }

    # Fallback from tracking records for players missing from metadata team roster.
    for fr in ds.records:
        for player in getattr(fr, "players_data", {}).keys():
            pid = _nonempty_str(getattr(player, "player_id", None))
            if not pid:
                continue
            rec = out.setdefault(
                pid,
                {
                    "player_id": pid,
                    "player_name": None,
                    "player_position": None,
                    "jersey_no": float("nan"),
                    "team_side": _team_side_from_player(player),
                    "team_name": None,
                },
            )
            if rec.get("player_name") is None:
                rec["player_name"] = _pick_player_name(player)
            if rec.get("player_position") is None:
                rec["player_position"] = _pick_player_position(player)
            if not math.isfinite(float(rec.get("jersey_no", np.nan))):
                rec["jersey_no"] = _jersey_sort_value(getattr(player, "jersey_no", np.nan))

    return out


def _build_clip_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for clip_idx, g in df.groupby("clip_idx", sort=True):
        obs = g[g["is_observed_row"]]
        if obs.empty:
            continue
        obs_row = obs.iloc[0]

        local_vals = g.loc[g["variant_group"] == "local", "delta_vs_observed"].to_numpy(dtype=np.float64)
        global_vals = g.loc[g["variant_group"] == "global", "delta_vs_observed"].to_numpy(dtype=np.float64)
        all_vals = g.loc[g["variant_group"].isin(["local", "global"]), "delta_vs_observed"].to_numpy(dtype=np.float64)

        team_side = None if pd.isna(obs_row["passer_team_side"]) else str(obs_row["passer_team_side"])
        home_team = None if pd.isna(obs_row["home_team_label"]) else str(obs_row["home_team_label"])
        away_team = None if pd.isna(obs_row["away_team_label"]) else str(obs_row["away_team_label"])
        if team_side == "home":
            team_label = home_team
        elif team_side == "away":
            team_label = away_team
        else:
            team_label = None

        rows.append(
            {
                "clip_idx": int(clip_idx),
                "match_id": str(obs_row["match_id"]),
                "passer_entity_idx": int(obs_row["passer_entity_idx"]),
                "passer_team_side": team_side,
                "passer_team_label": team_label,
                "home_team_label": home_team,
                "away_team_label": away_team,
                "passer_player_id": None if ("passer_player_id" not in obs_row.index or pd.isna(obs_row["passer_player_id"])) else str(obs_row["passer_player_id"]),
                "passer_player_name": None if pd.isna(obs_row["passer_player_name"]) else str(obs_row["passer_player_name"]),
                "passer_inferred_band": None if pd.isna(obs_row["passer_inferred_band"]) else str(obs_row["passer_inferred_band"]),
                "passer_inferred_lateral_band": None
                if pd.isna(obs_row["passer_inferred_lateral_band"])
                else str(obs_row["passer_inferred_lateral_band"]),
                "passer_formation_slot_idx": _safe_float(obs_row["passer_formation_slot_idx"]),
                "observed_pv_net": _safe_float(obs_row["observed_pv_net_for_clip"]),
                "num_local_variants": int(np.sum(np.isfinite(local_vals))),
                "num_global_variants": int(np.sum(np.isfinite(global_vals))),
                "num_hypothetical_variants": int(np.sum(np.isfinite(all_vals))),
                # delta_vs_observed = variant_pv - observed_pv, so observed - agg(variant) = -agg(delta).
                "obs_minus_hyp_mean_local": -_nanmean(local_vals),
                "obs_minus_hyp_median_local": -_nanmedian(local_vals),
                "obs_minus_hyp_mean_global": -_nanmean(global_vals),
                "obs_minus_hyp_median_global": -_nanmedian(global_vals),
                "obs_minus_hyp_mean_combined": -_nanmean(all_vals),
                "obs_minus_hyp_median_combined": -_nanmedian(all_vals),
                "observed_pct_local": _percentile_of_zero(local_vals),
                "observed_pct_global": _percentile_of_zero(global_vals),
                "observed_pct_combined": _percentile_of_zero(all_vals),
            }
        )
    return pd.DataFrame(rows)


def _apply_name_resolution(clip_df: pd.DataFrame, use_name_resolution: bool) -> pd.DataFrame:
    if clip_df.empty:
        return clip_df

    clip_df = clip_df.copy()
    clip_df["passer_display_name"] = clip_df["passer_player_name"]
    clip_df["passer_name_source"] = np.where(
        clip_df["passer_player_name"].notna() & (clip_df["passer_player_name"].astype(str).str.strip() != ""),
        "csv_name",
        "fallback_slot",
    )
    clip_df["passer_team_name"] = clip_df["passer_team_label"]
    if "passer_player_id" not in clip_df.columns:
        clip_df["passer_player_id"] = None
    clip_df["passer_player_id"] = clip_df["passer_player_id"].where(clip_df["passer_player_id"].notna(), None)
    clip_df["passer_position_kloppy"] = None
    clip_df["passer_jersey_no"] = np.nan

    if not use_name_resolution:
        missing = clip_df["passer_display_name"].isna() | (clip_df["passer_display_name"].astype(str).str.strip() == "")
        clip_df.loc[missing, "passer_display_name"] = clip_df.loc[missing].apply(
            lambda r: f"{r['match_id']}:{r['passer_team_side']}_slot_{int(r['passer_entity_idx']):02d}",
            axis=1,
        )
        return clip_df

    unique_match_ids = sorted(set(str(m) for m in clip_df["match_id"].dropna().astype(str).tolist()))
    match_slot_meta: Dict[str, Dict[int, Dict[str, object]]] = {}
    match_player_meta: Dict[str, Dict[str, Dict[str, object]]] = {}

    for match_id in unique_match_ids:
        if not match_id:
            continue
        try:
            match_slot_meta[match_id] = _load_sportec_slot_metadata(match_id=match_id)
            match_player_meta[match_id] = _load_sportec_player_metadata(match_id=match_id)
            print(
                f"[rank_passers] name mapping loaded for {match_id} "
                f"(slots={len(match_slot_meta[match_id])}, players={len(match_player_meta[match_id])})"
            )
        except Exception as exc:
            print(f"[rank_passers] warning: could not load name mapping for {match_id}: {type(exc).__name__}: {exc}")

    for i in clip_df.index.tolist():
        row = clip_df.loc[i]
        match_id = str(row["match_id"])
        player_meta = match_player_meta.get(match_id, {})
        slot_meta = match_slot_meta.get(match_id, {})

        pid_row = _nonempty_str(row.get("passer_player_id"))
        meta = player_meta.get(pid_row or "")

        if meta is None:
            try:
                slot = int(row["passer_entity_idx"])
            except Exception:
                slot = -1
            meta = slot_meta.get(slot)

        if meta is None:
            continue

        pid = _nonempty_str(meta.get("player_id"))
        if pid:
            clip_df.at[i, "passer_player_id"] = pid

        team_name = _nonempty_str(meta.get("team_name"))
        if team_name:
            clip_df.at[i, "passer_team_name"] = team_name

        pos = _nonempty_str(meta.get("player_position"))
        if pos:
            clip_df.at[i, "passer_position_kloppy"] = pos

        jersey = _safe_float(meta.get("jersey_no"))
        if math.isfinite(jersey):
            clip_df.at[i, "passer_jersey_no"] = jersey

        nm = _nonempty_str(meta.get("player_name"))
        cur_name = _nonempty_str(clip_df.at[i, "passer_display_name"])
        if nm and (cur_name is None or pid_row is None):
            clip_df.at[i, "passer_display_name"] = nm
            clip_df.at[i, "passer_name_source"] = "kloppy_metadata"

    missing = clip_df["passer_display_name"].isna() | (clip_df["passer_display_name"].astype(str).str.strip() == "")
    clip_df.loc[missing, "passer_display_name"] = clip_df.loc[missing].apply(
        lambda r: f"{r['match_id']}:{r['passer_team_side']}_slot_{int(r['passer_entity_idx']):02d}",
        axis=1,
    )

    return clip_df


def _build_rankings(clip_df: pd.DataFrame) -> pd.DataFrame:
    if clip_df.empty:
        return pd.DataFrame()

    clip_df = clip_df.copy()
    clip_df["passer_player_id_norm"] = clip_df["passer_player_id"].apply(_nonempty_str)
    clip_df["passer_group_key"] = clip_df.apply(
        lambda r: (
            f"{r['match_id']}:{r['passer_player_id_norm']}"
            if r["passer_player_id_norm"] is not None
            else f"{r['match_id']}:{r['passer_team_side']}:{int(r['passer_entity_idx'])}"
        ),
        axis=1,
    )

    group_cols = ["match_id", "passer_group_key"]

    def _agg_one(g: pd.DataFrame) -> pd.Series:
        ent_idx = _safe_float(_first_non_null(g["passer_entity_idx"]))
        return pd.Series(
            {
                "passer_team_side": _first_non_null(g["passer_team_side"]),
                "passer_entity_idx": int(ent_idx) if math.isfinite(ent_idx) else -1,
                "passer_display_name": _first_non_null(g["passer_display_name"]),
                "passer_name_source": _first_non_null(g["passer_name_source"]),
                "passer_player_id": _first_non_null(g["passer_player_id"]),
                "passer_team_name": _first_non_null(g["passer_team_name"]),
                "passer_position_kloppy": _first_non_null(g["passer_position_kloppy"]),
                "passer_inferred_band": _first_non_null(g["passer_inferred_band"]),
                "passer_inferred_lateral_band": _first_non_null(g["passer_inferred_lateral_band"]),
                "passer_formation_slot_idx": _safe_float(_first_non_null(g["passer_formation_slot_idx"])),
                "passer_jersey_no": _safe_float(_first_non_null(g["passer_jersey_no"])),
                "num_passes": int(g["clip_idx"].nunique()),
                "avg_obs_minus_hyp_mean_combined": float(g["obs_minus_hyp_mean_combined"].mean(skipna=True)),
                "avg_obs_minus_hyp_median_combined": float(g["obs_minus_hyp_median_combined"].mean(skipna=True)),
                "avg_obs_minus_hyp_mean_local": float(g["obs_minus_hyp_mean_local"].mean(skipna=True)),
                "avg_obs_minus_hyp_median_local": float(g["obs_minus_hyp_median_local"].mean(skipna=True)),
                "avg_obs_minus_hyp_mean_global": float(g["obs_minus_hyp_mean_global"].mean(skipna=True)),
                "avg_obs_minus_hyp_median_global": float(g["obs_minus_hyp_median_global"].mean(skipna=True)),
                "avg_observed_pct_combined": float(g["observed_pct_combined"].mean(skipna=True)),
                "avg_observed_pct_local": float(g["observed_pct_local"].mean(skipna=True)),
                "avg_observed_pct_global": float(g["observed_pct_global"].mean(skipna=True)),
                "median_observed_pct_combined": float(g["observed_pct_combined"].median(skipna=True)),
                "std_observed_pct_combined": float(g["observed_pct_combined"].std(skipna=True)),
                "avg_observed_pv_net": float(g["observed_pv_net"].mean(skipna=True)),
                "avg_num_hypothetical_variants": float(g["num_hypothetical_variants"].mean(skipna=True)),
            }
        )

    gb = clip_df.groupby(group_cols, dropna=False, sort=False)
    try:
        out = gb.apply(_agg_one, include_groups=False).reset_index()
    except TypeError:
        out = gb.apply(_agg_one).reset_index()
    out = out.drop(columns=["passer_group_key"], errors="ignore")
    out = out.sort_values(
        by=["avg_obs_minus_hyp_mean_combined", "avg_obs_minus_hyp_median_combined", "num_passes", "avg_observed_pv_net"],
        ascending=[False, False, False, False],
        kind="mergesort",
    ).reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1, dtype=np.int64))
    return out


def _build_rankings_by_metric(rank_df: pd.DataFrame, metric_col: str) -> pd.DataFrame:
    if rank_df.empty:
        return rank_df.copy()
    out = rank_df.drop(columns=["rank"], errors="ignore").copy()
    extra_metric = None
    if metric_col == "avg_obs_minus_hyp_mean_local":
        extra_metric = "avg_obs_minus_hyp_median_local"
    elif metric_col == "avg_obs_minus_hyp_mean_global":
        extra_metric = "avg_obs_minus_hyp_median_global"
    elif metric_col == "avg_obs_minus_hyp_mean_combined":
        extra_metric = "avg_obs_minus_hyp_median_combined"

    sort_cols = [metric_col, "num_passes", "avg_observed_pv_net"]
    sort_asc = [False, False, False]
    if extra_metric and extra_metric in out.columns:
        sort_cols.insert(1, extra_metric)
        sort_asc.insert(1, False)
    out = out.sort_values(by=sort_cols, ascending=sort_asc, kind="mergesort").reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1, dtype=np.int64))
    out.insert(1, "rank_metric", metric_col)
    return out


def main() -> None:
    args = _parse_args()
    if not args.input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.input_csv}")

    usecols = [
        "clip_idx",
        "match_id",
        "variant_group",
        "delta_vs_observed",
        "is_observed_row",
        "observed_pv_net_for_clip",
        "passer_entity_idx",
        "passer_team_side",
        "passer_player_id",
        "passer_player_name",
        "passer_inferred_band",
        "passer_inferred_lateral_band",
        "passer_formation_slot_idx",
        "home_team_label",
        "away_team_label",
    ]
    input_cols = pd.read_csv(args.input_csv, nrows=0).columns.tolist()
    read_cols = [c for c in usecols if c in input_cols]
    df = pd.read_csv(args.input_csv, usecols=read_cols)
    if "passer_player_id" not in df.columns:
        df["passer_player_id"] = None
    df["is_observed_row"] = _coerce_bool_series(df["is_observed_row"])
    df["delta_vs_observed"] = pd.to_numeric(df["delta_vs_observed"], errors="coerce")
    df["clip_idx"] = pd.to_numeric(df["clip_idx"], errors="coerce").astype("Int64")
    df = df[df["clip_idx"].notna()].copy()
    df["clip_idx"] = df["clip_idx"].astype(np.int64)
    df["passer_entity_idx"] = pd.to_numeric(df["passer_entity_idx"], errors="coerce").fillna(-1).astype(np.int64)

    clip_df = _build_clip_table(df)
    clip_df = _apply_name_resolution(clip_df, use_name_resolution=(not args.no_name_resolution))
    rank_df = _build_rankings(clip_df)
    local_rank_df = _build_rankings_by_metric(rank_df, "avg_obs_minus_hyp_mean_local")
    global_rank_df = _build_rankings_by_metric(rank_df, "avg_obs_minus_hyp_mean_global")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    rank_df.to_csv(args.output_csv, index=False)
    args.output_csv_local.parent.mkdir(parents=True, exist_ok=True)
    local_rank_df.to_csv(args.output_csv_local, index=False)
    args.output_csv_global.parent.mkdir(parents=True, exist_ok=True)
    global_rank_df.to_csv(args.output_csv_global, index=False)

    if args.clip_output_csv:
        args.clip_output_csv.parent.mkdir(parents=True, exist_ok=True)
        clip_df.to_csv(args.clip_output_csv, index=False)

    print(
        f"[rank_passers] clips={len(clip_df)} passers={len(rank_df)} "
        f"output_combined={args.output_csv} output_local={args.output_csv_local} "
        f"output_global={args.output_csv_global} clip_output={args.clip_output_csv}"
    )


if __name__ == "__main__":
    main()
