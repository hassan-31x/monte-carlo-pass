#!/usr/bin/env python3
"""Build ballPlayerTrajModel-compatible preprocessed clips from kloppy open tracking data."""

from __future__ import annotations

import argparse
import csv
import io
import json
import random
import re
import sys
import urllib.request
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from tqdm.auto import tqdm

from kloppy import metrica, sportec

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset import (  # noqa: E402
    ClipEntry,
    STOP_CLASS_NAMES,
    STOP_CONTINUE,
    STOP_EGO_CORNER,
    STOP_EGO_FREE_KICK,
    STOP_EGO_GOAL_KICK,
    STOP_EGO_THROW_IN,
    STOP_OPP_CORNER,
    STOP_OPP_FREE_KICK,
    STOP_OPP_GOAL_KICK,
    STOP_OPP_THROW_IN,
    STOP_UNKNOWN_BREAK,
    compute_normalization_stats,
    save_stats,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert kloppy open tracking data into preprocessed clip artifacts."
    )
    parser.add_argument("--provider", choices=["metrica", "sportec"], required=True)
    parser.add_argument(
        "--match-ids",
        type=str,
        default="",
        help=(
            "Comma-separated match ids. Defaults: metrica=1,2,3; "
            "sportec=all 7 IDSSE Bundesliga matches."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Output preprocessed dir containing train/val/test manifests and clips.",
    )

    parser.add_argument("--history-frames", type=int, default=512)
    parser.add_argument("--future-frames", type=int, default=1024)

    parser.add_argument(
        "--ball-mode",
        choices=["xy", "xyz"],
        default="xy",
        help="xy: features [x,y,vx,vy], xyz: features [x,y,vx,vy,z,vz] (ball z/vz only).",
    )

    parser.add_argument("--anchor-contact-radius", type=float, default=1.6)
    parser.add_argument("--anchor-speed-min", type=float, default=1.5)
    parser.add_argument("--anchor-dv-min", type=float, default=0.5)
    parser.add_argument("--anchor-refractory-frames", type=int, default=8)
    parser.add_argument("--anchor-stride-fallback", type=int, default=18)
    parser.add_argument(
        "--pass-anchor-tolerance-sec",
        type=float,
        default=0.6,
        help="Max |tracking_time - pass_event_time| for event anchor matching.",
    )

    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--max-frames-per-match", type=int, default=None)
    parser.add_argument("--rebuild", action="store_true")

    return parser.parse_args()


def _default_match_ids(provider: str) -> List[str]:
    if provider == "metrica":
        return ["1", "2", "3"]
    # Seven open IDSSE Bundesliga matches.
    return ["J03WMX", "J03WN1", "J03WPY", "J03WOH", "J03WQQ", "J03WOY", "J03WR9"]


def _split_match_ids(
    match_ids: Sequence[str],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[List[str], List[str], List[str]]:
    ids = list(match_ids)
    if not ids:
        return [], [], []

    rng = random.Random(seed)
    rng.shuffle(ids)

    n_total = len(ids)
    n_test = int(round(n_total * test_ratio))
    n_val = int(round(n_total * val_ratio))

    if n_total >= 2:
        if test_ratio > 0.0 and n_test < 1:
            n_test = 1
        if val_ratio > 0.0 and n_val < 1 and n_total >= 3:
            n_val = 1
        if n_test + n_val >= n_total:
            n_val = max(0, n_total - n_test - 1)
            if n_test + n_val >= n_total:
                n_test = max(0, n_total - n_val - 1)

    test_ids = ids[:n_test]
    val_ids = ids[n_test : n_test + n_val]
    train_ids = ids[n_test + n_val :]

    if not train_ids and val_ids:
        train_ids.append(val_ids.pop())
    if not train_ids and test_ids:
        train_ids.append(test_ids.pop())

    return train_ids, val_ids, test_ids


def _resolve_side(player: Any) -> Optional[str]:
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

    team_id = str(getattr(team, "team_id", "")).lower()
    if "home" in team_id:
        return "home"
    if "away" in team_id:
        return "away"
    return None


def _to_centered_xy(provider: str, x: float, y: float) -> Tuple[float, float]:
    if provider == "metrica":
        # Metrica open tracking is normalized [0,1] with (0,0) at top-left.
        if -1e-3 <= x <= 1.001 and -1e-3 <= y <= 1.001:
            return (x - 0.5) * 105.0, (0.5 - y) * 68.0
    return x, y


def _causal_impute(features: np.ndarray, observed_mask: np.ndarray) -> np.ndarray:
    out = features.copy()
    t_steps, n_entities, n_feats = out.shape
    for n in range(n_entities):
        last = np.zeros(n_feats, dtype=np.float32)
        has_last = False
        for t in range(t_steps):
            if observed_mask[t, n]:
                last = out[t, n]
                has_last = True
            elif has_last:
                out[t, n] = last
    return out


def _build_slot_mapping(frame_rows: List[Dict[str, Any]]) -> Tuple[Dict[str, int], np.ndarray, int]:
    side_counts = {"home": Counter(), "away": Counter()}
    side_jersey: Dict[str, Dict[str, float]] = {"home": {}, "away": {}}

    for row in frame_rows:
        for p in row["players"]:
            side = p["side"]
            pid = p["player_id"]
            side_counts[side][pid] += 1
            if pid not in side_jersey[side]:
                side_jersey[side][pid] = float(p["jersey_no"])

    selected: List[Tuple[str, str]] = []
    for side in ["home", "away"]:
        ranked = sorted(
            side_counts[side].items(),
            key=lambda kv: (
                -kv[1],
                side_jersey[side].get(kv[0], 99.0),
                kv[0],
            ),
        )
        for pid, _ in ranked[:11]:
            selected.append((side, pid))

    real_player_count = len(selected)

    while len([s for s, _ in selected if s == "home"]) < 11:
        i = len([1 for s, _ in selected if s == "home"])
        selected.append(("home", f"__PAD_HOME_{i}__"))
    while len([s for s, _ in selected if s == "away"]) < 11:
        i = len([1 for s, _ in selected if s == "away"])
        selected.append(("away", f"__PAD_AWAY_{i}__"))

    # Enforce 22 player slots: home first, away second.
    home = [pid for side, pid in selected if side == "home"][:11]
    away = [pid for side, pid in selected if side == "away"][:11]
    ordered = home + away

    player_to_slot = {pid: idx for idx, pid in enumerate(ordered) if not pid.startswith("__PAD_")}

    entity_type = np.full((23,), 3, dtype=np.int64)
    entity_type[:11] = 0
    entity_type[11:22] = 1
    entity_type[22] = 2

    return player_to_slot, entity_type, real_player_count


def _detect_anchor_indices(
    features: np.ndarray,
    mask: np.ndarray,
    entity_type: np.ndarray,
    contact_radius: float,
    speed_min: float,
    dv_min: float,
    refractory_frames: int,
) -> List[int]:
    ball_idx = 22
    ball_valid = mask[:, ball_idx]
    if ball_valid.sum() < 2:
        return []

    ball_pos = features[:, ball_idx, :2]
    ball_vel = features[:, ball_idx, 2:4]
    speed = np.sqrt(np.square(ball_vel).sum(axis=1))
    speed_prev = np.concatenate([speed[:1], speed[:-1]], axis=0)
    dv = speed - speed_prev

    player_pos = features[:, :22, :2]
    player_valid = mask[:, :22] & (entity_type[:22][None, :] != 3)

    nearest = np.full((features.shape[0],), np.nan, dtype=np.float32)
    for t in range(features.shape[0]):
        valid = player_valid[t]
        if not np.any(valid):
            continue
        dif = player_pos[t, valid] - ball_pos[t]
        nearest[t] = float(np.sqrt(np.square(dif).sum(axis=1)).min())

    candidate = np.where(
        ball_valid
        & np.isfinite(nearest)
        & (nearest <= contact_radius)
        & (speed >= speed_min)
        & (dv >= dv_min)
    )[0]

    if candidate.size == 0:
        return []

    score = dv + 0.1 * speed
    picked: List[int] = []
    last = -10**9
    for idx in candidate.tolist():
        if idx - last <= refractory_frames:
            if picked and score[idx] > score[picked[-1]]:
                picked[-1] = idx
                last = idx
            continue
        picked.append(idx)
        last = idx
    return picked


def _nearest_side_at_frame(
    features: np.ndarray,
    mask: np.ndarray,
    frame_idx: int,
) -> Optional[str]:
    if frame_idx < 0 or frame_idx >= features.shape[0]:
        return None
    if not bool(mask[frame_idx, 22]):
        return None
    ball_xy = features[frame_idx, 22, :2]
    best_side = None
    best_dist = np.inf
    for idx in range(22):
        if not bool(mask[frame_idx, idx]):
            continue
        side = "home" if idx < 11 else "away"
        d = float(np.linalg.norm(features[frame_idx, idx, :2] - ball_xy))
        if d < best_dist:
            best_dist = d
            best_side = side
    return best_side


def _infer_restart_kind_from_ball_xy(x: float, y: float) -> str:
    ax = abs(float(x))
    ay = abs(float(y))
    # 105x68 centered pitch bounds.
    if ay >= 31.0 and ax <= 50.0:
        return "throw_in"
    if ax >= 50.0 and ay >= 31.0:
        return "corner"
    if ax >= 46.0 and ay <= 20.0:
        return "goal_kick"
    return "free_kick"


def _map_restart_to_stop_id(kind: str, restart_side: Optional[str], ego_side: Optional[str]) -> int:
    if restart_side is None or ego_side is None:
        return int(STOP_UNKNOWN_BREAK)
    is_ego = str(restart_side) == str(ego_side)
    if kind == "throw_in":
        return int(STOP_EGO_THROW_IN if is_ego else STOP_OPP_THROW_IN)
    if kind == "corner":
        return int(STOP_EGO_CORNER if is_ego else STOP_OPP_CORNER)
    if kind == "goal_kick":
        return int(STOP_EGO_GOAL_KICK if is_ego else STOP_OPP_GOAL_KICK)
    if kind == "free_kick":
        return int(STOP_EGO_FREE_KICK if is_ego else STOP_OPP_FREE_KICK)
    return int(STOP_UNKNOWN_BREAK)


def _figshare_file_id_from_url(url: str) -> str:
    m = re.search(r"/files/(\d+)$", str(url))
    if not m:
        # Newer kloppy releases serve IDSSE files from Hugging Face directly.
        return str(url)
    return m.group(1)


def _figshare_api_url(file_id: str) -> str:
    if str(file_id).startswith(("http://", "https://")):
        return str(file_id)
    return f"https://api.figshare.com/v2/file/download/{file_id}"


def _resolve_side_from_team(team: Any) -> Optional[str]:
    ground = getattr(team, "ground", None)
    if ground is not None:
        g = str(ground).lower()
        if "home" in g:
            return "home"
        if "away" in g:
            return "away"

    team_id = str(getattr(team, "team_id", "")).lower()
    if "home" in team_id:
        return "home"
    if "away" in team_id:
        return "away"
    return None


def _extract_pass_events_from_kloppy_event_dataset(event_dataset: Any) -> List[Dict[str, Any]]:
    team_side: Dict[str, str] = {}
    for team in getattr(getattr(event_dataset, "metadata", None), "teams", []) or []:
        side = _resolve_side_from_team(team)
        if side is None:
            continue
        team_side[str(getattr(team, "team_id", ""))] = side

    out: List[Dict[str, Any]] = []
    for event in getattr(event_dataset, "events", []) or []:
        event_type = str(getattr(getattr(event, "event_type", None), "name", "")).upper()
        if event_type != "PASS":
            continue
        period = int(getattr(getattr(event, "period", None), "id", 1) or 1)
        ts = getattr(event, "timestamp", None)
        if ts is None:
            continue
        t_sec = float(ts.total_seconds())
        if not np.isfinite(t_sec):
            continue
        team = getattr(event, "team", None)
        team_id = str(getattr(team, "team_id", "") or "")
        side = team_side.get(team_id)
        if side is None and team is not None:
            side = _resolve_side_from_team(team)
        out.append(
            {
                "period": period,
                "time": float(t_sec),
                "team_id": team_id,
                "side": side,
            }
        )
    return out


def _load_metrica_csv_pass_events(match_id: str) -> List[Dict[str, Any]]:
    url = (
        "https://raw.githubusercontent.com/metrica-sports/sample-data/master/"
        f"data/Sample_Game_{match_id}/Sample_Game_{match_id}_RawEventsData.csv"
    )
    with urllib.request.urlopen(url, timeout=30) as resp:
        text = io.TextIOWrapper(resp, encoding="utf-8")
        reader = csv.DictReader(text)
        period_min_time: Dict[int, float] = {}
        pass_rows: List[Dict[str, Any]] = []
        for row in reader:
            try:
                period = int(float(row.get("Period", "1") or 1))
            except Exception:
                period = 1
            try:
                t_abs = float(row.get("Start Time [s]", "nan"))
            except Exception:
                t_abs = np.nan
            if np.isfinite(t_abs):
                prev = period_min_time.get(period, np.inf)
                if float(t_abs) < float(prev):
                    period_min_time[period] = float(t_abs)

            typ = str(row.get("Type", "")).strip().upper()
            if typ != "PASS":
                continue
            team_raw = str(row.get("Team", "")).strip().lower()
            side: Optional[str]
            if team_raw.startswith("home"):
                side = "home"
            elif team_raw.startswith("away"):
                side = "away"
            else:
                side = None
            pass_rows.append(
                {
                    "period": int(period),
                    "time_abs": float(t_abs) if np.isfinite(t_abs) else np.nan,
                    "side": side,
                }
            )

    pass_events: List[Dict[str, Any]] = []
    for row in pass_rows:
        period = int(row["period"])
        t_abs = float(row["time_abs"])
        if not np.isfinite(t_abs):
            continue
        period_start = float(period_min_time.get(period, 0.0))
        t_rel = float(t_abs - period_start)
        if t_rel < 0.0:
            t_rel = 0.0
        pass_events.append(
            {
                "period": period,
                "time": t_rel,
                "team_id": str(row.get("side") or ""),
                "side": row.get("side"),
            }
        )
    return pass_events


def _load_provider_pass_events(provider: str, match_id: str) -> List[Dict[str, Any]]:
    if provider == "metrica":
        if str(match_id) in {"1", "2"}:
            return _load_metrica_csv_pass_events(match_id=str(match_id))
        if str(match_id) == "3":
            base = "https://raw.githubusercontent.com/metrica-sports/sample-data/master/data/Sample_Game_3/"
            event_ds = metrica.load_event(
                event_data=base + "Sample_Game_3_events.json",
                meta_data=base + "Sample_Game_3_metadata.xml",
            )
            return _extract_pass_events_from_kloppy_event_dataset(event_ds)
        return []

    from kloppy._providers.sportec import get_IDSSE_url  # local import to avoid private dependency at module load

    meta_url = _figshare_api_url(_figshare_file_id_from_url(get_IDSSE_url(match_id, "meta")))
    event_url = _figshare_api_url(_figshare_file_id_from_url(get_IDSSE_url(match_id, "event")))
    event_ds = sportec.load_event(
        event_data=event_url,
        meta_data=meta_url,
    )
    return _extract_pass_events_from_kloppy_event_dataset(event_ds)


def _match_pass_events_to_frame_indices(
    pass_events: Sequence[Dict[str, Any]],
    periods: np.ndarray,
    frame_times: np.ndarray,
    tolerance_sec: float,
    refractory_frames: int,
) -> List[Dict[str, Any]]:
    if not pass_events:
        return []

    periods = np.asarray(periods, dtype=np.int64).reshape(-1)
    frame_times = np.asarray(frame_times, dtype=np.float32).reshape(-1)
    if periods.size == 0 or frame_times.size == 0 or periods.size != frame_times.size:
        return []

    events_by_period: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for ev in pass_events:
        try:
            period = int(ev.get("period", 1))
        except Exception:
            period = 1
        try:
            t_event = float(ev.get("time", np.nan))
        except Exception:
            t_event = np.nan
        if not np.isfinite(t_event):
            continue
        events_by_period[period].append(
            {
                "period": period,
                "time": float(t_event),
                "team_id": str(ev.get("team_id", "") or ""),
                "side": ev.get("side"),
            }
        )

    picked: List[Dict[str, Any]] = []
    tol = float(max(tolerance_sec, 1e-6))
    for period, rows in events_by_period.items():
        idx = np.where((periods == int(period)) & np.isfinite(frame_times))[0]
        if idx.size == 0:
            continue
        t_valid = frame_times[idx]

        for ev in rows:
            t_event = float(ev["time"])
            right = int(np.searchsorted(t_valid, t_event, side="left"))
            cand: List[int] = []
            if 0 <= right < t_valid.shape[0]:
                cand.append(right)
            if right - 1 >= 0:
                cand.append(right - 1)
            if not cand:
                continue
            best_local = min(cand, key=lambda j: abs(float(t_valid[j]) - t_event))
            best_idx = int(idx[best_local])
            err = abs(float(frame_times[best_idx]) - t_event)
            if err <= tol:
                picked.append(
                    {
                        "idx": best_idx,
                        "err": float(err),
                        "period": int(period),
                        "time": float(t_event),
                        "team_id": ev["team_id"],
                        "side": ev["side"],
                    }
                )

    if not picked:
        return []

    by_idx: Dict[int, Dict[str, Any]] = {}
    for row in picked:
        idx = int(row["idx"])
        prev = by_idx.get(idx)
        if prev is None or float(row["err"]) < float(prev["err"]):
            by_idx[idx] = row

    matched = sorted(by_idx.values(), key=lambda r: int(r["idx"]))
    refractory = max(int(refractory_frames), 0)
    if refractory <= 0:
        return matched

    filtered: List[Dict[str, Any]] = []
    for row in matched:
        if not filtered:
            filtered.append(row)
            continue
        if int(row["idx"]) - int(filtered[-1]["idx"]) > refractory:
            filtered.append(row)
            continue
        if float(row["err"]) < float(filtered[-1]["err"]):
            filtered[-1] = row
    return filtered


def _load_tracking_dataset(provider: str, match_id: str, limit: Optional[int]) -> Any:
    if provider == "metrica":
        return metrica.load_open_data(match_id=match_id, limit=limit)
    # kloppy's open sportec helper currently resolves to ndownloader URLs that
    # can return empty 202 payloads. Resolve file IDs and load via Figshare API URLs.
    from kloppy._providers.sportec import get_IDSSE_url  # local import to avoid private dependency at module load

    meta_file_id = _figshare_file_id_from_url(get_IDSSE_url(match_id, "meta"))
    tracking_file_id = _figshare_file_id_from_url(get_IDSSE_url(match_id, "tracking"))
    meta_url = _figshare_api_url(meta_file_id)
    tracking_url = _figshare_api_url(tracking_file_id)
    return sportec.load_tracking(
        meta_data=meta_url,
        raw_data=tracking_url,
        limit=limit,
        only_alive=False,
    )


def _dataset_to_tensor(
    dataset_obj: Any,
    provider: str,
    ball_mode: str,
    max_frames: Optional[int],
) -> Optional[Dict[str, Any]]:
    records = list(dataset_obj.records)
    if max_frames is not None:
        records = records[: max(0, int(max_frames))]
    if len(records) < 2:
        return None

    frame_rows: List[Dict[str, Any]] = []
    for fr in records:
        period = int(getattr(getattr(fr, "period", None), "id", 1) or 1)
        timestamp = getattr(fr, "timestamp", None)
        t_sec = float(timestamp.total_seconds()) if timestamp is not None else float(len(frame_rows) * 0.04)

        players = []
        for player, pdata in getattr(fr, "players_data", {}).items():
            coord = getattr(pdata, "coordinates", None)
            if coord is None:
                continue
            side = _resolve_side(player)
            if side not in {"home", "away"}:
                continue

            pid = str(getattr(player, "player_id", str(player)))
            jersey_no = getattr(player, "jersey_no", np.nan)
            try:
                jersey_no = float(jersey_no)
            except Exception:
                jersey_no = np.nan

            x, y = _to_centered_xy(provider, float(coord.x), float(coord.y))
            players.append(
                {
                    "side": side,
                    "player_id": pid,
                    "jersey_no": jersey_no,
                    "x": float(x),
                    "y": float(y),
                }
            )

        ball_coord = getattr(fr, "ball_coordinates", None)
        ball = None
        if ball_coord is not None:
            bx, by = _to_centered_xy(provider, float(ball_coord.x), float(ball_coord.y))
            bz = float(getattr(ball_coord, "z", 0.0) or 0.0)
            ball = {"x": bx, "y": by, "z": bz}

        frame_rows.append(
            {
                "frame_id": int(getattr(fr, "frame_id", len(frame_rows))),
                "period": period,
                "time": t_sec,
                "players": players,
                "ball": ball,
            }
        )

    if len(frame_rows) < 2:
        return None

    player_to_slot, entity_type, real_player_count = _build_slot_mapping(frame_rows)

    t_steps = len(frame_rows)
    feat_dim = 4 if ball_mode == "xy" else 6
    features = np.zeros((t_steps, 23, feat_dim), dtype=np.float32)
    mask = np.zeros((t_steps, 23), dtype=bool)
    frames = np.zeros((t_steps,), dtype=np.int64)
    periods = np.zeros((t_steps,), dtype=np.int64)

    pos = np.zeros((t_steps, 23, 2), dtype=np.float32)
    ball_z = np.zeros((t_steps,), dtype=np.float32)
    ball_has_z = np.zeros((t_steps,), dtype=bool)
    times = np.zeros((t_steps,), dtype=np.float32)

    for t, row in enumerate(frame_rows):
        frames[t] = int(row["frame_id"])
        periods[t] = int(row["period"])
        times[t] = float(row["time"])

        for p in row["players"]:
            pid = p["player_id"]
            if pid not in player_to_slot:
                continue
            idx = int(player_to_slot[pid])
            pos[t, idx, 0] = float(p["x"])
            pos[t, idx, 1] = float(p["y"])
            mask[t, idx] = True

        ball = row["ball"]
        if ball is not None:
            pos[t, 22, 0] = float(ball["x"])
            pos[t, 22, 1] = float(ball["y"])
            mask[t, 22] = True
            ball_z[t] = float(ball["z"])
            ball_has_z[t] = True

    vel = np.zeros((t_steps, 23, 2), dtype=np.float32)
    vz = np.zeros((t_steps,), dtype=np.float32)
    for t in range(1, t_steps):
        dt = float(times[t] - times[t - 1])
        if not np.isfinite(dt) or dt <= 1e-6:
            dt = 0.04
        valid = mask[t] & mask[t - 1]
        if np.any(valid):
            vel[t, valid] = (pos[t, valid] - pos[t - 1, valid]) / dt

        if ball_has_z[t] and ball_has_z[t - 1] and mask[t, 22] and mask[t - 1, 22]:
            vz[t] = (ball_z[t] - ball_z[t - 1]) / dt

    features[:, :, :2] = pos
    features[:, :, 2:4] = vel
    if ball_mode == "xyz":
        features[:, 22, 4] = ball_z
        features[:, 22, 5] = vz

    features = _causal_impute(features=features, observed_mask=mask)

    return {
        "features": features,
        "mask": mask,
        "entity_type": entity_type,
        "frames": frames,
        "periods": periods,
        "times": times,
        "real_player_count": int(real_player_count),
    }


def _save_manifest(entries: Sequence[ClipEntry], manifest_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(asdict(entry)) + "\n")


def _extract_clips(
    match_data: Dict[str, Any],
    provider: str,
    match_id: str,
    out_split_dir: Path,
    history: int,
    future: int,
    anchor_contact_radius: float,
    anchor_speed_min: float,
    anchor_dv_min: float,
    anchor_refractory_frames: int,
    anchor_stride_fallback: int,
    pass_anchor_tolerance_sec: float,
) -> List[ClipEntry]:
    features = match_data["features"]
    mask = match_data["mask"]
    entity_type = match_data["entity_type"]
    frames = match_data["frames"]
    periods = match_data["periods"]
    times = match_data["times"]
    pass_events = list(match_data.get("pass_events", []) or [])
    real_player_count = int(match_data["real_player_count"])

    anchor_rows = _match_pass_events_to_frame_indices(
        pass_events=pass_events,
        periods=periods,
        frame_times=times,
        tolerance_sec=float(pass_anchor_tolerance_sec),
        refractory_frames=int(anchor_refractory_frames),
    )

    if not anchor_rows:
        heuristic_anchors = _detect_anchor_indices(
            features=features,
            mask=mask,
            entity_type=entity_type,
            contact_radius=float(anchor_contact_radius),
            speed_min=float(anchor_speed_min),
            dv_min=float(anchor_dv_min),
            refractory_frames=int(anchor_refractory_frames),
        )
        anchor_rows = [
            {
                "idx": int(a),
                "side": None,
                "team_id": "",
                "time": float(times[int(a)]) if np.isfinite(times[int(a)]) else np.nan,
                "period": int(periods[int(a)]),
            }
            for a in heuristic_anchors
        ]

    if not anchor_rows:
        lo = history - 1
        hi = features.shape[0] - future - 1
        if hi >= lo:
            anchor_rows = [
                {
                    "idx": int(a),
                    "side": None,
                    "team_id": "",
                    "time": float(times[int(a)]) if np.isfinite(times[int(a)]) else np.nan,
                    "period": int(periods[int(a)]),
                }
                for a in range(lo, hi + 1, max(1, int(anchor_stride_fallback)))
            ]

    entries: List[ClipEntry] = []
    clip_idx = 0
    out_split_dir.mkdir(parents=True, exist_ok=True)

    for anchor_row in anchor_rows:
        anchor = int(anchor_row["idx"])
        start = int(anchor - history + 1)
        end = int(anchor + future)
        if start < 0 or end >= features.shape[0]:
            continue

        clip_features = features[start : end + 1].astype(np.float32)
        clip_mask = mask[start : end + 1].astype(bool)
        clip_frames = frames[start : end + 1].astype(np.int64)
        clip_times = times[start : end + 1].astype(np.float32)
        clip_periods = periods[start : end + 1].astype(np.int16)

        if clip_features.shape[0] <= 1:
            continue

        entity_delta = clip_features[1:, :, :2] - clip_features[:-1, :, :2]
        entity_delta_mask = clip_mask[1:, :] & clip_mask[:-1, :]

        # Stop-event labeling for both context and future windows.
        stop_event_id = np.full((clip_features.shape[0],), STOP_CONTINUE, dtype=np.uint8)

        frame_gap = np.diff(clip_frames.astype(np.int64))
        time_gap = np.diff(clip_times.astype(np.float32))
        bad_gap = (frame_gap > 1) | (~np.isfinite(time_gap)) | (time_gap > (0.08 * np.maximum(frame_gap, 1)))
        stop_event_id[1:][bad_gap] = np.uint8(STOP_UNKNOWN_BREAK)

        ego_side = anchor_row.get("side")
        if ego_side not in {"home", "away"}:
            ego_side = _nearest_side_at_frame(features=features, mask=mask, frame_idx=int(anchor))

        ball_obs = clip_mask[:, 22]
        ball_xy = clip_features[:, 22, :2]
        ball_out = ball_obs & (
            (np.abs(ball_xy[:, 0]) > 52.5) | (np.abs(ball_xy[:, 1]) > 34.0)
        )

        # Out-of-play onset: generic break.
        out_on = (~ball_out[:-1]) & ball_out[1:] & ball_obs[:-1] & ball_obs[1:]
        stop_event_id[1:][out_on] = np.uint8(STOP_UNKNOWN_BREAK)

        # Restart onset (back in play): classify restart type + side at restart frame.
        out_off = ball_out[:-1] & (~ball_out[1:]) & ball_obs[:-1] & ball_obs[1:]
        restart_locals = np.where(out_off)[0] + 1
        for local_idx in restart_locals.tolist():
            bx = float(ball_xy[local_idx, 0])
            by = float(ball_xy[local_idx, 1])
            kind = _infer_restart_kind_from_ball_xy(x=bx, y=by)
            restart_side = _nearest_side_at_frame(
                features=clip_features,
                mask=clip_mask,
                frame_idx=int(local_idx),
            )
            stop_event_id[local_idx] = np.uint8(
                _map_restart_to_stop_id(
                    kind=kind,
                    restart_side=restart_side,
                    ego_side=ego_side,
                )
            )

        clip_path = out_split_dir / f"{provider}_{match_id}_clip{clip_idx:06d}.npz"
        np.savez_compressed(
            clip_path,
            features=clip_features,
            mask=clip_mask.astype(np.uint8),
            entity_type=entity_type.astype(np.int64),
            frames=clip_frames,
            periods=clip_periods,
            times=clip_times,
            entity_delta=entity_delta.astype(np.float32),
            entity_delta_mask=entity_delta_mask.astype(np.uint8),
            stop_event_id=stop_event_id.astype(np.uint8),
            kick_frame=int(clip_frames[history - 1]),
            kick_frame_local=int(history - 1),
        )

        entries.append(
            ClipEntry(
                clip_path=str(clip_path),
                length=int(clip_features.shape[0]),
                source_file=f"{provider}:{match_id}",
                clip_index=int(clip_idx),
                start_frame=int(clip_frames[0]),
                end_frame=int(clip_frames[-1]),
                kick_frame=int(clip_frames[history - 1]),
                real_player_count=real_player_count,
            )
        )
        clip_idx += 1

    return entries


def main() -> None:
    args = parse_args()

    if args.history_frames < 2:
        raise ValueError("--history-frames must be >= 2")
    if args.future_frames < 1:
        raise ValueError("--future-frames must be >= 1")

    if args.match_ids.strip():
        requested_match_ids = [m.strip() for m in args.match_ids.split(",") if m.strip()]
    else:
        requested_match_ids = _default_match_ids(args.provider)

    # Provider-specific split defaults.
    test_ratio = args.test_ratio
    if test_ratio is None:
        test_ratio = 0.2 if args.provider == "sportec" else 0.0

    out_dir = Path(args.out_dir)
    if args.rebuild and out_dir.exists():
        for old in out_dir.rglob("*.npz"):
            old.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)

    loaded: Dict[str, Dict[str, Any]] = {}
    failures: Dict[str, str] = {}
    pass_event_counts: Dict[str, int] = {}
    pass_event_failures: Dict[str, str] = {}

    for match_id in tqdm(requested_match_ids, desc=f"Loading {args.provider}", unit="match"):
        try:
            ds = _load_tracking_dataset(
                provider=args.provider,
                match_id=match_id,
                limit=args.max_frames_per_match,
            )
            tensor_data = _dataset_to_tensor(
                dataset_obj=ds,
                provider=args.provider,
                ball_mode=args.ball_mode,
                max_frames=args.max_frames_per_match,
            )
            if tensor_data is None:
                failures[match_id] = "insufficient frames"
                continue
            pass_events: List[Dict[str, Any]] = []
            try:
                pass_events = _load_provider_pass_events(provider=args.provider, match_id=str(match_id))
            except Exception as exc:  # pragma: no cover - provider event feed variability
                pass_event_failures[str(match_id)] = f"{type(exc).__name__}: {exc}"
                pass_events = []
            tensor_data["pass_events"] = pass_events
            pass_event_counts[str(match_id)] = int(len(pass_events))
            loaded[match_id] = tensor_data
        except Exception as exc:  # pragma: no cover - external IO/provider failures
            failures[match_id] = f"{type(exc).__name__}: {exc}"

    if not loaded:
        raise RuntimeError(
            "No matches loaded successfully. "
            f"Provider failures: {json.dumps(failures, indent=2)}"
        )

    train_ids, val_ids, test_ids = _split_match_ids(
        match_ids=list(loaded.keys()),
        val_ratio=float(args.val_ratio),
        test_ratio=float(test_ratio),
        seed=int(args.seed),
    )

    split_to_ids = {
        "train": train_ids,
        "val": val_ids,
        "test": test_ids,
    }

    split_entries: Dict[str, List[ClipEntry]] = {"train": [], "val": [], "test": []}
    for split, ids in split_to_ids.items():
        split_dir = out_dir / split
        for match_id in tqdm(ids, desc=f"Clipping {split}", unit="match"):
            split_entries[split].extend(
                _extract_clips(
                    match_data=loaded[match_id],
                    provider=args.provider,
                    match_id=match_id,
                    out_split_dir=split_dir,
                    history=int(args.history_frames),
                    future=int(args.future_frames),
                    anchor_contact_radius=float(args.anchor_contact_radius),
                    anchor_speed_min=float(args.anchor_speed_min),
                    anchor_dv_min=float(args.anchor_dv_min),
                    anchor_refractory_frames=int(args.anchor_refractory_frames),
                    anchor_stride_fallback=int(args.anchor_stride_fallback),
                    pass_anchor_tolerance_sec=float(args.pass_anchor_tolerance_sec),
                )
            )

    if not split_entries["train"]:
        raise RuntimeError("No training clips were produced from loaded public matches.")

    for split, entries in split_entries.items():
        _save_manifest(entries, out_dir / split / "manifest.jsonl")

    stats = compute_normalization_stats(split_entries["train"], use_tqdm=True, tqdm_desc="Stats")
    save_stats(stats, out_dir / "normalization_stats.json")

    meta = {
        "provider": args.provider,
        "out_dir": str(out_dir),
        "match_ids_requested": requested_match_ids,
        "match_ids_loaded": sorted(list(loaded.keys())),
        "match_ids_failed": failures,
        "split_match_ids": split_to_ids,
        "history_frames": int(args.history_frames),
        "future_frames": int(args.future_frames),
        "ball_mode": args.ball_mode,
        "coordinate_standard": {
            "target": "centered_meters_on_105x68",
            "metrica_conversion": "(x-0.5)*105, (0.5-y)*68",
            "sportec_conversion": "identity (already centered)",
        },
        "anchor_params": {
            "contact_radius": float(args.anchor_contact_radius),
            "speed_min": float(args.anchor_speed_min),
            "dv_min": float(args.anchor_dv_min),
            "refractory_frames": int(args.anchor_refractory_frames),
            "stride_fallback": int(args.anchor_stride_fallback),
            "pass_event_tolerance_sec": float(args.pass_anchor_tolerance_sec),
        },
        "anchor_priority": "pass_event_then_heuristic_then_stride_fallback",
        "pass_events": {
            "matches_with_pass_events": int(sum(1 for n in pass_event_counts.values() if int(n) > 0)),
            "matches_without_pass_events": int(sum(1 for n in pass_event_counts.values() if int(n) <= 0)),
            "pass_event_counts_by_match": pass_event_counts,
            "pass_event_failures_by_match": pass_event_failures,
        },
        "stop_event_classes": list(STOP_CLASS_NAMES),
        "stop_event_labeling": {
            "frame_gap_or_time_gap": "unknown_break",
            "ball_out_to_in_transition": "throw/corner/goal_kick/free_kick (team via nearest player, ego via anchor side)",
        },
        "num_clips_train": len(split_entries["train"]),
        "num_clips_val": len(split_entries["val"]),
        "num_clips_test": len(split_entries["test"]),
        "num_frames_train": int(sum(e.length for e in split_entries["train"])),
        "num_frames_val": int(sum(e.length for e in split_entries["val"])),
        "num_frames_test": int(sum(e.length for e in split_entries["test"])),
    }
    with (out_dir / "preprocessed_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(json.dumps(
        {
            "out_dir": str(out_dir),
            "provider": args.provider,
            "loaded_matches": len(loaded),
            "failed_matches": len(failures),
            "num_clips_train": len(split_entries["train"]),
            "num_clips_val": len(split_entries["val"]),
            "num_clips_test": len(split_entries["test"]),
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
