#!/usr/bin/env python3
"""Dataset utilities for kick-anchored soccer trajectory modeling (players + ball)."""

from __future__ import annotations

import json
from bisect import bisect_right
from collections import Counter, OrderedDict, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm

BALL_TEAM_SENTINEL = "bbbbbbbbbbbbbbbbbbbbbbbbb"
BALL_PLAYER_SENTINEL = "aaaaaaaaaaaaaaaaaaaaaaaaa"

STOP_CONTINUE = 0
STOP_OPP_THROW_IN = 1
STOP_EGO_THROW_IN = 2
STOP_OPP_CORNER = 3
STOP_EGO_CORNER = 4
STOP_OPP_GOAL_KICK = 5
STOP_EGO_GOAL_KICK = 6
STOP_OPP_FREE_KICK = 7
STOP_EGO_FREE_KICK = 8
STOP_UNKNOWN_BREAK = 9
STOP_PAD = 10

STOP_CLASS_NAMES: Tuple[str, ...] = (
    "continue",
    "opp_throw_in",
    "ego_throw_in",
    "opp_corner_kick",
    "ego_corner_kick",
    "opp_goal_kick",
    "ego_goal_kick",
    "opp_free_kick",
    "ego_free_kick",
    "unknown_break",
)
STOP_CLASS_COUNT = len(STOP_CLASS_NAMES)
STOP_TOKEN_VOCAB_SIZE = STOP_PAD + 1
BASE_FEATURE_DIM_XY = 4
BASE_FEATURE_DIM_XYZ = 6
BALL_3D_OBS_FLAG_DIM = 2
BASE_FEATURE_DIM_XYZ_WITH_FLAGS = BASE_FEATURE_DIM_XYZ + BALL_3D_OBS_FLAG_DIM

REQUIRED_COLUMNS: Tuple[str, ...] = (
    "current_phase",
    "timeelapsed",
    "frame_count",
    "team_id_opta",
    "player_id",
    "jersey_no",
    "pos_x",
    "pos_y",
    "speed_x",
    "speed_y",
)


@dataclass
class PreprocessConfig:
    max_players: int = 22
    min_players: int = 20
    min_segment_frames: int = 80
    max_roster_diff: int = 2
    max_player_step: float = 6.0
    max_ball_step: float = 18.0
    max_frames_per_file: Optional[int] = None

    history_frames: int = 24
    future_frames: int = 72

    # Kick-anchor heuristics (calibrated from raw tracking distributions).
    kick_contact_radius: float = 1.2
    kick_speed_min: float = 6.0
    kick_dv_min: float = 1.5
    kick_speed_ratio_min: float = 1.15
    kick_refractory_frames: int = 12

    # Pass-event anchor settings (preferred over heuristic kick detection).
    use_pass_event_anchors: bool = True
    pass_anchor_tolerance_sec: float = 0.6
    pass_anchor_refractory_frames: int = 4
    fallback_to_heuristic_when_no_pass: bool = False

    # If True, tolerate tracking frame gaps inside segments and represent them as stop events.
    allow_noncontiguous_windows: bool = True


@dataclass
class ClipEntry:
    clip_path: str
    length: int
    source_file: str
    clip_index: int
    start_frame: int
    end_frame: int
    kick_frame: int
    real_player_count: int


@dataclass
class NormalizationStats:
    mean: np.ndarray
    std: np.ndarray
    delta_scale: np.ndarray

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "delta_scale": self.delta_scale.tolist(),
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "NormalizationStats":
        return NormalizationStats(
            mean=np.asarray(data["mean"], dtype=np.float32),
            std=np.asarray(data["std"], dtype=np.float32),
            delta_scale=np.asarray(data["delta_scale"], dtype=np.float32),
        )


def _is_ball(player_id: str, team_id_opta: str) -> bool:
    if player_id == BALL_PLAYER_SENTINEL or team_id_opta == BALL_TEAM_SENTINEL:
        return True
    # Fallback for similarly encoded sentinels.
    return (
        bool(player_id)
        and bool(team_id_opta)
        and set(player_id) == {"a"}
        and set(team_id_opta) == {"b"}
    )


def discover_tracking_files(
    data_dir: str | Path,
    file_indices: Optional[Sequence[int]] = None,
) -> List[Path]:
    data_dir = Path(data_dir)
    discovered = sorted(data_dir.glob("*_tracking.parquet"), key=lambda p: int(p.name.split("_")[0]))
    if file_indices is None:
        return discovered

    requested = set(int(i) for i in file_indices)
    return [p for p in discovered if int(p.name.split("_")[0]) in requested]


def split_files(
    files: Sequence[Path],
    val_ratio: float,
    seed: int,
) -> Tuple[List[Path], List[Path]]:
    files = list(files)
    if not files:
        return [], []

    idx = np.arange(len(files))
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)

    val_count = max(1, int(round(len(files) * val_ratio))) if len(files) > 1 else 0
    val_idx = set(idx[:val_count].tolist())

    train_files = [f for i, f in enumerate(files) if i not in val_idx]
    val_files = [f for i, f in enumerate(files) if i in val_idx]

    if not train_files and val_files:
        train_files = val_files
        val_files = []

    return train_files, val_files


def _float_or(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _causal_impute(features: np.ndarray, observed_mask: np.ndarray) -> np.ndarray:
    """Forward-fill missing values using past only. Initial missing values stay zero."""
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


def _should_break_segment(
    frame_gap: int,
    phase_changed: bool,
    prev_players: Dict[str, np.ndarray],
    curr_players: Dict[str, np.ndarray],
    prev_ball: np.ndarray,
    curr_ball: np.ndarray,
    cfg: PreprocessConfig,
) -> bool:
    if frame_gap <= 0:
        return True
    if (frame_gap != 1) and (not cfg.allow_noncontiguous_windows):
        return True
    if phase_changed:
        return True

    prev_ids = set(prev_players)
    curr_ids = set(curr_players)
    if len(prev_ids.symmetric_difference(curr_ids)) >= cfg.max_roster_diff:
        return True

    common = prev_ids.intersection(curr_ids)
    if common:
        max_player_step = max(
            float(np.linalg.norm(curr_players[pid][:2] - prev_players[pid][:2])) for pid in common
        )
        step_per_frame = max_player_step / float(max(frame_gap, 1))
        if step_per_frame > cfg.max_player_step:
            return True

    if not np.isnan(prev_ball[0]) and not np.isnan(curr_ball[0]):
        ball_step = float(np.linalg.norm(curr_ball[:2] - prev_ball[:2]))
        ball_step_per_frame = ball_step / float(max(frame_gap, 1))
        if ball_step_per_frame > cfg.max_ball_step:
            return True

    return False


def _finalize_segment(
    frames: List[int],
    times: List[float],
    phases: List[int],
    players_seq: List[Dict[str, np.ndarray]],
    ball_seq: List[np.ndarray],
    pid_counts: Counter,
    team_votes: Dict[str, Counter],
    jersey_votes: Dict[str, Counter],
    cfg: PreprocessConfig,
) -> Optional[Dict[str, Any]]:
    min_frames_needed = max(cfg.min_segment_frames, cfg.history_frames + cfg.future_frames + 1)
    if len(frames) < min_frames_needed:
        return None

    selected = [pid for pid, _ in pid_counts.most_common(cfg.max_players)]
    if len(selected) < cfg.min_players:
        return None

    player_team: Dict[str, str] = {}
    for pid in selected:
        vote = team_votes.get(pid)
        player_team[pid] = vote.most_common(1)[0][0] if vote else "UNK_TEAM"

    team_counter = Counter(player_team.values())
    top_teams = [tid for tid, _ in team_counter.most_common(2)]
    if len(top_teams) == 0:
        top_teams = ["TEAM_A", "TEAM_B"]
    elif len(top_teams) == 1:
        top_teams.append("TEAM_B")
    team_to_type = {top_teams[0]: 0, top_teams[1]: 1}

    def jersey_key(pid: str) -> float:
        vote = jersey_votes.get(pid)
        if not vote:
            return 99.0
        jersey = vote.most_common(1)[0][0]
        return float(jersey) if np.isfinite(jersey) else 99.0

    selected.sort(key=lambda pid: (team_to_type.get(player_team.get(pid, ""), 1), jersey_key(pid), pid))
    real_player_count = len(selected)

    if len(selected) < cfg.max_players:
        for i in range(cfg.max_players - len(selected)):
            selected.append(f"__PAD_{i}__")

    n_entities = cfg.max_players + 1
    t_steps = len(frames)
    features = np.zeros((t_steps, n_entities, 4), dtype=np.float32)
    observed_mask = np.zeros((t_steps, n_entities), dtype=bool)

    for t, frame_players in enumerate(players_seq):
        for i, pid in enumerate(selected):
            if pid in frame_players:
                features[t, i] = frame_players[pid]
                observed_mask[t, i] = True

        ball_feat = ball_seq[t]
        if not np.isnan(ball_feat[0]):
            features[t, cfg.max_players] = ball_feat
            observed_mask[t, cfg.max_players] = True

    features = _causal_impute(features, observed_mask)

    entity_type = np.full((n_entities,), 3, dtype=np.int64)
    for i, pid in enumerate(selected):
        if not pid.startswith("__PAD_"):
            entity_type[i] = team_to_type.get(player_team.get(pid, ""), 1)
    entity_type[cfg.max_players] = 2  # ball

    return {
        "features": features,
        "mask": observed_mask,
        "entity_type": entity_type,
        "frames": np.asarray(frames, dtype=np.int64),
        "timeelapsed": np.asarray(times, dtype=np.float32),
        "phase": int(phases[0]) if phases else -1,
        "real_player_count": int(real_player_count),
    }


def _nearest_player_distance(
    features: np.ndarray,
    mask: np.ndarray,
    entity_type: np.ndarray,
) -> np.ndarray:
    ball_pos = features[:, 22, :2]
    player_pos = features[:, :22, :2]
    player_valid = mask[:, :22] & (entity_type[:22][None, :] != 3)
    out = np.full((features.shape[0],), np.nan, dtype=np.float32)

    for t in range(features.shape[0]):
        valid = player_valid[t]
        if not np.any(valid):
            continue
        dif = player_pos[t, valid] - ball_pos[t]
        out[t] = float(np.sqrt(np.square(dif).sum(axis=1)).min())
    return out


def _detect_kick_indices(
    features: np.ndarray,
    mask: np.ndarray,
    entity_type: np.ndarray,
    cfg: PreprocessConfig,
) -> List[int]:
    ball_valid = mask[:, 22]
    if ball_valid.sum() < 2:
        return []

    ball_vel = features[:, 22, 2:4]
    speed = np.sqrt(np.square(ball_vel).sum(axis=1))
    speed_prev = np.concatenate([speed[:1], speed[:-1]], axis=0)
    dv = speed - speed_prev
    nearest = _nearest_player_distance(features=features, mask=mask, entity_type=entity_type)

    valid = ball_valid & np.isfinite(nearest)
    candidate = np.where(
        valid
        & (nearest <= cfg.kick_contact_radius)
        & (speed >= cfg.kick_speed_min)
        & (dv >= cfg.kick_dv_min)
        & (speed >= np.maximum(1e-6, speed_prev * cfg.kick_speed_ratio_min))
    )[0]
    if candidate.size == 0:
        return []

    # Non-max suppression with refractory window; keep strongest acceleration evidence.
    score = dv + 0.05 * speed
    picked: List[int] = []
    last = -10**9
    for idx in candidate.tolist():
        if idx - last < cfg.kick_refractory_frames:
            if picked and score[idx] > score[picked[-1]]:
                picked[-1] = idx
                last = idx
            continue
        picked.append(idx)
        last = idx
    return picked


def _parse_event_time_rel_sec(event: Dict[str, Any], period: int) -> Optional[float]:
    try:
        t_min = int(event.get("timeMin", event.get("min", 0)))
        t_sec = int(event.get("timeSec", event.get("sec", 0)))
    except Exception:
        return None
    t_abs = float(t_min * 60 + t_sec)
    t_rel = t_abs if period == 1 else t_abs - 2700.0
    if t_rel < 0.0:
        return None
    return t_rel


def load_match_events(pass_json_path: str | Path) -> Dict[int, List[Dict[str, Any]]]:
    """Load Opta-style events by period with times aligned to tracking half clocks."""
    pass_json_path = Path(pass_json_path)
    if not pass_json_path.exists():
        return {}

    with pass_json_path.open("r", encoding="utf-8") as f:
        root = json.load(f)
    events = root.get("liveData", {}).get("event", [])
    if not isinstance(events, list):
        return {}

    by_period: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    seen = set()
    for e in events:
        if not isinstance(e, dict):
            continue

        period = int(e.get("periodId", -1))
        if period not in (1, 2):
            continue

        t_rel = _parse_event_time_rel_sec(e, period=period)
        if t_rel is None:
            continue

        type_id = int(e.get("typeId", -1))
        team = str(e.get("contestantId", "") or "")
        qualifier_ids: List[int] = []
        for q in e.get("qualifier", []) or []:
            if not isinstance(q, dict):
                continue
            qid = q.get("qualifierId", None)
            if qid is None:
                continue
            try:
                qualifier_ids.append(int(qid))
            except Exception:
                continue

        event_id = e.get("eventId", e.get("id", None))
        dedup_key = (period, event_id, type_id, team, int(round(float(t_rel) * 1000.0)))
        if dedup_key in seen:
            continue
        seen.add(dedup_key)

        by_period[period].append(
            {
                "time": float(t_rel),
                "team": team,
                "type_id": int(type_id),
                "qualifier_ids": qualifier_ids,
            }
        )

    out: Dict[int, List[Dict[str, Any]]] = {}
    for period, rows in by_period.items():
        rows = sorted(rows, key=lambda r: (float(r["time"]), int(r["type_id"])))
        out[int(period)] = rows
    return out


def load_pass_event_seconds(pass_json_path: str | Path) -> Dict[int, List[float]]:
    """Backward-compatible helper used by older scripts."""
    events = load_match_events(pass_json_path)
    out: Dict[int, List[float]] = {}
    for period, rows in events.items():
        vals = sorted(float(r["time"]) for r in rows if int(r.get("type_id", -1)) == 1)
        dedup_vals: List[float] = []
        prev = None
        for v in vals:
            if prev is None or abs(v - prev) > 1e-9:
                dedup_vals.append(v)
                prev = v
        out[int(period)] = dedup_vals
    return out


def _detect_pass_anchor_matches(
    frame_times: np.ndarray,
    phase: int,
    events_by_period: Dict[int, List[Dict[str, Any]]],
    cfg: PreprocessConfig,
) -> List[Dict[str, Any]]:
    if phase not in events_by_period:
        return []
    if frame_times.size == 0:
        return []

    pass_events = [e for e in events_by_period.get(phase, []) if int(e.get("type_id", -1)) == 1]
    if not pass_events:
        return []

    pass_times = np.asarray([float(e.get("time", np.nan)) for e in pass_events], dtype=np.float32)
    if pass_times.size == 0:
        return []

    valid_frame = np.isfinite(frame_times)
    if not valid_frame.any():
        return []

    # Use nearest frame-time with bounded tolerance.
    idx_valid = np.where(valid_frame)[0]
    t_valid = frame_times[idx_valid]
    picked: List[Tuple[int, float, Dict[str, Any]]] = []

    for event, t_event in zip(pass_events, pass_times.tolist()):
        right = int(np.searchsorted(t_valid, t_event, side="left"))
        cand = []
        if right < t_valid.size:
            cand.append(right)
        if right - 1 >= 0:
            cand.append(right - 1)
        if not cand:
            continue
        best_local = min(cand, key=lambda j: abs(float(t_valid[j]) - float(t_event)))
        best_idx = int(idx_valid[best_local])
        err = abs(float(frame_times[best_idx]) - float(t_event))
        if err <= float(cfg.pass_anchor_tolerance_sec):
            picked.append((best_idx, err, event))

    if not picked:
        return []

    # Keep best (lowest error) event per matched frame.
    by_idx: Dict[int, Tuple[float, Dict[str, Any]]] = {}
    for idx, err, event in picked:
        if idx not in by_idx or err < by_idx[idx][0]:
            by_idx[idx] = (float(err), event)
    matched = [(idx, by_idx[idx][0], by_idx[idx][1]) for idx in sorted(by_idx)]

    # Suppress duplicates from second-level event timestamps.
    refractory = max(int(cfg.pass_anchor_refractory_frames), 0)
    if refractory <= 0:
        return [
            {
                "idx": int(idx),
                "team": str(event.get("team", "") or ""),
                "time": float(event.get("time", np.nan)),
            }
            for idx, _err, event in matched
        ]

    filtered: List[Tuple[int, float, Dict[str, Any]]] = []
    for idx, err, event in matched:
        if not filtered:
            filtered.append((idx, err, event))
            continue
        if idx - filtered[-1][0] > refractory:
            filtered.append((idx, err, event))
        elif err < filtered[-1][1]:
            filtered[-1] = (idx, err, event)
    return [
        {
            "idx": int(idx),
            "team": str(event.get("team", "") or ""),
            "time": float(event.get("time", np.nan)),
        }
        for idx, _err, event in filtered
    ]


def _classify_restart_kind(event: Dict[str, Any]) -> Optional[str]:
    type_id = int(event.get("type_id", -1))
    qids = set(int(q) for q in event.get("qualifier_ids", []) if q is not None)

    # Restart-taken pass events in this feed are encoded as pass + set-piece qualifier.
    if type_id == 1:
        if 107 in qids:
            return "throw_in"
        if 6 in qids:
            return "corner"
        if 124 in qids:
            return "goal_kick"
        if 5 in qids:
            return "free_kick"

    # Fallback for a few feeds where corner can appear as non-pass event.
    if type_id == 6:
        return "corner"
    return None


def _map_restart_kind_to_stop_id(kind: str, restart_team: str, ego_team: str) -> int:
    if not kind:
        return STOP_UNKNOWN_BREAK
    if not restart_team or not ego_team:
        return STOP_UNKNOWN_BREAK
    is_ego = str(restart_team) == str(ego_team)
    if kind == "throw_in":
        return STOP_EGO_THROW_IN if is_ego else STOP_OPP_THROW_IN
    if kind == "corner":
        return STOP_EGO_CORNER if is_ego else STOP_OPP_CORNER
    if kind == "goal_kick":
        return STOP_EGO_GOAL_KICK if is_ego else STOP_OPP_GOAL_KICK
    if kind == "free_kick":
        return STOP_EGO_FREE_KICK if is_ego else STOP_OPP_FREE_KICK
    return STOP_UNKNOWN_BREAK


def _infer_ball_out_index(
    *,
    restart_idx: int,
    frame_counts: np.ndarray,
    frame_times: np.ndarray,
    ball_xy: np.ndarray,
    ball_mask: np.ndarray,
) -> int:
    """Heuristically move a mapped restart frame back to ball-out timing.

    Priority:
    1) Last discontinuity before restart (frame/time gap).
    2) Latest transition into out-of-bounds before restart.
    3) Fallback to one frame before restart.
    """
    n = int(frame_times.shape[0])
    if n <= 1:
        return int(np.clip(restart_idx, 0, max(n - 1, 0)))
    restart_idx = int(np.clip(restart_idx, 0, n - 1))
    if restart_idx <= 0:
        return 0

    valid_t = np.isfinite(frame_times)
    dt = np.diff(frame_times[valid_t])
    dt = dt[np.isfinite(dt) & (dt > 0)]
    nominal_dt = float(np.median(dt)) if dt.size > 0 else 0.04
    lookback_steps = int(max(8, min(400, round(6.0 / max(nominal_dt, 1e-3)))))
    start = max(0, restart_idx - lookback_steps)

    frame_gap = np.diff(frame_counts.astype(np.int64))
    time_gap = np.diff(frame_times.astype(np.float32))
    bad_gap = (frame_gap > 1) | (~np.isfinite(time_gap)) | (time_gap > (0.08 * np.maximum(frame_gap, 1)))
    if restart_idx > start:
        bad_local = np.where(bad_gap[start:restart_idx])[0]
        if bad_local.size > 0:
            # Gap at i means transition from i -> i+1 is broken; mark i as last in-play frame.
            return int(start + bad_local[-1])

    if ball_xy.ndim == 2 and ball_xy.shape[0] == n:
        finite_ball = (
            np.asarray(ball_mask, dtype=bool).reshape(-1)[:n]
            & np.isfinite(ball_xy[:, 0])
            & np.isfinite(ball_xy[:, 1])
        )
        oob = finite_ball & ((np.abs(ball_xy[:, 0]) > 52.5) | (np.abs(ball_xy[:, 1]) > 34.0))
        for idx in range(restart_idx, start, -1):
            if idx < oob.shape[0] and oob[idx] and (not oob[idx - 1]):
                return int(idx)
        for idx in range(restart_idx, start - 1, -1):
            if idx < oob.shape[0] and oob[idx]:
                run_start = idx
                while run_start > start and oob[run_start - 1]:
                    run_start -= 1
                return int(run_start)

    return int(max(0, restart_idx - 1))


def _map_restart_events_to_frame_indices(
    frame_times: np.ndarray,
    frame_counts: np.ndarray,
    ball_xy: np.ndarray,
    ball_mask: np.ndarray,
    phase: int,
    events_by_period: Optional[Dict[int, List[Dict[str, Any]]]],
    cfg: PreprocessConfig,
) -> Dict[int, List[Dict[str, Any]]]:
    if not events_by_period or phase not in events_by_period or frame_times.size == 0:
        return {}
    valid_frame = np.isfinite(frame_times)
    if not valid_frame.any():
        return {}

    idx_valid = np.where(valid_frame)[0]
    t_valid = frame_times[idx_valid]
    mapped: Dict[int, List[Dict[str, Any]]] = defaultdict(list)

    for event in events_by_period.get(phase, []):
        kind = _classify_restart_kind(event)
        if kind is None:
            continue
        t_event = float(event.get("time", np.nan))
        if not np.isfinite(t_event):
            continue

        right = int(np.searchsorted(t_valid, t_event, side="left"))
        cand: List[int] = []
        if right < t_valid.size:
            cand.append(right)
        if right - 1 >= 0:
            cand.append(right - 1)
        if not cand:
            continue
        best_local = min(cand, key=lambda j: abs(float(t_valid[j]) - t_event))
        best_idx = int(idx_valid[best_local])  # restart frame
        err = abs(float(frame_times[best_idx]) - t_event)
        if err > float(cfg.pass_anchor_tolerance_sec):
            continue

        out_idx = _infer_ball_out_index(
            restart_idx=best_idx,
            frame_counts=frame_counts,
            frame_times=frame_times,
            ball_xy=ball_xy,
            ball_mask=ball_mask,
        )
        mapped[int(out_idx)].append(
            {
                "kind": kind,
                "team": str(event.get("team", "") or ""),
                "time": float(t_event),
                "restart_idx": int(best_idx),
                "out_idx": int(out_idx),
            }
        )
    return dict(mapped)


def tokenize_stop_sequence(stop_ids: np.ndarray) -> np.ndarray:
    """Tokenize stop ids into causal train targets with PAD-after-stop behavior.

    Output vocabulary:
      0..9  : raw stop ids (continue/restart/unknown)
      10    : PAD (all frames strictly after the first stop frame)
    """
    arr = np.asarray(stop_ids, dtype=np.int64).reshape(-1)
    if arr.size == 0:
        return arr.astype(np.int64)

    out = np.full_like(arr, STOP_CONTINUE, dtype=np.int64)
    non_continue = np.where(arr != STOP_CONTINUE)[0]
    if non_continue.size == 0:
        return out

    first = int(non_continue[0])
    stop_type = int(arr[first])
    if stop_type < 0 or stop_type >= STOP_CLASS_COUNT:
        stop_type = STOP_UNKNOWN_BREAK
    out[first] = stop_type
    if first + 1 < out.shape[0]:
        out[first + 1 :] = STOP_PAD
    return out


def _build_clip_stop_event_ids(
    clip_frames: np.ndarray,
    clip_times: np.ndarray,
    start_idx: int,
    end_idx: int,
    restart_events_by_index: Dict[int, List[Dict[str, Any]]],
    ego_team: str,
) -> np.ndarray:
    n = int(clip_frames.shape[0])
    stop_ids = np.full((n,), STOP_CONTINUE, dtype=np.uint8)
    if n <= 1:
        return stop_ids

    frame_gap = np.diff(clip_frames.astype(np.int64))
    time_gap = np.diff(clip_times.astype(np.float32))
    bad_gap = (frame_gap > 1) | (~np.isfinite(time_gap)) | (time_gap > (0.08 * np.maximum(frame_gap, 1)))
    stop_ids[1:][bad_gap] = STOP_UNKNOWN_BREAK

    for global_idx in range(int(start_idx), int(end_idx) + 1):
        if global_idx not in restart_events_by_index:
            continue
        local_idx = int(global_idx - start_idx)
        if local_idx < 0 or local_idx >= n:
            continue
        chosen = stop_ids[local_idx]
        for ev in restart_events_by_index[global_idx]:
            mapped = _map_restart_kind_to_stop_id(
                kind=str(ev.get("kind", "")),
                restart_team=str(ev.get("team", "") or ""),
                ego_team=str(ego_team or ""),
            )
            # Prefer concrete restart labels over generic unknown break.
            if mapped != STOP_UNKNOWN_BREAK:
                chosen = int(mapped)
                break
            if chosen == STOP_CONTINUE:
                chosen = int(mapped)
        stop_ids[local_idx] = np.uint8(chosen)
    return stop_ids


def process_parquet_to_clips(
    parquet_path: str | Path,
    out_dir: str | Path,
    cfg: PreprocessConfig,
    match_events_by_period: Optional[Dict[int, List[Dict[str, Any]]]] = None,
    overwrite_existing: bool = False,
) -> List[ClipEntry]:
    parquet_path = Path(parquet_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(parquet_path, columns=list(REQUIRED_COLUMNS))
    if df.empty:
        return []

    df = df.dropna(subset=["frame_count", "player_id", "team_id_opta", "pos_x", "pos_y"]).copy()
    if df.empty:
        return []

    df["frame_count"] = df["frame_count"].astype(np.int64)
    df.sort_values("frame_count", inplace=True)

    if cfg.max_frames_per_file is not None:
        unique_frames = df["frame_count"].drop_duplicates().to_numpy()
        if len(unique_frames) > cfg.max_frames_per_file:
            cutoff = int(unique_frames[cfg.max_frames_per_file - 1])
            df = df[df["frame_count"] <= cutoff]

    clip_entries: List[ClipEntry] = []
    clip_idx = 0

    frames: List[int] = []
    times: List[float] = []
    phases: List[int] = []
    players_seq: List[Dict[str, np.ndarray]] = []
    ball_seq: List[np.ndarray] = []
    pid_counts: Counter = Counter()
    team_votes: Dict[str, Counter] = defaultdict(Counter)
    jersey_votes: Dict[str, Counter] = defaultdict(Counter)

    prev_frame: Optional[int] = None
    prev_phase: Optional[int] = None
    prev_players: Dict[str, np.ndarray] = {}
    prev_ball = np.full((4,), np.nan, dtype=np.float32)

    def flush_segment() -> None:
        nonlocal frames, times, phases, players_seq, ball_seq, pid_counts, team_votes, jersey_votes, clip_idx

        seg = _finalize_segment(
            frames=frames,
            times=times,
            phases=phases,
            players_seq=players_seq,
            ball_seq=ball_seq,
            pid_counts=pid_counts,
            team_votes=team_votes,
            jersey_votes=jersey_votes,
            cfg=cfg,
        )

        frames = []
        times = []
        phases = []
        players_seq = []
        ball_seq = []
        pid_counts = Counter()
        team_votes = defaultdict(Counter)
        jersey_votes = defaultdict(Counter)

        if seg is None:
            return

        anchor_matches: List[Dict[str, Any]] = []
        if cfg.use_pass_event_anchors and match_events_by_period is not None:
            anchor_matches = _detect_pass_anchor_matches(
                frame_times=seg["timeelapsed"],
                phase=int(seg["phase"]),
                events_by_period=match_events_by_period,
                cfg=cfg,
            )
            if not anchor_matches and cfg.fallback_to_heuristic_when_no_pass:
                kick_idx_list = _detect_kick_indices(
                    features=seg["features"],
                    mask=seg["mask"],
                    entity_type=seg["entity_type"],
                    cfg=cfg,
                )
                anchor_matches = [{"idx": int(i), "team": "", "time": np.nan} for i in kick_idx_list]
        else:
            kick_idx_list = _detect_kick_indices(
                features=seg["features"],
                mask=seg["mask"],
                entity_type=seg["entity_type"],
                cfg=cfg,
            )
            anchor_matches = [{"idx": int(i), "team": "", "time": np.nan} for i in kick_idx_list]

        if not anchor_matches:
            return

        h = int(cfg.history_frames)
        r = int(cfg.future_frames)
        restart_events_by_index = _map_restart_events_to_frame_indices(
            frame_times=seg["timeelapsed"],
            frame_counts=seg["frames"],
            ball_xy=seg["features"][:, 22, :2],
            ball_mask=seg["mask"][:, 22],
            phase=int(seg["phase"]),
            events_by_period=match_events_by_period,
            cfg=cfg,
        )

        for anchor in anchor_matches:
            kick_idx = int(anchor["idx"])
            ego_team = str(anchor.get("team", "") or "")
            start = int(kick_idx - h + 1)
            end = int(kick_idx + r)
            if start < 0 or end >= int(seg["features"].shape[0]):
                continue

            clip_features = seg["features"][start : end + 1].astype(np.float32)
            clip_mask = seg["mask"][start : end + 1].astype(bool)
            clip_frames = seg["frames"][start : end + 1].astype(np.int64)

            if clip_features.shape[0] <= 1:
                continue
            entity_delta = clip_features[1:, :, :2] - clip_features[:-1, :, :2]
            entity_delta_mask = clip_mask[1:, :] & clip_mask[:-1, :]
            stop_event_id = _build_clip_stop_event_ids(
                clip_frames=clip_frames,
                clip_times=seg["timeelapsed"][start : end + 1].astype(np.float32),
                start_idx=start,
                end_idx=end,
                restart_events_by_index=restart_events_by_index,
                ego_team=ego_team,
            )

            clip_path = out_dir / f"{parquet_path.stem}_kickclip{clip_idx:06d}.npz"
            if overwrite_existing or not clip_path.exists():
                np.savez_compressed(
                    clip_path,
                    features=clip_features,
                    mask=clip_mask.astype(np.uint8),
                    entity_type=seg["entity_type"].astype(np.int64),
                    frames=clip_frames,
                    entity_delta=entity_delta.astype(np.float32),
                    entity_delta_mask=entity_delta_mask.astype(np.uint8),
                    stop_event_id=stop_event_id.astype(np.uint8),
                    kick_frame=int(seg["frames"][kick_idx]),
                    kick_frame_local=int(h - 1),
                )

            clip_entries.append(
                ClipEntry(
                    clip_path=str(clip_path),
                    length=int(clip_features.shape[0]),
                    source_file=parquet_path.name,
                    clip_index=int(clip_idx),
                    start_frame=int(clip_frames[0]),
                    end_frame=int(clip_frames[-1]),
                    kick_frame=int(seg["frames"][kick_idx]),
                    real_player_count=int(seg["real_player_count"]),
                )
            )
            clip_idx += 1

    grouped = df.groupby("frame_count", sort=True, observed=True)
    for frame, frame_df in grouped:
        phase_values = frame_df["current_phase"].dropna()
        phase = int(phase_values.iloc[0]) if not phase_values.empty else -1
        time_values = frame_df["timeelapsed"].dropna()
        timeelapsed = float(time_values.iloc[0]) if not time_values.empty else np.nan

        frame_players: Dict[str, np.ndarray] = {}
        frame_player_team: Dict[str, str] = {}
        frame_player_jersey: Dict[str, float] = {}
        ball = np.full((4,), np.nan, dtype=np.float32)

        for row in frame_df.itertuples(index=False):
            player_id = str(row.player_id)
            team_id_opta = str(row.team_id_opta)

            feat = np.array(
                [
                    _float_or(row.pos_x, 0.0),
                    _float_or(row.pos_y, 0.0),
                    _float_or(row.speed_x, 0.0),
                    _float_or(row.speed_y, 0.0),
                ],
                dtype=np.float32,
            )

            if _is_ball(player_id, team_id_opta):
                ball = feat
                continue

            if player_id in frame_players:
                continue

            frame_players[player_id] = feat
            frame_player_team[player_id] = team_id_opta
            frame_player_jersey[player_id] = _float_or(row.jersey_no, np.nan)

        if prev_frame is not None:
            should_break = _should_break_segment(
                frame_gap=int(frame - prev_frame),
                phase_changed=bool(phase != prev_phase),
                prev_players=prev_players,
                curr_players=frame_players,
                prev_ball=prev_ball,
                curr_ball=ball,
                cfg=cfg,
            )
            if should_break:
                flush_segment()

        frames.append(int(frame))
        times.append(float(timeelapsed))
        phases.append(int(phase))
        players_seq.append(frame_players)
        ball_seq.append(ball)

        for pid in frame_players:
            pid_counts[pid] += 1
            team_votes[pid][frame_player_team[pid]] += 1
            jersey = frame_player_jersey[pid]
            if np.isfinite(jersey):
                jersey_votes[pid][float(jersey)] += 1

        prev_frame = int(frame)
        prev_phase = phase
        prev_players = frame_players
        prev_ball = ball

    flush_segment()
    return clip_entries


def _load_manifest(manifest_path: Path) -> List[ClipEntry]:
    if not manifest_path.exists():
        return []
    entries: List[ClipEntry] = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entries.append(ClipEntry(**json.loads(line)))
    return entries


def _save_manifest(entries: Sequence[ClipEntry], manifest_path: Path) -> None:
    with manifest_path.open("w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(asdict(entry)) + "\n")


def build_cache_for_files(
    files: Sequence[Path],
    cache_dir: str | Path,
    cfg: PreprocessConfig,
    match_events_by_file: Optional[Dict[str, Dict[int, List[Dict[str, Any]]]]] = None,
    rebuild: bool = False,
    use_tqdm: bool = True,
    tqdm_desc: str = "Preprocessing files",
) -> List[ClipEntry]:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = cache_dir / "manifest.jsonl"
    config_path = cache_dir / "preprocess_config.json"
    files_path = cache_dir / "source_files.json"

    if not rebuild and manifest_path.exists() and config_path.exists() and files_path.exists():
        with config_path.open("r", encoding="utf-8") as f:
            cached_cfg = json.load(f)
        with files_path.open("r", encoding="utf-8") as f:
            cached_files = json.load(f)

        requested_files = [str(p) for p in files]
        if cached_cfg == asdict(cfg) and cached_files == requested_files:
            return _load_manifest(manifest_path)

    for old_npz in cache_dir.glob("*.npz"):
        old_npz.unlink()

    all_entries: List[ClipEntry] = []
    iterator = tqdm(files, desc=tqdm_desc, unit="file", disable=not use_tqdm)
    for parquet_path in iterator:
        file_key = parquet_path.name
        stem_key = parquet_path.stem.split("_")[0]
        match_events = None
        if match_events_by_file is not None:
            match_events = (
                match_events_by_file.get(file_key)
                or match_events_by_file.get(stem_key)
            )
        all_entries.extend(
            process_parquet_to_clips(
                parquet_path=parquet_path,
                out_dir=cache_dir,
                cfg=cfg,
                match_events_by_period=match_events,
                overwrite_existing=True,
            )
        )

    _save_manifest(all_entries, manifest_path)
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)
    with files_path.open("w", encoding="utf-8") as f:
        json.dump([str(p) for p in files], f, indent=2)

    return all_entries


def _append_stop_features(
    features: np.ndarray,
    stop_event_id: Optional[np.ndarray],
    n_entities: int = 23,
) -> np.ndarray:
    """Append stop-event one-hot channels as global context on the ball token."""
    if stop_event_id is None:
        return features
    if features.ndim != 3:
        return features
    t_steps = int(features.shape[0])
    if t_steps == 0:
        return features

    stop_ids = np.asarray(stop_event_id, dtype=np.int64).reshape(-1)
    if stop_ids.shape[0] != t_steps:
        return features
    stop_ids = np.clip(stop_ids, 0, STOP_CLASS_COUNT - 1)
    onehot = np.eye(STOP_CLASS_COUNT, dtype=np.float32)[stop_ids]  # [T, C]

    extra = np.zeros((t_steps, n_entities, STOP_CLASS_COUNT), dtype=np.float32)
    ball_idx = min(max(n_entities - 1, 0), 22)
    extra[:, ball_idx, :] = onehot
    return np.concatenate([features.astype(np.float32), extra], axis=-1)


def _canonicalize_features_for_model(
    raw_features: np.ndarray,
    stop_event_id: Optional[np.ndarray],
    target_feat_dim: int,
    mask: Optional[np.ndarray] = None,
    n_entities: int = 23,
) -> np.ndarray:
    """Map mixed-source raw feature layouts into a fixed model layout.

    Canonical layouts:
      - 14D: [x, y, vx, vy, stop_onehot(10)]
      - 16D: [x, y, vx, vy, z, vz, stop_onehot(10)]
    """
    if raw_features.ndim != 3:
        return raw_features.astype(np.float32, copy=False)

    t_steps, n_curr, raw_dim = raw_features.shape
    n_out = int(n_entities if n_entities > 0 else n_curr)
    out = np.zeros((t_steps, n_out, int(target_feat_dim)), dtype=np.float32)

    # Always place planar kinematics first.
    copy_xyv = min(4, raw_dim, out.shape[-1])
    if copy_xyv > 0:
        out[:, :n_curr, :copy_xyv] = raw_features[:, :, :copy_xyv]

    has_xyz_layout = int(target_feat_dim) >= (BASE_FEATURE_DIM_XYZ + STOP_CLASS_COUNT)
    has_xyz_obs_flags = int(target_feat_dim) >= (BASE_FEATURE_DIM_XYZ_WITH_FLAGS + STOP_CLASS_COUNT)
    if has_xyz_layout:
        if raw_dim >= 5 and out.shape[-1] > 4:
            out[:, :n_curr, 4] = raw_features[:, :, 4]
        if raw_dim >= 6 and out.shape[-1] > 5:
            out[:, :n_curr, 5] = raw_features[:, :, 5]
        if has_xyz_obs_flags:
            ball_idx = min(max(n_out - 1, 0), 22)
            if mask is not None and mask.shape[:2] == raw_features.shape[:2]:
                ball_obs = mask[:, ball_idx].astype(bool)
            else:
                ball_obs = np.ones((t_steps,), dtype=bool)
            if raw_dim >= 5 and out.shape[-1] > 6:
                z_obs = ball_obs & np.isfinite(raw_features[:, ball_idx, 4])
                out[:, ball_idx, 6] = z_obs.astype(np.float32)
            if raw_dim >= 6 and out.shape[-1] > 7:
                vz_obs = ball_obs & np.isfinite(raw_features[:, ball_idx, 5])
                out[:, ball_idx, 7] = vz_obs.astype(np.float32)
            stop_base = BASE_FEATURE_DIM_XYZ_WITH_FLAGS
        else:
            stop_base = BASE_FEATURE_DIM_XYZ
    else:
        stop_base = BASE_FEATURE_DIM_XY

    if stop_base + STOP_CLASS_COUNT <= out.shape[-1] and stop_event_id is not None:
        stop_ids = np.asarray(stop_event_id, dtype=np.int64).reshape(-1)
        if stop_ids.shape[0] == t_steps:
            stop_ids = np.clip(stop_ids, 0, STOP_CLASS_COUNT - 1)
            onehot = np.eye(STOP_CLASS_COUNT, dtype=np.float32)[stop_ids]  # [T, C]
            ball_idx = min(max(n_out - 1, 0), 22)
            out[:, ball_idx, stop_base : stop_base + STOP_CLASS_COUNT] = onehot

    return out


def _infer_expected_feature_dim_from_entries(
    entries: Sequence[ClipEntry],
    sample_limit: int = 256,
) -> int:
    if not entries:
        return BASE_FEATURE_DIM_XY + STOP_CLASS_COUNT
    n = len(entries)
    take = min(max(int(sample_limit), 1), n)
    if take == n:
        sample_idx = list(range(n))
    else:
        sample_idx = np.linspace(0, n - 1, num=take, dtype=np.int64).tolist()
    max_raw_dim = 0
    for i in sample_idx:
        try:
            with np.load(entries[int(i)].clip_path, allow_pickle=False) as data:
                max_raw_dim = max(max_raw_dim, int(data["features"].shape[-1]))
        except Exception:
            continue
    if max_raw_dim >= 5:
        return BASE_FEATURE_DIM_XYZ_WITH_FLAGS + STOP_CLASS_COUNT
    return BASE_FEATURE_DIM_XY + STOP_CLASS_COUNT


def _maybe_convert_xy_normalized_to_centered_meters(
    features: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Convert normalized [0,1]-like XY(+VXY) features to centered meter coordinates.

    Some public Sportec exports arrive normalized despite metadata claiming centered meters.
    We auto-detect by observed XY range and convert in-memory at load-time.
    """
    if features.ndim != 3 or features.shape[-1] < 2:
        return features
    if mask.shape[:2] != features.shape[:2]:
        return features

    obs = mask.astype(bool)
    if not obs.any():
        return features

    x_obs = features[..., 0][obs]
    y_obs = features[..., 1][obs]
    if x_obs.size < 128 or y_obs.size < 128:
        return features

    x_min = float(np.nanmin(x_obs))
    x_max = float(np.nanmax(x_obs))
    y_min = float(np.nanmin(y_obs))
    y_max = float(np.nanmax(y_obs))

    looks_normalized = (
        x_min >= -0.25
        and x_max <= 1.25
        and y_min >= -0.25
        and y_max <= 1.25
        and max(abs(x_min), abs(x_max), abs(y_min), abs(y_max)) < 2.5
    )
    if not looks_normalized:
        return features

    out = features.copy()
    out[..., 0] = (out[..., 0] - 0.5) * 105.0
    out[..., 1] = (0.5 - out[..., 1]) * 68.0
    if out.shape[-1] >= 3:
        out[..., 2] = out[..., 2] * 105.0
    if out.shape[-1] >= 4:
        out[..., 3] = -out[..., 3] * 68.0
    return out


def compute_normalization_stats(
    entries: Sequence[ClipEntry],
    use_tqdm: bool = False,
    tqdm_desc: str = "Computing normalization",
) -> NormalizationStats:
    if not entries:
        raise RuntimeError("Cannot compute normalization stats from empty entries.")

    # Determine whether any source includes ball z/vz so feature layout is stable.
    max_raw_dim = 0
    pre_iter = tqdm(entries, desc=f"{tqdm_desc} (shape scan)", unit="clip", disable=not use_tqdm)
    for entry in pre_iter:
        with np.load(entry.clip_path, allow_pickle=False) as data:
            max_raw_dim = max(max_raw_dim, int(data["features"].shape[-1]))
    feat_dim = (
        BASE_FEATURE_DIM_XYZ_WITH_FLAGS + STOP_CLASS_COUNT
        if max_raw_dim >= 5
        else BASE_FEATURE_DIM_XY + STOP_CLASS_COUNT
    )

    sum_x = np.zeros((feat_dim,), dtype=np.float64)
    sumsq_x = np.zeros((feat_dim,), dtype=np.float64)
    count_x = np.zeros((feat_dim,), dtype=np.float64)

    # Delta normalization is always 3D target space: dx, dy, dz.
    sum_d = np.zeros((3,), dtype=np.float64)
    sumsq_d = np.zeros((3,), dtype=np.float64)
    count_d = np.zeros((3,), dtype=np.float64)

    iterator = tqdm(entries, desc=tqdm_desc, unit="clip", disable=not use_tqdm)
    for entry in iterator:
        with np.load(entry.clip_path, allow_pickle=False) as data:
            raw_features = data["features"].astype(np.float32)
            mask = data["mask"].astype(bool)
            entity_type = data["entity_type"].astype(np.int64)
            stop_event_id = data["stop_event_id"].astype(np.int64) if "stop_event_id" in data.files else None
        raw_features = _maybe_convert_xy_normalized_to_centered_meters(
            features=raw_features,
            mask=mask,
        )

        features = _canonicalize_features_for_model(
            raw_features=raw_features,
            stop_event_id=stop_event_id,
            target_feat_dim=feat_dim,
            mask=mask,
            n_entities=int(raw_features.shape[1]),
        )

        obs = mask[..., None]
        sum_x += (features * obs).sum(axis=(0, 1))
        sumsq_x += ((features ** 2) * obs).sum(axis=(0, 1))
        count_x += obs.sum(axis=(0, 1))

        if features.shape[0] <= 1:
            continue
        delta = features[1:, :, :2] - features[:-1, :, :2]  # [T-1, 23, 2]
        delta_valid = (mask[1:, :] & mask[:-1, :]) & (entity_type[None, :] != 3)
        if not delta_valid.any():
            continue

        for dim in range(2):
            vals = delta[..., dim][delta_valid]
            if vals.size == 0:
                continue
            sum_d[dim] += vals.sum(dtype=np.float64)
            sumsq_d[dim] += np.square(vals, dtype=np.float64).sum(dtype=np.float64)
            count_d[dim] += vals.size

        # dz: only meaningful when ball z exists in features (sportec xyz mode).
        if feat_dim >= (BASE_FEATURE_DIM_XYZ_WITH_FLAGS + STOP_CLASS_COUNT) and raw_features.shape[-1] >= 5:
            ball_valid = mask[1:, 22] & mask[:-1, 22]
            if ball_valid.any():
                dz = features[1:, 22, 4] - features[:-1, 22, 4]
                vals = dz[ball_valid]
                if vals.size > 0:
                    sum_d[2] += vals.sum(dtype=np.float64)
                    sumsq_d[2] += np.square(vals, dtype=np.float64).sum(dtype=np.float64)
                    count_d[2] += vals.size

    count_x = np.maximum(count_x, 1.0)
    mean = sum_x / count_x
    var = np.maximum(sumsq_x / count_x - np.square(mean), 1e-6)
    std = np.sqrt(var)

    # Keep discrete stop channels as raw 0/1 inputs (no z-score amplification).
    stop_base = (
        BASE_FEATURE_DIM_XYZ_WITH_FLAGS
        if feat_dim >= (BASE_FEATURE_DIM_XYZ_WITH_FLAGS + STOP_CLASS_COUNT)
        else BASE_FEATURE_DIM_XYZ if feat_dim >= (BASE_FEATURE_DIM_XYZ + STOP_CLASS_COUNT)
        else BASE_FEATURE_DIM_XY
    )
    stop_end = min(stop_base + STOP_CLASS_COUNT, feat_dim)
    if stop_end > stop_base:
        mean[stop_base:stop_end] = 0.0
        std[stop_base:stop_end] = 1.0
    # Keep observation flags as raw binary semantics when present.
    if feat_dim >= (BASE_FEATURE_DIM_XYZ_WITH_FLAGS + STOP_CLASS_COUNT):
        mean[6:8] = 0.0
        std[6:8] = 1.0

    count_d = np.maximum(count_d, 1.0)
    mean_d = sum_d / count_d
    var_d = np.maximum(sumsq_d / count_d - np.square(mean_d), 1e-6)
    delta_scale = np.sqrt(var_d)
    delta_scale = np.maximum(delta_scale, 1e-3)

    return NormalizationStats(
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
        delta_scale=delta_scale.astype(np.float32),
    )


def save_stats(stats: NormalizationStats, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(stats.to_dict(), f, indent=2)


def load_stats(path: str | Path) -> NormalizationStats:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    stats = NormalizationStats.from_dict(data)
    # Backward compatibility: older artifacts may have only dx,dy scale.
    if stats.delta_scale.shape[0] == 2:
        stats = NormalizationStats(
            mean=stats.mean,
            std=stats.std,
            delta_scale=np.asarray([stats.delta_scale[0], stats.delta_scale[1], 1.0], dtype=np.float32),
        )
    return stats


def _stats_is_valid(stats: Optional[NormalizationStats]) -> bool:
    if stats is None:
        return False
    mean = np.asarray(stats.mean, dtype=np.float32).reshape(-1)
    std = np.asarray(stats.std, dtype=np.float32).reshape(-1)
    delta = np.asarray(stats.delta_scale, dtype=np.float32).reshape(-1)
    if mean.size == 0 or std.size == 0 or delta.size == 0:
        return False
    if not np.isfinite(mean).all():
        return False
    if not np.isfinite(std).all():
        return False
    if not np.isfinite(delta).all():
        return False
    if (std <= 0.0).any():
        return False
    if (delta <= 0.0).any():
        return False
    return True


class TrajectoryWindowDataset(Dataset):
    """Sliding-window dataset over preprocessed kick-anchored clips."""

    def __init__(
        self,
        entries: Sequence[ClipEntry],
        history: int,
        mode: str,
        stats: NormalizationStats,
        rollout_steps: int = 1,
        clip_cache_size: int = 4,
        preload_clips: bool = False,
        preload_progress: bool = False,
    ) -> None:
        if mode not in {"multi", "single"}:
            raise ValueError("mode must be 'multi' or 'single'.")

        self.entries = list(entries)
        self.history = int(history)
        self.rollout_steps = int(rollout_steps)
        self.mode = mode
        self.stats = stats
        self.clip_cache_size = max(int(clip_cache_size), 1)
        self.preload_clips = bool(preload_clips)
        self.preload_progress = bool(preload_progress)

        self.mean = stats.mean.astype(np.float32)
        self.std = np.maximum(stats.std.astype(np.float32), 1e-6)
        self.delta_scale = np.maximum(stats.delta_scale.astype(np.float32), 1e-6)
        self.delta_dim = int(self.delta_scale.shape[0])

        self.n_entities = 23
        self._clip_cache: OrderedDict[int, Dict[str, np.ndarray]] = OrderedDict()
        self._preloaded_clips: Optional[List[Dict[str, np.ndarray]]] = None

        self._windows_per_clip = [
            max(0, int(entry.length) - self.history - self.rollout_steps + 1)
            for entry in self.entries
        ]
        self._cum_windows = np.cumsum(self._windows_per_clip, dtype=np.int64)
        self.num_windows = int(np.sum(self._windows_per_clip, dtype=np.int64))
        self._total = self.num_windows if self.mode == "multi" else self.num_windows * self.n_entities

        if self.preload_clips:
            self._preloaded_clips = []
            iterator: Iterable[int] = range(len(self.entries))
            if self.preload_progress:
                iterator = tqdm(iterator, desc="Preloading clips", unit="clip")
            for clip_idx in iterator:
                self._preloaded_clips.append(self._load_clip_from_disk(clip_idx))

    def __len__(self) -> int:
        return self._total

    def _resolve_window(self, window_idx: int) -> Tuple[int, int]:
        if window_idx < 0 or window_idx >= self.num_windows:
            raise IndexError(f"window_idx {window_idx} out of range (num_windows={self.num_windows})")

        clip_idx = bisect_right(self._cum_windows, window_idx)
        prev_cum = 0 if clip_idx == 0 else int(self._cum_windows[clip_idx - 1])
        local_start = int(window_idx - prev_cum)
        return clip_idx, local_start

    def _prepare_clip_arrays(
        self,
        features: np.ndarray,
        mask: np.ndarray,
        entity_type: np.ndarray,
        frames: np.ndarray,
        stop_event_id: Optional[np.ndarray] = None,
        raw_feat_dim: Optional[int] = None,
        delta_raw: Optional[np.ndarray] = None,
        delta_mask_raw: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        mask = mask.astype(bool, copy=True)
        if raw_feat_dim is None:
            raw_feat_dim = int(features.shape[-1])
        features = _maybe_convert_xy_normalized_to_centered_meters(
            features=features,
            mask=mask,
        )
        target_feat_dim = int(self.mean.shape[0])
        features = _canonicalize_features_for_model(
            raw_features=features,
            stop_event_id=stop_event_id,
            target_feat_dim=target_feat_dim,
            mask=mask,
            n_entities=int(features.shape[1]),
        )

        # Public sources can contain ball rows marked observed with invalid XY.
        # Drop those observations from the mask and sanitize feature values.
        if features.shape[-1] >= 2:
            finite_xy = np.isfinite(features[..., 0]) & np.isfinite(features[..., 1])
            bad_xy_obs = mask & (~finite_xy)
            if np.any(bad_xy_obs):
                mask[bad_xy_obs] = False
        finite_feat = np.isfinite(features).all(axis=-1)
        bad_feat_obs = mask & (~finite_feat)
        if np.any(bad_feat_obs):
            mask[bad_feat_obs] = False
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

        if features.shape[0] > 1:
            # Always recompute from features to support mixed legacy/new npz layouts.
            entity_delta = np.zeros((features.shape[0] - 1, self.n_entities, self.delta_dim), dtype=np.float32)
            entity_delta_mask = mask[1:, :] & mask[:-1, :]
            entity_delta_dim_mask = np.zeros(
                (features.shape[0] - 1, self.n_entities, self.delta_dim),
                dtype=bool,
            )

            # dx, dy for all entities.
            entity_delta[:, :, :2] = features[1:, :, :2] - features[:-1, :, :2]

            active_entities = entity_type != 3
            for e in range(self.n_entities):
                if not active_entities[e]:
                    entity_delta[:, e] = 0.0
                    entity_delta_mask[:, e] = True
                    entity_delta_dim_mask[:, e, :] = False
                    continue

                obs = mask[:, e]
                if not obs.any():
                    entity_delta[:, e] = 0.0
                    entity_delta_mask[:, e] = True
                    entity_delta_dim_mask[:, e, :] = False
                    continue

                first_obs = int(np.argmax(obs))
                if first_obs > 0:
                    entity_delta_mask[:first_obs, e] = False
                entity_delta_mask[first_obs:, e] = True
                entity_delta_dim_mask[:, e, :2] = entity_delta_mask[:, e][:, None]

            # dz only for ball when z channel exists.
            if self.delta_dim >= 3 and target_feat_dim >= 5 and int(raw_feat_dim) >= 5:
                ball_idx = 22
                dz = features[1:, ball_idx, 4] - features[:-1, ball_idx, 4]
                dz_valid = mask[1:, ball_idx] & mask[:-1, ball_idx] & (entity_type[ball_idx] != 3)
                entity_delta[:, ball_idx, 2] = dz.astype(np.float32)
                entity_delta_dim_mask[:, ball_idx, 2] = dz_valid.astype(bool)

            entity_delta_norm = entity_delta / self.delta_scale[None, None, :]
        else:
            entity_delta = np.zeros((0, self.n_entities, self.delta_dim), dtype=np.float32)
            entity_delta_norm = np.zeros((0, self.n_entities, self.delta_dim), dtype=np.float32)
            entity_delta_mask = np.zeros((0, self.n_entities), dtype=bool)
            entity_delta_dim_mask = np.zeros((0, self.n_entities, self.delta_dim), dtype=bool)

        return {
            "features": features.astype(np.float32),
            "mask": mask.astype(bool),
            "entity_type": entity_type.astype(np.int64),
            "frames": frames.astype(np.int64),
            "stop_event_id": (
                np.asarray(stop_event_id, dtype=np.uint8).copy()
                if stop_event_id is not None
                else np.full((features.shape[0],), STOP_CONTINUE, dtype=np.uint8)
            ),
            "entity_delta": entity_delta.astype(np.float32),
            "entity_delta_norm": entity_delta_norm.astype(np.float32),
            "entity_delta_mask": entity_delta_mask.astype(bool),
            "entity_delta_dim_mask": entity_delta_dim_mask.astype(bool),
        }

    def _load_clip_from_disk(self, clip_idx: int) -> Dict[str, np.ndarray]:
        entry = self.entries[clip_idx]
        with np.load(entry.clip_path, allow_pickle=False) as data:
            features = data["features"].astype(np.float32)
            mask = data["mask"].astype(bool)
            entity_type = data["entity_type"].astype(np.int64)
            frames = data["frames"].astype(np.int64)
            stop_event_id = data["stop_event_id"].astype(np.uint8) if "stop_event_id" in data.files else None
        raw_feat_dim = int(features.shape[-1])
        return self._prepare_clip_arrays(
            features=features,
            mask=mask,
            entity_type=entity_type,
            frames=frames,
            stop_event_id=stop_event_id,
            raw_feat_dim=raw_feat_dim,
            delta_raw=None,
            delta_mask_raw=None,
        )

    def _load_clip(self, clip_idx: int) -> Dict[str, np.ndarray]:
        if self._preloaded_clips is not None:
            return self._preloaded_clips[clip_idx]

        if clip_idx in self._clip_cache:
            self._clip_cache.move_to_end(clip_idx)
            return self._clip_cache[clip_idx]

        clip = self._load_clip_from_disk(clip_idx)
        self._clip_cache[clip_idx] = clip
        self._clip_cache.move_to_end(clip_idx)

        while len(self._clip_cache) > self.clip_cache_size:
            self._clip_cache.popitem(last=False)
        return clip

    def _sample_by_window(self, window_idx: int, target_entity: Optional[int] = None) -> Dict[str, torch.Tensor]:
        clip_idx, local_start = self._resolve_window(window_idx)
        clip = self._load_clip(clip_idx)

        h = self.history
        r = self.rollout_steps
        hist_slice = slice(local_start, local_start + h)
        future_slice = slice(local_start + h, local_start + h + r)

        x_raw = clip["features"][hist_slice]       # [H, 23, 4]
        x_mask = clip["mask"][hist_slice]          # [H, 23]

        target_t0 = local_start + h - 1
        target_t1 = target_t0 + r
        target_delta = clip["entity_delta_norm"][target_t0:target_t1]  # [R, 23, D_out]
        target_mask = clip["entity_delta_mask"][target_t0:target_t1]   # [R, 23]
        target_dim_mask = clip["entity_delta_dim_mask"][target_t0:target_t1]  # [R, 23, D_out]
        future_features = clip["features"][future_slice]                # [R, 23, 4]
        future_mask = clip["mask"][future_slice]                        # [R, 23]
        history_stop_event = clip["stop_event_id"][hist_slice]          # [H]
        future_stop_event = clip["stop_event_id"][future_slice]         # [R]
        history_stop_token = tokenize_stop_sequence(history_stop_event)  # [H]
        future_stop_token = tokenize_stop_sequence(future_stop_event)    # [R]
        future_continue_mask = future_stop_token == STOP_CONTINUE
        prev_pos = clip["features"][local_start + h - 1, :, :2]

        sample: Dict[str, torch.Tensor] = {
            "x": torch.from_numpy(x_raw).float(),
            "obs_mask": torch.from_numpy(x_mask).bool(),
            "entity_type": torch.from_numpy(clip["entity_type"]).long(),
            "target_delta": torch.from_numpy(target_delta).float(),
            "target_mask": torch.from_numpy(target_mask).bool(),
            "target_dim_mask": torch.from_numpy(target_dim_mask).bool(),
            "future_features": torch.from_numpy(future_features).float(),
            "future_mask": torch.from_numpy(future_mask).bool(),
            "history_stop_event": torch.from_numpy(history_stop_event.astype(np.int64)).long(),
            "future_stop_event": torch.from_numpy(future_stop_event.astype(np.int64)).long(),
            "history_stop_token": torch.from_numpy(history_stop_token.astype(np.int64)).long(),
            "future_stop_token": torch.from_numpy(future_stop_token.astype(np.int64)).long(),
            "future_continue_mask": torch.from_numpy(future_continue_mask.astype(bool)).bool(),
            "current_pos": torch.from_numpy(prev_pos).float(),
        }

        if target_entity is not None:
            sample["target_entity_idx"] = torch.tensor(int(target_entity), dtype=torch.long)
            sample["single_target_delta"] = torch.from_numpy(target_delta[:, target_entity]).float()
            sample["single_target_mask"] = torch.from_numpy(target_mask[:, target_entity]).bool()
            sample["single_target_dim_mask"] = torch.from_numpy(target_dim_mask[:, target_entity]).bool()

        return sample

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.mode == "multi":
            return self._sample_by_window(window_idx=int(idx), target_entity=None)

        window_idx = int(idx) // self.n_entities
        target_entity = int(idx) % self.n_entities
        return self._sample_by_window(window_idx=window_idx, target_entity=target_entity)

    def get_rollout_context(self, window_idx: int, rollout_steps: int) -> Dict[str, np.ndarray]:
        clip_idx, local_start = self._resolve_window(window_idx)
        clip = self._load_clip(clip_idx)

        h = self.history
        target_t = local_start + h
        end_t = min(target_t + rollout_steps, clip["features"].shape[0])

        return {
            "history_features": clip["features"][local_start:target_t].copy(),
            "history_mask": clip["mask"][local_start:target_t].copy(),
            "history_stop_event": clip["stop_event_id"][local_start:target_t].copy(),
            "history_stop_token": tokenize_stop_sequence(clip["stop_event_id"][local_start:target_t]).copy(),
            "entity_type": clip["entity_type"].copy(),
            "future_features": clip["features"][target_t:end_t].copy(),
            "future_mask": clip["mask"][target_t:end_t].copy(),
            "future_stop_event": clip["stop_event_id"][target_t:end_t].copy(),
            "future_stop_token": tokenize_stop_sequence(clip["stop_event_id"][target_t:end_t]).copy(),
            "future_players": clip["features"][target_t:end_t, :22, :2].copy(),
            "future_players_mask": clip["mask"][target_t:end_t, :22].copy(),
            "future_ball": clip["features"][target_t:end_t, 22, :].copy(),
            "future_ball_mask": clip["mask"][target_t:end_t, 22].copy(),
            "available_steps": np.asarray([end_t - target_t], dtype=np.int64),
        }


def create_train_val_datasets(
    data_dir: str | Path,
    cache_dir: str | Path,
    history: int,
    mode: str,
    rollout_steps: int,
    val_ratio: float,
    seed: int,
    file_indices: Optional[Sequence[int]] = None,
    max_files: Optional[int] = None,
    preprocess_cfg: Optional[PreprocessConfig] = None,
    rebuild_cache: bool = False,
    clip_cache_size: int = 4,
    preload_clips: bool = False,
    preload_progress: bool = False,
    preload_train_clips: Optional[bool] = None,
    preload_val_clips: Optional[bool] = None,
) -> Tuple[TrajectoryWindowDataset, Optional[TrajectoryWindowDataset], NormalizationStats, Dict[str, Any]]:
    files = discover_tracking_files(data_dir=data_dir, file_indices=file_indices)
    if max_files is not None:
        files = files[: max(0, int(max_files))]

    if not files:
        raise RuntimeError(f"No tracking parquet files found in {data_dir}.")

    train_files, val_files = split_files(files=files, val_ratio=val_ratio, seed=seed)
    cfg = preprocess_cfg or PreprocessConfig()

    cache_dir = Path(cache_dir)
    train_cache = cache_dir / "train"
    val_cache = cache_dir / "val"
    stats_path = cache_dir / "normalization_stats.json"

    train_entries = build_cache_for_files(train_files, train_cache, cfg=cfg, rebuild=rebuild_cache)
    val_entries = build_cache_for_files(val_files, val_cache, cfg=cfg, rebuild=rebuild_cache) if val_files else []
    if not train_entries:
        raise RuntimeError("No usable clips found in training split. Relax kick/segment filters.")

    if stats_path.exists() and not rebuild_cache:
        stats = load_stats(stats_path)
    else:
        stats = compute_normalization_stats(train_entries)
        save_stats(stats, stats_path)

    preload_train = preload_clips if preload_train_clips is None else bool(preload_train_clips)
    preload_val = preload_clips if preload_val_clips is None else bool(preload_val_clips)

    train_dataset = TrajectoryWindowDataset(
        entries=train_entries,
        history=history,
        rollout_steps=rollout_steps,
        mode=mode,
        stats=stats,
        clip_cache_size=clip_cache_size,
        preload_clips=preload_train,
        preload_progress=preload_progress,
    )

    val_dataset = None
    if val_entries:
        val_dataset = TrajectoryWindowDataset(
            entries=val_entries,
            history=history,
            rollout_steps=rollout_steps,
            mode=mode,
            stats=stats,
            clip_cache_size=clip_cache_size,
            preload_clips=preload_val,
            preload_progress=preload_progress,
        )

    meta: Dict[str, Any] = {
        "num_files_total": len(files),
        "num_files_train": len(train_files),
        "num_files_val": len(val_files),
        "num_clips_train": len(train_entries),
        "num_clips_val": len(val_entries),
        "num_windows_train": train_dataset.num_windows,
        "num_windows_val": val_dataset.num_windows if val_dataset is not None else 0,
    }
    return train_dataset, val_dataset, stats, meta


def _read_json_if_exists(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _infer_ballplayer_root(preprocessed_dir: Path) -> Optional[Path]:
    candidates = [preprocessed_dir] + list(preprocessed_dir.parents)
    for cand in candidates:
        if (cand / "dataset.py").exists() and (cand / "train.py").exists():
            return cand
    return None


def _discover_external_source_dirs(preprocessed_dir: Path) -> Dict[str, Path]:
    root = _infer_ballplayer_root(preprocessed_dir)
    if root is None:
        return {}
    candidates = {
        "current": root / "preprocessed",
        "metrica_xy": root / "publicData" / "preprocessed" / "metrica_xy",
        "sportec_xyz": root / "publicData" / "preprocessed" / "sportec_xyz",
    }
    out: Dict[str, Path] = {}
    for name, path in candidates.items():
        if path.exists() and path.is_dir():
            out[name] = path
    return out


def _pad_to_dim(x: np.ndarray, dim: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if x.shape[0] == dim:
        return x
    out = np.zeros((dim,), dtype=np.float32)
    take = min(dim, x.shape[0])
    out[:take] = x[:take]
    return out


def _upgrade_stats_feature_dim_fast(
    stats: NormalizationStats,
    target_feat_dim: int,
) -> NormalizationStats:
    cur_dim = int(np.asarray(stats.mean).shape[0])
    target_feat_dim = int(target_feat_dim)
    if cur_dim == target_feat_dim:
        return stats

    mean = np.asarray(stats.mean, dtype=np.float32).reshape(-1)
    std = np.asarray(stats.std, dtype=np.float32).reshape(-1)

    if cur_dim > target_feat_dim:
        mean = mean[:target_feat_dim]
        std = std[:target_feat_dim]
    else:
        pad_dim = target_feat_dim - cur_dim
        mean = np.concatenate([mean, np.zeros((pad_dim,), dtype=np.float32)], axis=0)
        # Added channels are indicator-like; default to unit scale to avoid destabilizing normalization.
        std = np.concatenate([std, np.ones((pad_dim,), dtype=np.float32)], axis=0)

    std = np.maximum(std, 1e-3)
    delta = np.maximum(np.asarray(stats.delta_scale, dtype=np.float32).reshape(-1), 1e-6)
    return NormalizationStats(mean=mean.astype(np.float32), std=std.astype(np.float32), delta_scale=delta.astype(np.float32))


def _merge_stats_from_source_dirs(source_dirs: Sequence[Path]) -> Optional[NormalizationStats]:
    items: List[Tuple[NormalizationStats, float]] = []
    max_feat_dim = 0
    max_delta_dim = 0

    for src in source_dirs:
        stats_path = src / "normalization_stats.json"
        if not stats_path.exists():
            continue
        try:
            stats = load_stats(stats_path)
        except Exception:
            continue
        if not _stats_is_valid(stats):
            continue

        meta = _read_json_if_exists(src / "preprocessed_meta.json")
        weight = float(meta.get("num_frames_train", 1.0) or 1.0)
        weight = max(weight, 1.0)

        max_feat_dim = max(max_feat_dim, int(stats.mean.shape[0]))
        max_delta_dim = max(max_delta_dim, int(stats.delta_scale.shape[0]))
        items.append((stats, weight))

    if not items:
        return None

    if max_feat_dim <= 0:
        max_feat_dim = 1
    if max_delta_dim <= 0:
        max_delta_dim = 2

    total_w = float(sum(w for _s, w in items))
    mean_acc = np.zeros((max_feat_dim,), dtype=np.float64)
    second_acc = np.zeros((max_feat_dim,), dtype=np.float64)
    delta_acc = np.zeros((max_delta_dim,), dtype=np.float64)

    for stats, w in items:
        mean_i = _pad_to_dim(stats.mean, max_feat_dim).astype(np.float64)
        std_i = np.maximum(_pad_to_dim(stats.std, max_feat_dim).astype(np.float64), 1e-6)
        var_i = np.square(std_i)
        mean_acc += w * mean_i
        second_acc += w * (var_i + np.square(mean_i))

        d_i = np.maximum(_pad_to_dim(stats.delta_scale, max_delta_dim).astype(np.float64), 1e-6)
        delta_acc += w * d_i

    mean = mean_acc / max(total_w, 1.0)
    var = np.maximum(second_acc / max(total_w, 1.0) - np.square(mean), 1e-6)
    std = np.sqrt(var)
    delta_scale = np.maximum(delta_acc / max(total_w, 1.0), 1e-6)

    return NormalizationStats(
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
        delta_scale=delta_scale.astype(np.float32),
    )


def _resolve_external_preprocessed_fallback(
    preprocessed_dir: Path,
    need_stats: bool,
) -> Optional[Dict[str, Any]]:
    source_dirs = _discover_external_source_dirs(preprocessed_dir)
    if not source_dirs:
        return None

    def split_entries(src: Path, split: str) -> List[ClipEntry]:
        return _load_manifest(src / split / "manifest.jsonl")

    train_entries: List[ClipEntry] = []
    for key in ("current", "metrica_xy", "sportec_xyz"):
        src = source_dirs.get(key, None)
        if src is None:
            continue
        train_entries.extend(split_entries(src, "train"))
    if not train_entries:
        return None

    sportec_dir = source_dirs.get("sportec_xyz", None)
    if sportec_dir is not None:
        val_entries = split_entries(sportec_dir, "val")
        test_entries = split_entries(sportec_dir, "test")
    else:
        val_entries = []
        test_entries = []

    if not val_entries:
        for key in ("current", "metrica_xy"):
            src = source_dirs.get(key, None)
            if src is None:
                continue
            val_entries.extend(split_entries(src, "val"))

    stats: Optional[NormalizationStats] = None
    if need_stats:
        stage_stats_path = preprocessed_dir / "normalization_stats.json"
        if stage_stats_path.exists():
            try:
                stats = load_stats(stage_stats_path)
                if not _stats_is_valid(stats):
                    stats = None
            except Exception:
                stats = None
        if stats is None:
            merged = _merge_stats_from_source_dirs(list(source_dirs.values()))
            if merged is None:
                try:
                    merged = compute_normalization_stats(
                        train_entries,
                        use_tqdm=True,
                        tqdm_desc=f"Computing fallback normalization ({preprocessed_dir.name})",
                    )
                except Exception:
                    merged = None
            stats = merged
            if stats is not None:
                try:
                    save_stats(stats, stage_stats_path)
                except Exception:
                    pass

    meta = _read_json_if_exists(preprocessed_dir / "preprocessed_meta.json")
    meta.setdefault("fallback_mode", "external_sources")
    meta["fallback_sources"] = {k: str(v) for k, v in source_dirs.items()}
    meta.setdefault("num_clips_train", len(train_entries))
    meta.setdefault("num_clips_val", len(val_entries))
    meta.setdefault("num_clips_test", len(test_entries))
    meta["preprocessed_dir"] = str(preprocessed_dir)

    return {
        "train_entries": train_entries,
        "val_entries": val_entries,
        "test_entries": test_entries,
        "stats": stats,
        "meta": meta,
    }


def create_datasets_from_preprocessed(
    preprocessed_dir: str | Path,
    history: int,
    mode: str,
    rollout_steps: int = 1,
    clip_cache_size: int = 4,
    preload_clips: bool = False,
    preload_progress: bool = False,
    preload_train_clips: Optional[bool] = None,
    preload_val_clips: Optional[bool] = None,
) -> Tuple[TrajectoryWindowDataset, Optional[TrajectoryWindowDataset], NormalizationStats, Dict[str, Any]]:
    preprocessed_dir = Path(preprocessed_dir)
    stats_path = preprocessed_dir / "normalization_stats.json"
    train_manifest = preprocessed_dir / "train" / "manifest.jsonl"
    val_manifest = preprocessed_dir / "val" / "manifest.jsonl"
    meta_path = preprocessed_dir / "preprocessed_meta.json"

    stats: Optional[NormalizationStats] = None
    if stats_path.exists():
        try:
            stats = load_stats(stats_path)
            if not _stats_is_valid(stats):
                stats = None
        except Exception:
            stats = None
    train_entries = _load_manifest(train_manifest) if train_manifest.exists() else []
    val_entries = _load_manifest(val_manifest) if val_manifest.exists() else []

    if stats is None or not train_entries:
        fallback = _resolve_external_preprocessed_fallback(
            preprocessed_dir=preprocessed_dir,
            need_stats=(stats is None),
        )
        if fallback is not None:
            if not train_entries:
                train_entries = fallback["train_entries"]
                val_entries = fallback["val_entries"]
            if stats is None:
                stats = fallback["stats"]

    if stats is None:
        raise RuntimeError(f"Missing normalization stats at {stats_path}. Run preprocess.py first.")
    if not train_entries:
        raise RuntimeError(f"Missing train manifest at {train_manifest}. Run preprocess.py first.")

    expected_feat_dim = int(_infer_expected_feature_dim_from_entries(train_entries))
    current_feat_dim = int(np.asarray(stats.mean).shape[0])
    if current_feat_dim != expected_feat_dim:
        # Fast-path for small schema extensions (e.g. z_obs/vz_obs flags) to avoid full rescans.
        if current_feat_dim < expected_feat_dim and (expected_feat_dim - current_feat_dim) <= 2:
            stats = _upgrade_stats_feature_dim_fast(stats, target_feat_dim=expected_feat_dim)
        else:
            stats = compute_normalization_stats(
                train_entries,
                use_tqdm=True,
                tqdm_desc=f"Upgrading normalization ({preprocessed_dir.name})",
            )
        try:
            save_stats(stats, stats_path)
        except Exception:
            pass

    if not train_entries:
        raise RuntimeError("Preprocessed train split is empty. Re-run preprocess.py with relaxed filters.")

    preload_train = preload_clips if preload_train_clips is None else bool(preload_train_clips)
    preload_val = preload_clips if preload_val_clips is None else bool(preload_val_clips)

    train_dataset = TrajectoryWindowDataset(
        entries=train_entries,
        history=history,
        rollout_steps=rollout_steps,
        mode=mode,
        stats=stats,
        clip_cache_size=clip_cache_size,
        preload_clips=preload_train,
        preload_progress=preload_progress,
    )

    val_dataset = None
    if val_entries:
        val_dataset = TrajectoryWindowDataset(
            entries=val_entries,
            history=history,
            rollout_steps=rollout_steps,
            mode=mode,
            stats=stats,
            clip_cache_size=clip_cache_size,
            preload_clips=preload_val,
            preload_progress=preload_progress,
        )

    meta = _read_json_if_exists(meta_path)
    if not meta:
        fallback_meta = _resolve_external_preprocessed_fallback(
            preprocessed_dir=preprocessed_dir,
            need_stats=False,
        )
        if fallback_meta is not None:
            meta = dict(fallback_meta.get("meta", {}))

    meta.setdefault("num_clips_train", len(train_entries))
    meta.setdefault("num_clips_val", len(val_entries))
    meta["num_windows_train"] = train_dataset.num_windows
    meta["num_windows_val"] = val_dataset.num_windows if val_dataset is not None else 0
    meta["preprocessed_dir"] = str(preprocessed_dir)

    return train_dataset, val_dataset, stats, meta


def create_split_dataset_from_preprocessed(
    preprocessed_dir: str | Path,
    split: str,
    history: int,
    mode: str,
    rollout_steps: int = 1,
    clip_cache_size: int = 4,
    preload_clips: bool = False,
    preload_progress: bool = False,
    stats_override: Optional[NormalizationStats] = None,
) -> Tuple[Optional[TrajectoryWindowDataset], NormalizationStats, Dict[str, Any]]:
    if split not in {"train", "val", "test"}:
        raise ValueError("split must be one of {'train','val','test'}.")

    preprocessed_dir = Path(preprocessed_dir)
    stats_path = preprocessed_dir / "normalization_stats.json"
    manifest_path = preprocessed_dir / split / "manifest.jsonl"
    meta_path = preprocessed_dir / "preprocessed_meta.json"

    if stats_override is not None:
        stats = stats_override if _stats_is_valid(stats_override) else None
    else:
        stats = None
        if stats_path.exists():
            try:
                stats = load_stats(stats_path)
                if not _stats_is_valid(stats):
                    stats = None
            except Exception:
                stats = None
    entries = _load_manifest(manifest_path) if manifest_path.exists() else []

    if stats is None or not entries:
        fallback = _resolve_external_preprocessed_fallback(
            preprocessed_dir=preprocessed_dir,
            need_stats=(stats is None and stats_override is None),
        )
        if fallback is not None:
            if not entries:
                entries = list(fallback.get(f"{split}_entries", []))
            if stats is None:
                stats = fallback.get("stats", None)

    if stats is None:
        raise RuntimeError(f"Missing normalization stats at {stats_path}.")
    dataset: Optional[TrajectoryWindowDataset] = None
    if entries:
        dataset = TrajectoryWindowDataset(
            entries=entries,
            history=history,
            rollout_steps=rollout_steps,
            mode=mode,
            stats=stats,
            clip_cache_size=clip_cache_size,
            preload_clips=preload_clips,
            preload_progress=preload_progress,
        )

    meta = _read_json_if_exists(meta_path)
    if not meta:
        fallback_meta = _resolve_external_preprocessed_fallback(
            preprocessed_dir=preprocessed_dir,
            need_stats=False,
        )
        if fallback_meta is not None:
            meta = dict(fallback_meta.get("meta", {}))
    meta["preprocessed_dir"] = str(preprocessed_dir)
    meta[f"num_clips_{split}"] = len(entries)
    meta[f"num_windows_{split}"] = dataset.num_windows if dataset is not None else 0
    return dataset, stats, meta
