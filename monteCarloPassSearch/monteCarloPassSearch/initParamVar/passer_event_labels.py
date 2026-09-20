from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Dict, List, Optional, Sequence

import numpy as np


def _parse_match_id(source_file: object) -> str:
    s = str(source_file or "")
    if ":" in s:
        return s.split(":", 1)[1]
    return s


def _figshare_file_id_from_url(url: str) -> str:
    m = re.search(r"/files/(\d+)$", str(url))
    if not m:
        return str(url)
    return m.group(1)


def _figshare_api_url(file_id: str) -> str:
    if str(file_id).startswith(("http://", "https://")):
        return str(file_id)
    return f"https://api.figshare.com/v2/file/download/{file_id}"


def _resolve_side_from_team(team: object) -> Optional[str]:
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


def _safe_total_seconds(ts: object) -> float:
    try:
        out = float(ts.total_seconds())  # type: ignore[attr-defined]
    except Exception:
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _load_tracking_frame_table(match_id: str) -> Dict[str, np.ndarray]:
    from kloppy import sportec
    from kloppy._providers.sportec import get_IDSSE_url

    meta_file_id = _figshare_file_id_from_url(get_IDSSE_url(str(match_id), "meta"))
    tracking_file_id = _figshare_file_id_from_url(get_IDSSE_url(str(match_id), "tracking"))
    meta_url = _figshare_api_url(meta_file_id)
    tracking_url = _figshare_api_url(tracking_file_id)

    ds = sportec.load_tracking(meta_data=meta_url, raw_data=tracking_url, limit=None, only_alive=False)
    frames: List[int] = []
    periods: List[int] = []
    times: List[float] = []
    for fr in ds.records:
        frames.append(int(getattr(fr, "frame_id", len(frames))))
        periods.append(int(getattr(getattr(fr, "period", None), "id", 1) or 1))
        times.append(_safe_total_seconds(getattr(fr, "timestamp", None)))
    return {
        "frames": np.asarray(frames, dtype=np.int64),
        "periods": np.asarray(periods, dtype=np.int64),
        "times": np.asarray(times, dtype=np.float64),
    }


def _load_pass_events(match_id: str) -> Dict[str, object]:
    from kloppy import sportec
    from kloppy._providers.sportec import get_IDSSE_url

    meta_file_id = _figshare_file_id_from_url(get_IDSSE_url(str(match_id), "meta"))
    event_file_id = _figshare_file_id_from_url(get_IDSSE_url(str(match_id), "event"))
    meta_url = _figshare_api_url(meta_file_id)
    event_url = _figshare_api_url(event_file_id)

    ds = sportec.load_event(event_data=event_url, meta_data=meta_url)

    side_to_team_name: Dict[str, str] = {}
    for team in getattr(getattr(ds, "metadata", None), "teams", []) or []:
        side = _resolve_side_from_team(team)
        if side in {"home", "away"}:
            nm = getattr(team, "name", None)
            if nm is not None and str(nm).strip():
                side_to_team_name[side] = str(nm).strip()

    events: List[Dict[str, object]] = []
    for ev in getattr(ds, "events", []) or []:
        ev_type = str(getattr(getattr(ev, "event_type", None), "name", "")).upper()
        if ev_type != "PASS":
            continue
        period = int(getattr(getattr(ev, "period", None), "id", 1) or 1)
        t_sec = _safe_total_seconds(getattr(ev, "timestamp", None))
        if not math.isfinite(t_sec):
            continue

        team = getattr(ev, "team", None)
        player = getattr(ev, "player", None)
        team_side = _resolve_side_from_team(team)
        team_name = str(getattr(team, "name", "")).strip() if team is not None else ""

        player_id = str(getattr(player, "player_id", "") or "").strip() if player is not None else ""
        player_name = ""
        if player is not None:
            for attr in ("full_name", "name"):
                v = getattr(player, attr, None)
                if v is not None and str(v).strip():
                    player_name = str(v).strip()
                    break

        events.append(
            {
                "period": int(period),
                "time": float(t_sec),
                "team_side": team_side,
                "team_name": team_name if team_name else side_to_team_name.get(team_side or "", ""),
                "player_id": player_id,
                "player_name": player_name,
                "event_id": str(getattr(ev, "event_id", "") or ""),
            }
        )

    return {"events": events, "side_to_team_name": side_to_team_name}


def _match_pass_events_to_frame_indices(
    pass_events: Sequence[Dict[str, object]],
    periods: np.ndarray,
    frame_times: np.ndarray,
    tolerance_sec: float,
    refractory_frames: int,
) -> List[Dict[str, object]]:
    if not pass_events:
        return []

    periods = np.asarray(periods, dtype=np.int64).reshape(-1)
    frame_times = np.asarray(frame_times, dtype=np.float64).reshape(-1)
    if periods.size == 0 or frame_times.size == 0 or periods.size != frame_times.size:
        return []

    events_by_period: Dict[int, List[Dict[str, object]]] = defaultdict(list)
    for ev in pass_events:
        try:
            period = int(ev.get("period", 1))
        except Exception:
            period = 1
        try:
            t_event = float(ev.get("time", np.nan))
        except Exception:
            t_event = np.nan
        if not math.isfinite(t_event):
            continue
        x = dict(ev)
        x["period"] = int(period)
        x["time"] = float(t_event)
        events_by_period[int(period)].append(x)

    picked: List[Dict[str, object]] = []
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
                row = dict(ev)
                row["idx"] = int(best_idx)
                row["err"] = float(err)
                picked.append(row)

    if not picked:
        return []

    by_idx: Dict[int, Dict[str, object]] = {}
    for row in picked:
        idx = int(row["idx"])
        prev = by_idx.get(idx)
        if prev is None or float(row["err"]) < float(prev["err"]):
            by_idx[idx] = row

    matched = sorted(by_idx.values(), key=lambda r: int(r["idx"]))
    refractory = max(int(refractory_frames), 0)
    if refractory <= 0:
        return matched

    filtered: List[Dict[str, object]] = []
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


def _nearest_event_by_time(
    *,
    events: Sequence[Dict[str, object]],
    period: int,
    time_sec: float,
    max_err_sec: float,
) -> Optional[Dict[str, object]]:
    best = None
    best_err = float("inf")
    for ev in events:
        if int(ev.get("period", -1)) != int(period):
            continue
        t = float(ev.get("time", np.nan))
        if not math.isfinite(t):
            continue
        e = abs(float(time_sec) - t)
        if e < best_err:
            best = ev
            best_err = e
    if best is None or best_err > float(max_err_sec):
        return None
    out = dict(best)
    out["err"] = float(best_err)
    return out


def build_manifest_passer_labels(
    manifest_entries: Sequence[dict],
    *,
    tolerance_sec: float = 0.6,
    refractory_frames: int = 8,
) -> Dict[int, Dict[str, object]]:
    by_match: Dict[str, List[tuple[int, dict]]] = defaultdict(list)
    for clip_idx, entry in enumerate(manifest_entries):
        match_id = _parse_match_id(entry.get("source_file"))
        by_match[match_id].append((int(clip_idx), entry))

    out: Dict[int, Dict[str, object]] = {}
    for match_id, rows in by_match.items():
        if not str(match_id).strip():
            continue
        tr = _load_tracking_frame_table(match_id=str(match_id))
        periods = tr["periods"]
        times = tr["times"]
        frames = tr["frames"]
        frame_to_idx = {int(frames[i]): int(i) for i in range(frames.shape[0])}

        ev_pack = _load_pass_events(match_id=str(match_id))
        events = list(ev_pack.get("events", []))
        side_to_team_name = dict(ev_pack.get("side_to_team_name", {}))

        matched = _match_pass_events_to_frame_indices(
            pass_events=events,
            periods=periods,
            frame_times=times,
            tolerance_sec=float(tolerance_sec),
            refractory_frames=int(refractory_frames),
        )
        frame_to_event: Dict[int, Dict[str, object]] = {}
        for m in matched:
            idx = int(m["idx"])
            if idx < 0 or idx >= frames.shape[0]:
                continue
            frame_to_event[int(frames[idx])] = m

        for clip_idx, entry in rows:
            kick_frame = int(entry.get("kick_frame", -1))
            ev = frame_to_event.get(kick_frame)
            err = float(ev.get("err", np.nan)) if ev is not None else float("nan")

            if ev is None:
                idx = frame_to_idx.get(kick_frame)
                if idx is None and frames.size > 0:
                    idx = int(np.argmin(np.abs(frames.astype(np.int64) - int(kick_frame))))
                if idx is not None and 0 <= idx < frames.shape[0]:
                    ev = _nearest_event_by_time(
                        events=events,
                        period=int(periods[int(idx)]),
                        time_sec=float(times[int(idx)]),
                        max_err_sec=max(1.5, 2.0 * float(tolerance_sec)),
                    )
                    if ev is not None:
                        err = float(ev.get("err", np.nan))

            team_side = str(ev.get("team_side", "") or "").lower() if ev is not None else ""
            if team_side not in {"home", "away"}:
                team_side = ""
            team_name = str(ev.get("team_name", "") or "").strip() if ev is not None else ""
            if not team_name and team_side:
                team_name = str(side_to_team_name.get(team_side, "") or "").strip()

            out[int(clip_idx)] = {
                "passer_player_id": str(ev.get("player_id", "") or "").strip() if ev is not None else "",
                "passer_player_name": str(ev.get("player_name", "") or "").strip() if ev is not None else "",
                "passer_team_side": team_side,
                "passer_team_name": team_name,
                "home_team_name": str(side_to_team_name.get("home", "") or "").strip(),
                "away_team_name": str(side_to_team_name.get("away", "") or "").strip(),
                "passer_event_period": int(ev.get("period", -1)) if ev is not None else -1,
                "passer_event_time_sec": float(ev.get("time", np.nan)) if ev is not None else float("nan"),
                "passer_event_time_error_sec": float(err) if math.isfinite(err) else float("nan"),
                "passer_event_id": str(ev.get("event_id", "") or "").strip() if ev is not None else "",
            }

    return out
