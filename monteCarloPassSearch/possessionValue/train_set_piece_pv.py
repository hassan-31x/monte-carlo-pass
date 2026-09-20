#!/usr/bin/env python3
"""Train set-piece PV estimators.

Implements:
  1) Throw-ins: logistic regression with only horizontal ball coordinate x.
  2) Corners / Goal-kicks: counting-based expected xG within 10 seconds.

Training data:
  - Sportec (IDSSE): set-piece starts from kloppy Sportec events.
  - StatsBomb open data: set-piece starts from kloppy StatsBomb events.
"""

from __future__ import annotations

import argparse
import bisect
import json
import random
import re
import urllib.request
from collections import defaultdict
from datetime import timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from tqdm.auto import tqdm

from kloppy import sportec, statsbomb
from kloppy._providers import sportec as sportec_provider


STATSBOMB_COMPETITIONS_URL = (
    "https://raw.githubusercontent.com/statsbomb/open-data/master/data/competitions.json"
)
STATSBOMB_MATCHES_URL_TMPL = (
    "https://raw.githubusercontent.com/statsbomb/open-data/master/data/matches/"
    "{competition_id}/{season_id}.json"
)


def _json_url(url: str):
    with urllib.request.urlopen(url, timeout=30) as f:
        return json.load(f)


def _timestamp_to_seconds(ts) -> Optional[float]:
    if ts is None:
        return None
    if isinstance(ts, timedelta):
        return float(ts.total_seconds())
    if isinstance(ts, (int, float)):
        return float(ts)
    s = str(ts).strip()
    if not s:
        return None
    # HH:MM:SS(.sss)
    m = re.match(r"^(\d+):(\d+):(\d+(?:\.\d+)?)$", s)
    if m:
        h, mm, sec = m.groups()
        return float(h) * 3600.0 + float(mm) * 60.0 + float(sec)
    return None


def _first_shot_xg(
    shot_list: Sequence[Tuple[float, float]],
    t0: float,
    horizon: float,
) -> float:
    if not shot_list:
        return 0.0
    times = [v[0] for v in shot_list]
    i = bisect.bisect_left(times, t0)
    while i < len(shot_list):
        t, xg = shot_list[i]
        if t > t0 + horizon:
            break
        if t >= t0:
            return float(max(0.0, xg))
        i += 1
    return 0.0


def _sportec_file_url(match_id: str, kind: str) -> str:
    src = sportec_provider.get_IDSSE_url(match_id, kind)
    fid = re.search(r"/files/(\d+)$", src)
    if fid is None:
        return src
    return f"https://ndownloader.figshare.com/files/{fid.group(1)}"


def _classify_sportec_restart(x: float, y: float) -> Optional[str]:
    """Heuristic set-piece classification from static start location."""
    sideline_tol = 3.0
    endline_tol = 8.0
    goal_kick_y_min = 18.0
    goal_kick_y_max = 62.0

    on_sideline = (y <= sideline_tol) or (y >= 80.0 - sideline_tol)
    near_endline = (x <= endline_tol) or (x >= 120.0 - endline_tol)

    if on_sideline:
        if near_endline:
            return "corner"
        return "throw_in"
    if near_endline and (goal_kick_y_min <= y <= goal_kick_y_max):
        return "goal_kick"
    return None


def _load_sportec_shots(path: str) -> Dict[str, Dict[Tuple[str, int], List[Tuple[float, float]]]]:
    with open(path, "r") as f:
        rows = json.load(f)

    out: Dict[str, Dict[Tuple[str, int], List[Tuple[float, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for r in rows:
        mid = str(r.get("match_id", "")).strip()
        if not mid:
            continue
        side = str(r.get("side", "")).strip().lower()
        if side not in ("home", "away"):
            continue
        try:
            period = int(r.get("period"))
            tsec = float(r.get("time_sec"))
            xg = float(r.get("xG", 0.0))
        except Exception:
            continue
        out[mid][(side, period)].append((tsec, max(0.0, xg)))

    for mid in out:
        for key in out[mid]:
            out[mid][key].sort(key=lambda v: v[0])
    return out


def collect_sportec_rows(
    *,
    horizon_seconds: float,
    sportec_shots_path: str,
) -> List[dict]:
    shot_lookup = _load_sportec_shots(sportec_shots_path)
    match_ids = sorted(shot_lookup.keys())
    rows: List[dict] = []

    print(f"[set_piece_pv] Loading Sportec matches: {match_ids}")
    for mid in tqdm(match_ids, desc="Sportec", unit="match"):
        try:
            event_url = _sportec_file_url(mid, "event")
            meta_url = _sportec_file_url(mid, "meta")
            ds = sportec.load_event(
                event_data=event_url,
                meta_data=meta_url,
                coordinates="statsbomb",
            )
        except Exception as e:
            print(f"[set_piece_pv] Sportec match {mid} failed: {e}")
            continue

        seen_event_ids = set()
        for ev in ds.events:
            if ev.event_type.value != "PASS":
                continue
            raw = ev.raw_event if isinstance(ev.raw_event, dict) else {}
            if str(raw.get("FromOpenPlay", "")).lower() != "false":
                continue
            if ev.coordinates is None or ev.team is None:
                continue

            event_id = raw.get("EventId")
            if event_id is not None and event_id in seen_event_ids:
                continue
            if event_id is not None:
                seen_event_ids.add(event_id)

            team_side = str(getattr(ev.team, "ground", "")).lower()
            if team_side not in ("home", "away"):
                continue
            period = int(ev.period.id) if ev.period is not None else None
            if period is None:
                continue

            tsec = _timestamp_to_seconds(ev.timestamp)
            if tsec is None or tsec < 0.0:
                continue

            x = float(ev.coordinates.x)
            y = float(ev.coordinates.y)
            kind = _classify_sportec_restart(x, y)
            if kind is None:
                continue

            shots = shot_lookup.get(mid, {}).get((team_side, period), [])
            shot_xg = _first_shot_xg(shots, tsec, horizon_seconds)
            rows.append(
                {
                    "source": "sportec",
                    "match_id": mid,
                    "kind": kind,
                    "x": x,
                    "shot_xg": shot_xg,
                }
            )

    return rows


def _statsbomb_all_match_ids(seed: int = 42) -> List[int]:
    comps = _json_url(STATSBOMB_COMPETITIONS_URL)
    match_ids = set()
    for c in comps:
        cid = c.get("competition_id")
        sid = c.get("season_id")
        if cid is None or sid is None:
            continue
        url = STATSBOMB_MATCHES_URL_TMPL.format(competition_id=cid, season_id=sid)
        try:
            matches = _json_url(url)
        except Exception:
            continue
        for m in matches:
            mid = m.get("match_id")
            if mid is not None:
                match_ids.add(int(mid))
    ids = sorted(match_ids)
    rng = random.Random(seed)
    rng.shuffle(ids)
    return ids


def collect_statsbomb_rows(
    *,
    horizon_seconds: float,
    max_matches: Optional[int],
    explicit_match_ids: Optional[Sequence[int]],
    seed: int,
) -> List[dict]:
    if explicit_match_ids:
        match_ids = [int(m) for m in explicit_match_ids]
    else:
        all_ids = _statsbomb_all_match_ids(seed=seed)
        match_ids = all_ids[: max_matches or len(all_ids)]

    rows: List[dict] = []
    print(f"[set_piece_pv] Loading StatsBomb matches: {len(match_ids)}")
    for mid in tqdm(match_ids, desc="StatsBomb", unit="match"):
        try:
            ds = statsbomb.load_open_data(match_id=mid, coordinates="statsbomb")
        except Exception:
            continue

        setpieces: List[dict] = []
        shots_by_team_period: Dict[Tuple[str, int], List[Tuple[float, float]]] = defaultdict(list)
        seen_setpiece = set()
        seen_shot = set()

        for ev in ds.events:
            raw = ev.raw_event if isinstance(ev.raw_event, dict) else {}
            et_name = ((raw.get("type") or {}).get("name") or "").strip()
            event_id = raw.get("id")

            period = raw.get("period")
            if period is None and ev.period is not None:
                period = ev.period.id
            if period is None:
                continue
            period = int(period)

            tsec = _timestamp_to_seconds(raw.get("timestamp"))
            if tsec is None:
                tsec = _timestamp_to_seconds(ev.timestamp)
            if tsec is None:
                continue

            team_obj = raw.get("team") or {}
            team_id = str(team_obj.get("id", "")).strip()
            if not team_id:
                continue

            if et_name == "Shot":
                if event_id is not None and event_id in seen_shot:
                    continue
                if event_id is not None:
                    seen_shot.add(event_id)
                shot = raw.get("shot") or {}
                try:
                    xg = float(shot.get("statsbomb_xg", 0.0))
                except Exception:
                    xg = 0.0
                shots_by_team_period[(team_id, period)].append((tsec, max(0.0, xg)))
                continue

            if et_name != "Pass":
                continue
            ptype = (((raw.get("pass") or {}).get("type") or {}).get("name") or "").strip()
            kind = None
            if ptype == "Throw-in":
                kind = "throw_in"
            elif ptype == "Corner":
                kind = "corner"
            elif ptype == "Goal Kick":
                kind = "goal_kick"
            if kind is None:
                continue

            if event_id is not None and event_id in seen_setpiece:
                continue
            if event_id is not None:
                seen_setpiece.add(event_id)

            loc = raw.get("location") or []
            if len(loc) >= 1 and loc[0] is not None:
                x = float(loc[0])
            elif ev.coordinates is not None:
                x = float(ev.coordinates.x)
            else:
                x = float("nan")

            setpieces.append(
                {
                    "kind": kind,
                    "team_id": team_id,
                    "period": period,
                    "time_sec": tsec,
                    "x": x,
                }
            )

        for key in shots_by_team_period:
            shots_by_team_period[key].sort(key=lambda v: v[0])

        for sp in setpieces:
            shot_xg = _first_shot_xg(
                shots_by_team_period.get((sp["team_id"], sp["period"]), []),
                float(sp["time_sec"]),
                horizon_seconds,
            )
            rows.append(
                {
                    "source": "statsbomb",
                    "match_id": str(mid),
                    "kind": sp["kind"],
                    "x": float(sp["x"]),
                    "shot_xg": float(shot_xg),
                }
            )
    return rows


def _kind_stats(rows: Iterable[dict], kind: str) -> dict:
    vals = [r for r in rows if r["kind"] == kind]
    n = len(vals)
    xgs = np.asarray([float(v["shot_xg"]) for v in vals], dtype=np.float64)
    is_shot = xgs > 0.0
    n_shot = int(is_shot.sum())
    shot_rate = float(n_shot / n) if n > 0 else 0.0
    mean_shot_xg = float(xgs[is_shot].mean()) if n_shot > 0 else 0.0
    value_xg = float(xgs.mean()) if n > 0 else 0.0
    return {
        "n_samples": n,
        "n_shots": n_shot,
        "shot_rate": shot_rate,
        "mean_shot_xg": mean_shot_xg,
        "value_xg": value_xg,
    }


def train_and_save(
    *,
    rows: List[dict],
    out_dir: Path,
    horizon_seconds: float,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    throw_rows = [r for r in rows if r["kind"] == "throw_in" and np.isfinite(r["x"])]
    if not throw_rows:
        raise RuntimeError("No throw-in rows found; cannot train logistic regression.")

    x_throw = np.asarray([float(r["x"]) for r in throw_rows], dtype=np.float64).reshape(-1, 1)
    y_throw = np.asarray([1.0 if float(r["shot_xg"]) > 0.0 else 0.0 for r in throw_rows], dtype=np.float64)

    n_pos = int(y_throw.sum())
    fallback_prob = float(y_throw.mean())
    mean_shot_xg = float(
        np.mean([float(r["shot_xg"]) for r in throw_rows if float(r["shot_xg"]) > 0.0])
    ) if n_pos > 0 else 0.0

    if n_pos > 0 and n_pos < len(y_throw):
        throw_lr = LogisticRegression(solver="lbfgs", max_iter=500)
        throw_lr.fit(x_throw, y_throw)
        intercept = float(throw_lr.intercept_[0])
        coef_x = float(throw_lr.coef_[0, 0])
        joblib.dump(throw_lr, out_dir / "throw_in_logreg.joblib")
    else:
        throw_lr = None
        intercept = None
        coef_x = None

    corner_stats = _kind_stats(rows, "corner")
    goal_kick_stats = _kind_stats(rows, "goal_kick")

    by_source = defaultdict(lambda: defaultdict(int))
    for r in rows:
        by_source[r["source"]]["total"] += 1
        by_source[r["source"]][r["kind"]] += 1

    model_payload = {
        "version": 1,
        "horizon_seconds": float(horizon_seconds),
        "models": {
            "throw_in": {
                "type": "logistic_x_only",
                "intercept": intercept,
                "coef_x": coef_x,
                "fallback_prob": fallback_prob,
                "mean_shot_xg": mean_shot_xg,
                "n_samples": int(len(throw_rows)),
                "n_shots": int(n_pos),
            },
            "corner": {
                "type": "empirical_mean_xg",
                **corner_stats,
            },
            "goal_kick": {
                "type": "empirical_mean_xg",
                **goal_kick_stats,
            },
        },
        "summary": {
            "n_rows_total": int(len(rows)),
            "by_source": {k: dict(v) for k, v in by_source.items()},
        },
    }

    with (out_dir / "model.json").open("w") as f:
        json.dump(model_payload, f, indent=2)

    print("[set_piece_pv] Saved:", out_dir / "model.json")
    print("[set_piece_pv] Throw-in:",
          f"n={len(throw_rows)} shots={n_pos} shot_rate={fallback_prob:.4f} "
          f"mean_shot_xg={mean_shot_xg:.4f} "
          f"coef_x={coef_x if coef_x is not None else 'None'}")
    print("[set_piece_pv] Corner:",
          f"n={corner_stats['n_samples']} shots={corner_stats['n_shots']} "
          f"value_xg={corner_stats['value_xg']:.4f}")
    print("[set_piece_pv] Goal-kick:",
          f"n={goal_kick_stats['n_samples']} shots={goal_kick_stats['n_shots']} "
          f"value_xg={goal_kick_stats['value_xg']:.4f}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--out-dir",
        default="/mnt/data/remains/opta2026/possessionValue/checkpoints/set_piece_pv_v1",
    )
    p.add_argument(
        "--sportec-shots",
        default="/mnt/data/remains/opta2026/possessionValue/sportec_shots.json",
    )
    p.add_argument("--horizon-seconds", type=float, default=10.0)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--no-sportec", action="store_true")
    p.add_argument("--no-statsbomb", action="store_true")
    p.add_argument("--statsbomb-max-matches", type=int, default=300)
    p.add_argument(
        "--statsbomb-match-ids",
        type=str,
        default="",
        help="Optional comma-separated match ids. If set, overrides --statsbomb-max-matches.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    rows: List[dict] = []

    if not args.no_sportec:
        sportec_rows = collect_sportec_rows(
            horizon_seconds=float(args.horizon_seconds),
            sportec_shots_path=args.sportec_shots,
        )
        rows.extend(sportec_rows)
        print(f"[set_piece_pv] Sportec rows: {len(sportec_rows)}")

    if not args.no_statsbomb:
        explicit_ids = None
        if args.statsbomb_match_ids.strip():
            explicit_ids = [
                int(v.strip())
                for v in args.statsbomb_match_ids.split(",")
                if v.strip()
            ]
        statsbomb_rows = collect_statsbomb_rows(
            horizon_seconds=float(args.horizon_seconds),
            max_matches=None if explicit_ids else int(args.statsbomb_max_matches),
            explicit_match_ids=explicit_ids,
            seed=int(args.seed),
        )
        rows.extend(statsbomb_rows)
        print(f"[set_piece_pv] StatsBomb rows: {len(statsbomb_rows)}")

    if not rows:
        raise RuntimeError("No rows collected. Check data/network configuration.")

    train_and_save(
        rows=rows,
        out_dir=Path(args.out_dir),
        horizon_seconds=float(args.horizon_seconds),
    )


if __name__ == "__main__":
    main()
