#!/usr/bin/env python3
"""Export Sportec shot events in the format expected by PV preprocessing."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path


def download_url(url: str) -> str:
    match = re.search(r"/files/(\d+)$", str(url))
    if match:
        return f"https://api.figshare.com/v2/file/download/{match.group(1)}"
    return str(url)


def side(team: object) -> str:
    ground = str(getattr(team, "ground", "")).lower()
    if ground in {"home", "away"}:
        return ground
    team_id = str(getattr(team, "team_id", "")).lower()
    if "home" in team_id:
        return "home"
    if "away" in team_id:
        return "away"
    return "unknown"


def first_number(raw: dict, names: tuple[str, ...], default: float | None = None) -> float | None:
    lowered = {str(key).lower(): value for key, value in raw.items()}
    for name in names:
        value = lowered.get(name.lower())
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            return number
    return default


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--match-id", default="J03WN1")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    from kloppy import sportec
    from kloppy._providers.sportec import get_IDSSE_url

    meta = download_url(get_IDSSE_url(args.match_id, "meta"))
    events = download_url(get_IDSSE_url(args.match_id, "event"))
    dataset = sportec.load_event(event_data=events, meta_data=meta)

    output = []
    for event in dataset.events:
        event_name = str(getattr(getattr(event, "event_type", None), "name", "")).upper()
        raw = event.raw_event if isinstance(event.raw_event, dict) else {}
        raw_text = " ".join(str(value).upper() for value in raw.values())
        if "SHOT" not in event_name and "SHOT" not in raw_text and "GOAL" not in event_name:
            continue
        calculated_frame = first_number(raw, ("CalculatedFrame", "Frame", "N"))
        if calculated_frame is None:
            continue
        xg = first_number(raw, ("xG", "ExpectedGoals", "GoalExpectancy"), 0.05)
        timestamp = getattr(event, "timestamp", None)
        time_sec = float(timestamp.total_seconds()) if timestamp is not None else math.nan
        output.append({
            "match_id": args.match_id,
            "CalculatedFrame": int(calculated_frame),
            "xG": float(xg),
            "side": side(getattr(event, "team", None)),
            "period": int(getattr(getattr(event, "period", None), "id", 1) or 1),
            "time_sec": time_sec,
            "result": event_name,
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
    print(json.dumps({"match_id": args.match_id, "shots": len(output), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
