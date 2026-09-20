from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple


CSV_COLUMNS: List[str] = [
    "clip_idx",
    "clip_path",
    "source_file",
    "match_id",
    "kick_frame_local_refined",
    "variant_id",
    "variant_group",
    "seed",
    "v0x",
    "v0y",
    "v0z",
    "speed_xy",
    "speed_3d",
    "spin_scalar",
    "observed_fit_rmse_xy",
    "observed_fit_rmse_z",
    "observed_fit_near_zero",
    "observed_fit_endpoint_xy_err",
    "observed_fit_endpoint_z_err",
    "observed_fit_endpoint_enforced",
    "attacking_side_at_kick",
    "defending_side_at_kick",
    "home_team_label",
    "away_team_label",
    "passer_entity_idx",
    "passer_team_side",
    "passer_player_name",
    "passer_player_id",
    "passer_position_label",
    "passer_inferred_band",
    "passer_inferred_lateral_band",
    "passer_formation_slot_idx",
    "touch_found",
    "touch_frame_rel",
    "touch_entity_idx",
    "touch_team_side",
    "touch_player_name",
    "touch_position_label",
    "touch_inferred_band",
    "touch_inferred_lateral_band",
    "touch_formation_slot_idx",
    "receiver_proxy_entity_idx",
    "receiver_proxy_team_side",
    "receiver_proxy_player_name",
    "receiver_proxy_position_label",
    "receiver_proxy_inferred_band",
    "receiver_proxy_inferred_lateral_band",
    "receiver_proxy_formation_slot_idx",
    "ball_out",
    "out_frame_rel",
    "restart_kind",
    "taking_side",
    "restart_reason",
    "score_frame_rel",
    "score_mode",
    "pv_home",
    "pv_away",
    "pv_net",
    "pre_pass_pv_home",
    "pre_pass_pv_away",
    "pre_pass_pv_net",
    "pv_added_vs_prepass",
    "observed_pv_added_vs_prepass_for_clip",
    "delta_pv_added_vs_observed",
    "observed_pv_net_for_clip",
    "delta_vs_observed",
    "is_observed_row",
    "player_fallback_gt",
    "status_code",
    "error_msg",
]


def load_done_keys(csv_path: str | Path) -> Set[Tuple[int, str]]:
    p = Path(csv_path)
    if not p.exists():
        return set()
    done: Set[Tuple[int, str]] = set()
    with p.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                k = (int(row["clip_idx"]), str(row["variant_id"]))
            except Exception:
                continue
            done.add(k)
    return done


class CsvStreamWriter:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        file_exists = self.path.exists() and self.path.stat().st_size > 0
        self._f = self.path.open("a", newline="")
        self._writer = csv.DictWriter(self._f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        if not file_exists:
            self._writer.writeheader()

    def write_rows(self, rows: Iterable[Dict]) -> int:
        n = 0
        for r in rows:
            self._writer.writerow(r)
            n += 1
        return n

    def flush(self) -> None:
        self._f.flush()

    def close(self) -> None:
        self._f.flush()
        self._f.close()
