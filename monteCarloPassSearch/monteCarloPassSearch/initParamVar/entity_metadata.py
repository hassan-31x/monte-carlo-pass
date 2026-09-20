from __future__ import annotations

from typing import Dict, Optional

import numpy as np


def _team_side(et: int) -> Optional[str]:
    if int(et) == 0:
        return "home"
    if int(et) == 1:
        return "away"
    return None


def _quantile_band(v: float, q1: float, q2: float, low: str, mid: str, high: str) -> str:
    if not np.isfinite(v):
        return "unknown"
    if v <= q1:
        return low
    if v >= q2:
        return high
    return mid


def build_clip_entity_table(
    entity_type: np.ndarray,
    player_pos_at_kick: np.ndarray,
) -> Dict[int, dict]:
    """Build stable per-entity metadata for CSV enrichment.

    Names are unavailable in the current Sportec clip tensors, so this table
    provides deterministic entity/team/role proxies.
    """
    et = np.asarray(entity_type).astype(np.int64)
    pos = np.asarray(player_pos_at_kick).astype(np.float32)

    out: Dict[int, dict] = {}
    for n in range(len(et)):
        side = _team_side(int(et[n]))
        rec = {
            "entity_idx": int(n),
            "team_side": side,
            "entity_type": int(et[n]),
            "player_name": None,
            "player_position_label": None,
            "formation_slot_idx": None,
            "inferred_position_band": "unknown",
            "inferred_lateral_band": "unknown",
        }
        out[int(n)] = rec

    for side in ("home", "away"):
        idx = [i for i in range(len(et)) if _team_side(int(et[i])) == side]
        if not idx:
            continue
        x = np.asarray([pos[i, 0] for i in idx], dtype=np.float32)
        y = np.asarray([pos[i, 1] for i in idx], dtype=np.float32)
        qx1, qx2 = np.quantile(x, [1.0 / 3.0, 2.0 / 3.0]) if len(x) >= 3 else (np.min(x), np.max(x))
        qy1, qy2 = np.quantile(y, [1.0 / 3.0, 2.0 / 3.0]) if len(y) >= 3 else (np.min(y), np.max(y))

        order_y = np.argsort(y)
        for rank, j in enumerate(order_y):
            n = int(idx[int(j)])
            out[n]["formation_slot_idx"] = int(rank)
            pos_band = _quantile_band(float(x[int(j)]), float(qx1), float(qx2), "defensive", "middle", "attacking")
            lat_band = _quantile_band(float(y[int(j)]), float(qy1), float(qy2), "left", "center", "right")
            out[n]["inferred_position_band"] = pos_band
            out[n]["inferred_lateral_band"] = lat_band
            out[n]["player_position_label"] = f"{pos_band}_{lat_band}"

    return out

