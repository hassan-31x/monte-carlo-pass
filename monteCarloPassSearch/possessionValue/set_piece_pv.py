#!/usr/bin/env python3
"""Set-piece possession value model.

Throw-ins:
  - Logistic regression on static horizontal ball coordinate only.
  - Output PV = P(shot within 10s) * mean_xG_given_shot.

Corners / Goal-kicks:
  - Counting-based expected xG within 10s.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

PITCH_X_METRES = 105.0
HALF_X = PITCH_X_METRES / 2.0
STATSBOMB_X_MAX = 120.0


@dataclass
class ThrowInModel:
    intercept: Optional[float]
    coef_x: Optional[float]
    mean_shot_xg: float
    fallback_prob: float
    n_samples: int
    n_shots: int

    def predict_prob_shot(self, x_statsbomb: Optional[float]) -> float:
        if x_statsbomb is None or self.intercept is None or self.coef_x is None:
            return float(self.fallback_prob)
        z = float(self.intercept) + float(self.coef_x) * float(x_statsbomb)
        # Numerically stable sigmoid.
        if z >= 0:
            ez = math.exp(-z)
            return float(1.0 / (1.0 + ez))
        ez = math.exp(z)
        return float(ez / (1.0 + ez))

    def predict_xg(self, x_statsbomb: Optional[float]) -> float:
        p_shot = self.predict_prob_shot(x_statsbomb)
        return float(max(0.0, p_shot) * max(0.0, self.mean_shot_xg))


@dataclass
class EmpiricalModel:
    value_xg: float
    n_samples: int
    n_shots: int
    shot_rate: float
    mean_shot_xg: float

    def predict_xg(self) -> float:
        return float(max(0.0, self.value_xg))


class SetPiecePVModel:
    def __init__(self, payload: dict):
        models = payload.get("models", {})
        throw = models.get("throw_in", {})
        corner = models.get("corner", {})
        goal_kick = models.get("goal_kick", {})

        self.payload = payload
        self.throw_in = ThrowInModel(
            intercept=throw.get("intercept"),
            coef_x=throw.get("coef_x"),
            mean_shot_xg=float(throw.get("mean_shot_xg", 0.0)),
            fallback_prob=float(throw.get("fallback_prob", 0.0)),
            n_samples=int(throw.get("n_samples", 0)),
            n_shots=int(throw.get("n_shots", 0)),
        )
        self.corner = EmpiricalModel(
            value_xg=float(corner.get("value_xg", 0.0)),
            n_samples=int(corner.get("n_samples", 0)),
            n_shots=int(corner.get("n_shots", 0)),
            shot_rate=float(corner.get("shot_rate", 0.0)),
            mean_shot_xg=float(corner.get("mean_shot_xg", 0.0)),
        )
        self.goal_kick = EmpiricalModel(
            value_xg=float(goal_kick.get("value_xg", 0.0)),
            n_samples=int(goal_kick.get("n_samples", 0)),
            n_shots=int(goal_kick.get("n_shots", 0)),
            shot_rate=float(goal_kick.get("shot_rate", 0.0)),
            mean_shot_xg=float(goal_kick.get("mean_shot_xg", 0.0)),
        )
        self.horizon_seconds = float(payload.get("horizon_seconds", 10.0))

    def predict_expected_xg(
        self,
        kind: str,
        *,
        ball_x_statsbomb: Optional[float] = None,
    ) -> float:
        k = str(kind).strip().lower().replace("-", "_")
        if k == "throw_in":
            return self.throw_in.predict_xg(ball_x_statsbomb)
        if k == "corner":
            return self.corner.predict_xg()
        if k == "goal_kick":
            return self.goal_kick.predict_xg()
        return 0.0

    def predict_team_values(
        self,
        *,
        kind: str,
        taking_side: str,
        ball_x_statsbomb: Optional[float] = None,
    ) -> Tuple[float, float, float]:
        """Return (pv_home, pv_away, pv_net=home-away)."""
        v = float(self.predict_expected_xg(kind, ball_x_statsbomb=ball_x_statsbomb))
        side = str(taking_side).strip().lower()
        if side == "away":
            home, away = 0.0, v
        else:
            home, away = v, 0.0
        return home, away, home - away


def meters_x_to_statsbomb_x(x_m: float) -> float:
    """Convert centred metres x in [-52.5, 52.5] to StatsBomb x in [0, 120]."""
    x = ((float(x_m) + HALF_X) / PITCH_X_METRES) * STATSBOMB_X_MAX
    return float(min(STATSBOMB_X_MAX, max(0.0, x)))


def load_set_piece_pv_model(path: str | Path) -> SetPiecePVModel:
    p = Path(path)
    with p.open("r") as f:
        payload = json.load(f)
    return SetPiecePVModel(payload)
