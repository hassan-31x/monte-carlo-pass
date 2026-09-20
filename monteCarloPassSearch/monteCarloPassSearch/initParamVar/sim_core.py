from __future__ import annotations

import json
import math
import os
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from entity_metadata import build_clip_entity_table
from variant_sampling import (
    VariationCaps,
    infer_observed_variant,
    infer_observed_variant_fitted,
    sample_global_variants,
    sample_local_variants,
)

_THIS_DIR = Path(__file__).resolve().parent
_OPTA_DIR = _THIS_DIR.parent.parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from sim_viz_runtime import (  # noqa: E402
    AIR_RESISTANCE,
    FPS,
    GRAVITY,
    GROUND_FRICTION,
    H_HIST,
    HALF_X,
    HALF_Y,
    K_TOUCH,
    RESTITUTION,
    ROLLING_FRICTION,
    ROLLING_Z_THRESH,
    VEL_SCALE,
    _apply_ptt_classifier_constraints,
    _apply_ptt_survival_constraints,
    _ball_feats,
    _ptt_geometry_valid_mask,
    refine_kick_frame_local,
    smart_predict_positions,
    try_load_bat_model,
    try_load_pv_model,
    try_load_set_piece_pv_model,
    try_load_smart_model,
    try_load_touch_model,
)
from set_piece_pv import meters_x_to_statsbomb_x  # noqa: E402


TOUCH_SKIP = 5
SPIN_ACCEL_COEF = 0.14


@dataclass
class RunnerConfig:
    context_len: int
    rollout_len: int
    physics_len: int
    eval_offset_after_touch: int
    touch_threshold: float
    local_variants: int
    global_variants: int
    include_observed: bool
    observed_use_gt_players: bool
    require_near_zero_fit: bool
    seed: int


@dataclass
class ModelPaths:
    smart_checkpoint: str
    vocab_dir: str
    touch_checkpoint: str
    bat_checkpoint: str
    pv_checkpoint: str
    set_piece_pv_model: str


@dataclass
class WorkerState:
    gpu_id: int
    device: torch.device
    config: RunnerConfig
    model_paths: ModelPaths
    smart_result: Optional[tuple]
    touch_result: Optional[tuple]
    bat_result: Optional[tuple]
    pv_result: Optional[tuple]
    set_piece_model: Optional[object]


def _team_side(entity_type: np.ndarray, entity_idx: Optional[int]) -> Optional[str]:
    if entity_idx is None:
        return None
    if int(entity_idx) < 0 or int(entity_idx) >= len(entity_type):
        return None
    et = int(entity_type[int(entity_idx)])
    if et == 0:
        return "home"
    if et == 1:
        return "away"
    return None


def _opposite_side(side: Optional[str]) -> Optional[str]:
    if side == "home":
        return "away"
    if side == "away":
        return "home"
    return None


def _parse_match_id(source_file: str) -> str:
    if ":" in str(source_file):
        return str(source_file).split(":", 1)[1]
    return str(source_file)


def _clip_features_to_metres(features: np.ndarray) -> np.ndarray:
    out = features.astype(np.float32).copy()
    xy_range = float(out[:, :, :2].max() - out[:, :, :2].min())
    if xy_range < 2.0:
        out[:, :, 0] = (out[:, :, 0] - 0.5) * 105.0
        out[:, :, 1] = (out[:, :, 1] - 0.5) * 68.0
    return out


def _simulate_ball_with_spin(
    start_pos: np.ndarray,
    start_vel: np.ndarray,
    spin_scalar: float,
    n_frames: int,
) -> np.ndarray:
    dt = 1.0 / FPS
    pos = start_pos.astype(np.float64).copy()
    vel = start_vel.astype(np.float64).copy()
    arr = np.zeros((n_frames, 3), dtype=np.float32)

    for i in range(n_frames):
        arr[i] = pos.astype(np.float32)

        speed_xy = float(np.linalg.norm(vel[:2]))
        if speed_xy > 1e-6 and abs(float(spin_scalar)) > 1e-8:
            perp = np.asarray([-vel[1], vel[0]], dtype=np.float64) / speed_xy
            a_spin = float(spin_scalar) * SPIN_ACCEL_COEF * speed_xy
            vel[0] += perp[0] * a_spin * dt
            vel[1] += perp[1] * a_spin * dt

        vel *= (1.0 - AIR_RESISTANCE)
        if pos[2] < ROLLING_Z_THRESH and abs(vel[2]) < 1.0:
            vel[0] *= (1.0 - ROLLING_FRICTION)
            vel[1] *= (1.0 - ROLLING_FRICTION)

        vel[2] -= GRAVITY * dt
        pos += vel * dt

        if pos[2] < 0:
            pos[2] = 0.0
            vel[2] = -RESTITUTION * vel[2]
            vel[0] *= (1.0 - GROUND_FRICTION)
            vel[1] *= (1.0 - GROUND_FRICTION)

    return arr


def _ball_vel_from_positions(ball_pos: np.ndarray, start_vel: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = ball_pos.shape[0]
    vxy = np.zeros((n, 2), dtype=np.float32)
    vz = np.zeros((n,), dtype=np.float32)
    vxy[0] = start_vel[:2]
    vz[0] = float(start_vel[2])
    if n > 1:
        vxy[1:] = (ball_pos[1:, :2] - ball_pos[:-1, :2]) * FPS
        vz[1:] = (ball_pos[1:, 2] - ball_pos[:-1, 2]) * FPS
    return vxy, vz


def _get_player_history_features(
    full_player_pos: np.ndarray,
    full_player_vel: np.ndarray,
    abs_idx: int,
    player_idx: int,
    ball_pos: np.ndarray,
) -> List[float]:
    feats: List[float] = []
    for h in range(H_HIST):
        t = abs_idx - (H_HIST - 1) + h
        if t < 0:
            feats.extend([0.0, 0.0, 0.0, 0.0])
            continue
        t = int(min(t, full_player_pos.shape[0] - 1))
        pp = full_player_pos[t, player_idx]
        pv = full_player_vel[t, player_idx]
        feats.extend(
            [
                (float(pp[0]) - float(ball_pos[0])) / HALF_X,
                (float(pp[1]) - float(ball_pos[1])) / HALF_Y,
                float(pv[0]) / VEL_SCALE if np.isfinite(pv[0]) else 0.0,
                float(pv[1]) / VEL_SCALE if np.isfinite(pv[1]) else 0.0,
            ]
        )
    return feats


def _predict_touch(
    *,
    state: WorkerState,
    entity_type: np.ndarray,
    player_mask: np.ndarray,
    passer_entity: int,
    pre_ball_pos: np.ndarray,
    pre_ball_vel_xy: np.ndarray,
    pre_ball_vz: np.ndarray,
    full_player_pos: np.ndarray,
    full_player_vel: np.ndarray,
    ctx_len_actual: int,
    sim_len: int,
) -> Tuple[Optional[int], Optional[int], Optional[str]]:
    if state.touch_result is None:
        return None, None, None

    touch_model, _touch_args, touch_model_type, ptt_ball_dim = state.touch_result
    device = state.device
    anchor = pre_ball_pos[0].copy()
    survival_prob = 1.0

    for t in range(min(sim_len + 1, pre_ball_pos.shape[0])):
        bp = pre_ball_pos[t]
        # Once the ball leaves the field, no legal touch can occur.
        if abs(float(bp[0])) > HALF_X or abs(float(bp[1])) > HALF_Y:
            return None, None, None
        bv = pre_ball_vel_xy[t]
        bvz = pre_ball_vz[t]
        if np.any(np.isnan(bp)):
            continue

        player_indices = np.where(player_mask)[0]
        non_passer = player_indices[player_indices != int(passer_entity)]
        if len(non_passer) == 0:
            non_passer = player_indices
        player_pos_t = full_player_pos[min(ctx_len_actual + t, full_player_pos.shape[0] - 1), non_passer, :]
        dists = np.linalg.norm(player_pos_t[:, :2] - bp[None, :2], axis=1)
        knn_local = np.argsort(dists)[:K_TOUCH]
        knn_global = non_passer[knn_local]
        if len(knn_global) == 0:
            continue
        if len(knn_global) < K_TOUCH:
            knn_global = np.concatenate([knn_global, np.full(K_TOUCH - len(knn_global), knn_global[-1])])
        knn_pos = full_player_pos[min(ctx_len_actual + t, full_player_pos.shape[0] - 1), knn_global, :]
        valid_mask = _ptt_geometry_valid_mask(bp, knn_pos)
        # Skip expensive model calls when no candidate can physically touch.
        if not bool(np.any(valid_mask)):
            continue

        bf = torch.from_numpy(
            _ball_feats(
                bp,
                bv,
                anchor,
                ball_z=float(bp[2]),
                ball_vz=float(bvz),
            )
        ).unsqueeze(0).to(device)

        abs_idx = ctx_len_actual + t
        pf = torch.tensor(
            [_get_player_history_features(full_player_pos, full_player_vel, abs_idx, int(n_idx), bp) for n_idx in knn_global],
            dtype=torch.float32,
        ).unsqueeze(0).to(device)

        with torch.no_grad():
            bf_ptt = bf[:, :ptt_ball_dim]
            if touch_model_type == "survival":
                out = touch_model(bf_ptt, pf)
                h_t = torch.sigmoid(out["hazard"]).squeeze().item()
                p_logits = out["player_logits"].squeeze(0)
                p_probs = F.softmax(p_logits, dim=-1).cpu().numpy()
                h_eff, p_probs_eff, _ = _apply_ptt_survival_constraints(h_t, p_probs, valid_mask)
                survival_prob *= (1.0 - h_eff)
                cum_touch_prob = 1.0 - survival_prob
                if t >= TOUCH_SKIP and cum_touch_prob >= state.config.touch_threshold and float(np.sum(p_probs_eff)) > 1e-12:
                    local_idx = int(np.argmax(p_probs_eff))
                    toucher = int(knn_global[local_idx])
                    return t, toucher, _team_side(entity_type, toucher)
            else:
                logits = touch_model(bf_ptt, pf).squeeze(0)
                probs_raw = F.softmax(logits, dim=-1).cpu().numpy()
                probs = _apply_ptt_classifier_constraints(probs_raw, valid_mask)
                max_player_prob = float(np.max(probs[:K_TOUCH])) if probs.shape[0] > 0 else 0.0
                if t >= TOUCH_SKIP and max_player_prob >= state.config.touch_threshold:
                    local_idx = int(np.argmax(probs[:K_TOUCH]))
                    toucher = int(knn_global[local_idx])
                    return t, toucher, _team_side(entity_type, toucher)

    return None, None, None


def _predict_bat_velocity(
    *,
    state: WorkerState,
    touch_frame_rel: int,
    touch_player_n: int,
    ball_pos_touch: np.ndarray,
    ball_vel_touch: np.ndarray,
    ball_vz_touch: float,
    anchor_at_touch: np.ndarray,
    full_player_pos: np.ndarray,
    full_player_vel: np.ndarray,
    abs_touch_idx: int,
    player_mask: np.ndarray,
    n_entities: int,
) -> Optional[np.ndarray]:
    if state.bat_result is None:
        return None
    bat_model, _bat_args, k_other, bat_model_type, bat_ball_dim = state.bat_result
    device = state.device

    bf = torch.from_numpy(
        _ball_feats(
            ball_pos_touch,
            ball_vel_touch,
            anchor_at_touch,
            ball_z=float(ball_pos_touch[2]),
            ball_vz=float(ball_vz_touch),
        )
    ).unsqueeze(0).to(device)

    def _hist(n_idx: int) -> List[float]:
        return _get_player_history_features(full_player_pos, full_player_vel, abs_touch_idx, n_idx, ball_pos_touch)

    sf = torch.tensor([_hist(int(touch_player_n))], dtype=torch.float32).to(device)

    player_indices = np.where(player_mask)[0]
    other = player_indices[player_indices != int(touch_player_n)]
    if len(other) == 0:
        other = player_indices
    pos_t = full_player_pos[min(abs_touch_idx, full_player_pos.shape[0] - 1), other, :]
    dists = np.linalg.norm(pos_t[:, :2] - ball_pos_touch[None, :2], axis=1)
    other_nn = other[np.argsort(dists)[:k_other]]
    if len(other_nn) == 0:
        return None
    if len(other_nn) < k_other:
        other_nn = np.concatenate([other_nn, np.full(k_other - len(other_nn), other_nn[-1])])
    of = torch.tensor([_hist(int(n)) for n in other_nn], dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        raw_out = bat_model(bf[:, :bat_ball_dim], sf, of)

    if bat_model_type == "gaussian":
        vel_pred_norm = raw_out["mu"].squeeze(0).cpu().numpy()
    else:
        vel_pred_norm = raw_out.squeeze(0).cpu().numpy()
    vel_pred_ms = vel_pred_norm.astype(np.float32) * VEL_SCALE

    if abs(float(vel_pred_ms[2])) < 0.5:
        pass_dist = float(np.linalg.norm(vel_pred_ms[:2])) / FPS * 25.0
        vel_pred_ms[2] = max(0.5, pass_dist * 0.05)
    return vel_pred_ms.astype(np.float32)


def _compute_pv_numeric(
    *,
    pv_result: Optional[tuple],
    entity_type: np.ndarray,
    player_pos_roll: np.ndarray,
    ball_seq: np.ndarray,
    ball_start_vel: np.ndarray,
    device: torch.device,
) -> Tuple[float, float, float]:
    if pv_result is None:
        return math.nan, math.nan, math.nan

    pv_model, pv_meta = pv_result
    L = int(ball_seq.shape[0])
    N = int(player_pos_roll.shape[1])
    ball_idx_arr = np.where(entity_type == 2)[0]
    if len(ball_idx_arr) == 0:
        return math.nan, math.nan, math.nan
    ball_idx = int(ball_idx_arr[0])

    pos_seq = np.zeros((L, N, 3), dtype=np.float32)
    for t in range(L):
        pf = min(t, player_pos_roll.shape[0] - 1)
        pos_seq[t] = player_pos_roll[pf]
        pos_seq[t, ball_idx] = ball_seq[t]

    feat_seq = np.zeros((L, N, 6), dtype=np.float32)
    feat_seq[:, :, 0:2] = pos_seq[:, :, 0:2]
    feat_seq[:, :, 4] = pos_seq[:, :, 2]
    if L > 1:
        feat_seq[1:, :, 2] = (pos_seq[1:, :, 0] - pos_seq[:-1, :, 0]) * FPS
        feat_seq[1:, :, 3] = (pos_seq[1:, :, 1] - pos_seq[:-1, :, 1]) * FPS
        feat_seq[1:, :, 5] = (pos_seq[1:, :, 2] - pos_seq[:-1, :, 2]) * FPS
    feat_seq[0, ball_idx, 2] = float(ball_start_vel[0])
    feat_seq[0, ball_idx, 3] = float(ball_start_vel[1])
    feat_seq[0, ball_idx, 5] = float(ball_start_vel[2])

    mask_seq = np.ones((L, N), dtype=np.float32)
    mask_seq[:, entity_type == 3] = 0.0

    W = int(pv_meta.get("window_size", 64))
    if L >= W:
        feat_win = feat_seq[-W:]
        mask_win = mask_seq[-W:]
    else:
        pad = W - L
        feat_win = np.concatenate([np.repeat(feat_seq[:1], pad, axis=0), feat_seq], axis=0)
        mask_win = np.concatenate([np.repeat(mask_seq[:1], pad, axis=0), mask_seq], axis=0)

    feat_t = torch.from_numpy(feat_win).unsqueeze(0).to(device)
    mask_t = torch.from_numpy(mask_win).unsqueeze(0).to(device)
    et_t = torch.from_numpy(entity_type.astype(np.int64)).to(device)
    with torch.no_grad():
        out = pv_model(feat_t, mask_t, et_t)
    return float(out["pv_home"].item()), float(out["pv_away"].item()), float(out["pv"].item())


def _compute_pre_pass_pv(
    *,
    pv_result: Optional[tuple],
    features_m: np.ndarray,
    entity_type: np.ndarray,
    ball_idx: int,
    kfl: int,
    device: torch.device,
) -> Tuple[float, float, float]:
    """Compute PV at the pre-pass state (kick anchor frame)."""
    if pv_result is None:
        return math.nan, math.nan, math.nan
    T, N, _ = features_m.shape
    kk = int(np.clip(int(kfl), 0, max(0, T - 1)))

    player_pos_roll = np.zeros((1, N, 3), dtype=np.float32)
    player_pos_roll[0, :, 0:2] = features_m[kk, :, 0:2]
    if features_m.shape[-1] >= 5:
        player_pos_roll[0, :, 2] = features_m[kk, :, 4]

    ball_seq = player_pos_roll[:, ball_idx, :].copy()
    ball_start_vel = np.zeros((3,), dtype=np.float32)

    if kk > 0:
        p0 = np.zeros((3,), dtype=np.float32)
        p1 = np.zeros((3,), dtype=np.float32)
        p0[:2] = features_m[kk - 1, ball_idx, :2]
        p1[:2] = features_m[kk, ball_idx, :2]
        if features_m.shape[-1] >= 5:
            p0[2] = features_m[kk - 1, ball_idx, 4]
            p1[2] = features_m[kk, ball_idx, 4]
        if np.all(np.isfinite(p0)) and np.all(np.isfinite(p1)):
            ball_start_vel = (p1 - p0) * FPS
    elif kk + 1 < T:
        p0 = np.zeros((3,), dtype=np.float32)
        p1 = np.zeros((3,), dtype=np.float32)
        p0[:2] = features_m[kk, ball_idx, :2]
        p1[:2] = features_m[kk + 1, ball_idx, :2]
        if features_m.shape[-1] >= 5:
            p0[2] = features_m[kk, ball_idx, 4]
            p1[2] = features_m[kk + 1, ball_idx, 4]
        if np.all(np.isfinite(p0)) and np.all(np.isfinite(p1)):
            ball_start_vel = (p1 - p0) * FPS

    return _compute_pv_numeric(
        pv_result=pv_result,
        entity_type=entity_type,
        player_pos_roll=player_pos_roll,
        ball_seq=ball_seq,
        ball_start_vel=ball_start_vel,
        device=device,
    )


def _detect_oob(ball_seq: np.ndarray) -> Tuple[Optional[int], Optional[np.ndarray]]:
    for i in range(ball_seq.shape[0]):
        p = ball_seq[i]
        if abs(float(p[0])) > HALF_X or abs(float(p[1])) > HALF_Y:
            return i, p
    return None, None


def _estimate_team_directions(
    entity_type: np.ndarray,
    player_pos_at_kick: np.ndarray,
) -> Dict[str, int]:
    home_idx = np.where(entity_type == 0)[0]
    away_idx = np.where(entity_type == 1)[0]
    if len(home_idx) == 0 or len(away_idx) == 0:
        return {"home_attack_dir": 1, "away_attack_dir": -1}
    home_x = float(np.nanmean(player_pos_at_kick[home_idx, 0]))
    away_x = float(np.nanmean(player_pos_at_kick[away_idx, 0]))
    home_attack = 1 if home_x <= away_x else -1
    return {"home_attack_dir": int(home_attack), "away_attack_dir": int(-home_attack)}


def _infer_restart_from_oob(
    *,
    exit_pos: np.ndarray,
    last_touch_side: Optional[str],
    team_dirs: Dict[str, int],
) -> Tuple[str, Optional[str], str]:
    x = float(exit_pos[0])
    y = float(exit_pos[1])
    if abs(y) > HALF_Y:
        taking = _opposite_side(last_touch_side)
        return "throw_in", taking, "sideline_exit"

    if abs(x) <= HALF_X:
        return "throw_in", _opposite_side(last_touch_side), "fallback_non_boundary"

    exit_sign = 1 if x >= 0 else -1
    home_own_goal_sign = -int(team_dirs.get("home_attack_dir", 1))
    away_own_goal_sign = -int(team_dirs.get("away_attack_dir", -1))
    if home_own_goal_sign == exit_sign:
        defending = "home"
    elif away_own_goal_sign == exit_sign:
        defending = "away"
    else:
        defending = "home"

    if last_touch_side == defending:
        return "corner", _opposite_side(defending), "goal_line_exit_last_touch_defender"
    return "goal_kick", defending, "goal_line_exit_last_touch_attacker"


def _angle_delta_deg(v_prev: np.ndarray, v_cur: np.ndarray) -> float:
    n1 = float(np.linalg.norm(v_prev))
    n2 = float(np.linalg.norm(v_cur))
    if n1 <= 1e-8 or n2 <= 1e-8:
        return 0.0
    c = float(np.dot(v_prev, v_cur) / (n1 * n2))
    c = float(np.clip(c, -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def _detect_gt_touch(
    *,
    entity_type: np.ndarray,
    ball_pos_roll: np.ndarray,
    passer_entity: int,
    player_roll_pos: np.ndarray,
    min_t: int = TOUCH_SKIP,
    hard_dist: float = 1.05,
    soft_dist: float = 1.55,
    dv_thresh: float = 2.0,
    ang_thresh: float = 25.0,
) -> Tuple[Optional[int], Optional[int], Optional[str]]:
    if ball_pos_roll.shape[0] <= 1:
        return None, None, None
    player_mask = (entity_type == 0) | (entity_type == 1)
    player_idx = np.where(player_mask)[0]
    if len(player_idx) == 0:
        return None, None, None

    vel_xy = np.zeros((ball_pos_roll.shape[0], 2), dtype=np.float32)
    if ball_pos_roll.shape[0] > 1:
        vel_xy[1:] = (ball_pos_roll[1:, :2] - ball_pos_roll[:-1, :2]) * FPS

    start_t = int(max(1, min_t))
    for t in range(start_t, int(ball_pos_roll.shape[0])):
        bp = ball_pos_roll[t, :2]
        if not np.all(np.isfinite(bp)):
            continue
        pp = player_roll_pos[min(t, player_roll_pos.shape[0] - 1), player_idx, :2]
        if pp.size == 0:
            continue
        d = np.linalg.norm(pp - bp[None, :], axis=1)
        nearest_local = int(np.argmin(d))
        nearest_d = float(d[nearest_local])
        nearest_entity = int(player_idx[nearest_local])
        if nearest_entity == int(passer_entity) and t <= (start_t + 1):
            continue
        if nearest_d <= float(hard_dist):
            return int(t), int(nearest_entity), _team_side(entity_type, nearest_entity)

        dv = float(np.linalg.norm(vel_xy[t] - vel_xy[t - 1]))
        dang = _angle_delta_deg(vel_xy[t - 1], vel_xy[t])
        if nearest_d <= float(soft_dist) and (dv >= float(dv_thresh) or dang >= float(ang_thresh)):
            return int(t), int(nearest_entity), _team_side(entity_type, nearest_entity)

    return None, None, None


def _entity_fields(entity_table: Dict[int, dict], entity_idx: Optional[int], prefix: str) -> Dict[str, object]:
    rec = entity_table.get(int(entity_idx), {}) if entity_idx is not None else {}
    return {
        f"{prefix}_entity_idx": None if entity_idx is None else int(entity_idx),
        f"{prefix}_team_side": rec.get("team_side"),
        f"{prefix}_player_name": rec.get("player_name"),
        f"{prefix}_position_label": rec.get("player_position_label"),
        f"{prefix}_inferred_band": rec.get("inferred_position_band"),
        f"{prefix}_inferred_lateral_band": rec.get("inferred_lateral_band"),
        f"{prefix}_formation_slot_idx": rec.get("formation_slot_idx"),
    }


def _receiver_proxy(
    *,
    entity_type: np.ndarray,
    player_pos_at_score: np.ndarray,
    ball_at_score: np.ndarray,
    attacking_side: Optional[str],
    passer_entity: Optional[int],
) -> Optional[int]:
    if attacking_side not in ("home", "away"):
        return None
    team_val = 0 if attacking_side == "home" else 1
    idx = np.where(entity_type == team_val)[0]
    if passer_entity is not None:
        idx = idx[idx != int(passer_entity)]
    if len(idx) == 0:
        return None
    pos = player_pos_at_score[idx, :]
    d = np.linalg.norm(pos[:, :2] - ball_at_score[None, :2], axis=1)
    return int(idx[int(np.argmin(d))])


def _evaluate_variant(
    *,
    state: WorkerState,
    clip_idx: int,
    clip_path: str,
    source_file: str,
    match_id: str,
    entity_type: np.ndarray,
    ball_idx: int,
    kfl: int,
    sim_len: int,
    ctx_len_actual: int,
    player_roll_pos: np.ndarray,
    full_player_pos: np.ndarray,
    full_player_vel: np.ndarray,
    passer_entity: int,
    attacking_side: Optional[str],
    defending_side: Optional[str],
    entity_table: Dict[int, dict],
    team_dirs: Dict[str, int],
    variant: Dict[str, object],
    passer_event_label: Optional[Dict[str, object]] = None,
    observed_fit_meta: Optional[Dict[str, object]] = None,
    observed_use_gt_ball: bool = False,
    gt_ball_roll: Optional[np.ndarray] = None,
    pre_pass_pv_home: float = math.nan,
    pre_pass_pv_away: float = math.nan,
    pre_pass_pv_net: float = math.nan,
) -> Dict[str, object]:
    v0 = np.asarray([variant["v0x"], variant["v0y"], variant["v0z"]], dtype=np.float32)
    spin_scalar = float(variant["spin_scalar"])
    start_pos = full_player_pos[ctx_len_actual, ball_idx].astype(np.float32).copy()
    is_observed = str(variant.get("variant_group", "")) == "observed"
    use_gt_ball_observed = bool(observed_use_gt_ball) and is_observed and gt_ball_roll is not None

    if use_gt_ball_observed:
        pre_ball_pos = np.asarray(gt_ball_roll, dtype=np.float32)
        if pre_ball_pos.shape[0] < (sim_len + 1):
            pad_n = (sim_len + 1) - int(pre_ball_pos.shape[0])
            pre_ball_pos = np.concatenate([pre_ball_pos, np.repeat(pre_ball_pos[-1:, :], pad_n, axis=0)], axis=0)
        else:
            pre_ball_pos = pre_ball_pos[: sim_len + 1]

        v0_for_gt = np.asarray(v0, dtype=np.float32)
        if pre_ball_pos.shape[0] > 1:
            v0_for_gt = (pre_ball_pos[1] - pre_ball_pos[0]) * FPS
        pre_ball_vel_xy, pre_ball_vz = _ball_vel_from_positions(pre_ball_pos, v0_for_gt)
        touch_frame_rel, touch_player_n, touch_side = _detect_gt_touch(
            entity_type=entity_type,
            ball_pos_roll=pre_ball_pos,
            passer_entity=passer_entity,
            player_roll_pos=player_roll_pos,
        )
    else:
        pre_ball_pos = _simulate_ball_with_spin(start_pos=start_pos, start_vel=v0, spin_scalar=spin_scalar, n_frames=sim_len + 1)
        pre_ball_vel_xy, pre_ball_vz = _ball_vel_from_positions(pre_ball_pos, v0)

        touch_frame_rel, touch_player_n, touch_side = _predict_touch(
            state=state,
            entity_type=entity_type,
            player_mask=((entity_type == 0) | (entity_type == 1)),
            passer_entity=passer_entity,
            pre_ball_pos=pre_ball_pos,
            pre_ball_vel_xy=pre_ball_vel_xy,
            pre_ball_vz=pre_ball_vz,
            full_player_pos=full_player_pos,
            full_player_vel=full_player_vel,
            ctx_len_actual=ctx_len_actual,
            sim_len=sim_len,
        )

    post_ball_pos = None
    bat_v0 = None
    if (not use_gt_ball_observed) and touch_frame_rel is not None and touch_player_n is not None:
        abs_touch = ctx_len_actual + int(touch_frame_rel)
        anchor_touch = pre_ball_pos[max(0, int(touch_frame_rel) - 1)] if int(touch_frame_rel) > 0 else pre_ball_pos[0]
        bat_v0 = _predict_bat_velocity(
            state=state,
            touch_frame_rel=int(touch_frame_rel),
            touch_player_n=int(touch_player_n),
            ball_pos_touch=pre_ball_pos[int(touch_frame_rel)],
            ball_vel_touch=pre_ball_vel_xy[int(touch_frame_rel)],
            ball_vz_touch=float(pre_ball_vz[int(touch_frame_rel)]),
            anchor_at_touch=anchor_touch,
            full_player_pos=full_player_pos,
            full_player_vel=full_player_vel,
            abs_touch_idx=abs_touch,
            player_mask=((entity_type == 0) | (entity_type == 1)),
            n_entities=entity_type.shape[0],
        )
        if bat_v0 is not None:
            phy_len = max(int(state.config.physics_len), int(state.config.eval_offset_after_touch))
            post_ball_pos = _simulate_ball_with_spin(
                start_pos=pre_ball_pos[int(touch_frame_rel)],
                start_vel=bat_v0,
                spin_scalar=0.0,
                n_frames=max(1, phy_len),
            )

    if touch_frame_rel is not None:
        score_frame_rel = int(touch_frame_rel) + max(0, int(state.config.eval_offset_after_touch))
        score_mode = "touch_offset"
    else:
        score_frame_rel = int(sim_len)
        score_mode = "terminal"

    score_frame_rel = int(min(score_frame_rel, pre_ball_pos.shape[0] - 1))
    ball_seq = np.zeros((score_frame_rel + 1, 3), dtype=np.float32)
    if use_gt_ball_observed:
        ball_seq[:] = pre_ball_pos[: score_frame_rel + 1]
    else:
        for t in range(score_frame_rel + 1):
            if touch_frame_rel is None or t <= int(touch_frame_rel):
                ball_seq[t] = pre_ball_pos[min(t, pre_ball_pos.shape[0] - 1)]
            else:
                idx = t - int(touch_frame_rel) - 1
                if post_ball_pos is None:
                    ball_seq[t] = pre_ball_pos[min(t, pre_ball_pos.shape[0] - 1)]
                else:
                    ball_seq[t] = post_ball_pos[min(idx, post_ball_pos.shape[0] - 1)]

    oob_frame, oob_pos = _detect_oob(ball_seq)
    restart_kind = None
    taking_side = None
    restart_reason = None

    if oob_frame is not None and oob_pos is not None:
        last_touch_side = _team_side(entity_type, passer_entity)
        if touch_frame_rel is not None and oob_frame > int(touch_frame_rel):
            last_touch_side = touch_side if touch_side is not None else last_touch_side
        restart_kind, taking_side, restart_reason = _infer_restart_from_oob(
            exit_pos=oob_pos,
            last_touch_side=last_touch_side,
            team_dirs=team_dirs,
        )
        score_mode = "set_piece"
        score_frame_rel = int(oob_frame)

    if score_mode == "set_piece":
        pv_home = pv_away = pv_net = math.nan
        if state.set_piece_model is not None and restart_kind is not None and taking_side is not None:
            ball_x_sb = meters_x_to_statsbomb_x(float(oob_pos[0])) if restart_kind == "throw_in" and oob_pos is not None else None
            pv_home, pv_away, pv_net = state.set_piece_model.predict_team_values(
                kind=restart_kind,
                taking_side=taking_side,
                ball_x_statsbomb=ball_x_sb,
            )
    else:
        ball_start_vel_for_pv = v0
        if use_gt_ball_observed and pre_ball_pos.shape[0] > 1:
            ball_start_vel_for_pv = (pre_ball_pos[1] - pre_ball_pos[0]) * FPS
        pv_home, pv_away, pv_net = _compute_pv_numeric(
            pv_result=state.pv_result,
            entity_type=entity_type,
            player_pos_roll=player_roll_pos,
            ball_seq=ball_seq,
            ball_start_vel=ball_start_vel_for_pv,
            device=state.device,
        )

    player_score_frame = min(int(score_frame_rel), player_roll_pos.shape[0] - 1)
    receiver_proxy = _receiver_proxy(
        entity_type=entity_type,
        player_pos_at_score=player_roll_pos[player_score_frame],
        ball_at_score=ball_seq[min(score_frame_rel, ball_seq.shape[0] - 1)],
        attacking_side=attacking_side,
        passer_entity=passer_entity,
    )

    row: Dict[str, object] = {
        "clip_idx": int(clip_idx),
        "clip_path": str(clip_path),
        "source_file": str(source_file),
        "match_id": str(match_id),
        "kick_frame_local_refined": int(kfl),
        "variant_id": variant["variant_id"],
        "variant_group": variant["variant_group"],
        "seed": int(state.config.seed),
        "v0x": float(v0[0]),
        "v0y": float(v0[1]),
        "v0z": float(v0[2]),
        "speed_xy": float(np.linalg.norm(v0[:2])),
        "speed_3d": float(np.linalg.norm(v0)),
        "spin_scalar": float(spin_scalar),
        "observed_fit_rmse_xy": math.nan,
        "observed_fit_rmse_z": math.nan,
        "observed_fit_near_zero": 0,
        "observed_fit_endpoint_xy_err": math.nan,
        "observed_fit_endpoint_z_err": math.nan,
        "observed_fit_endpoint_enforced": 0,
        "attacking_side_at_kick": attacking_side,
        "defending_side_at_kick": defending_side,
        "home_team_label": "home",
        "away_team_label": "away",
        "passer_player_id": None,
        "touch_found": bool(touch_frame_rel is not None),
        "touch_frame_rel": None if touch_frame_rel is None else int(touch_frame_rel),
        "ball_out": bool(oob_frame is not None),
        "out_frame_rel": None if oob_frame is None else int(oob_frame),
        "restart_kind": restart_kind,
        "taking_side": taking_side,
        "restart_reason": restart_reason,
        "score_frame_rel": int(score_frame_rel),
        "score_mode": score_mode,
        "pv_home": float(pv_home) if np.isfinite(pv_home) else math.nan,
        "pv_away": float(pv_away) if np.isfinite(pv_away) else math.nan,
        "pv_net": float(pv_net) if np.isfinite(pv_net) else math.nan,
        "pre_pass_pv_home": float(pre_pass_pv_home) if np.isfinite(pre_pass_pv_home) else math.nan,
        "pre_pass_pv_away": float(pre_pass_pv_away) if np.isfinite(pre_pass_pv_away) else math.nan,
        "pre_pass_pv_net": float(pre_pass_pv_net) if np.isfinite(pre_pass_pv_net) else math.nan,
        "pv_added_vs_prepass": (float(pv_net) - float(pre_pass_pv_net))
        if np.isfinite(pv_net) and np.isfinite(pre_pass_pv_net)
        else math.nan,
        "observed_pv_added_vs_prepass_for_clip": math.nan,
        "delta_pv_added_vs_observed": math.nan,
        "observed_pv_net_for_clip": math.nan,
        "delta_vs_observed": math.nan,
        "is_observed_row": bool(variant["variant_group"] == "observed"),
        "player_fallback_gt": 0,
        "status_code": "ok_set_piece" if score_mode == "set_piece" else ("no_touch_terminal" if touch_frame_rel is None else "ok_inplay"),
        "error_msg": "",
    }
    if observed_fit_meta:
        row["observed_fit_rmse_xy"] = float(observed_fit_meta.get("fit_rmse_xy", math.nan))
        row["observed_fit_rmse_z"] = float(observed_fit_meta.get("fit_rmse_z", math.nan))
        row["observed_fit_near_zero"] = int(observed_fit_meta.get("fit_near_zero", 0) or 0)
        row["observed_fit_endpoint_xy_err"] = float(observed_fit_meta.get("fit_endpoint_xy_err", math.nan))
        row["observed_fit_endpoint_z_err"] = float(observed_fit_meta.get("fit_endpoint_z_err", math.nan))
        row["observed_fit_endpoint_enforced"] = int(observed_fit_meta.get("fit_endpoint_enforced", 0) or 0)
    row.update(_entity_fields(entity_table, passer_entity, "passer"))
    row.update(_entity_fields(entity_table, touch_player_n, "touch"))
    row.update(_entity_fields(entity_table, receiver_proxy, "receiver_proxy"))
    if passer_event_label:
        pid = str(passer_event_label.get("passer_player_id", "") or "").strip()
        pname = str(passer_event_label.get("passer_player_name", "") or "").strip()
        pside = str(passer_event_label.get("passer_team_side", "") or "").strip().lower()
        home_name = str(passer_event_label.get("home_team_name", "") or "").strip()
        away_name = str(passer_event_label.get("away_team_name", "") or "").strip()
        if pid:
            row["passer_player_id"] = pid
        if pname:
            row["passer_player_name"] = pname
        if pside in {"home", "away"}:
            row["passer_team_side"] = pside
        if home_name:
            row["home_team_label"] = home_name
        if away_name:
            row["away_team_label"] = away_name
    return row


def init_worker(gpu_id: int, model_paths: Dict[str, str], config: Dict[str, object]) -> WorkerState:
    cfg = RunnerConfig(**config)
    mpaths = ModelPaths(**model_paths)
    use_cuda = torch.cuda.is_available()
    device = torch.device(f"cuda:{gpu_id}" if use_cuda else "cpu")

    smart_result = try_load_smart_model(mpaths.smart_checkpoint, mpaths.vocab_dir, device)
    touch_result = try_load_touch_model(mpaths.touch_checkpoint, device)
    bat_result = try_load_bat_model(mpaths.bat_checkpoint, device)
    pv_result = try_load_pv_model(mpaths.pv_checkpoint, device)
    set_piece_model = try_load_set_piece_pv_model(mpaths.set_piece_pv_model)

    missing = []
    for name, loaded in (
        ("SMART", smart_result),
        ("Player-to-Touch", touch_result),
        ("Ball-at-Touch", bat_result),
        ("Possession Value", pv_result),
    ):
        if loaded is None:
            missing.append(name)
    if missing:
        raise FileNotFoundError(
            "Required MCPS checkpoints failed to load: " + ", ".join(missing)
        )

    return WorkerState(
        gpu_id=int(gpu_id),
        device=device,
        config=cfg,
        model_paths=mpaths,
        smart_result=smart_result,
        touch_result=touch_result,
        bat_result=bat_result,
        pv_result=pv_result,
        set_piece_model=set_piece_model,
    )


def process_clip_task(task: Dict[str, object], state: WorkerState) -> Dict[str, object]:
    clip_idx = int(task["clip_idx"])
    entry = task["entry"]
    skip_variant_ids = set(task.get("skip_variant_ids", []))
    passer_event_label = dict(task.get("passer_event_label") or {})
    caps = VariationCaps(**task["caps"])

    try:
        data = np.load(entry["clip_path"])
        clip = {k: data[k] for k in data.files}
        features = _clip_features_to_metres(clip["features"])
        entity_type = clip["entity_type"].astype(np.int64)
        mask = clip.get("mask", np.ones(features.shape[:2], dtype=np.uint8)).astype(bool)
        T, N, _ = features.shape

        ball_idx_arr = np.where(entity_type == 2)[0]
        if len(ball_idx_arr) == 0:
            raise RuntimeError("No ball entity found in clip.")
        ball_idx = int(ball_idx_arr[0])

        kfl = refine_kick_frame_local(clip)
        kfl = int(np.clip(kfl, 0, T - 1))
        ctx_start = max(0, kfl - int(state.config.context_len))
        ctx_end = min(T - 1, kfl + int(state.config.rollout_len))
        sim_len = int(ctx_end - kfl)
        ctx_len_actual = int(kfl - ctx_start)

        pos_ctx = np.zeros((ctx_end - ctx_start + 1, N, 3), dtype=np.float32)
        pos_ctx[:, :, :2] = features[ctx_start : ctx_end + 1, :, :2]
        if features.shape[-1] >= 5:
            pos_ctx[:, :, 2] = features[ctx_start : ctx_end + 1, :, 4]

        player_fallback_gt = 0

        # Ground-truth player rollout tensors (used for observed variant when configured).
        gt_player_roll = pos_ctx[ctx_len_actual : ctx_len_actual + sim_len + 1]
        full_player_pos_gt = pos_ctx.copy()
        full_player_vel_gt = np.zeros((full_player_pos_gt.shape[0], N, 2), dtype=np.float32)
        if full_player_pos_gt.shape[0] > 1:
            full_player_vel_gt[1:] = (full_player_pos_gt[1:, :, :2] - full_player_pos_gt[:-1, :, :2]) * FPS

        player_mask = (entity_type == 0) | (entity_type == 1)
        player_idx = np.where(player_mask)[0]
        event_side = str(passer_event_label.get("passer_team_side", "") or "").strip().lower()
        if event_side in {"home", "away"}:
            event_team_val = 0 if event_side == "home" else 1
            event_team_idx = np.where(entity_type == event_team_val)[0]
            if len(event_team_idx) > 0:
                player_idx = event_team_idx
        kick_ball = full_player_pos_gt[ctx_len_actual, ball_idx]
        kick_players = full_player_pos_gt[ctx_len_actual, player_idx]
        d0 = np.linalg.norm(kick_players[:, :2] - kick_ball[None, :2], axis=1)
        passer_entity = int(player_idx[int(np.argmin(d0))]) if len(player_idx) > 0 else -1

        attacking_side = event_side if event_side in {"home", "away"} else _team_side(entity_type, passer_entity)
        defending_side = _opposite_side(attacking_side)
        entity_table = build_clip_entity_table(entity_type, full_player_pos_gt[ctx_len_actual])
        team_dirs = _estimate_team_directions(entity_type, full_player_pos_gt[ctx_len_actual])

        rng = np.random.default_rng(int(state.config.seed + clip_idx * 100_003))
        observed_override = task.get("observed_param_override")
        observed: Dict[str, object]
        if isinstance(observed_override, dict) and observed_override:
            observed = dict(observed_override)
            observed["variant_id"] = "observed"
            observed["variant_group"] = "observed"
        else:
            observed = infer_observed_variant_fitted(
                features=features,
                ball_idx=ball_idx,
                kfl=kfl,
                caps=caps,
                rng=rng,
                max_fit_frames=max(28, min(int(sim_len), 96)),
                player_pos_seq=full_player_pos_gt[ctx_len_actual : ctx_len_actual + sim_len + 1],
                entity_type=entity_type,
            )
        # Safety fallback (should be rare): keep legacy finite-difference estimate.
        if not all(np.isfinite([observed.get("v0x"), observed.get("v0y"), observed.get("v0z"), observed.get("spin_scalar", 0.0)])):
            observed = infer_observed_variant(features, ball_idx, kfl)

        obs_v = np.asarray([observed["v0x"], observed["v0y"], observed["v0z"]], dtype=np.float32)
        pre_pass_pv_home, pre_pass_pv_away, pre_pass_pv_net = _compute_pre_pass_pv(
            pv_result=state.pv_result,
            features_m=features,
            entity_type=entity_type,
            ball_idx=ball_idx,
            kfl=kfl,
            device=state.device,
        )
        observed_fit_meta = {
            "fit_rmse_xy": float(observed.get("fit_rmse_xy", math.nan)),
            "fit_rmse_z": float(observed.get("fit_rmse_z", math.nan)),
            "fit_near_zero": int(observed.get("fit_near_zero", 0) or 0),
            "fit_endpoint_xy_err": float(observed.get("fit_endpoint_xy_err", math.nan)),
            "fit_endpoint_z_err": float(observed.get("fit_endpoint_z_err", math.nan)),
            "fit_endpoint_enforced": int(observed.get("fit_endpoint_enforced", 0) or 0),
        }
        if state.config.require_near_zero_fit and not bool(observed_fit_meta["fit_near_zero"]):
            rejected = {
                "clip_idx": clip_idx,
                "clip_path": str(entry["clip_path"]),
                "source_file": str(entry.get("source_file", "")),
                "match_id": _parse_match_id(str(entry.get("source_file", ""))),
                "kick_frame_local_refined": kfl,
                "variant_id": "rejected_fit",
                "variant_group": "rejected",
                "seed": int(state.config.seed),
                "v0x": observed.get("v0x"),
                "v0y": observed.get("v0y"),
                "v0z": observed.get("v0z"),
                "spin_scalar": observed.get("spin_scalar"),
                "observed_fit_rmse_xy": observed_fit_meta["fit_rmse_xy"],
                "observed_fit_rmse_z": observed_fit_meta["fit_rmse_z"],
                "observed_fit_near_zero": 0,
                "observed_fit_endpoint_xy_err": observed_fit_meta["fit_endpoint_xy_err"],
                "observed_fit_endpoint_z_err": observed_fit_meta["fit_endpoint_z_err"],
                "observed_fit_endpoint_enforced": observed_fit_meta["fit_endpoint_enforced"],
                "attacking_side_at_kick": attacking_side,
                "defending_side_at_kick": defending_side,
                "is_observed_row": False,
                "player_fallback_gt": 0,
                "status_code": "rejected_fit",
                "error_msg": "Observed kick parameters did not meet near-zero fit thresholds",
            }
            rejected.update(_entity_fields(entity_table, passer_entity, "passer"))
            return {"clip_idx": clip_idx, "rows": [rejected], "ok": True, "error": ""}
        variants: List[Dict[str, object]] = []
        if state.config.include_observed:
            variants.append(observed)
        variants.extend(sample_local_variants(obs_v, state.config.local_variants, caps, rng))
        variants.extend(sample_global_variants(obs_v, state.config.global_variants, caps, rng))
        variants = [v for v in variants if str(v["variant_id"]) not in skip_variant_ids]

        rows: List[Dict[str, object]] = []
        match_id = _parse_match_id(str(entry.get("source_file", "")))
        for v in variants:
            try:
                use_gt_players = bool(state.config.observed_use_gt_players) and str(v.get("variant_group")) == "observed"
                if use_gt_players:
                    pred_pos = gt_player_roll
                    full_player_pos = full_player_pos_gt
                    full_player_vel = full_player_vel_gt
                else:
                    candidate_v0 = np.asarray([v["v0x"], v["v0y"], v["v0z"]], dtype=np.float32)
                    candidate_ball = _simulate_ball_with_spin(
                        start_pos=full_player_pos_gt[ctx_len_actual, ball_idx],
                        start_vel=candidate_v0,
                        spin_scalar=float(v.get("spin_scalar", 0.0)),
                        n_frames=sim_len + 1,
                    )
                    pred_pos = smart_predict_positions(
                        state.smart_result,
                        features,
                        entity_type,
                        mask,
                        ctx_start,
                        kfl,
                        sim_len,
                        state.device,
                        candidate_ball_pos=candidate_ball,
                        temperature=0.8,
                        top_k=10,
                        seed=int(state.config.seed + clip_idx * 9973 + len(rows) * 37),
                    )
                    if pred_pos is None:
                        raise RuntimeError("SMART rollout could not be constructed from the available history")
                    full_player_pos = pos_ctx.copy()
                    full_player_pos[ctx_len_actual : ctx_len_actual + pred_pos.shape[0]] = pred_pos
                    full_player_vel = np.zeros((full_player_pos.shape[0], N, 2), dtype=np.float32)
                    if full_player_pos.shape[0] > 1:
                        full_player_vel[1:] = (
                            full_player_pos[1:, :, :2] - full_player_pos[:-1, :, :2]
                        ) * FPS
                row = _evaluate_variant(
                    state=state,
                    clip_idx=clip_idx,
                    clip_path=str(entry["clip_path"]),
                    source_file=str(entry.get("source_file", "")),
                    match_id=match_id,
                    entity_type=entity_type,
                    ball_idx=ball_idx,
                    kfl=kfl,
                    sim_len=sim_len,
                    ctx_len_actual=ctx_len_actual,
                    player_roll_pos=gt_player_roll if use_gt_players else pred_pos,
                    full_player_pos=full_player_pos_gt if use_gt_players else full_player_pos,
                    full_player_vel=full_player_vel_gt if use_gt_players else full_player_vel,
                    passer_entity=passer_entity,
                    attacking_side=attacking_side,
                    defending_side=defending_side,
                    entity_table=entity_table,
                    team_dirs=team_dirs,
                    variant=v,
                    passer_event_label=passer_event_label,
                    observed_fit_meta=observed_fit_meta,
                    observed_use_gt_ball=False,
                    gt_ball_roll=full_player_pos_gt[ctx_len_actual : ctx_len_actual + sim_len + 1, ball_idx, :],
                    pre_pass_pv_home=float(pre_pass_pv_home),
                    pre_pass_pv_away=float(pre_pass_pv_away),
                    pre_pass_pv_net=float(pre_pass_pv_net),
                )
                row["player_fallback_gt"] = int(player_fallback_gt)
            except Exception as ve:
                row = {
                    "clip_idx": int(clip_idx),
                    "clip_path": str(entry["clip_path"]),
                    "source_file": str(entry.get("source_file", "")),
                    "match_id": match_id,
                    "kick_frame_local_refined": int(kfl),
                    "variant_id": v.get("variant_id"),
                    "variant_group": v.get("variant_group"),
                    "seed": int(state.config.seed),
                    "v0x": v.get("v0x"),
                    "v0y": v.get("v0y"),
                    "v0z": v.get("v0z"),
                    "speed_xy": math.nan,
                    "speed_3d": math.nan,
                    "spin_scalar": v.get("spin_scalar"),
                    "observed_fit_rmse_xy": float(observed_fit_meta.get("fit_rmse_xy", math.nan)),
                    "observed_fit_rmse_z": float(observed_fit_meta.get("fit_rmse_z", math.nan)),
                    "observed_fit_near_zero": int(observed_fit_meta.get("fit_near_zero", 0) or 0),
                    "observed_fit_endpoint_xy_err": float(observed_fit_meta.get("fit_endpoint_xy_err", math.nan)),
                    "observed_fit_endpoint_z_err": float(observed_fit_meta.get("fit_endpoint_z_err", math.nan)),
                    "observed_fit_endpoint_enforced": int(observed_fit_meta.get("fit_endpoint_enforced", 0) or 0),
                    "attacking_side_at_kick": attacking_side,
                    "defending_side_at_kick": defending_side,
                    "home_team_label": str(passer_event_label.get("home_team_name", "") or "home"),
                    "away_team_label": str(passer_event_label.get("away_team_name", "") or "away"),
                    "passer_player_id": None,
                    "touch_found": False,
                    "touch_frame_rel": None,
                    "ball_out": False,
                    "out_frame_rel": None,
                    "restart_kind": None,
                    "taking_side": None,
                    "restart_reason": None,
                    "score_frame_rel": None,
                    "score_mode": "error",
                    "pv_home": math.nan,
                    "pv_away": math.nan,
                    "pv_net": math.nan,
                    "pre_pass_pv_home": float(pre_pass_pv_home) if np.isfinite(pre_pass_pv_home) else math.nan,
                    "pre_pass_pv_away": float(pre_pass_pv_away) if np.isfinite(pre_pass_pv_away) else math.nan,
                    "pre_pass_pv_net": float(pre_pass_pv_net) if np.isfinite(pre_pass_pv_net) else math.nan,
                    "pv_added_vs_prepass": math.nan,
                    "observed_pv_added_vs_prepass_for_clip": math.nan,
                    "delta_pv_added_vs_observed": math.nan,
                    "observed_pv_net_for_clip": math.nan,
                    "delta_vs_observed": math.nan,
                    "is_observed_row": bool(v.get("variant_group") == "observed"),
                    "player_fallback_gt": int(player_fallback_gt),
                    "status_code": "variant_error",
                    "error_msg": f"{type(ve).__name__}: {ve}",
                }
                row.update(_entity_fields(entity_table, passer_entity, "passer"))
                row.update(_entity_fields(entity_table, None, "touch"))
                row.update(_entity_fields(entity_table, None, "receiver_proxy"))
                pid = str(passer_event_label.get("passer_player_id", "") or "").strip()
                pname = str(passer_event_label.get("passer_player_name", "") or "").strip()
                pside = str(passer_event_label.get("passer_team_side", "") or "").strip().lower()
                if pid:
                    row["passer_player_id"] = pid
                if pname:
                    row["passer_player_name"] = pname
                if pside in {"home", "away"}:
                    row["passer_team_side"] = pside
            rows.append(row)

        obs_rows = [r for r in rows if bool(r.get("is_observed_row"))]
        obs_pv = float(obs_rows[0]["pv_net"]) if obs_rows and np.isfinite(obs_rows[0].get("pv_net", math.nan)) else math.nan
        obs_pv_added = (
            float(obs_rows[0]["pv_added_vs_prepass"])
            if obs_rows and np.isfinite(obs_rows[0].get("pv_added_vs_prepass", math.nan))
            else math.nan
        )
        for r in rows:
            r["observed_pv_net_for_clip"] = obs_pv
            r_pv = float(r.get("pv_net", math.nan))
            r["delta_vs_observed"] = (r_pv - obs_pv) if np.isfinite(r_pv) and np.isfinite(obs_pv) else math.nan
            r["observed_pv_added_vs_prepass_for_clip"] = obs_pv_added
            r_added = float(r.get("pv_added_vs_prepass", math.nan))
            r["delta_pv_added_vs_observed"] = (
                r_added - obs_pv_added
                if np.isfinite(r_added) and np.isfinite(obs_pv_added)
                else math.nan
            )

        return {"clip_idx": clip_idx, "rows": rows, "ok": True, "error": ""}
    except Exception as e:
        return {
            "clip_idx": clip_idx,
            "rows": [
                {
                    "clip_idx": clip_idx,
                    "clip_path": str(entry.get("clip_path", "")),
                    "source_file": str(entry.get("source_file", "")),
                    "match_id": _parse_match_id(str(entry.get("source_file", ""))),
                    "kick_frame_local_refined": None,
                    "variant_id": "__clip_error__",
                    "variant_group": "error",
                    "seed": int(state.config.seed),
                    "v0x": math.nan,
                    "v0y": math.nan,
                    "v0z": math.nan,
                    "speed_xy": math.nan,
                    "speed_3d": math.nan,
                    "spin_scalar": math.nan,
                    "observed_fit_rmse_xy": math.nan,
                    "observed_fit_rmse_z": math.nan,
                    "observed_fit_near_zero": 0,
                    "observed_fit_endpoint_xy_err": math.nan,
                    "observed_fit_endpoint_z_err": math.nan,
                    "observed_fit_endpoint_enforced": 0,
                    "attacking_side_at_kick": None,
                    "defending_side_at_kick": None,
                    "home_team_label": "home",
                    "away_team_label": "away",
                    "passer_entity_idx": None,
                    "passer_team_side": None,
                    "passer_player_name": None,
                    "passer_player_id": None,
                    "passer_position_label": None,
                    "passer_inferred_band": None,
                    "touch_found": False,
                    "touch_frame_rel": None,
                    "touch_entity_idx": None,
                    "touch_team_side": None,
                    "touch_player_name": None,
                    "touch_position_label": None,
                    "touch_inferred_band": None,
                    "receiver_proxy_entity_idx": None,
                    "receiver_proxy_team_side": None,
                    "receiver_proxy_player_name": None,
                    "receiver_proxy_position_label": None,
                    "receiver_proxy_inferred_band": None,
                    "ball_out": False,
                    "out_frame_rel": None,
                    "restart_kind": None,
                    "taking_side": None,
                    "restart_reason": None,
                    "score_frame_rel": None,
                    "score_mode": "error",
                    "pv_home": math.nan,
                    "pv_away": math.nan,
                    "pv_net": math.nan,
                    "pre_pass_pv_home": math.nan,
                    "pre_pass_pv_away": math.nan,
                    "pre_pass_pv_net": math.nan,
                    "pv_added_vs_prepass": math.nan,
                    "observed_pv_added_vs_prepass_for_clip": math.nan,
                    "delta_pv_added_vs_observed": math.nan,
                    "observed_pv_net_for_clip": math.nan,
                    "delta_vs_observed": math.nan,
                    "is_observed_row": False,
                    "player_fallback_gt": 0,
                    "status_code": "clip_error",
                    "error_msg": f"{type(e).__name__}: {e}",
                }
            ],
            "ok": False,
            "error": traceback.format_exc(limit=2),
        }
