from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

FPS = 25.0
HALF_X = 105.0 / 2.0
HALF_Y = 68.0 / 2.0
GRAVITY = 9.81
AIR_RESISTANCE = 0.00648
RESTITUTION = 0.109
GROUND_FRICTION = 0.0
ROLLING_FRICTION = 0.02731
ROLLING_Z_THRESH = 0.054
SPIN_ACCEL_COEF = 0.14


@dataclass
class VariationCaps:
    speed_xy_max: float
    vz_abs_max: float
    spin_abs_max: float


def _clip_xy_speed(vxy: np.ndarray, cap: float) -> np.ndarray:
    speed = float(np.linalg.norm(vxy))
    if speed <= cap or speed <= 1e-8:
        return vxy
    return vxy * (cap / speed)


def _rotate(vxy: np.ndarray, angle_deg: float) -> np.ndarray:
    th = np.deg2rad(float(angle_deg))
    c, s = np.cos(th), np.sin(th)
    rot = np.asarray([[c, -s], [s, c]], dtype=np.float32)
    return (rot @ vxy.astype(np.float32)).astype(np.float32)


def _infer_ball_release_velocity_ms(features: np.ndarray, ball_idx: int, kfl: int) -> np.ndarray:
    """Infer release velocity in m/s from finite differences."""
    t = int(np.clip(kfl, 0, features.shape[0] - 1))
    if t <= 0:
        if features.shape[-1] >= 6:
            return np.asarray(
                [features[t, ball_idx, 2] * FPS, features[t, ball_idx, 3] * FPS, features[t, ball_idx, 5] * FPS],
                dtype=np.float32,
            )
        return np.asarray([0.0, 0.0, 0.0], dtype=np.float32)

    p_now = features[t, ball_idx, :2].astype(np.float32)
    p_prev = features[t - 1, ball_idx, :2].astype(np.float32)
    vxy = (p_now - p_prev) * FPS

    z_now = float(features[t, ball_idx, 4]) if features.shape[-1] >= 5 else 0.0
    z_prev = float(features[t - 1, ball_idx, 4]) if features.shape[-1] >= 5 else z_now
    vz = (z_now - z_prev) * FPS
    return np.asarray([vxy[0], vxy[1], vz], dtype=np.float32)


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

    for i in range(int(n_frames)):
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


def _ball_vel(ball_seq: np.ndarray) -> np.ndarray:
    v = np.zeros((ball_seq.shape[0], 3), dtype=np.float32)
    if ball_seq.shape[0] > 1:
        v[1:] = (ball_seq[1:] - ball_seq[:-1]) * FPS
    return v


def _angle_delta_deg(v_prev: np.ndarray, v_cur: np.ndarray) -> float:
    n1 = float(np.linalg.norm(v_prev))
    n2 = float(np.linalg.norm(v_cur))
    if n1 <= 1e-8 or n2 <= 1e-8:
        return 0.0
    c = float(np.dot(v_prev, v_cur) / (n1 * n2))
    c = float(np.clip(c, -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def _detect_fit_horizon(
    ball_gt: np.ndarray,
    *,
    max_fit_frames: int,
    entity_type: Optional[np.ndarray] = None,
    player_pos_seq: Optional[np.ndarray] = None,
    z_ground_thresh: float = 0.06,
    touch_hard_dist_thresh: float = 1.55,
    touch_dist_thresh: float = 1.6,
    touch_dv_thresh: float = 2.0,
    touch_angle_thresh: float = 25.0,
) -> int:
    n = int(ball_gt.shape[0])
    if n <= 2:
        return max(1, n - 1)

    finite_xy = np.isfinite(ball_gt[:, 0]) & np.isfinite(ball_gt[:, 1])
    if not bool(finite_xy[0]):
        return 1
    valid_end = 0
    for t in range(n):
        if not bool(finite_xy[t]):
            break
        valid_end = t

    end = int(min(valid_end, max(2, int(max_fit_frames))))

    # Out-of-bounds onset.
    for t in range(1, end + 1):
        if abs(float(ball_gt[t, 0])) > HALF_X or abs(float(ball_gt[t, 1])) > HALF_Y:
            end = t
            break

    # First bounce contact (if z available).
    z = ball_gt[:, 2] if ball_gt.shape[1] >= 3 else np.zeros((n,), dtype=np.float32)
    for t in range(1, end + 1):
        if not (np.isfinite(z[t - 1]) and np.isfinite(z[t])):
            continue
        if float(z[t - 1]) > float(z_ground_thresh) and float(z[t]) <= float(z_ground_thresh):
            end = t
            break

    # Touch-like event: close to player + abrupt velocity/direction change.
    if player_pos_seq is not None and entity_type is not None and player_pos_seq.shape[0] >= (end + 1):
        mask = (np.asarray(entity_type).astype(np.int64) == 0) | (np.asarray(entity_type).astype(np.int64) == 1)
        if bool(np.any(mask)):
            vel = _ball_vel(ball_gt)
            for t in range(3, end + 1):
                bp = ball_gt[t, :2]
                if not np.all(np.isfinite(bp)):
                    continue
                pp = player_pos_seq[t, mask, :2]
                if pp.size == 0:
                    continue
                d = np.linalg.norm(pp - bp[None, :], axis=1)
                nearest = float(np.min(d)) if d.size > 0 else math.inf
                if nearest <= float(touch_hard_dist_thresh):
                    end = t
                    break
                dv = float(np.linalg.norm(vel[t, :2] - vel[t - 1, :2]))
                dang = _angle_delta_deg(vel[t - 1, :2], vel[t, :2])
                if nearest <= float(touch_dist_thresh) and (dv >= float(touch_dv_thresh) or dang >= float(touch_angle_thresh)):
                    end = t
                    break
                # Very abrupt trajectory change often indicates a hidden interaction.
                if dv >= 4.0 or dang >= 45.0:
                    end = t
                    break

    return int(max(2, end))


def _fit_loss(sim_ball: np.ndarray, gt_ball: np.ndarray) -> Tuple[float, float, float]:
    n = int(min(sim_ball.shape[0], gt_ball.shape[0]))
    if n <= 1:
        return 1e9, 1e9, 1e9
    sim = sim_ball[:n]
    gt = gt_ball[:n]
    valid_xy = np.isfinite(gt[:, 0]) & np.isfinite(gt[:, 1]) & np.isfinite(sim[:, 0]) & np.isfinite(sim[:, 1])
    if not bool(np.any(valid_xy)):
        return 1e9, 1e9, 1e9

    sim = sim[valid_xy]
    gt = gt[valid_xy]
    m = sim.shape[0]
    t = np.arange(m, dtype=np.float32)
    w = 1.0 + 2.0 * (1.0 - (t / max(1.0, float(m - 1))))
    e_xy = (sim[:, 0] - gt[:, 0]) ** 2 + (sim[:, 1] - gt[:, 1]) ** 2
    rmse_xy = float(np.sqrt(float(np.mean(e_xy))))

    valid_z = np.isfinite(gt[:, 2]) & np.isfinite(sim[:, 2])
    if bool(np.any(valid_z)):
        e_z = np.zeros((m,), dtype=np.float32)
        e_z[valid_z] = (sim[valid_z, 2] - gt[valid_z, 2]) ** 2
        rmse_z = float(np.sqrt(float(np.mean(e_z[valid_z]))))
    else:
        e_z = np.zeros((m,), dtype=np.float32)
        rmse_z = 0.0

    wsum = float(np.sum(w))
    loss_xy = float(np.sum(w * e_xy) / max(1.0, wsum))
    loss_z = float(np.sum(w * e_z) / max(1.0, wsum))
    # Prefer tight XY alignment while still matching vertical arc.
    loss = loss_xy + 0.60 * loss_z + 2e-3 * rmse_xy + 1e-3 * rmse_z
    return float(loss), rmse_xy, rmse_z


def _clip_v0(
    v0: np.ndarray,
    *,
    speed_cap: float,
    vz_cap: float,
) -> np.ndarray:
    out = v0.astype(np.float32).copy()
    out[:2] = _clip_xy_speed(out[:2], float(speed_cap))
    out[2] = float(np.clip(out[2], -float(vz_cap), float(vz_cap)))
    return out


def estimate_caps_from_manifest(
    manifest_entries: Iterable[dict],
    *,
    pctl: float = 99.0,
    margin: float = 1.05,
    max_entries: int | None = None,
) -> VariationCaps:
    """Estimate hard caps from observed test-pass releases."""
    speeds: List[float] = []
    vz_abs: List[float] = []

    for i, entry in enumerate(manifest_entries):
        if max_entries is not None and i >= int(max_entries):
            break
        clip = np.load(entry["clip_path"])
        features = clip["features"].astype(np.float32)
        et = clip["entity_type"].astype(np.int64)
        ball_idx_arr = np.where(et == 2)[0]
        if len(ball_idx_arr) == 0:
            continue
        ball_idx = int(ball_idx_arr[0])
        kfl = int(clip.get("kick_frame_local", 0))
        v = _infer_ball_release_velocity_ms(features, ball_idx, kfl)
        speeds.append(float(np.linalg.norm(v[:2])))
        vz_abs.append(abs(float(v[2])))

    if not speeds:
        return VariationCaps(speed_xy_max=25.0, vz_abs_max=12.0, spin_abs_max=0.25)

    speed_cap = max(10.0, float(np.percentile(np.asarray(speeds), pctl)) * float(margin))
    vz_cap = max(4.0, float(np.percentile(np.asarray(vz_abs), pctl)) * float(margin))
    # No direct spin observation exists in clips; keep a strict small cap.
    spin_cap = 0.25
    return VariationCaps(speed_xy_max=speed_cap, vz_abs_max=vz_cap, spin_abs_max=spin_cap)


def infer_observed_variant(features: np.ndarray, ball_idx: int, kfl: int) -> Dict[str, float | str]:
    v0 = _infer_ball_release_velocity_ms(features, ball_idx, kfl)
    return {
        "variant_id": "observed",
        "variant_group": "observed",
        "v0x": float(v0[0]),
        "v0y": float(v0[1]),
        "v0z": float(v0[2]),
        "spin_scalar": 0.0,
    }


def infer_observed_variant_fitted(
    *,
    features: np.ndarray,
    ball_idx: int,
    kfl: int,
    caps: Optional[VariationCaps],
    rng: np.random.Generator,
    max_fit_frames: int = 28,
    player_pos_seq: Optional[np.ndarray] = None,
    entity_type: Optional[np.ndarray] = None,
    near_zero_xy: float = 0.03,
    near_zero_z: float = 0.10,
    near_zero_endpoint_xy: float = 0.02,
    near_zero_endpoint_z: float = 0.03,
    high_error_xy: float = 0.25,
    high_error_z: float = 0.35,
    max_adaptive_rounds: int = 5,
    allow_horizon_backoff: bool = True,
    max_horizon_backoff: int = 10,
    horizon_backoff_depth: int = 0,
    allow_endpoint_enforce: bool = True,
) -> Dict[str, float | str]:
    """Search release parameters (v0, spin) that best reproduce GT ball prefix.

    The fit horizon ends at the earliest plausible touch/bounce/out event so
    the inferred parameters capture the pre-interaction flight.
    """
    obs = infer_observed_variant(features=features, ball_idx=ball_idx, kfl=kfl)

    def _with_diag(
        row: Dict[str, float | str],
        *,
        fit_h: int = 0,
        fit_evals: int = 0,
        rmse_xy: float = float("nan"),
        rmse_z: float = float("nan"),
        loss: float = float("nan"),
        near_zero_flag: int = 0,
        near_zero_rmse_flag: int = 0,
        adaptive_rounds: int = 0,
        speed_cap_used: float = float("nan"),
        vz_cap_used: float = float("nan"),
        spin_cap_used: float = float("nan"),
        backoff_depth: int = 0,
        backoff_from_frames: int = 0,
        endpoint_xy_err: float = float("nan"),
        endpoint_z_err: float = float("nan"),
        endpoint_enforced: int = 0,
    ) -> Dict[str, float | str]:
        out = dict(row)
        out.update(
            {
                "fit_rmse_xy": float(rmse_xy),
                "fit_rmse_z": float(rmse_z),
                "fit_loss": float(loss),
                "fit_frames": int(fit_h),
                "fit_evals": int(fit_evals),
                "fit_near_zero": int(near_zero_flag),
                "fit_near_zero_rmse": int(near_zero_rmse_flag),
                "fit_adaptive_rounds": int(adaptive_rounds),
                "fit_speed_cap_used": float(speed_cap_used),
                "fit_vz_cap_used": float(vz_cap_used),
                "fit_spin_cap_used": float(spin_cap_used),
                "fit_horizon_backoff_depth": int(backoff_depth),
                "fit_horizon_backoff_from_frames": int(backoff_from_frames),
                "fit_endpoint_xy_err": float(endpoint_xy_err),
                "fit_endpoint_z_err": float(endpoint_z_err),
                "fit_endpoint_enforced": int(endpoint_enforced),
            }
        )
        return out

    def _is_rmse_near_zero(rmse_xy_val: float, rmse_z_val: float) -> bool:
        return bool(rmse_xy_val <= float(near_zero_xy) and rmse_z_val <= float(near_zero_z))

    def _is_true_near_zero(
        rmse_xy_val: float,
        rmse_z_val: float,
        endpoint_xy_err_val: float,
        endpoint_z_err_val: float,
    ) -> bool:
        return bool(
            _is_rmse_near_zero(rmse_xy_val, rmse_z_val)
            and np.isfinite(float(endpoint_xy_err_val))
            and np.isfinite(float(endpoint_z_err_val))
            and float(endpoint_xy_err_val) <= float(near_zero_endpoint_xy)
            and float(endpoint_z_err_val) <= float(near_zero_endpoint_z)
        )

    naive_v0 = np.asarray([obs["v0x"], obs["v0y"], obs["v0z"]], dtype=np.float32)

    # GT trajectory from kick anchor onward.
    ball_gt = np.zeros((features.shape[0] - int(kfl), 3), dtype=np.float32)
    ball_gt[:, :2] = features[int(kfl) :, int(ball_idx), :2]
    if features.shape[-1] >= 5:
        ball_gt[:, 2] = features[int(kfl) :, int(ball_idx), 4]

    if ball_gt.shape[0] < 3 or not np.all(np.isfinite(ball_gt[0, :2])):
        return _with_diag(obs)

    fit_h = _detect_fit_horizon(
        ball_gt,
        max_fit_frames=int(max_fit_frames),
        entity_type=entity_type,
        player_pos_seq=player_pos_seq,
    )
    fit_h = int(min(fit_h, ball_gt.shape[0] - 1))
    if fit_h < 2:
        return _with_diag(obs, fit_h=int(max(1, fit_h + 1)))

    gt_fit = ball_gt[: fit_h + 1]
    start_pos = gt_fit[0]

    # Regression-style init from early displacement.
    dt = (np.arange(1, fit_h + 1, dtype=np.float32) / FPS).reshape(-1, 1)
    dxyz = gt_fit[1:] - gt_fit[0:1]
    with np.errstate(invalid="ignore", divide="ignore"):
        vel_est = dxyz / dt
    vxy_reg = np.nanmedian(vel_est[:, :2], axis=0)
    vz_reg = float(np.nanmedian(vel_est[:, 2])) if np.isfinite(np.nanmedian(vel_est[:, 2])) else float(naive_v0[2])
    reg_v0 = np.asarray([vxy_reg[0], vxy_reg[1], vz_reg], dtype=np.float32)
    if not np.all(np.isfinite(reg_v0)):
        reg_v0 = naive_v0.copy()

    speed_step = float(np.linalg.norm((gt_fit[1, :2] - gt_fit[0, :2]) * FPS))
    speed_cap = max(
        18.0,
        min(
            250.0,
            max(speed_step * 2.2 + 4.0, float(np.linalg.norm(naive_v0[:2])) * 2.8 + 8.0),
        ),
    )
    if caps is not None:
        speed_cap = max(float(speed_cap), float(caps.speed_xy_max) * 2.2)
        spin_cap = max(0.06, float(caps.spin_abs_max))
        vz_cap = max(float(caps.vz_abs_max) * 2.5, 8.0)
    else:
        spin_cap = 0.25
        vz_cap = max(abs(float(naive_v0[2])) * 2.6 + 4.0, 8.0)

    k_prior = int(np.clip(min(6, fit_h), 1, fit_h))
    disp = gt_fit[k_prior, :2] - gt_fit[0, :2]
    disp_norm = float(np.linalg.norm(disp))
    if disp_norm > 1e-6:
        ang_prior = float(np.arctan2(float(disp[1]), float(disp[0])))
        speed_prior = float(disp_norm * FPS / max(1.0, float(k_prior)))
    else:
        ang_prior = float(np.arctan2(float(naive_v0[1]), float(naive_v0[0]) + 1e-9))
        speed_prior = float(max(1e-3, np.linalg.norm(naive_v0[:2])))
    dt_prior = float(k_prior) / FPS
    vz_prior = float((float(gt_fit[k_prior, 2] - gt_fit[0, 2]) / max(1e-5, dt_prior)) + 0.5 * GRAVITY * dt_prior)
    vz_prior = float(np.clip(vz_prior, -vz_cap, vz_cap))

    init_candidates = [
        _clip_v0(naive_v0, speed_cap=speed_cap, vz_cap=vz_cap),
        _clip_v0(reg_v0, speed_cap=speed_cap, vz_cap=vz_cap),
        _clip_v0(0.5 * (naive_v0 + reg_v0), speed_cap=speed_cap, vz_cap=vz_cap),
    ]
    for kk in range(1, int(min(fit_h, 8)) + 1):
        dt_k = float(kk) / FPS
        if dt_k <= 1e-6:
            continue
        dxyz = (gt_fit[kk] - gt_fit[0]) / dt_k
        cand = np.asarray(
            [float(dxyz[0]), float(dxyz[1]), float(dxyz[2] + 0.5 * GRAVITY * dt_k)],
            dtype=np.float32,
        )
        if np.all(np.isfinite(cand)):
            init_candidates.append(_clip_v0(cand, speed_cap=speed_cap, vz_cap=vz_cap))

    best_v0 = init_candidates[0]
    best_spin = 0.0
    best_loss = float("inf")
    best_rmse_xy = float("inf")
    best_rmse_z = float("inf")
    fit_evals = 0
    adaptive_rounds = 0

    def _eval(v0_c: np.ndarray, spin_c: float) -> Tuple[float, float, float]:
        nonlocal fit_evals
        fit_evals += 1
        sim = _simulate_ball_with_spin(start_pos=start_pos, start_vel=v0_c, spin_scalar=spin_c, n_frames=fit_h + 1)
        return _fit_loss(sim_ball=sim, gt_ball=gt_fit)

    def _maybe_update(v0_c: np.ndarray, spin_c: float) -> None:
        nonlocal best_v0, best_spin, best_loss, best_rmse_xy, best_rmse_z
        loss, rmse_xy, rmse_z = _eval(v0_c, spin_c)
        if loss < best_loss:
            best_loss = float(loss)
            best_rmse_xy = float(rmse_xy)
            best_rmse_z = float(rmse_z)
            best_v0 = v0_c.copy()
            best_spin = float(spin_c)

    for v0_c in init_candidates:
        for spin0 in (0.0, -0.06, 0.06):
            _maybe_update(v0_c, float(np.clip(spin0, -spin_cap, spin_cap)))

    # Directed global exploration around observed displacement priors.
    for _ in range(192):
        speed = float(np.clip(rng.normal(speed_prior, max(2.0, 0.45 * max(1.0, speed_prior))), 0.2, speed_cap))
        ang = float(ang_prior + np.deg2rad(rng.normal(0.0, 35.0)))
        vxy = np.asarray([math.cos(ang) * speed, math.sin(ang) * speed], dtype=np.float32)
        vz = float(np.clip(rng.normal(vz_prior, max(1.8, 0.55 * max(1.0, abs(vz_prior)))), -vz_cap, vz_cap))
        spin = float(np.clip(rng.normal(0.0, 0.08), -spin_cap, spin_cap))
        cand_v0 = _clip_v0(np.asarray([vxy[0], vxy[1], vz], dtype=np.float32), speed_cap=speed_cap, vz_cap=vz_cap)
        _maybe_update(cand_v0, spin)

    # Uniform global exploration to escape local basins.
    for _ in range(192):
        speed = float(rng.uniform(0.2, speed_cap))
        ang = float(np.deg2rad(rng.uniform(-180.0, 180.0)))
        vxy = np.asarray([math.cos(ang) * speed, math.sin(ang) * speed], dtype=np.float32)
        vz = float(rng.uniform(-vz_cap, vz_cap))
        spin = float(rng.uniform(-spin_cap, spin_cap))
        cand_v0 = _clip_v0(np.asarray([vxy[0], vxy[1], vz], dtype=np.float32), speed_cap=speed_cap, vz_cap=vz_cap)
        _maybe_update(cand_v0, spin)

    stage_cfg = [
        # n_samples, speed_sigma, angle_sigma_deg, vz_sigma, spin_sigma
        (160, 0.28, 20.0, 3.0, 0.09),
        (128, 0.12, 8.0, 1.3, 0.05),
        (96, 0.05, 3.0, 0.45, 0.02),
    ]

    for n_samples, speed_sigma, ang_sigma, vz_sigma, spin_sigma in stage_cfg:
        base_v0 = best_v0.copy()
        base_spin = float(best_spin)
        base_speed = float(max(0.2, np.linalg.norm(base_v0[:2])))
        for _ in range(int(n_samples)):
            speed_scale = float(np.clip(rng.normal(1.0, speed_sigma), 0.2, 3.0))
            ang = float(rng.normal(0.0, ang_sigma))
            vxy = _rotate((base_v0[:2] / base_speed) * (base_speed * speed_scale), ang)
            vz = float(base_v0[2] + rng.normal(0.0, vz_sigma))
            spin = float(np.clip(base_spin + rng.normal(0.0, spin_sigma), -spin_cap, spin_cap))
            cand_v0 = _clip_v0(np.asarray([vxy[0], vxy[1], vz], dtype=np.float32), speed_cap=speed_cap, vz_cap=vz_cap)
            _maybe_update(cand_v0, spin)

        # Early stop once the pre-event prefix is well matched.
        if best_rmse_xy <= 0.08 and best_rmse_z <= 0.20:
            break

    # Population-based anchored refinement (CEM-style) for stubborn clips.
    if best_rmse_xy > 0.04 or best_rmse_z > 0.12:
        mean_v = best_v0.copy()
        mean_s = float(best_spin)
        std_v = np.asarray(
            [
                max(0.8, 0.18 * max(1.0, abs(float(mean_v[0])))),
                max(0.8, 0.18 * max(1.0, abs(float(mean_v[1])))),
                max(0.6, 0.20 * max(1.0, abs(float(mean_v[2])))),
            ],
            dtype=np.float32,
        )
        std_s = max(0.02, 0.25 * max(0.03, abs(float(mean_s))))

        for _ in range(7):
            n_pop = 144
            cand_v = np.zeros((n_pop, 3), dtype=np.float32)
            cand_s = np.zeros((n_pop,), dtype=np.float32)
            losses = np.full((n_pop,), np.inf, dtype=np.float64)

            for i in range(n_pop):
                if i == 0:
                    v = mean_v.copy()
                    s = float(mean_s)
                elif i == 1:
                    v = best_v0.copy()
                    s = float(best_spin)
                else:
                    v = np.asarray(
                        [
                            float(rng.normal(float(mean_v[0]), float(std_v[0]))),
                            float(rng.normal(float(mean_v[1]), float(std_v[1]))),
                            float(rng.normal(float(mean_v[2]), float(std_v[2]))),
                        ],
                        dtype=np.float32,
                    )
                    s = float(rng.normal(float(mean_s), float(std_s)))
                v = _clip_v0(v, speed_cap=speed_cap, vz_cap=vz_cap)
                s = float(np.clip(s, -spin_cap, spin_cap))
                cand_v[i] = v
                cand_s[i] = s
                loss_i, rm_xy_i, rm_z_i = _eval(v, s)
                losses[i] = float(loss_i)
                if loss_i < best_loss:
                    best_loss = float(loss_i)
                    best_rmse_xy = float(rm_xy_i)
                    best_rmse_z = float(rm_z_i)
                    best_v0 = v.copy()
                    best_spin = float(s)

            elite_n = 18
            elite_idx = np.argsort(losses)[:elite_n]
            elite_v = cand_v[elite_idx]
            elite_s = cand_s[elite_idx]
            w = np.linspace(1.0, 2.0, elite_n, dtype=np.float32)
            w /= float(np.sum(w))

            mean_v = np.sum(elite_v * w[:, None], axis=0).astype(np.float32)
            mean_s = float(np.sum(elite_s * w))
            std_v = np.sqrt(np.sum(((elite_v - mean_v[None, :]) ** 2) * w[:, None], axis=0)).astype(np.float32)
            std_v = np.maximum(std_v * 0.85, np.asarray([0.18, 0.18, 0.12], dtype=np.float32))
            std_s = max(0.008, float(np.sqrt(np.sum(((elite_s - mean_s) ** 2) * w)) * 0.85))

            if best_rmse_xy <= 0.03 and best_rmse_z <= 0.08:
                break

    # Final coordinate-pattern local search.
    if best_rmse_xy > 0.02 or best_rmse_z > 0.05:
        step = np.asarray(
            [
                max(0.30, 0.06 * max(1.0, abs(float(best_v0[0])))),
                max(0.30, 0.06 * max(1.0, abs(float(best_v0[1])))),
                max(0.22, 0.06 * max(1.0, abs(float(best_v0[2])))),
                max(0.010, 0.06 * max(0.02, abs(float(best_spin)))),
            ],
            dtype=np.float32,
        )
        for _ in range(6):
            improved = False
            base = np.asarray([best_v0[0], best_v0[1], best_v0[2], best_spin], dtype=np.float32)
            for d in range(4):
                for sgn in (-1.0, 1.0):
                    cand = base.copy()
                    cand[d] = float(cand[d] + sgn * step[d])
                    cand_v0 = _clip_v0(cand[:3], speed_cap=speed_cap, vz_cap=vz_cap)
                    cand_spin = float(np.clip(cand[3], -spin_cap, spin_cap))
                    loss_i, rm_xy_i, rm_z_i = _eval(cand_v0, cand_spin)
                    if loss_i < best_loss:
                        best_loss = float(loss_i)
                        best_rmse_xy = float(rm_xy_i)
                        best_rmse_z = float(rm_z_i)
                        best_v0 = cand_v0.copy()
                        best_spin = float(cand_spin)
                        improved = True
            if not improved:
                step *= 0.55
            if best_rmse_xy <= 0.02 and best_rmse_z <= 0.05:
                break

    # Adaptive long-search for stubborn high-error clips.
    if best_rmse_xy > float(high_error_xy) or best_rmse_z > float(high_error_z):
        for rr in range(int(max_adaptive_rounds)):
            if _is_rmse_near_zero(best_rmse_xy, best_rmse_z):
                break
            adaptive_rounds += 1

            # Expand parameter-space bounds for this round.
            speed_cap = min(
                420.0,
                max(float(speed_cap) * (1.20 + 0.07 * rr), float(np.linalg.norm(best_v0[:2])) * 1.25 + 2.0),
            )
            vz_cap = min(
                120.0,
                max(float(vz_cap) * (1.18 + 0.04 * rr), abs(float(best_v0[2])) * 1.35 + 1.2),
            )
            spin_cap = min(
                1.20,
                max(float(spin_cap) * (1.28 + 0.10 * rr), abs(float(best_spin)) * 1.55 + 0.08),
            )

            base_speed = float(max(0.2, np.linalg.norm(best_v0[:2])))
            base_ang = float(np.arctan2(float(best_v0[1]), float(best_v0[0]) + 1e-9))
            base_vz = float(best_v0[2])
            n_directed = 260 + 80 * rr
            n_uniform = 220 + 70 * rr

            for _ in range(int(n_directed)):
                use_best = bool(rng.random() < 0.70)
                c_speed = base_speed if use_best else speed_prior
                c_ang = base_ang if use_best else ang_prior
                c_vz = base_vz if use_best else vz_prior

                speed = float(np.clip(rng.normal(c_speed, max(1.2, 0.35 * max(1.0, c_speed))), 0.2, speed_cap))
                ang = float(c_ang + np.deg2rad(rng.normal(0.0, 12.0 + 6.0 * rr)))
                vxy = np.asarray([math.cos(ang) * speed, math.sin(ang) * speed], dtype=np.float32)
                vz = float(np.clip(rng.normal(c_vz, max(0.8, 0.22 * max(1.0, abs(c_vz)))), -vz_cap, vz_cap))
                spin = float(np.clip(rng.normal(best_spin, 0.09 + 0.03 * rr), -spin_cap, spin_cap))
                cand_v0 = _clip_v0(np.asarray([vxy[0], vxy[1], vz], dtype=np.float32), speed_cap=speed_cap, vz_cap=vz_cap)
                _maybe_update(cand_v0, spin)

            for _ in range(int(n_uniform)):
                speed = float(rng.uniform(0.2, speed_cap))
                ang = float(np.deg2rad(rng.uniform(-180.0, 180.0)))
                vxy = np.asarray([math.cos(ang) * speed, math.sin(ang) * speed], dtype=np.float32)
                vz = float(rng.uniform(-vz_cap, vz_cap))
                spin = float(rng.uniform(-spin_cap, spin_cap))
                cand_v0 = _clip_v0(np.asarray([vxy[0], vxy[1], vz], dtype=np.float32), speed_cap=speed_cap, vz_cap=vz_cap)
                _maybe_update(cand_v0, spin)

            # Anchored CEM refinement with enlarged population.
            mean_v = best_v0.copy()
            mean_s = float(best_spin)
            std_v = np.asarray(
                [
                    max(0.45, 0.14 * max(1.0, abs(float(mean_v[0])))),
                    max(0.45, 0.14 * max(1.0, abs(float(mean_v[1])))),
                    max(0.30, 0.14 * max(1.0, abs(float(mean_v[2])))),
                ],
                dtype=np.float32,
            )
            std_s = max(0.015, 0.30 * max(0.02, abs(float(mean_s))))

            n_cem_iter = 4 + rr
            n_pop = 224 + 64 * rr
            elite_n = 24 + 4 * rr
            for _ in range(int(n_cem_iter)):
                cand_v = np.zeros((int(n_pop), 3), dtype=np.float32)
                cand_s = np.zeros((int(n_pop),), dtype=np.float32)
                losses = np.full((int(n_pop),), np.inf, dtype=np.float64)

                for i in range(int(n_pop)):
                    if i == 0:
                        v = mean_v.copy()
                        s = float(mean_s)
                    elif i == 1:
                        v = best_v0.copy()
                        s = float(best_spin)
                    else:
                        v = np.asarray(
                            [
                                float(rng.normal(float(mean_v[0]), float(std_v[0]))),
                                float(rng.normal(float(mean_v[1]), float(std_v[1]))),
                                float(rng.normal(float(mean_v[2]), float(std_v[2]))),
                            ],
                            dtype=np.float32,
                        )
                        s = float(rng.normal(float(mean_s), float(std_s)))
                    v = _clip_v0(v, speed_cap=speed_cap, vz_cap=vz_cap)
                    s = float(np.clip(s, -spin_cap, spin_cap))
                    cand_v[i] = v
                    cand_s[i] = s
                    loss_i, rm_xy_i, rm_z_i = _eval(v, s)
                    losses[i] = float(loss_i)
                    if loss_i < best_loss:
                        best_loss = float(loss_i)
                        best_rmse_xy = float(rm_xy_i)
                        best_rmse_z = float(rm_z_i)
                        best_v0 = v.copy()
                        best_spin = float(s)

                elite_idx = np.argsort(losses)[: int(elite_n)]
                elite_v = cand_v[elite_idx]
                elite_s = cand_s[elite_idx]
                w = np.linspace(1.0, 2.0, int(elite_n), dtype=np.float32)
                w /= float(np.sum(w))
                mean_v = np.sum(elite_v * w[:, None], axis=0).astype(np.float32)
                mean_s = float(np.sum(elite_s * w))
                std_v = np.sqrt(np.sum(((elite_v - mean_v[None, :]) ** 2) * w[:, None], axis=0)).astype(np.float32)
                std_v = np.maximum(std_v * 0.82, np.asarray([0.14, 0.14, 0.10], dtype=np.float32))
                std_s = max(0.006, float(np.sqrt(np.sum(((elite_s - mean_s) ** 2) * w)) * 0.82))
                if _is_rmse_near_zero(best_rmse_xy, best_rmse_z):
                    break

            # Final local coordinate descent each round.
            step = np.asarray(
                [
                    max(0.12, 0.045 * max(1.0, abs(float(best_v0[0])))),
                    max(0.12, 0.045 * max(1.0, abs(float(best_v0[1])))),
                    max(0.08, 0.045 * max(1.0, abs(float(best_v0[2])))),
                    max(0.004, 0.050 * max(0.02, abs(float(best_spin)))),
                ],
                dtype=np.float32,
            )
            for _ in range(10):
                improved = False
                base = np.asarray([best_v0[0], best_v0[1], best_v0[2], best_spin], dtype=np.float32)
                for d in range(4):
                    for sgn in (-1.0, 1.0):
                        cand = base.copy()
                        cand[d] = float(cand[d] + sgn * step[d])
                        cand_v0 = _clip_v0(cand[:3], speed_cap=speed_cap, vz_cap=vz_cap)
                        cand_spin = float(np.clip(cand[3], -spin_cap, spin_cap))
                        loss_i, rm_xy_i, rm_z_i = _eval(cand_v0, cand_spin)
                        if loss_i < best_loss:
                            best_loss = float(loss_i)
                            best_rmse_xy = float(rm_xy_i)
                            best_rmse_z = float(rm_z_i)
                            best_v0 = cand_v0.copy()
                            best_spin = float(cand_spin)
                            improved = True
                if not improved:
                    step *= 0.62
                if _is_rmse_near_zero(best_rmse_xy, best_rmse_z):
                    break

    # Residual-driven horizon backoff fallback for stubborn clips:
    # if the model cannot match the longer prefix, iteratively trim the fit
    # horizon to the first sustained mismatch frame and re-run the optimizer.
    near_zero_now = _is_rmse_near_zero(best_rmse_xy, best_rmse_z)
    if (
        (not near_zero_now)
        and bool(allow_horizon_backoff)
        and int(horizon_backoff_depth) < int(max_horizon_backoff)
        and int(fit_h) > 2
    ):
        sim_best = _simulate_ball_with_spin(
            start_pos=start_pos,
            start_vel=best_v0,
            spin_scalar=float(best_spin),
            n_frames=fit_h + 1,
        )
        err_xy = np.linalg.norm(sim_best[:, :2] - gt_fit[:, :2], axis=1)
        err_xy = np.asarray(err_xy, dtype=np.float32)
        thr = float(max(0.05, 1.5 * float(near_zero_xy)))
        bad = np.where(err_xy > thr)[0]
        if bad.size > 0:
            cut = int(max(2, int(bad[0]) - 1))
        else:
            cut = int(max(2, int(round(0.75 * float(fit_h)))))
        if cut < int(fit_h):
            child = infer_observed_variant_fitted(
                features=features,
                ball_idx=ball_idx,
                kfl=kfl,
                caps=caps,
                rng=rng,
                max_fit_frames=int(cut),
                player_pos_seq=player_pos_seq,
                entity_type=entity_type,
                near_zero_xy=float(near_zero_xy),
                near_zero_z=float(near_zero_z),
                high_error_xy=float(high_error_xy),
                high_error_z=float(high_error_z),
                max_adaptive_rounds=max(int(max_adaptive_rounds), 6),
                allow_horizon_backoff=bool(allow_horizon_backoff),
                max_horizon_backoff=int(max_horizon_backoff),
                horizon_backoff_depth=int(horizon_backoff_depth) + 1,
                allow_endpoint_enforce=bool(allow_endpoint_enforce),
            )
            child_rmse_xy = float(child.get("fit_rmse_xy", float("inf")))
            child_rmse_z = float(child.get("fit_rmse_z", float("inf")))
            child_near = bool(int(child.get("fit_near_zero_rmse", 0) or 0) == 1) or _is_rmse_near_zero(child_rmse_xy, child_rmse_z)
            if child_near or (child_rmse_xy + 1e-8 < float(best_rmse_xy)):
                child = dict(child)
                child["fit_horizon_backoff_depth"] = int(max(int(child.get("fit_horizon_backoff_depth", 0) or 0), int(horizon_backoff_depth) + 1))
                child["fit_horizon_backoff_from_frames"] = int(max(int(child.get("fit_horizon_backoff_from_frames", 0) or 0), int(fit_h + 1)))
                return child

    out = {
        "variant_id": "observed",
        "variant_group": "observed",
        "v0x": float(best_v0[0]),
        "v0y": float(best_v0[1]),
        "v0z": float(best_v0[2]),
        "spin_scalar": float(best_spin),
    }
    sim_best = _simulate_ball_with_spin(
        start_pos=start_pos,
        start_vel=best_v0,
        spin_scalar=float(best_spin),
        n_frames=fit_h + 1,
    )
    end_xy_err = float(np.linalg.norm(sim_best[-1, :2] - gt_fit[-1, :2]))
    end_z_err = float(abs(float(sim_best[-1, 2] - gt_fit[-1, 2])))
    endpoint_enforced = 0

    # Endpoint enforcement fallback: for practically-unfit clips,
    # explicitly match the final pre-touch/bounce position.
    if (
        (not _is_true_near_zero(best_rmse_xy, best_rmse_z, end_xy_err, end_z_err))
        and bool(allow_endpoint_enforce)
        and (
            best_rmse_xy > float(high_error_xy)
            or best_rmse_z > float(high_error_z)
            or end_xy_err > float(near_zero_endpoint_xy)
            or end_z_err > float(near_zero_endpoint_z)
        )
    ):
        target = gt_fit[-1]
        dt_end = max(1.0 / FPS, float(fit_h) / FPS)
        end_seed_v0 = np.asarray(
            [
                float((target[0] - start_pos[0]) / dt_end),
                float((target[1] - start_pos[1]) / dt_end),
                float((target[2] - start_pos[2]) / dt_end + 0.5 * GRAVITY * dt_end),
            ],
            dtype=np.float32,
        )
        end_seed_v0 = _clip_v0(end_seed_v0, speed_cap=speed_cap, vz_cap=vz_cap)

        best_ep_v0 = best_v0.copy()
        best_ep_spin = float(best_spin)
        best_ep_loss = float("inf")
        best_ep_rmse_xy = float(best_rmse_xy)
        best_ep_rmse_z = float(best_rmse_z)
        best_ep_end_xy = float(end_xy_err)
        best_ep_end_z = float(end_z_err)

        def _eval_endpoint(v0_c: np.ndarray, spin_c: float) -> Tuple[float, float, float, float, float]:
            sim = _simulate_ball_with_spin(start_pos=start_pos, start_vel=v0_c, spin_scalar=spin_c, n_frames=fit_h + 1)
            loss_full, rm_xy, rm_z = _fit_loss(sim_ball=sim, gt_ball=gt_fit)
            ep_xy = float(np.linalg.norm(sim[-1, :2] - gt_fit[-1, :2]))
            ep_z = float(abs(float(sim[-1, 2] - gt_fit[-1, 2])))
            # Prioritize endpoint fidelity while keeping overall path sane.
            ep_loss = 9.0 * (ep_xy ** 2) + 12.0 * (ep_z ** 2) + 0.04 * float(loss_full)
            return float(ep_loss), float(rm_xy), float(rm_z), float(ep_xy), float(ep_z)

        def _maybe_update_endpoint(v0_c: np.ndarray, spin_c: float) -> None:
            nonlocal best_ep_v0, best_ep_spin, best_ep_loss, best_ep_rmse_xy, best_ep_rmse_z, best_ep_end_xy, best_ep_end_z
            nonlocal fit_evals
            fit_evals += 1
            ep_loss, rm_xy, rm_z, ep_xy, ep_z = _eval_endpoint(v0_c, spin_c)
            if ep_loss < best_ep_loss:
                best_ep_loss = float(ep_loss)
                best_ep_rmse_xy = float(rm_xy)
                best_ep_rmse_z = float(rm_z)
                best_ep_end_xy = float(ep_xy)
                best_ep_end_z = float(ep_z)
                best_ep_v0 = v0_c.copy()
                best_ep_spin = float(spin_c)

        for cand_v0 in (best_v0, end_seed_v0, _clip_v0(0.5 * (best_v0 + end_seed_v0), speed_cap=speed_cap, vz_cap=vz_cap)):
            for s0 in (best_spin, 0.0, float(np.clip(best_spin * 0.5, -spin_cap, spin_cap))):
                _maybe_update_endpoint(cand_v0, float(np.clip(s0, -spin_cap, spin_cap)))

        # Directed endpoint-centered samples.
        for _ in range(260):
            base = best_ep_v0 if bool(rng.random() < 0.72) else end_seed_v0
            v = np.asarray(
                [
                    float(rng.normal(float(base[0]), max(0.5, 0.12 * max(1.0, abs(float(base[0])))))),
                    float(rng.normal(float(base[1]), max(0.5, 0.12 * max(1.0, abs(float(base[1])))))),
                    float(rng.normal(float(base[2]), max(0.35, 0.12 * max(1.0, abs(float(base[2])))))),
                ],
                dtype=np.float32,
            )
            s = float(rng.normal(float(best_ep_spin), 0.08))
            v = _clip_v0(v, speed_cap=speed_cap, vz_cap=vz_cap)
            s = float(np.clip(s, -spin_cap, spin_cap))
            _maybe_update_endpoint(v, s)

        # CEM endpoint-focused refinement.
        mean_v = best_ep_v0.copy()
        mean_s = float(best_ep_spin)
        std_v = np.asarray(
            [
                max(0.35, 0.15 * max(1.0, abs(float(mean_v[0])))),
                max(0.35, 0.15 * max(1.0, abs(float(mean_v[1])))),
                max(0.22, 0.15 * max(1.0, abs(float(mean_v[2])))),
            ],
            dtype=np.float32,
        )
        std_s = max(0.010, 0.25 * max(0.02, abs(float(mean_s))))
        for _ in range(6):
            n_pop = 192
            elite_n = 20
            cand_v = np.zeros((n_pop, 3), dtype=np.float32)
            cand_s = np.zeros((n_pop,), dtype=np.float32)
            losses = np.full((n_pop,), np.inf, dtype=np.float64)

            for i in range(n_pop):
                if i == 0:
                    v = mean_v.copy()
                    s = float(mean_s)
                elif i == 1:
                    v = best_ep_v0.copy()
                    s = float(best_ep_spin)
                else:
                    v = np.asarray(
                        [
                            float(rng.normal(float(mean_v[0]), float(std_v[0]))),
                            float(rng.normal(float(mean_v[1]), float(std_v[1]))),
                            float(rng.normal(float(mean_v[2]), float(std_v[2]))),
                        ],
                        dtype=np.float32,
                    )
                    s = float(rng.normal(float(mean_s), float(std_s)))
                v = _clip_v0(v, speed_cap=speed_cap, vz_cap=vz_cap)
                s = float(np.clip(s, -spin_cap, spin_cap))
                cand_v[i] = v
                cand_s[i] = s
                fit_evals += 1
                ep_loss, rm_xy, rm_z, ep_xy, ep_z = _eval_endpoint(v, s)
                losses[i] = float(ep_loss)
                if ep_loss < best_ep_loss:
                    best_ep_loss = float(ep_loss)
                    best_ep_rmse_xy = float(rm_xy)
                    best_ep_rmse_z = float(rm_z)
                    best_ep_end_xy = float(ep_xy)
                    best_ep_end_z = float(ep_z)
                    best_ep_v0 = v.copy()
                    best_ep_spin = float(s)

            elite_idx = np.argsort(losses)[:elite_n]
            elite_v = cand_v[elite_idx]
            elite_s = cand_s[elite_idx]
            w = np.linspace(1.0, 2.0, elite_n, dtype=np.float32)
            w /= float(np.sum(w))
            mean_v = np.sum(elite_v * w[:, None], axis=0).astype(np.float32)
            mean_s = float(np.sum(elite_s * w))
            std_v = np.sqrt(np.sum(((elite_v - mean_v[None, :]) ** 2) * w[:, None], axis=0)).astype(np.float32)
            std_v = np.maximum(std_v * 0.80, np.asarray([0.12, 0.12, 0.08], dtype=np.float32))
            std_s = max(0.005, float(np.sqrt(np.sum(((elite_s - mean_s) ** 2) * w)) * 0.80))

        if (best_ep_end_xy + 1e-7 < end_xy_err) or (best_ep_end_z + 1e-7 < end_z_err):
            best_v0 = best_ep_v0.copy()
            best_spin = float(best_ep_spin)
            best_rmse_xy = float(best_ep_rmse_xy)
            best_rmse_z = float(best_ep_rmse_z)
            best_loss = float(
                _fit_loss(
                    sim_ball=_simulate_ball_with_spin(
                        start_pos=start_pos,
                        start_vel=best_v0,
                        spin_scalar=best_spin,
                        n_frames=fit_h + 1,
                    ),
                    gt_ball=gt_fit,
                )[0]
            )
            end_xy_err = float(best_ep_end_xy)
            end_z_err = float(best_ep_end_z)
            endpoint_enforced = 1

    true_near_zero_flag = int(_is_true_near_zero(best_rmse_xy, best_rmse_z, end_xy_err, end_z_err))
    rmse_near_zero_flag = int(_is_rmse_near_zero(best_rmse_xy, best_rmse_z))
    out = _with_diag(
        out,
        fit_h=int(fit_h + 1),
        fit_evals=int(fit_evals),
        rmse_xy=float(best_rmse_xy),
        rmse_z=float(best_rmse_z),
        loss=float(best_loss),
        near_zero_flag=int(true_near_zero_flag),
        near_zero_rmse_flag=int(rmse_near_zero_flag),
        adaptive_rounds=int(adaptive_rounds),
        speed_cap_used=float(speed_cap),
        vz_cap_used=float(vz_cap),
        spin_cap_used=float(spin_cap),
        backoff_depth=int(horizon_backoff_depth),
        backoff_from_frames=0,
        endpoint_xy_err=float(end_xy_err),
        endpoint_z_err=float(end_z_err),
        endpoint_enforced=int(endpoint_enforced),
    )
    return out


def sample_local_variants(
    observed_v0: np.ndarray,
    n: int,
    caps: VariationCaps,
    rng: np.random.Generator,
) -> List[Dict[str, float | str]]:
    out: List[Dict[str, float | str]] = []
    obs_vxy = observed_v0[:2].astype(np.float32)
    obs_vz = float(observed_v0[2])
    obs_speed = float(np.linalg.norm(obs_vxy))
    base_dir = obs_vxy / max(obs_speed, 1e-6)
    noise_scale = float(max(0.10, float(os.getenv("MCPS_LOCAL_NOISE_SCALE", "1.0"))))
    speed_sigma = 0.05 * noise_scale
    speed_lo = max(0.15, 1.0 - 0.20 * noise_scale)
    speed_hi = min(3.00, 1.0 + 0.20 * noise_scale)
    angle_sigma = 5.0 * noise_scale
    angle_clip = max(15.0, 15.0 * noise_scale)
    spin_sigma = 0.035 * noise_scale
    vz_sigma = max(0.25, 0.05 * max(abs(obs_vz), 1.0)) * noise_scale

    for i in range(int(n)):
        speed_scale = float(np.clip(rng.normal(1.0, speed_sigma), speed_lo, speed_hi))
        angle = float(np.clip(rng.normal(0.0, angle_sigma), -angle_clip, angle_clip))
        spin = float(np.clip(rng.normal(0.0, spin_sigma), -caps.spin_abs_max, caps.spin_abs_max))

        speed = max(0.5, obs_speed * speed_scale)
        vxy = base_dir * speed
        vxy = _rotate(vxy, angle)
        vxy = _clip_xy_speed(vxy, caps.speed_xy_max)
        vz = float(np.clip(rng.normal(obs_vz, vz_sigma), -caps.vz_abs_max, caps.vz_abs_max))

        out.append(
            {
                "variant_id": f"local_{i:03d}",
                "variant_group": "local",
                "v0x": float(vxy[0]),
                "v0y": float(vxy[1]),
                "v0z": float(vz),
                "spin_scalar": float(spin),
            }
        )
    return out


def sample_global_variants(
    observed_v0: np.ndarray,
    n: int,
    caps: VariationCaps,
    rng: np.random.Generator,
) -> List[Dict[str, float | str]]:
    out: List[Dict[str, float | str]] = []
    obs_speed = float(np.linalg.norm(observed_v0[:2]))
    speed_ref = max(obs_speed, 2.0)
    obs_vz = float(observed_v0[2])

    for i in range(int(n)):
        angle = float(rng.uniform(-180.0, 180.0))
        # Around 30% spread, wide angle support.
        speed_scale = float(np.clip(rng.normal(1.0, 0.30), 0.3, 1.8))
        speed = min(caps.speed_xy_max, max(0.5, speed_ref * speed_scale))
        dir_xy = np.asarray([np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))], dtype=np.float32)
        vxy = dir_xy * speed
        vxy = _clip_xy_speed(vxy, caps.speed_xy_max)

        vz = float(np.clip(rng.normal(obs_vz, max(0.8, 0.20 * max(abs(obs_vz), 1.0))), -caps.vz_abs_max, caps.vz_abs_max))
        spin = float(np.clip(rng.normal(0.0, 0.04), -caps.spin_abs_max, caps.spin_abs_max))

        out.append(
            {
                "variant_id": f"global_{i:03d}",
                "variant_group": "global",
                "v0x": float(vxy[0]),
                "v0y": float(vxy[1]),
                "v0z": float(vz),
                "spin_scalar": float(spin),
            }
        )
    return out
