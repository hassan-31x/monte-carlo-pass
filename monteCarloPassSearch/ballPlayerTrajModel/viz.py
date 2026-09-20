#!/usr/bin/env python3
"""Rollout and visualization utility for joint player+ball trajectory models."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D

from dataset import (
    NormalizationStats,
    STOP_CLASS_COUNT,
    STOP_CLASS_NAMES,
    STOP_CONTINUE,
    STOP_PAD,
    create_datasets_from_preprocessed,
)
from model import TrajectoryTransformer, sample_multi, sample_single


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize rollouts from ballPlayerTrajModel checkpoint.")

    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--preprocessed-dir", type=str, default=None)
    parser.add_argument("--split", type=str, choices=["train", "val"], default="val")
    parser.add_argument("--window-idx", type=int, default=0)
    parser.add_argument(
        "--canonical-feature-normalization",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Match training-time canonical normalization for mixed-source inputs "
            "(unit-scale stop channels and z-observation flags when present)."
        ),
    )

    parser.add_argument("--history", type=int, default=None)
    parser.add_argument(
        "--gt-ball-conditioning-steps",
        type=int,
        default=None,
        help="Override checkpoint-configured number of initial ball-conditioned steps.",
    )
    parser.add_argument(
        "--rollout-steps",
        type=int,
        default=None,
        help=(
            "Rollout horizon for main plots. If omitted, uses checkpoint training horizon "
            "(one_shot_frames for one-shot checkpoints, rollout_steps otherwise)."
        ),
    )
    parser.add_argument(
        "--synthetic-rollout-steps",
        type=int,
        default=120,
        help="Fixed horizon for synthetic counterfactual panels.",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--diffusion-steps",
        type=int,
        default=1000,
        help="Number of denoising steps for one-shot diffusion during visualization.",
    )
    parser.add_argument("--mode", type=str, choices=["auto", "multi", "single"], default="auto")

    parser.add_argument(
        "--kick-speed-scale",
        type=float,
        default=1.0,
        help="Scale factor for kick-frame ball velocity in the history context.",
    )
    parser.add_argument(
        "--kick-angle-deg",
        type=float,
        default=0.0,
        help="Angle offset (degrees) applied to kick-frame ball velocity in history context.",
    )
    parser.add_argument(
        "--kick-spin",
        type=float,
        default=0.0,
        help=(
            "Pseudo-spin control: perturbs the pre-kick velocity direction to introduce rotational "
            "history cues."
        ),
    )

    parser.add_argument("--output", type=str, default="/mnt/data/remains/opta2026/ballPlayerTrajModel/rollout.png")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def infer_model_mode(requested: str, ckpt_args: Dict) -> str:
    if requested != "auto":
        return requested
    return str(ckpt_args.get("mode", "multi"))


def _rotation_matrix_deg(angle_deg: float) -> np.ndarray:
    th = np.deg2rad(float(angle_deg))
    c, s = np.cos(th), np.sin(th)
    return np.asarray([[c, -s], [s, c]], dtype=np.float32)


def apply_ball_kick_counterfactual(
    history_features: np.ndarray,
    kick_speed_scale: float,
    kick_angle_deg: float,
    kick_spin: float,
) -> np.ndarray:
    """Modify last history frames to emulate kick counterfactuals."""
    out = history_features.copy()
    if out.shape[0] < 2:
        return out

    ball = 22
    v = out[-1, ball, :2] - out[-2, ball, :2]
    if float(np.linalg.norm(v)) < 1e-6:
        v = out[-1, ball, 2:4].copy()
    if float(np.linalg.norm(v)) < 1e-6:
        v = np.asarray([0.1, 0.0], dtype=np.float32)

    v_new = (_rotation_matrix_deg(kick_angle_deg) @ v.astype(np.float32)) * float(kick_speed_scale)
    out[-1, ball, 2:4] = v_new
    out[-1, ball, :2] = out[-2, ball, :2] + v_new

    # Inject pseudo-spin by perturbing the previous frame's velocity.
    if out.shape[0] >= 3 and abs(float(kick_spin)) > 1e-8:
        perp = np.asarray([-v_new[1], v_new[0]], dtype=np.float32)
        v_prev = v_new - float(kick_spin) * perp
        out[-2, ball, 2:4] = v_prev
        out[-2, ball, :2] = out[-3, ball, :2] + v_prev

    return out


def _make_diffusion_alpha_schedule(
    steps: int,
    beta_start: float = 1e-4,
    beta_end: float = 2e-2,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    betas = torch.linspace(beta_start, beta_end, steps, device=device, dtype=dtype)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return betas, alphas, alpha_bars


def _fit_delta_scale(delta_scale: np.ndarray, delta_dim: int) -> np.ndarray:
    src = np.asarray(delta_scale, dtype=np.float32).reshape(-1)
    out = np.ones((int(delta_dim),), dtype=np.float32)
    take = min(out.shape[0], src.shape[0])
    out[:take] = src[:take]
    return out


def build_synthetic_ball_prefix(
    history_features: np.ndarray,
    steps: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Construct a synthetic ball trajectory prefix by constant-velocity extrapolation."""
    steps = max(0, int(steps))
    d_in = int(history_features.shape[-1])
    out = np.zeros((steps, d_in), dtype=np.float32)
    mask = np.ones((steps,), dtype=bool)
    if steps == 0:
        return out, mask

    ball_idx = 22
    prev = history_features[-1, ball_idx, :].astype(np.float32).copy()

    if history_features.shape[0] >= 2:
        vel_xy = (
            history_features[-1, ball_idx, :2].astype(np.float32)
            - history_features[-2, ball_idx, :2].astype(np.float32)
        )
    else:
        vel_xy = (
            history_features[-1, ball_idx, 2:4].astype(np.float32)
            if d_in >= 4
            else np.zeros((2,), dtype=np.float32)
        )
    vz = 0.0
    if d_in >= 5:
        if history_features.shape[0] >= 2:
            vz = float(history_features[-1, ball_idx, 4] - history_features[-2, ball_idx, 4])
        elif d_in >= 6:
            vz = float(history_features[-1, ball_idx, 5])

    for t in range(steps):
        nxt = prev.copy()
        nxt[:2] = prev[:2] + vel_xy
        if d_in >= 4:
            nxt[2:4] = vel_xy
        if d_in >= 5:
            nxt[4] = prev[4] + vz
            if d_in >= 6:
                nxt[5] = vz
        out[t] = nxt
        prev = nxt

    return out, mask


def build_forced_ball_target_norm(
    history_features: np.ndarray,
    forced_ball_features: np.ndarray,
    forced_ball_mask: np.ndarray,
    rollout_steps: int,
    delta_dim: int,
    delta_scale_np: np.ndarray,
    residual_model: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build forced ball deltas in one-shot target space (residual or absolute)."""
    steps = max(0, int(rollout_steps))
    out = np.zeros((steps, delta_dim), dtype=np.float32)
    valid = np.zeros((steps, delta_dim), dtype=bool)
    if steps == 0:
        return out, valid

    forced_ball_features = np.asarray(forced_ball_features, dtype=np.float32)
    forced_ball_mask = np.asarray(forced_ball_mask, dtype=bool).reshape(-1)
    force_steps = min(steps, int(forced_ball_features.shape[0]), int(forced_ball_mask.shape[0]))

    ball_idx = 22
    prev = history_features[-1, ball_idx, :].astype(np.float32).copy()
    for t in range(force_steps):
        if not bool(forced_ball_mask[t]):
            continue
        curr = forced_ball_features[t].astype(np.float32)
        dxy = curr[:2] - prev[:2]
        out[t, :2] = dxy / delta_scale_np[:2]
        valid[t, :2] = True
        if delta_dim >= 3 and history_features.shape[-1] >= 5 and curr.shape[0] >= 5:
            dz = float(curr[4] - prev[4])
            out[t, 2] = dz / float(max(delta_scale_np[2], 1e-6))
            valid[t, 2] = True
        prev = curr

    if residual_model:
        base = np.zeros((delta_dim,), dtype=np.float32)
        if history_features.shape[0] >= 2:
            base[:2] = (
                history_features[-1, ball_idx, :2].astype(np.float32)
                - history_features[-2, ball_idx, :2].astype(np.float32)
            )
        elif history_features.shape[-1] >= 4:
            base[:2] = history_features[-1, ball_idx, 2:4].astype(np.float32)
        if delta_dim >= 3 and history_features.shape[-1] >= 5:
            if history_features.shape[0] >= 2:
                base[2] = float(history_features[-1, ball_idx, 4] - history_features[-2, ball_idx, 4])
            elif history_features.shape[-1] >= 6:
                base[2] = float(history_features[-1, ball_idx, 5])
        base_norm = base / np.maximum(delta_scale_np[:delta_dim], 1e-6)
        out = out - base_norm[None, :]

    return out, valid


def rollout(
    model: TrajectoryTransformer,
    model_mode: str,
    residual_model: bool,
    one_shot_training: bool,
    one_shot_diffusion: bool,
    stats: NormalizationStats,
    history_features: np.ndarray,
    history_mask: np.ndarray,
    history_stop_token: np.ndarray | None,
    entity_type: np.ndarray,
    rollout_steps: int,
    gt_ball_conditioning_steps: int,
    forced_ball_features: np.ndarray | None,
    forced_ball_mask: np.ndarray | None,
    temperature: float,
    inference_diffusion_steps: int | None,
    canonical_feature_normalization: bool,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int | None]]:
    model.eval()

    d_in = int(stats.mean.shape[0])
    delta_dim = int(getattr(model, "delta_dim", int(np.asarray(stats.delta_scale).shape[0])))
    delta_scale_np = _fit_delta_scale(stats.delta_scale, delta_dim=delta_dim)
    mean_np = stats.mean.astype(np.float32).copy()
    std_np = np.maximum(stats.std.astype(np.float32).copy(), 1e-6)
    if bool(canonical_feature_normalization) and d_in >= (STOP_CLASS_COUNT + 4):
        has_xyz_obs_flags = d_in >= (STOP_CLASS_COUNT + 8)
        has_xyz_layout = d_in >= (STOP_CLASS_COUNT + 6)
        stop_base = 8 if has_xyz_obs_flags else 6 if has_xyz_layout else 4
        stop_end = min(stop_base + STOP_CLASS_COUNT, d_in)
        if stop_end > stop_base:
            mean_np[stop_base:stop_end] = 0.0
            std_np[stop_base:stop_end] = 1.0
        if has_xyz_obs_flags:
            mean_np[6:8] = 0.0
            std_np[6:8] = 1.0
    mean = torch.from_numpy(mean_np).to(device).view(1, 1, 1, d_in)
    std = torch.from_numpy(std_np).to(device).view(1, 1, 1, d_in)
    delta_scale = torch.from_numpy(delta_scale_np).to(device).view(1, 1, delta_dim)

    hist_feat = torch.from_numpy(history_features.astype(np.float32)).unsqueeze(0).to(device)
    hist_mask_t = torch.from_numpy(history_mask.astype(bool)).unsqueeze(0).to(device)
    ent_type_t = torch.from_numpy(entity_type.astype(np.int64)).unsqueeze(0).to(device)
    ball_idx = 22

    forced_steps = 0
    forced_target_norm_t = None
    forced_target_valid_t = None
    forced_ball_feat_t = None
    forced_ball_mask_t = None
    if forced_ball_features is not None and forced_ball_mask is not None and int(gt_ball_conditioning_steps) > 0:
        forced_steps = min(
            int(gt_ball_conditioning_steps),
            int(rollout_steps),
            int(np.asarray(forced_ball_features).shape[0]),
            int(np.asarray(forced_ball_mask).shape[0]),
        )
        if forced_steps > 0:
            forced_target_norm_np, forced_target_valid_np = build_forced_ball_target_norm(
                history_features=history_features,
                forced_ball_features=np.asarray(forced_ball_features, dtype=np.float32)[:forced_steps],
                forced_ball_mask=np.asarray(forced_ball_mask, dtype=bool)[:forced_steps],
                rollout_steps=rollout_steps,
                delta_dim=delta_dim,
                delta_scale_np=delta_scale_np,
                residual_model=residual_model,
            )
            forced_target_norm_t = torch.from_numpy(forced_target_norm_np).to(device=device, dtype=hist_feat.dtype).unsqueeze(0)
            forced_target_valid_t = torch.from_numpy(forced_target_valid_np).to(device=device, dtype=torch.bool).unsqueeze(0)
            forced_ball_feat_t = torch.from_numpy(np.asarray(forced_ball_features, dtype=np.float32)[:forced_steps]).to(device=device, dtype=hist_feat.dtype)
            forced_ball_mask_t = torch.from_numpy(np.asarray(forced_ball_mask, dtype=bool)[:forced_steps]).to(device=device, dtype=torch.bool)

    generated_players: List[np.ndarray] = []
    generated_ball: List[np.ndarray] = []
    stop_info: Dict[str, int | None] = {"pred_stop_index": None, "pred_stop_token": None}

    with torch.no_grad():
        if one_shot_training:
            steps = int(rollout_steps)
            x_norm = (hist_feat - mean) / std

            if residual_model:
                if hist_feat.shape[1] >= 2:
                    baseline_raw = hist_feat[:, -1, :, :2] - hist_feat[:, -2, :, :2]
                else:
                    baseline_raw = hist_feat[:, -1, :, 2:4]
                baseline_norm = torch.zeros((1, 23, delta_dim), device=device, dtype=hist_feat.dtype)
                baseline_norm[:, :, :2] = baseline_raw / delta_scale[:, :, :2]
                if delta_dim >= 3:
                    if hist_feat.shape[-1] >= 5:
                        if hist_feat.shape[1] >= 2:
                            baseline_z_raw = hist_feat[:, -1, :, 4] - hist_feat[:, -2, :, 4]
                        else:
                            baseline_z_raw = (
                                hist_feat[:, -1, :, 5]
                                if hist_feat.shape[-1] >= 6
                                else torch.zeros((1, 23), device=device, dtype=hist_feat.dtype)
                            )
                        baseline_norm[:, :, 2] = baseline_z_raw / delta_scale[:, :, 2]
            else:
                baseline_norm = torch.zeros((1, 23, delta_dim), device=device, dtype=hist_feat.dtype)

            if one_shot_diffusion:
                d_steps_model = int(getattr(model, "diffusion_steps", 1000))
                if inference_diffusion_steps is None:
                    d_steps = d_steps_model
                else:
                    d_steps = max(1, min(d_steps_model, int(inference_diffusion_steps)))
                betas, alphas, alpha_bars = _make_diffusion_alpha_schedule(
                    steps=d_steps,
                    device=device,
                    dtype=hist_feat.dtype,
                )

                if model_mode == "multi":
                    x_t = torch.randn((1, steps, 23, delta_dim), device=device, dtype=hist_feat.dtype) * temperature
                    cond_mask_ball = forced_target_valid_t if forced_target_valid_t is not None else None
                    cond_target_ball = forced_target_norm_t if forced_target_norm_t is not None else None
                    context_h_entities = None
                    for t_idx in range(d_steps - 1, -1, -1):
                        if cond_mask_ball is not None and cond_target_ball is not None:
                            alpha_bar_t = alpha_bars[t_idx]
                            eps_known = torch.randn_like(cond_target_ball) * temperature
                            x_known_t = torch.sqrt(alpha_bar_t) * cond_target_ball + torch.sqrt(1.0 - alpha_bar_t) * eps_known
                            x_ball = x_t[:, :, ball_idx, :]
                            x_ball = torch.where(cond_mask_ball, x_known_t, x_ball)
                            x_t[:, :, ball_idx, :] = x_ball

                        t_tensor = torch.tensor([t_idx], device=device, dtype=torch.long)
                        out = model.forward_multi_oneshot_diffusion(
                            x=x_norm,
                            obs_mask=hist_mask_t,
                            entity_type=ent_type_t,
                            noisy_target=x_t,
                            diffusion_timestep=t_tensor,
                            steps=steps,
                        )
                        context_h_entities = out.get("context_h_entities", None)
                        eps = out["pred_noise"]
                        alpha_t = alphas[t_idx]
                        alpha_bar_t = alpha_bars[t_idx]
                        mean_t = (x_t - ((1.0 - alpha_t) / torch.sqrt(1.0 - alpha_bar_t)) * eps) / torch.sqrt(alpha_t)
                        if t_idx > 0:
                            noise_t = torch.randn_like(x_t) * temperature
                            x_t = mean_t + torch.sqrt(betas[t_idx]) * noise_t
                        else:
                            x_t = mean_t

                        if cond_mask_ball is not None and cond_target_ball is not None:
                            if t_idx > 0:
                                alpha_bar_prev = alpha_bars[t_idx - 1]
                                eps_known = torch.randn_like(cond_target_ball) * temperature
                                x_known_prev = (
                                    torch.sqrt(alpha_bar_prev) * cond_target_ball
                                    + torch.sqrt(1.0 - alpha_bar_prev) * eps_known
                                )
                            else:
                                x_known_prev = cond_target_ball
                            x_ball = x_t[:, :, ball_idx, :]
                            x_ball = torch.where(cond_mask_ball, x_known_prev, x_ball)
                            x_t[:, :, ball_idx, :] = x_ball
                    delta_norm_seq = x_t

                    # Causal stop decoding from denoised one-shot trajectory.
                    if (
                        context_h_entities is not None
                        and hasattr(model, "forward_stop_tokens_from_context")
                        and hasattr(model, "decode_stop_tokens_from_logits")
                    ):
                        hist_stop_np = (
                            np.asarray(history_stop_token, dtype=np.int64).reshape(1, -1)
                            if history_stop_token is not None
                            else np.zeros((1, history_features.shape[0]), dtype=np.int64)
                        )
                        hist_stop_t = torch.from_numpy(hist_stop_np).to(device=device, dtype=torch.long)
                        stop_logits = model.forward_stop_tokens_from_context(
                            context_h_entities=context_h_entities,
                            traj_norm=delta_norm_seq,
                            history_stop_tokens=hist_stop_t,
                            steps=steps,
                        )
                        stop_tokens = model.decode_stop_tokens_from_logits(stop_logits)[0].detach().cpu().numpy()
                        stop_idx = np.where(stop_tokens != 0)[0]
                        if stop_idx.size > 0:
                            first_idx = int(stop_idx[0])
                            stop_info["pred_stop_index"] = first_idx
                            stop_info["pred_stop_token"] = int(stop_tokens[first_idx])
                            keep_steps = int(max(first_idx, 0))
                            delta_norm_seq = delta_norm_seq[:, :keep_steps]
                else:
                    # Batched per-entity single-head decoding.
                    x_rep = x_norm.repeat(23, 1, 1, 1)
                    mask_rep = hist_mask_t.repeat(23, 1, 1)
                    type_rep = ent_type_t.repeat(23, 1)
                    target_idx = torch.arange(23, device=device, dtype=torch.long)

                    x_t = torch.randn((23, steps, delta_dim), device=device, dtype=hist_feat.dtype) * temperature
                    for t_idx in range(d_steps - 1, -1, -1):
                        t_tensor = torch.full((23,), t_idx, device=device, dtype=torch.long)
                        out = model.forward_single_oneshot_diffusion(
                            x=x_rep,
                            obs_mask=mask_rep,
                            entity_type=type_rep,
                            target_entity_idx=target_idx,
                            noisy_target=x_t,
                            diffusion_timestep=t_tensor,
                            steps=steps,
                        )
                        eps = out["pred_noise"]
                        alpha_t = alphas[t_idx]
                        alpha_bar_t = alpha_bars[t_idx]
                        mean_t = (x_t - ((1.0 - alpha_t) / torch.sqrt(1.0 - alpha_bar_t)) * eps) / torch.sqrt(alpha_t)
                        if t_idx > 0:
                            noise_t = torch.randn_like(x_t) * temperature
                            x_t = mean_t + torch.sqrt(betas[t_idx]) * noise_t
                        else:
                            x_t = mean_t
                    delta_norm_seq = x_t.permute(1, 0, 2).unsqueeze(0)
                    if forced_target_valid_t is not None and forced_target_norm_t is not None:
                        ball_delta = delta_norm_seq[:, :, ball_idx, :]
                        ball_delta = torch.where(forced_target_valid_t, forced_target_norm_t, ball_delta)
                        delta_norm_seq[:, :, ball_idx, :] = ball_delta
            else:
                if model_mode == "multi":
                    out_seq = model.forward_multi_oneshot(
                        x=x_norm,
                        obs_mask=hist_mask_t,
                        entity_type=ent_type_t,
                        steps=steps,
                    )
                    delta_norm_seq = model.decode_multi_delta_seq(
                        out=out_seq,
                        sample=True,
                        temperature=temperature,
                    )
                else:
                    x_rep = x_norm.repeat(23, 1, 1, 1)
                    mask_rep = hist_mask_t.repeat(23, 1, 1)
                    type_rep = ent_type_t.repeat(23, 1)
                    target_idx = torch.arange(23, device=device, dtype=torch.long)
                    out_seq = model.forward_single_oneshot(
                        x=x_rep,
                        obs_mask=mask_rep,
                        entity_type=type_rep,
                        target_entity_idx=target_idx,
                        steps=steps,
                    )
                    delta_single = model.decode_single_delta_seq(
                        out=out_seq,
                        sample=True,
                        temperature=temperature,
                    )  # [23,S,2]
                    delta_norm_seq = delta_single.permute(1, 0, 2).unsqueeze(0)

                if forced_target_valid_t is not None and forced_target_norm_t is not None:
                    ball_delta = delta_norm_seq[:, :, ball_idx, :]
                    ball_delta = torch.where(forced_target_valid_t, forced_target_norm_t, ball_delta)
                    delta_norm_seq[:, :, ball_idx, :] = ball_delta

            if residual_model:
                delta_norm_seq = delta_norm_seq + baseline_norm.unsqueeze(1)
            delta_seq = delta_norm_seq * delta_scale.unsqueeze(1)

            prev_pos = hist_feat[:, -1, :, :2]
            pos_seq = prev_pos.unsqueeze(1) + torch.cumsum(delta_seq[..., :2], dim=1)
            steps_eff = int(pos_seq.shape[1])
            if steps_eff > 0:
                generated_players = [pos_seq[0, t, :22, :].cpu().numpy() for t in range(steps_eff)]
                generated_ball = [pos_seq[0, t, 22, :].cpu().numpy() for t in range(steps_eff)]
            else:
                generated_players = []
                generated_ball = []
        else:
            active_entities = (ent_type_t != 3)
            for step_idx in range(rollout_steps):
                x_norm = (hist_feat - mean) / std

                if model_mode == "multi":
                    out = model.forward_multi(
                        x=x_norm,
                        obs_mask=hist_mask_t,
                        entity_type=ent_type_t,
                    )
                    delta_norm = sample_multi(
                        mu=out["mu"],
                        log_sigma=out["log_sigma"],
                        u=out["U"],
                        temperature=temperature,
                    ).view(1, 23, delta_dim)
                else:
                    parts = []
                    for entity_idx in range(23):
                        idx = torch.tensor([entity_idx], device=device, dtype=torch.long)
                        out_i = model.forward_single(
                            x=x_norm,
                            obs_mask=hist_mask_t,
                            entity_type=ent_type_t,
                            target_entity_idx=idx,
                        )
                        sample_i = sample_single(
                            mu=out_i["mu"],
                            log_sigma=out_i["log_sigma"],
                            temperature=temperature,
                        ).view(1, 1, delta_dim)
                        parts.append(sample_i)
                    delta_norm = torch.cat(parts, dim=1)

                if residual_model:
                    if hist_feat.shape[1] >= 2:
                        baseline_raw = hist_feat[:, -1, :, :2] - hist_feat[:, -2, :, :2]
                    else:
                        baseline_raw = hist_feat[:, -1, :, 2:4]
                    baseline_norm = torch.zeros((1, 23, delta_dim), device=device, dtype=hist_feat.dtype)
                    baseline_norm[:, :, :2] = baseline_raw / delta_scale[:, :, :2]
                    if delta_dim >= 3:
                        if hist_feat.shape[-1] >= 5:
                            if hist_feat.shape[1] >= 2:
                                baseline_z_raw = hist_feat[:, -1, :, 4] - hist_feat[:, -2, :, 4]
                            else:
                                baseline_z_raw = (
                                    hist_feat[:, -1, :, 5]
                                    if hist_feat.shape[-1] >= 6
                                    else torch.zeros((1, 23), device=device, dtype=hist_feat.dtype)
                                )
                            baseline_norm[:, :, 2] = baseline_z_raw / delta_scale[:, :, 2]
                    delta_norm = delta_norm + baseline_norm

                delta = delta_norm * delta_scale
                prev_frame = hist_feat[:, -1].clone()
                next_frame = prev_frame.clone()
                next_mask = hist_mask_t[:, -1].clone()

                next_frame[:, :, :2] = prev_frame[:, :, :2] + delta[:, :, :2]
                next_frame[:, :, 2:4] = delta[:, :, :2]
                if delta_dim >= 3 and next_frame.shape[-1] >= 5:
                    next_frame[:, :, 4] = prev_frame[:, :, 4] + delta[:, :, 2]
                    if next_frame.shape[-1] >= 6:
                        next_frame[:, :, 5] = delta[:, :, 2]
                next_mask[:, :] = active_entities

                if (
                    forced_steps > 0
                    and step_idx < forced_steps
                    and forced_ball_feat_t is not None
                    and forced_ball_mask_t is not None
                    and bool(forced_ball_mask_t[step_idx].item())
                ):
                    fb = forced_ball_feat_t[step_idx]
                    next_frame[:, ball_idx, :2] = fb[:2].view(1, 2)
                    vel_xy = fb[:2] - prev_frame[:, ball_idx, :2].squeeze(0)
                    next_frame[:, ball_idx, 2:4] = vel_xy.view(1, 2)
                    if next_frame.shape[-1] >= 5 and fb.shape[0] >= 5:
                        next_frame[:, ball_idx, 4] = fb[4].view(1)
                        if next_frame.shape[-1] >= 6:
                            next_frame[:, ball_idx, 5] = (fb[4] - prev_frame[:, ball_idx, 4].squeeze(0)).view(1)
                    next_mask[:, ball_idx] = True

                generated_players.append(next_frame[0, :22, :2].cpu().numpy())
                generated_ball.append(next_frame[0, 22, :2].cpu().numpy())

                hist_feat = torch.cat([hist_feat[:, 1:], next_frame.unsqueeze(1)], dim=1)
                hist_mask_t = torch.cat([hist_mask_t[:, 1:], next_mask.unsqueeze(1)], dim=1)

    if len(generated_players) == 0:
        return (
            np.zeros((0, 22, 2), dtype=np.float32),
            np.zeros((0, 2), dtype=np.float32),
            stop_info,
        )
    return np.stack(generated_players, axis=0), np.stack(generated_ball, axis=0), stop_info


def draw_pitch(ax: plt.Axes, length: float = 105.0, width: float = 68.0) -> None:
    half_l = length / 2.0
    half_w = width / 2.0
    line_color = "#466B4E"

    def rect(x0: float, y0: float, x1: float, y1: float, lw: float = 1.6) -> None:
        ax.plot([x0, x1], [y0, y0], color=line_color, lw=lw, zorder=0)
        ax.plot([x0, x1], [y1, y1], color=line_color, lw=lw, zorder=0)
        ax.plot([x0, x0], [y0, y1], color=line_color, lw=lw, zorder=0)
        ax.plot([x1, x1], [y0, y1], color=line_color, lw=lw, zorder=0)

    rect(-half_l, -half_w, half_l, half_w, lw=1.8)
    ax.plot([0.0, 0.0], [-half_w, half_w], color=line_color, lw=1.6, zorder=0)

    center_circle = plt.Circle((0.0, 0.0), 9.15, fill=False, ec=line_color, lw=1.5, zorder=0)
    ax.add_patch(center_circle)
    ax.scatter([0.0], [0.0], s=10, color=line_color, zorder=0)

    pa_depth = 16.5
    pa_width = 40.32 / 2.0
    rect(-half_l, -pa_width, -half_l + pa_depth, pa_width, lw=1.5)
    rect(half_l - pa_depth, -pa_width, half_l, pa_width, lw=1.5)

    ga_depth = 5.5
    ga_width = 18.32 / 2.0
    rect(-half_l, -ga_width, -half_l + ga_depth, ga_width, lw=1.4)
    rect(half_l - ga_depth, -ga_width, half_l, ga_width, lw=1.4)

    ax.scatter([-half_l + 11.0, half_l - 11.0], [0.0, 0.0], s=10, color=line_color, zorder=0)
    left_arc = matplotlib.patches.Arc(
        (-half_l + 11.0, 0.0), 18.3, 18.3, theta1=310, theta2=50, ec=line_color, lw=1.3, zorder=0
    )
    right_arc = matplotlib.patches.Arc(
        (half_l - 11.0, 0.0), 18.3, 18.3, theta1=130, theta2=230, ec=line_color, lw=1.3, zorder=0
    )
    ax.add_patch(left_arc)
    ax.add_patch(right_arc)

    ax.set_facecolor("#F7FBF4")
    ax.set_xlim(-60, 60)
    ax.set_ylim(-42, 42)


def _add_line_arrow(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    color: str,
    alpha: float = 0.9,
    lw: float = 1.5,
    frac: float = 0.9,
    step_back: int = 3,
) -> None:
    if x.size < 2 or y.size < 2:
        return
    valid = np.isfinite(x) & np.isfinite(y)
    idx = np.where(valid)[0]
    if idx.size < 2:
        return
    end_i = idx[-1]
    start_i = idx[max(0, idx.size - 1 - step_back)]
    if start_i == end_i:
        return
    x0, y0 = float(x[start_i]), float(y[start_i])
    x1, y1 = float(x[end_i]), float(y[end_i])
    if (x1 - x0) ** 2 + (y1 - y0) ** 2 < 1e-8:
        return
    xs = x0 + frac * (x1 - x0)
    ys = y0 + frac * (y1 - y0)
    ax.annotate(
        "",
        xy=(x1, y1),
        xytext=(xs, ys),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, alpha=alpha, mutation_scale=10),
    )


def plot_rollout(
    history_players: np.ndarray,
    history_ball: np.ndarray,
    history_mask: np.ndarray,
    generated_players: np.ndarray,
    generated_ball: np.ndarray,
    gt_players: np.ndarray,
    gt_mask: np.ndarray,
    gt_ball: np.ndarray,
    gt_ball_mask: np.ndarray,
    pred_stop_index: int | None,
    pred_stop_token: int | None,
    output_path: str | Path,
    title: str,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 7), dpi=140)
    draw_pitch(ax)

    history_steps = history_players.shape[0]
    n_steps = generated_players.shape[0]

    for i in range(22):
        color = "tab:blue" if i < 11 else "tab:red"
        valid_hist = history_mask[:history_steps, i]
        if valid_hist.any():
            hx = history_players[:history_steps, i, 0].copy()
            hy = history_players[:history_steps, i, 1].copy()
            hx[~valid_hist] = np.nan
            hy[~valid_hist] = np.nan
            ax.plot(hx, hy, color=color, alpha=0.55, linewidth=1.9, linestyle=":")
            _add_line_arrow(ax=ax, x=hx, y=hy, color=color, alpha=0.55, lw=1.1, frac=0.82, step_back=2)

    for i in range(22):
        color = "tab:blue" if i < 11 else "tab:red"
        gx = generated_players[:, i, 0]
        gy = generated_players[:, i, 1]
        ax.plot(gx, gy, color=color, alpha=0.88, linewidth=2.3)
        _add_line_arrow(ax=ax, x=gx, y=gy, color=color, alpha=0.9, lw=1.5, frac=0.86, step_back=3)

    if gt_players.size > 0:
        overlap = int(gt_players.shape[0])
        for i in range(22):
            color = "tab:blue" if i < 11 else "tab:red"
            valid = gt_mask[:overlap, i]
            x = gt_players[:overlap, i, 0].copy()
            y = gt_players[:overlap, i, 1].copy()
            if valid.any():
                x[~valid] = np.nan
                y[~valid] = np.nan
            else:
                finite = np.isfinite(x) & np.isfinite(y)
                if not finite.any():
                    continue
                x[~finite] = np.nan
                y[~finite] = np.nan
            ax.plot(x, y, color=color, alpha=0.72, linestyle="--", linewidth=2.0)

    hbx = history_ball[:, 0]
    hby = history_ball[:, 1]
    gbx = generated_ball[:, 0]
    gby = generated_ball[:, 1]
    ax.plot(hbx, hby, color="dimgray", linewidth=2.2, linestyle=":", alpha=0.95)
    ax.plot(gbx, gby, color="#d97706", linewidth=2.4, linestyle="-", alpha=0.95)
    _add_line_arrow(ax=ax, x=hbx, y=hby, color="dimgray", alpha=0.85, lw=1.2, frac=0.82, step_back=2)
    _add_line_arrow(ax=ax, x=gbx, y=gby, color="#d97706", alpha=0.92, lw=1.4, frac=0.86, step_back=3)

    if gt_ball.size > 0:
        bx = gt_ball[:, 0].copy()
        by = gt_ball[:, 1].copy()
        valid_b = gt_ball_mask.astype(bool)
        bx[~valid_b] = np.nan
        by[~valid_b] = np.nan
        ax.plot(bx, by, color="black", linewidth=2.0, linestyle="--", alpha=0.88)

    stop_label = None
    if pred_stop_index is not None:
        idx = int(pred_stop_index)
        if pred_stop_token is not None and 0 <= int(pred_stop_token) < len(STOP_CLASS_NAMES):
            stop_label = STOP_CLASS_NAMES[int(pred_stop_token)]
        else:
            stop_label = "stop"

        if idx <= 0:
            sx, sy = float(history_ball[-1, 0]), float(history_ball[-1, 1])
        elif generated_ball.shape[0] > 0:
            pick = min(idx - 1, generated_ball.shape[0] - 1)
            sx, sy = float(generated_ball[pick, 0]), float(generated_ball[pick, 1])
        else:
            sx, sy = float(history_ball[-1, 0]), float(history_ball[-1, 1])

        ax.scatter([sx], [sy], s=140, marker="*", color="#c026d3", edgecolors="black", linewidths=0.8, zorder=8)
        ax.annotate(
            f"pred stop: {stop_label} @ t+{idx}",
            xy=(sx, sy),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=8,
            color="#7e22ce",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#c026d3", alpha=0.85),
        )

    legend_handles = [
        Line2D([0], [0], color="tab:blue", lw=1.9, ls=":", alpha=0.8, label="Players GT past (initial window)"),
        Line2D([0], [0], color="tab:blue", lw=2.0, ls="--", alpha=0.85, label="Players GT future (unseen)"),
        Line2D([0], [0], color="tab:blue", lw=2.3, ls="-", alpha=0.9, label="Players predicted future"),
        Line2D([0], [0], color="dimgray", lw=2.2, ls=":", alpha=0.9, label="Ball past"),
        Line2D([0], [0], color="#d97706", lw=2.4, ls="-", alpha=0.95, label="Ball predicted future"),
        Line2D([0], [0], color="black", lw=2.0, ls="--", alpha=0.9, label="Ball GT future"),
    ]
    if pred_stop_index is not None:
        stop_handle_label = f"Pred stop @ t+{int(pred_stop_index)}"
        if stop_label is not None:
            stop_handle_label = f"Pred stop: {stop_label} @ t+{int(pred_stop_index)}"
        legend_handles.append(
            Line2D([0], [0], marker="*", color="#c026d3", markeredgecolor="black", markersize=10, linewidth=0, label=stop_handle_label)
        )

    ax.set_title(title)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(alpha=0.12)
    ax.axis("equal")
    ax.legend(handles=legend_handles, loc="best")

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_next_step_comparison(
    history_players: np.ndarray,
    history_mask: np.ndarray,
    pred_next_players: np.ndarray,
    gt_next_players: np.ndarray,
    gt_next_mask: np.ndarray,
    output_path: str | Path,
    title: str,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 7), dpi=140)
    draw_pitch(ax)

    last_hist = history_players[-1]
    last_hist_mask = history_mask[-1].astype(bool)
    gt_mask_bool = gt_next_mask.astype(bool)

    for i in range(22):
        base_color = "tab:blue" if i < 11 else "tab:red"
        if not last_hist_mask[i]:
            continue

        x0, y0 = last_hist[i, 0], last_hist[i, 1]
        ax.scatter(x0, y0, color=base_color, s=26, alpha=0.45)

        xp, yp = pred_next_players[i, 0], pred_next_players[i, 1]
        ax.plot([x0, xp], [y0, yp], color=base_color, linewidth=2.0, alpha=0.85)
        ax.scatter(xp, yp, color=base_color, s=70, alpha=0.95, marker="o")

        if gt_mask_bool[i]:
            xg, yg = gt_next_players[i, 0], gt_next_players[i, 1]
            ax.plot([x0, xg], [y0, yg], color=base_color, linewidth=2.0, alpha=0.85, linestyle="--")
            ax.scatter(xg, yg, color=base_color, s=95, alpha=0.95, marker="x", linewidths=2.0)

    legend_handles = [
        Line2D([0], [0], marker="o", color="tab:gray", markersize=6, linewidth=0, alpha=0.5, label="Last history frame"),
        Line2D([0], [0], marker="o", color="tab:gray", markersize=8, linewidth=2.0, label="Predicted next step"),
        Line2D([0], [0], marker="x", color="tab:gray", markersize=9, linewidth=2.0, linestyle="--", label="GT next step"),
    ]
    ax.legend(handles=legend_handles, loc="best")
    ax.set_title(title)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(alpha=0.12)
    ax.axis("equal")

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_predictions_only(
    history_players: np.ndarray,
    history_ball: np.ndarray,
    history_mask: np.ndarray,
    generated_players: np.ndarray,
    generated_ball: np.ndarray,
    output_path: str | Path,
    title: str,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 7), dpi=140)
    draw_pitch(ax)

    history_steps = history_players.shape[0]
    for i in range(22):
        color = "tab:blue" if i < 11 else "tab:red"
        valid_hist = history_mask[:history_steps, i]
        if valid_hist.any():
            hx = history_players[:history_steps, i, 0].copy()
            hy = history_players[:history_steps, i, 1].copy()
            hx[~valid_hist] = np.nan
            hy[~valid_hist] = np.nan
            ax.plot(hx, hy, color=color, alpha=0.52, linewidth=1.8, linestyle=":")
            _add_line_arrow(ax=ax, x=hx, y=hy, color=color, alpha=0.52, lw=1.1, frac=0.82, step_back=2)

    for i in range(22):
        color = "tab:blue" if i < 11 else "tab:red"
        gx = generated_players[:, i, 0]
        gy = generated_players[:, i, 1]
        ax.plot(gx, gy, color=color, alpha=0.92, linewidth=2.4)
        _add_line_arrow(ax=ax, x=gx, y=gy, color=color, alpha=0.9, lw=1.5, frac=0.86, step_back=3)

    hbx = history_ball[:, 0]
    hby = history_ball[:, 1]
    ax.plot(hbx, hby, color="dimgray", linewidth=2.0, linestyle=":", alpha=0.9)
    _add_line_arrow(ax=ax, x=hbx, y=hby, color="dimgray", alpha=0.85, lw=1.3, frac=0.82, step_back=2)

    ax.plot(generated_ball[:, 0], generated_ball[:, 1], color="#d97706", alpha=0.95, linewidth=2.5)
    _add_line_arrow(
        ax=ax,
        x=generated_ball[:, 0],
        y=generated_ball[:, 1],
        color="#d97706",
        alpha=0.95,
        lw=1.5,
        frac=0.86,
        step_back=3,
    )

    legend_handles = [
        Line2D([0], [0], color="tab:blue", lw=1.8, ls=":", alpha=0.8, label="Players GT past (initial window)"),
        Line2D([0], [0], color="tab:blue", lw=2.4, ls="-", alpha=0.95, label="Team 1 predicted rollout"),
        Line2D([0], [0], color="tab:red", lw=2.4, ls="-", alpha=0.95, label="Team 2 predicted rollout"),
        Line2D([0], [0], color="dimgray", lw=2.0, ls=":", alpha=0.9, label="Ball past"),
        Line2D([0], [0], color="#d97706", lw=2.5, ls="-", alpha=0.95, label="Ball predicted rollout"),
    ]
    ax.legend(handles=legend_handles, loc="best")
    ax.set_title(title)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(alpha=0.12)
    ax.axis("equal")

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_rollout_style_axis(
    ax: plt.Axes,
    history_players: np.ndarray,
    history_ball: np.ndarray,
    history_mask: np.ndarray,
    generated_players: np.ndarray,
    generated_ball: np.ndarray,
    panel_title: str,
) -> None:
    draw_pitch(ax)
    history_steps = history_players.shape[0]

    for i in range(22):
        color = "tab:blue" if i < 11 else "tab:red"
        valid_hist = history_mask[:history_steps, i]
        if valid_hist.any():
            hx = history_players[:history_steps, i, 0].copy()
            hy = history_players[:history_steps, i, 1].copy()
            hx[~valid_hist] = np.nan
            hy[~valid_hist] = np.nan
            ax.plot(hx, hy, color=color, alpha=0.55, linewidth=1.4, linestyle=":")
            _add_line_arrow(ax=ax, x=hx, y=hy, color=color, alpha=0.55, lw=1.1, frac=0.82, step_back=2)

    for i in range(22):
        color = "tab:blue" if i < 11 else "tab:red"
        gx = generated_players[:, i, 0]
        gy = generated_players[:, i, 1]
        ax.plot(gx, gy, color=color, alpha=0.88, linewidth=1.8)
        _add_line_arrow(ax=ax, x=gx, y=gy, color=color, alpha=0.9, lw=1.4, frac=0.86, step_back=3)

    hbx = history_ball[:, 0]
    hby = history_ball[:, 1]
    gbx = generated_ball[:, 0]
    gby = generated_ball[:, 1]
    ax.plot(hbx, hby, color="dimgray", linewidth=1.8, linestyle=":", alpha=0.9)
    ax.plot(gbx, gby, color="#d97706", linewidth=1.8, linestyle="-", alpha=0.9)
    _add_line_arrow(ax=ax, x=hbx, y=hby, color="dimgray", alpha=0.85, lw=1.3, frac=0.82, step_back=2)
    _add_line_arrow(ax=ax, x=gbx, y=gby, color="#d97706", alpha=0.9, lw=1.3, frac=0.86, step_back=3)

    ax.set_title(panel_title)
    ax.grid(alpha=0.12)
    ax.axis("equal")


def plot_synthetic_scenarios_grid(
    history_players: np.ndarray,
    history_ball: np.ndarray,
    history_mask: np.ndarray,
    scenarios: List[Dict[str, np.ndarray]],
    output_path: str | Path,
    title: str,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n = len(scenarios)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6.2 * ncols, 4.8 * nrows), dpi=140)
    axes_arr = np.array(axes, dtype=object).reshape(nrows, ncols)

    for idx, sc in enumerate(scenarios):
        r = idx // ncols
        c = idx % ncols
        ax = axes_arr[r, c]
        _plot_rollout_style_axis(
            ax=ax,
            history_players=history_players,
            history_ball=history_ball,
            history_mask=history_mask,
            generated_players=sc["players"],
            generated_ball=sc["ball"],
            panel_title=sc["label"],
        )

    for idx in range(n, nrows * ncols):
        r = idx // ncols
        c = idx % ncols
        axes_arr[r, c].axis("off")

    legend_handles = [
        Line2D([0], [0], color="tab:blue", lw=1.4, ls=":", alpha=0.8, label="Players GT past (initial window)"),
        Line2D([0], [0], color="tab:blue", lw=1.8, ls="-", alpha=0.9, label="Players predicted future"),
        Line2D([0], [0], color="dimgray", lw=1.8, ls=":", alpha=0.9, label="Ball past"),
        Line2D([0], [0], color="#d97706", lw=1.8, ls="-", alpha=0.9, label="Ball predicted future"),
    ]

    fig.suptitle(title)
    fig.legend(handles=legend_handles, loc="lower center", ncol=4, frameon=True, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=[0.0, 0.06, 1.0, 0.95])
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = pick_device(args.device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    ckpt_args = ckpt.get("args", {})

    model_mode = infer_model_mode(args.mode, ckpt_args)
    residual_model = bool(ckpt_args.get("residual_model", False))
    one_shot_training = bool(ckpt_args.get("one_shot_training", False))
    one_shot_diffusion = bool(one_shot_training and ckpt_args.get("one_shot_diffusion", False))
    gt_ball_conditioning_steps = (
        int(args.gt_ball_conditioning_steps)
        if args.gt_ball_conditioning_steps is not None
        else int(ckpt_args.get("gt_ball_conditioning_steps", 10))
    )
    gt_ball_conditioning_steps = max(0, gt_ball_conditioning_steps)

    trained_horizon = int(
        ckpt_args.get("one_shot_frames" if one_shot_training else "rollout_steps", 120)
    )
    history = args.history if args.history is not None else int(ckpt_args.get("history", 24))
    preprocessed_dir = args.preprocessed_dir or ckpt_args.get(
        "preprocessed_dir", "/mnt/data/remains/opta2026/ballPlayerTrajModel/preprocessed"
    )

    train_dataset, val_dataset, computed_stats, _ = create_datasets_from_preprocessed(
        preprocessed_dir=preprocessed_dir,
        history=history,
        rollout_steps=1,
        mode="multi",
    )
    stats = NormalizationStats.from_dict(ckpt["stats"]) if "stats" in ckpt else computed_stats
    model_state = ckpt.get("model_state", {})
    if "mu_head.2.weight" in model_state:
        model_delta_dim = int(model_state["mu_head.2.weight"].shape[0])
    elif "single_head.2.weight" in model_state:
        model_delta_dim = max(2, int(model_state["single_head.2.weight"].shape[0] // 2))
    else:
        model_delta_dim = int(ckpt_args.get("delta_dim", int(np.asarray(stats.delta_scale).shape[0])))

    dataset = val_dataset if args.split == "val" and val_dataset is not None else train_dataset
    if dataset is None or dataset.num_windows == 0:
        raise RuntimeError("Selected split has no windows.")

    window_idx = int(np.clip(args.window_idx, 0, dataset.num_windows - 1))
    requested_rollout_steps = int(args.rollout_steps) if args.rollout_steps is not None else trained_horizon
    requested_rollout_steps = max(1, requested_rollout_steps)
    if one_shot_training:
        oneshot_max_steps = max(1, int(ckpt_args.get("one_shot_frames", trained_horizon)))
        if requested_rollout_steps > oneshot_max_steps:
            print(
                f"Requested horizon {requested_rollout_steps} exceeds one-shot model max "
                f"{oneshot_max_steps}; using {oneshot_max_steps}."
            )
            requested_rollout_steps = oneshot_max_steps

    context = dataset.get_rollout_context(window_idx=window_idx, rollout_steps=requested_rollout_steps)
    available_steps = int(context["available_steps"][0])
    if available_steps <= 0:
        raise RuntimeError("Selected window does not have future frames for rollout.")

    rollout_steps = min(requested_rollout_steps, available_steps)
    if rollout_steps < requested_rollout_steps:
        print(
            f"Requested/trained horizon {requested_rollout_steps} exceeds available GT future "
            f"{available_steps}; using {rollout_steps}."
        )
    print(f"Visualization horizon: {rollout_steps} (checkpoint trained horizon: {trained_horizon})")
    print(f"Model delta dim: {model_delta_dim} | Stats delta dim: {int(np.asarray(stats.delta_scale).shape[0])}")
    print(
        "Ball conditioning phase: "
        f"{min(gt_ball_conditioning_steps, rollout_steps)} steps conditioned, "
        f"{max(rollout_steps - min(gt_ball_conditioning_steps, rollout_steps), 0)} steps free generation"
    )

    history_features_raw = context["history_features"]
    history_features = apply_ball_kick_counterfactual(
        history_features=history_features_raw,
        kick_speed_scale=float(args.kick_speed_scale),
        kick_angle_deg=float(args.kick_angle_deg),
        kick_spin=float(args.kick_spin),
    )
    history_mask = context["history_mask"]
    entity_type = context["entity_type"]

    model = TrajectoryTransformer(
        d_in=int(stats.mean.shape[0]),
        n_entities=23,
        delta_dim=model_delta_dim,
        d_model=int(ckpt_args.get("d_model", 256)),
        n_heads=int(ckpt_args.get("n_heads", 8)),
        n_layers=int(ckpt_args.get("n_layers", 8)),
        dropout=float(ckpt_args.get("dropout", 0.1)),
        max_history=max(history, 32),
        joint_rank=int(ckpt_args.get("joint_rank", 8)),
        use_rope=bool(ckpt_args.get("use_rope", False)),
        use_gqa=bool(ckpt_args.get("use_gqa", False)),
        gqa_groups=int(ckpt_args.get("gqa_groups", 1)),
        use_oneshot_head=one_shot_training,
        max_oneshot_steps=max(
            1,
            int(
                ckpt_args.get("one_shot_frames")
                if ckpt_args.get("one_shot_frames") is not None
                else trained_horizon
            ),
        ),
        use_diffusion_head=one_shot_diffusion,
        diffusion_steps=1000,
    ).to(device)

    load_result = model.load_state_dict(ckpt["model_state"], strict=False)
    allowed_prefixes = ("oneshot_step_emb.", "diffusion_")
    bad_missing = [k for k in load_result.missing_keys if not k.startswith(allowed_prefixes)]
    bad_unexpected = [k for k in load_result.unexpected_keys if not k.startswith(allowed_prefixes)]
    if bad_missing or bad_unexpected:
        raise RuntimeError(
            "Checkpoint/model mismatch is incompatible for visualization.\n"
            f"missing_keys={bad_missing}\n"
            f"unexpected_keys={bad_unexpected}"
        )
    if load_result.missing_keys or load_result.unexpected_keys:
        print(
            "Loaded checkpoint with compatible optional head mismatch "
            f"(missing={load_result.missing_keys}, unexpected={load_result.unexpected_keys})."
        )
    model.eval()

    generated_players, generated_ball, stop_info = rollout(
        model=model,
        model_mode=model_mode,
        residual_model=residual_model,
        one_shot_training=one_shot_training,
        one_shot_diffusion=one_shot_diffusion,
        stats=stats,
        history_features=history_features,
        history_mask=history_mask,
        history_stop_token=context.get("history_stop_token"),
        entity_type=entity_type,
        rollout_steps=rollout_steps,
        gt_ball_conditioning_steps=gt_ball_conditioning_steps,
        forced_ball_features=context["future_features"][:rollout_steps, 22, :],
        forced_ball_mask=context["future_mask"][:rollout_steps, 22],
        temperature=float(args.temperature),
        inference_diffusion_steps=int(args.diffusion_steps),
        canonical_feature_normalization=bool(args.canonical_feature_normalization),
        device=device,
    )

    gt_features = context["future_features"][:rollout_steps]
    gt_mask = context["future_mask"][:rollout_steps]
    gt_players = gt_features[:, :22, :2]
    gt_players_mask = gt_mask[:, :22]
    gt_ball = gt_features[:, 22, :2]
    gt_ball_mask = gt_mask[:, 22]

    history_players = history_features[:, :22, :2]
    history_ball = history_features[:, 22, :2]
    history_mask_players = history_mask[:, :22]

    metric_steps = min(int(generated_players.shape[0]), int(gt_players.shape[0]))
    if metric_steps > 0:
        errors = np.linalg.norm(generated_players[:metric_steps] - gt_players[:metric_steps], axis=-1)
        valid = gt_players_mask[:metric_steps].astype(bool)
        if valid.any():
            ade = float(errors[valid].mean())
            last_valid = valid[-1]
            fde = float(errors[-1][last_valid].mean()) if last_valid.any() else float("nan")
            print(f"Model ADE (valid points): {ade:.4f}")
            print(f"Model FDE (valid players at final step): {fde:.4f}")

            baseline_players = np.repeat(history_players[-1:, :, :], metric_steps, axis=0)
            baseline_errors = np.linalg.norm(baseline_players - gt_players[:metric_steps], axis=-1)
            baseline_ade = float(baseline_errors[valid].mean())
            baseline_fde = float(baseline_errors[-1][last_valid].mean()) if last_valid.any() else float("nan")
            print(f"Baseline ADE (stand still): {baseline_ade:.4f}")
            print(f"Baseline FDE (stand still): {baseline_fde:.4f}")

            cv_players = np.zeros((metric_steps, generated_players.shape[1], generated_players.shape[2]), dtype=generated_players.dtype)
            curr_pos = history_players[-1].copy()
            vel = np.zeros((22, 2), dtype=curr_pos.dtype)
            if history_players.shape[0] >= 2:
                vel = history_players[-1] - history_players[-2]
                vel_valid = history_mask_players[-1].astype(bool) & history_mask_players[-2].astype(bool)
                vel[~vel_valid] = 0.0
            for t in range(cv_players.shape[0]):
                curr_pos = curr_pos + vel
                cv_players[t] = curr_pos
            cv_errors = np.linalg.norm(cv_players - gt_players[:metric_steps], axis=-1)
            cv_ade = float(cv_errors[valid].mean())
            cv_fde = float(cv_errors[-1][last_valid].mean()) if last_valid.any() else float("nan")
            print(f"Baseline ADE (constant velocity): {cv_ade:.4f}")
            print(f"Baseline FDE (constant velocity): {cv_fde:.4f}")

    if metric_steps > 0:
        valid_ball = gt_ball_mask[:metric_steps].astype(bool)
        if valid_ball.any():
            ball_err = np.linalg.norm(generated_ball[:metric_steps] - gt_ball[:metric_steps], axis=-1)
            ball_ade = float(ball_err[valid_ball].mean())
            last_valid_ball_idx = int(np.where(valid_ball)[0][-1])
            ball_fde = float(ball_err[last_valid_ball_idx])
            print(f"Ball ADE (valid points): {ball_ade:.4f}")
            print(f"Ball FDE (last valid step): {ball_fde:.4f}")

    target_kind = "residual" if residual_model else "absolute"
    objective_kind = (
        "one-shot-diffusion"
        if one_shot_diffusion
        else ("one-shot-gaussian" if one_shot_training else "autoregressive")
    )
    title = (
        f"Rollout | mode={model_mode} | objective={objective_kind} | target={target_kind} | "
        f"steps={generated_players.shape[0]} | kick_scale={args.kick_speed_scale:.2f} "
        f"angle={args.kick_angle_deg:.1f} spin={args.kick_spin:.2f}"
    )

    pred_stop_index = stop_info.get("pred_stop_index")
    pred_stop_token = stop_info.get("pred_stop_token")
    if pred_stop_index is not None:
        stop_name = (
            STOP_CLASS_NAMES[int(pred_stop_token)]
            if pred_stop_token is not None and 0 <= int(pred_stop_token) < len(STOP_CLASS_NAMES)
            else "stop"
        )
        print(f"Predicted stop: {stop_name} at t+{int(pred_stop_index)}")
    else:
        print("Predicted stop: none within visualized horizon.")

    plot_rollout(
        history_players=history_players,
        history_ball=history_ball,
        history_mask=history_mask_players,
        generated_players=generated_players,
        generated_ball=generated_ball,
        gt_players=gt_players,
        gt_mask=gt_players_mask,
        gt_ball=gt_ball,
        gt_ball_mask=gt_ball_mask,
        pred_stop_index=int(pred_stop_index) if pred_stop_index is not None else None,
        pred_stop_token=int(pred_stop_token) if pred_stop_token is not None else None,
        output_path=args.output,
        title=title,
    )

    next_step_output = Path(args.output).with_name(f"{Path(args.output).stem}_next_step{Path(args.output).suffix}")
    pred_only_output = Path(args.output).with_name(f"{Path(args.output).stem}_pred_only{Path(args.output).suffix}")
    if gt_players.shape[0] > 0:
        pred_next_players = generated_players[0] if generated_players.shape[0] > 0 else history_players[-1]
        plot_next_step_comparison(
            history_players=history_players,
            history_mask=history_mask_players,
            pred_next_players=pred_next_players,
            gt_next_players=gt_players[0],
            gt_next_mask=gt_players_mask[0],
            output_path=next_step_output,
            title=f"Next-Step Comparison | mode={model_mode} | objective={objective_kind} | target={target_kind}",
        )
        print(f"Saved next-step plot to {next_step_output}")

    if generated_players.shape[0] > 0:
        pred_players_for_plot = generated_players
        pred_ball_for_plot = generated_ball
    else:
        pred_players_for_plot = np.repeat(history_players[-1:, :, :], 1, axis=0)
        pred_ball_for_plot = np.repeat(history_ball[-1:, :], 1, axis=0)
    if pred_players_for_plot.shape[0] > 0:
        plot_predictions_only(
            history_players=history_players,
            history_ball=history_ball,
            history_mask=history_mask_players,
            generated_players=pred_players_for_plot,
            generated_ball=pred_ball_for_plot,
            output_path=pred_only_output,
            title=f"Predictions Only | mode={model_mode} | objective={objective_kind} | target={target_kind}",
        )
        print(f"Saved prediction-only plot to {pred_only_output}")

    synthetic_rollout_steps = max(1, int(args.synthetic_rollout_steps))
    if one_shot_training:
        oneshot_max_steps = int(ckpt_args.get("one_shot_frames", synthetic_rollout_steps))
        if synthetic_rollout_steps > oneshot_max_steps:
            print(
                f"Synthetic horizon {synthetic_rollout_steps} exceeds one-shot model max "
                f"{oneshot_max_steps}; using {oneshot_max_steps}."
            )
            synthetic_rollout_steps = oneshot_max_steps
    print(f"Synthetic-ball horizon: {synthetic_rollout_steps}")

    synthetic_specs = [
        {"label": "Base", "speed": 1.0, "angle": 0.0, "spin": 0.0},
        {"label": "Speed +20%", "speed": 1.2, "angle": 0.0, "spin": 0.0},
        {"label": "Speed -20%", "speed": 0.8, "angle": 0.0, "spin": 0.0},
        {"label": "Angle +15°", "speed": 1.0, "angle": 15.0, "spin": 0.0},
        {"label": "Angle -15°", "speed": 1.0, "angle": -15.0, "spin": 0.0},
        {"label": "Spin +0.20", "speed": 1.0, "angle": 0.0, "spin": 0.2},
        {"label": "Spin -0.20", "speed": 1.0, "angle": 0.0, "spin": -0.2},
    ]
    synth_rollouts: List[Dict[str, np.ndarray]] = []
    for spec in synthetic_specs:
        h_cf = apply_ball_kick_counterfactual(
            history_features=history_features_raw,
            kick_speed_scale=float(spec["speed"] * args.kick_speed_scale),
            kick_angle_deg=float(spec["angle"] + args.kick_angle_deg),
            kick_spin=float(spec["spin"] + args.kick_spin),
        )
        synth_ball_prefix, synth_ball_prefix_mask = build_synthetic_ball_prefix(
            history_features=h_cf,
            steps=min(gt_ball_conditioning_steps, synthetic_rollout_steps),
        )
        p_cf, b_cf, _ = rollout(
            model=model,
            model_mode=model_mode,
            residual_model=residual_model,
            one_shot_training=one_shot_training,
            one_shot_diffusion=one_shot_diffusion,
            stats=stats,
            history_features=h_cf,
            history_mask=history_mask,
            history_stop_token=context.get("history_stop_token"),
            entity_type=entity_type,
            rollout_steps=synthetic_rollout_steps,
            gt_ball_conditioning_steps=gt_ball_conditioning_steps,
            forced_ball_features=synth_ball_prefix,
            forced_ball_mask=synth_ball_prefix_mask,
            temperature=float(args.temperature),
            inference_diffusion_steps=int(args.diffusion_steps),
            canonical_feature_normalization=bool(args.canonical_feature_normalization),
            device=device,
        )
        synth_rollouts.append({"label": spec["label"], "players": p_cf, "ball": b_cf})

    synthetic_output = Path(args.output).with_name(f"{Path(args.output).stem}_synthetic_ball{Path(args.output).suffix}")
    plot_synthetic_scenarios_grid(
        history_players=history_players,
        history_ball=history_ball,
        history_mask=history_mask_players,
        scenarios=synth_rollouts,
        output_path=synthetic_output,
        title=f"Kick Counterfactual Scenarios | mode={model_mode} | objective={objective_kind} | target={target_kind}",
    )
    print(f"Saved synthetic-ball plot to {synthetic_output}")
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
