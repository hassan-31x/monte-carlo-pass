from __future__ import annotations

"""Runtime helpers vendored from the old sim_viz stack.

This keeps monteCarloPassSearch/initParamVar self-contained with SMART-based
player rollout, player-to-touch, ball-at-touch, and PV loading, without
depending on trajModel_allData.
"""

import importlib.util
import json
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

_THIS_DIR = Path(__file__).resolve().parent
_OPTA_DIR = _THIS_DIR.parent.parent


def _add_to_path(path: Path) -> None:
    s = str(path)
    if s not in sys.path:
        sys.path.insert(0, s)


_add_to_path(_OPTA_DIR / "trajModel_smart")
_add_to_path(_OPTA_DIR / "possessionValue")


PITCH_X = 105.0
PITCH_Y = 68.0
HALF_X = PITCH_X / 2.0
HALF_Y = PITCH_Y / 2.0

VEL_SCALE = 10.0
Z_SCALE = 5.0
K_TOUCH = 8
H_HIST = 8
FPS = 25
PTT_MAX_TOUCH_XY_RADIUS = 2.0
PTT_MAX_TOUCH_Z_DIFF = 3.0

GRAVITY = 9.81
AIR_RESISTANCE = 0.00648
RESTITUTION = 0.109
GROUND_FRICTION = 0.0
ROLLING_FRICTION = 0.02731
ROLLING_Z_THRESH = 0.054
KICK_REFINE_BACK_WINDOW = 20
KICK_REFINE_FWD_WINDOW = 220
KICK_CONTACT_RADIUS = 1.2
KICK_OWNED_RADIUS = 1.6
KICK_RELEASE_RADIUS = 2.5
KICK_SPEED_MIN = 4.0
KICK_DV_MIN = 1.0
KICK_RELEASE_LOOKAHEAD = 12
KICK_CONTACT_TO_KICK_MAX = 120
KICK_MAX_SHIFT = 220


def refine_kick_frame_local(clip: dict) -> int:
    """Refine kick frame around clip['kick_frame_local'] using ball-player contact."""
    features = clip["features"].astype(np.float32)
    entity_type = clip["entity_type"]
    t_total = features.shape[0]
    nominal = int(np.clip(int(clip.get("kick_frame_local", 0)), 0, max(t_total - 1, 0)))

    ball_indices = np.where(entity_type == 2)[0]
    if len(ball_indices) == 0:
        return nominal
    ball_idx = int(ball_indices[0])

    player_indices = np.where((entity_type != 2) & (entity_type != 3))[0]
    if len(player_indices) == 0:
        return nominal

    mask = clip.get("mask", np.ones((t_total, features.shape[1]), dtype=bool)).astype(bool)
    ball_valid = mask[:, ball_idx]
    if not np.any(ball_valid):
        return nominal

    xy_range = float(features[:, :, :2].max() - features[:, :, :2].min())
    if xy_range < 2.0:
        x_scale = PITCH_X
        y_scale = PITCH_Y
    else:
        x_scale = 1.0
        y_scale = 1.0

    ball_vx = features[:, ball_idx, 2] * x_scale
    ball_vy = features[:, ball_idx, 3] * y_scale
    speed = np.sqrt(np.square(ball_vx) + np.square(ball_vy))
    speed_prev = np.concatenate([speed[:1], speed[:-1]], axis=0)
    dv = speed - speed_prev

    nearest_dist = np.full((t_total,), np.nan, dtype=np.float32)
    nearest_player = np.full((t_total,), -1, dtype=np.int64)
    for t in range(t_total):
        if not bool(ball_valid[t]):
            continue
        valid_players_t = player_indices[mask[t, player_indices]]
        if len(valid_players_t) == 0:
            continue
        bp = features[t, ball_idx, :2]
        pp = features[t, valid_players_t, :2]
        dx = (pp[:, 0] - bp[0]) * x_scale
        dy = (pp[:, 1] - bp[1]) * y_scale
        d = np.sqrt(np.square(dx) + np.square(dy))
        j = int(np.argmin(d))
        nearest_dist[t] = float(d[j])
        nearest_player[t] = int(valid_players_t[j])

    lo = max(1, nominal - KICK_REFINE_BACK_WINDOW)
    hi = min(t_total - 1, nominal + KICK_REFINE_FWD_WINDOW)
    if hi <= lo:
        return nominal

    contact_candidates = [
        t
        for t in range(nominal, hi + 1)
        if np.isfinite(nearest_dist[t]) and nearest_dist[t] <= KICK_CONTACT_RADIUS and nearest_player[t] >= 0
    ]

    if not contact_candidates:
        close_candidates = [
            t
            for t in range(nominal, hi + 1)
            if np.isfinite(nearest_dist[t]) and nearest_dist[t] <= KICK_RELEASE_RADIUS
        ]
        if close_candidates:
            contact_candidates = close_candidates[:1]

    for c in contact_candidates:
        owner = int(nearest_player[c]) if nearest_player[c] >= 0 else -1
        if owner < 0:
            continue
        t_end = min(hi, c + KICK_CONTACT_TO_KICK_MAX)
        for t in range(c, t_end + 1):
            if not bool(ball_valid[t]) or not bool(mask[t, owner]):
                continue

            bp = features[t, ball_idx, :2]
            op = features[t, owner, :2]
            d_owner_t = float(
                np.sqrt(np.square((op[0] - bp[0]) * x_scale) + np.square((op[1] - bp[1]) * y_scale))
            )
            if d_owner_t > KICK_OWNED_RADIUS:
                continue

            fast_now = (float(speed[t]) >= KICK_SPEED_MIN) or (
                float(speed[t]) >= (KICK_SPEED_MIN * 0.75) and float(dv[t]) >= KICK_DV_MIN
            )
            if not fast_now:
                continue

            rel_hi = min(hi, t + KICK_RELEASE_LOOKAHEAD)
            released = False
            for u in range(t + 1, rel_hi + 1):
                if not bool(ball_valid[u]) or not bool(mask[u, owner]):
                    continue
                bpu = features[u, ball_idx, :2]
                opu = features[u, owner, :2]
                d_owner_u = float(
                    np.sqrt(np.square((opu[0] - bpu[0]) * x_scale) + np.square((opu[1] - bpu[1]) * y_scale))
                )
                if d_owner_u >= KICK_RELEASE_RADIUS:
                    released = True
                    break
            if released:
                refined = t
                if abs(refined - nominal) > KICK_MAX_SHIFT:
                    refined = nominal + int(np.sign(refined - nominal) * KICK_MAX_SHIFT)
                return int(np.clip(refined, 0, t_total - 1))

    return nominal


def try_load_smart_model(checkpoint_path: Optional[str], vocab_dir: Optional[str], device: torch.device):
    if not checkpoint_path or not Path(checkpoint_path).exists():
        print(f"[mcps] SMART checkpoint not found ({checkpoint_path}), skipping.")
        return None
    try:
        from tokenizer import MotionTokenizer
        from model import SMARTTransformer

        if vocab_dir is None:
            smart_dir = Path(checkpoint_path).resolve().parent.parent.parent
            vocab_dir = str(smart_dir / "vocabs")
        vd = Path(vocab_dir)
        tokenizer = MotionTokenizer(vd / "player_vocab.npz", vd / "ball_vocab.npz")

        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        ckpt_args = ckpt.get("args", {})
        history = ckpt_args.get("history", 8)
        rollout = ckpt_args.get("rollout", 24)

        model = SMARTTransformer(
            d_model=ckpt_args.get("d_model", 256),
            n_heads=ckpt_args.get("n_heads", 8),
            n_layers=ckpt_args.get("n_layers", 6),
            dropout=0.0,
            use_rope=ckpt_args.get("use_rope", True),
            vocab_player=tokenizer.player_centroids.shape[0],
            vocab_ball=tokenizer.ball_centroids.shape[0],
            max_seq_len=history + rollout + 8,
        ).to(device).eval()
        model.load_state_dict(ckpt["model"])
        return model, tokenizer, ckpt_args
    except Exception as exc:
        print(f"[mcps] Failed to load SMART model: {exc}")
        return None


def smart_predict_positions(
    smart_result,
    features: np.ndarray,
    entity_type: np.ndarray,
    mask: np.ndarray,
    ctx_start: int,
    kfl: int,
    sim_len: int,
    device: torch.device,
    candidate_ball_pos: Optional[np.ndarray] = None,
    temperature: float = 0.8,
    top_k: int = 10,
    seed: int = 42,
) -> Optional[np.ndarray]:
    """Roll players forward conditioned on a proposed ball trajectory.

    Only frames through ``kfl`` are read from ``features``.  Hypothetical
    inference must provide ``candidate_ball_pos`` for the future; this keeps
    recorded future player and ball motion out of counterfactual rollouts.
    """
    from tokenizer import (
        BALL_CLAMP,
        PLAYER_CLAMP,
        STEPS_PER_TOKEN,
        extract_absolute_positions,
        extract_ball_displacements,
        extract_player_displacements,
    )
    from dataset import BALL_ENTITY_TYPE, DOWNSAMPLE_FACTOR, N_PLAYERS

    model, tokenizer, ckpt_args = smart_result
    history = ckpt_args.get("history", 8)
    rollout_cap = ckpt_args.get("rollout", 24)

    t_raw, n_entities, feat_dim = features.shape
    ball_idx = int(np.where(entity_type == BALL_ENTITY_TYPE)[0][0])
    player_idxs = np.where((entity_type == 0) | (entity_type == 1))[0]
    if len(player_idxs) > N_PLAYERS:
        player_idxs = player_idxs[:N_PLAYERS]

    frames_per_token = STEPS_PER_TOKEN * DOWNSAMPLE_FACTOR
    smart_rollout_tokens = min(rollout_cap, (sim_len + frames_per_token - 1) // frames_per_token + 1)
    smart_total_tokens = history + smart_rollout_tokens
    smart_window_25hz = smart_total_tokens * frames_per_token

    smart_start_25hz = kfl - history * frames_per_token
    if smart_start_25hz < 0:
        return None
    if candidate_ball_pos is None:
        raise ValueError("candidate_ball_pos is required for leakage-free SMART rollout")
    candidate_ball_pos = np.asarray(candidate_ball_pos, dtype=np.float32)
    if candidate_ball_pos.ndim != 2 or candidate_ball_pos.shape[1] != 3:
        raise ValueError("candidate_ball_pos must have shape [future_frames, 3]")
    if len(candidate_ball_pos) < sim_len + 1:
        raise ValueError("candidate_ball_pos is shorter than the requested rollout")

    # Historical data ends at the kick.  Append empty player frames and the
    # candidate ball flight; no recorded future is copied into model inputs.
    hist_feat = features[smart_start_25hz:kfl].astype(np.float32)
    hist_mask = (
        mask[smart_start_25hz:kfl].astype(bool)
        if mask is not None
        else np.ones(hist_feat.shape[:2], dtype=bool)
    )
    future_n = smart_rollout_tokens * frames_per_token
    future_feat = np.repeat(features[kfl:kfl + 1].astype(np.float32), future_n, axis=0)
    future_mask = np.repeat(hist_mask[-1:], future_n, axis=0)
    ball_future = candidate_ball_pos[:future_n]
    if len(ball_future) < future_n:
        ball_future = np.concatenate(
            [ball_future, np.repeat(ball_future[-1:], future_n - len(ball_future), axis=0)], axis=0
        )
    future_feat[:, ball_idx, :2] = ball_future[:, :2]
    if feat_dim >= 5:
        future_feat[:, ball_idx, 4] = ball_future[:, 2]
    win_feat = np.concatenate([hist_feat, future_feat], axis=0)
    win_mask = np.concatenate([hist_mask, future_mask], axis=0)

    win_ds = win_feat[::DOWNSAMPLE_FACTOR]
    mask_ds = win_mask[::DOWNSAMPLE_FACTOR]
    t_ds = win_ds.shape[0]
    n_token_steps = t_ds // STEPS_PER_TOKEN
    if n_token_steps < history + 2:
        return None

    actual_total = min(n_token_steps, smart_total_tokens)
    actual_rollout = actual_total - history

    ball_xy = win_ds[:, ball_idx, :2]
    ball_z = win_ds[:, ball_idx, 4:5] if feat_dim >= 6 else np.zeros((t_ds, 1), np.float32)
    ball_pos_3d = np.concatenate([ball_xy, ball_z], axis=1)
    ball_disps = extract_ball_displacements(ball_pos_3d, clamp=BALL_CLAMP)
    ball_abs_pos = extract_absolute_positions(ball_xy, STEPS_PER_TOKEN)
    ball_tokens = tokenizer.encode_ball(ball_disps[:actual_total])

    player_token_ids = np.zeros((actual_total, N_PLAYERS), dtype=np.int64)
    player_abs_pos = np.zeros((actual_total, N_PLAYERS, 2), dtype=np.float32)
    player_obs = np.zeros((actual_total, N_PLAYERS), dtype=bool)

    for j, pi in enumerate(player_idxs):
        p_xy = win_ds[:, pi, :2]
        p_mask = mask_ds[:, pi]
        p_disps = extract_player_displacements(p_xy, clamp=PLAYER_CLAMP)
        p_abs = extract_absolute_positions(p_xy, STEPS_PER_TOKEN)
        p_tokens = tokenizer.encode_player(p_disps[:actual_total])
        n_tok = min(len(p_tokens), actual_total)
        player_token_ids[:n_tok, j] = p_tokens[:n_tok]
        player_abs_pos[:n_tok, j] = p_abs[:n_tok]
        obs_indices = np.minimum(np.arange(actual_total) * STEPS_PER_TOKEN, len(p_mask) - 1)
        player_obs[:, j] = p_mask[obs_indices]

    ball_obs_indices = np.minimum(np.arange(actual_total) * STEPS_PER_TOKEN, mask_ds.shape[0] - 1)
    ball_obs = mask_ds[ball_obs_indices, ball_idx]

    obs_mask_full = np.zeros((actual_total, N_PLAYERS + 1), dtype=bool)
    obs_mask_full[:, 0] = ball_obs
    obs_mask_full[:, 1:] = player_obs

    entity_types_ordered = np.zeros(N_PLAYERS + 1, dtype=np.int64)
    entity_types_ordered[0] = BALL_ENTITY_TYPE
    for j, pi in enumerate(player_idxs):
        entity_types_ordered[j + 1] = int(entity_type[pi])

    torch.manual_seed(seed)
    np.random.seed(seed)

    pt = torch.from_numpy(player_token_ids).unsqueeze(0).to(device)
    bt = torch.from_numpy(ball_tokens).unsqueeze(0).to(device)
    pp = torch.from_numpy(player_abs_pos).unsqueeze(0).to(device)
    bp = torch.from_numpy(ball_abs_pos[:actual_total]).unsqueeze(0).to(device)
    et = torch.from_numpy(entity_types_ordered).unsqueeze(0).to(device)
    om = torch.from_numpy(obs_mask_full).unsqueeze(0).to(device)

    pred_pt = pt[:, :history].clone()
    # Positions used by the Fourier encoder are historical for the prefix and
    # autoregressively extended thereafter.
    pp = pp[:, :history].clone()
    om = om[:, :history].clone()

    with torch.no_grad():
        for step in range(actual_rollout):
            t_cur = history + step
            if t_cur >= actual_total:
                break
            cur_len = pred_pt.shape[1]
            player_logits, _ = model(
                pred_pt,
                bt[:, :cur_len],
                pp[:, :cur_len],
                bp[:, :cur_len],
                et,
                om[:, :cur_len],
                perm=None,
            )
            last_logits = player_logits[:, -1, :, :] / max(temperature, 1e-3)
            if top_k > 0:
                topk_vals, topk_idx = last_logits.topk(top_k, dim=-1)
                mask_t = torch.full_like(last_logits, float("-inf"))
                mask_t.scatter_(-1, topk_idx, topk_vals)
                last_logits = mask_t
            probs = torch.softmax(last_logits, dim=-1)
            sampled = torch.multinomial(probs.view(-1, probs.shape[-1]), 1).view(1, N_PLAYERS)
            pred_pt = torch.cat([pred_pt, sampled.unsqueeze(1)], dim=1)

            new_disps = tokenizer.decode_player(sampled.cpu().numpy()).reshape(1, N_PLAYERS, 5, 2)
            net_disp = new_disps.sum(axis=2)
            prev_pos = pp[:, t_cur - 1, :, :].cpu().numpy()
            new_pos = torch.from_numpy(prev_pos + net_disp).to(device)
            pp = torch.cat([pp, new_pos.unsqueeze(1)], dim=1)
            om = torch.cat([om, om[:, -1:]], dim=1)

    pred_tokens_np = pred_pt[0].cpu().numpy()
    n_pred = pred_tokens_np.shape[0]
    pos_12hz = np.zeros((n_pred * STEPS_PER_TOKEN + 1, N_PLAYERS, 2), np.float32)
    for t in range(history):
        for substep in range(STEPS_PER_TOKEN):
            idx_12 = t * STEPS_PER_TOKEN + substep
            if idx_12 < t_ds:
                for j, pi in enumerate(player_idxs):
                    pos_12hz[idx_12, j] = win_ds[idx_12, pi, :2]

    for t in range(history, n_pred):
        disps_all = tokenizer.decode_player(pred_tokens_np[t : t + 1]).reshape(N_PLAYERS, 5, 2)
        base_pos = pos_12hz[t * STEPS_PER_TOKEN - 1]
        for substep in range(STEPS_PER_TOKEN):
            idx_12 = t * STEPS_PER_TOKEN + substep
            base_pos = base_pos + disps_all[:, substep, :]
            pos_12hz[idx_12] = base_pos

    n_12 = n_pred * STEPS_PER_TOKEN
    n_25 = n_12 * DOWNSAMPLE_FACTOR
    pos_25hz = np.zeros((n_25, N_PLAYERS, 2), np.float32)
    for i in range(n_12):
        pos_25hz[i * DOWNSAMPLE_FACTOR] = pos_12hz[i]
        if i + 1 < n_12:
            pos_25hz[i * DOWNSAMPLE_FACTOR + 1] = 0.5 * (pos_12hz[i] + pos_12hz[i + 1])
        else:
            pos_25hz[i * DOWNSAMPLE_FACTOR + 1] = pos_12hz[i]

    rollout_start_25hz = history * frames_per_token
    pred_pos = np.zeros((sim_len + 1, n_entities, 3), np.float32)
    gt_pos = np.zeros((sim_len + 1, n_entities, 3), np.float32)
    feat_slice = features[kfl : kfl + sim_len + 1]
    gt_pos[: len(feat_slice), :, :2] = feat_slice[:, :, :2]
    if feat_dim >= 6:
        gt_pos[: len(feat_slice), :, 2] = feat_slice[:, :, 4]
    # Start from the kick state and fill only generated player coordinates.
    pred_pos[:] = gt_pos[:1]
    pred_pos[:, ball_idx, :] = candidate_ball_pos[: sim_len + 1]

    for frame_25 in range(sim_len + 1):
        smart_25_idx = rollout_start_25hz + frame_25
        if smart_25_idx < len(pos_25hz):
            for j, pi in enumerate(player_idxs):
                pred_pos[frame_25, pi, :2] = pos_25hz[smart_25_idx, j]

    return pred_pos


def try_load_touch_model(checkpoint_path: Optional[str], device: torch.device):
    if not checkpoint_path or not Path(checkpoint_path).exists():
        print(f"[mcps] touch checkpoint not found ({checkpoint_path}), skipping.")
        return None
    try:
        spec = importlib.util.spec_from_file_location(
            "_mcps_ptt_model",
            _OPTA_DIR / "playerToTouch" / "model.py",
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        ckpt = torch.load(checkpoint_path, map_location="cpu")
        saved_args = ckpt.get("args", {})
        model_type = saved_args.get("model_type", "classifier")
        state_dict = ckpt.get("model_state_dict", ckpt.get("model", {}))
        ball_dim = state_dict["ball_embed.weight"].shape[1] if "ball_embed.weight" in state_dict else 8

        if model_type == "survival":
            model = module.SurvivalTouchModel(
                ball_dim=ball_dim,
                player_hist_dim=H_HIST * 4,
                k=saved_args.get("k", K_TOUCH),
                d_model=saved_args.get("d_model", 128),
                n_heads=saved_args.get("n_heads", 4),
                n_layers=saved_args.get("n_layers", 4),
                dropout=0.0,
            ).to(device).eval()
        else:
            model = module.TouchPredictorModel(
                ball_dim=ball_dim,
                player_hist_dim=H_HIST * 4,
                k=saved_args.get("k", K_TOUCH),
                d_model=saved_args.get("d_model", 128),
                n_heads=saved_args.get("n_heads", 4),
                n_layers=saved_args.get("n_layers", 4),
                dropout=0.0,
            ).to(device).eval()
        model.load_state_dict(state_dict)
        return model, saved_args, model_type, ball_dim
    except Exception as exc:
        print(f"[mcps] Failed to load touch model: {exc}")
        return None


def try_load_bat_model(checkpoint_path: Optional[str], device: torch.device):
    if not checkpoint_path or not Path(checkpoint_path).exists():
        print(f"[mcps] bat checkpoint not found ({checkpoint_path}), skipping.")
        return None
    try:
        spec = importlib.util.spec_from_file_location(
            "_mcps_bat_model",
            _OPTA_DIR / "ballAtTouch" / "initParamVar" / "model.py",
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        ckpt = torch.load(checkpoint_path, map_location="cpu")
        saved_args = ckpt.get("args", {})
        k_other = saved_args.get("k_other", K_TOUCH)
        model_type = saved_args.get("model_type", "deterministic")
        state_dict = ckpt.get("model", ckpt.get("model_state_dict", {}))
        ball_dim = state_dict["ball_embed.weight"].shape[1] if "ball_embed.weight" in state_dict else 8
        kwargs = dict(
            ball_dim=ball_dim,
            self_dim=H_HIST * 4,
            other_dim=H_HIST * 4,
            k_other=k_other,
            d_model=saved_args.get("d_model", 128),
            n_heads=saved_args.get("n_heads", 4),
            n_layers=saved_args.get("n_layers", 4),
            dropout=0.0,
        )
        if model_type == "gaussian":
            model = module.GaussianBallAtTouchModel(**kwargs)
        else:
            model = module.BallAtTouchModel(**kwargs)
        model = model.to(device).eval()
        model.load_state_dict(state_dict)
        return model, saved_args, k_other, model_type, ball_dim
    except Exception as exc:
        print(f"[mcps] Failed to load bat model: {exc}")
        return None


def _find_latest_best_checkpoint(root_dir: Path) -> Optional[str]:
    if not root_dir.exists():
        return None
    cands = list(root_dir.glob("*/best.pt"))
    if not cands:
        return None
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return str(cands[0])


def _find_latest_set_piece_model(root_dir: Path) -> Optional[str]:
    if not root_dir.exists():
        return None
    cands = list(root_dir.glob("set_piece_pv*/model.json"))
    if not cands:
        cands = list(root_dir.glob("*/set_piece*/model.json"))
    if not cands:
        return None
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return str(cands[0])


def try_load_pv_model(checkpoint_path: Optional[str], device: torch.device):
    if not checkpoint_path:
        checkpoint_path = _find_latest_best_checkpoint(_OPTA_DIR / "possessionValue" / "checkpoints")
    if not checkpoint_path or not Path(checkpoint_path).exists():
        print(f"[mcps] PV checkpoint not found ({checkpoint_path}), skipping.")
        return None
    try:
        from pv_model import PossessionValueTransformer

        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        meta = ckpt.get("meta", {})
        args = ckpt.get("args", {})
        model = PossessionValueTransformer(
            feat_dim=meta.get("feat_dim", 6),
            n_entities=meta.get("n_entities", 23),
            d_model=args.get("d_model", 128),
            n_heads=args.get("n_heads", 4),
            n_layers=args.get("n_layers", 4),
            dropout=0.0,
            max_seq_len=meta.get("window_size", 64),
        ).to(device).eval()
        model.load_state_dict(ckpt["model_state_dict"])
        return model, meta
    except Exception as exc:
        print(f"[mcps] Failed to load PV model: {exc}")
        return None


def try_load_set_piece_pv_model(model_path: Optional[str]):
    if not model_path:
        model_path = _find_latest_set_piece_model(_OPTA_DIR / "possessionValue" / "checkpoints")
    if not model_path or not Path(model_path).exists():
        print(f"[mcps] set-piece PV model not found ({model_path}), skipping.")
        return None
    try:
        from set_piece_pv import load_set_piece_pv_model

        return load_set_piece_pv_model(model_path)
    except Exception as exc:
        print(f"[mcps] Failed to load set-piece PV model: {exc}")
        return None


def _ball_feats(
    ball_pos: np.ndarray,
    ball_vel: np.ndarray,
    anchor: np.ndarray,
    ball_z: float = 0.0,
    ball_vz: float = 0.0,
) -> np.ndarray:
    return np.array(
        [
            ball_pos[0] / HALF_X,
            ball_pos[1] / HALF_Y,
            ball_vel[0] / VEL_SCALE,
            ball_vel[1] / VEL_SCALE,
            (anchor[0] - ball_pos[0]) / HALF_X,
            (anchor[1] - ball_pos[1]) / HALF_Y,
            ball_z / Z_SCALE,
            ball_vz / VEL_SCALE,
        ],
        dtype=np.float32,
    )


def _ptt_geometry_valid_mask(ball_pos: np.ndarray, player_pos: np.ndarray) -> np.ndarray:
    if player_pos.size == 0:
        return np.zeros((0,), dtype=bool)
    xy_dist = np.linalg.norm(player_pos[:, :2] - ball_pos[None, :2], axis=1)
    z_diff = np.abs(player_pos[:, 2] - ball_pos[2])
    valid = (xy_dist <= PTT_MAX_TOUCH_XY_RADIUS) & (z_diff <= PTT_MAX_TOUCH_Z_DIFF)
    valid &= ~np.isnan(xy_dist) & ~np.isnan(z_diff)
    return valid


def _apply_ptt_survival_constraints(
    hazard: float,
    player_probs: np.ndarray,
    valid_mask: np.ndarray,
) -> Tuple[float, np.ndarray, np.ndarray]:
    joint_player = float(hazard) * np.asarray(player_probs, dtype=np.float64)
    joint_player = np.where(valid_mask, joint_player, 0.0)
    effective_hazard = float(joint_player.sum())
    no_touch = float(np.clip(1.0 - effective_hazard, 0.0, 1.0))

    if effective_hazard > 1e-12:
        player_probs_cond = (joint_player / effective_hazard).astype(np.float32)
    else:
        player_probs_cond = np.zeros_like(player_probs, dtype=np.float32)
    viz_probs = np.concatenate([joint_player.astype(np.float32), [np.float32(no_touch)]])
    return effective_hazard, player_probs_cond, viz_probs


def _apply_ptt_classifier_constraints(probs_kplus1: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    probs = np.asarray(probs_kplus1, dtype=np.float64).copy()
    player_probs = probs[: len(valid_mask)]
    invalid_mass = float(player_probs[~valid_mask].sum())
    player_probs[~valid_mask] = 0.0
    probs[: len(valid_mask)] = player_probs
    probs[len(valid_mask)] += invalid_mass
    total = float(probs.sum())
    if total > 1e-12:
        probs /= total
    return probs.astype(np.float32)
