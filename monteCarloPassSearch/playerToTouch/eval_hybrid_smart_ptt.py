#!/usr/bin/env python3
"""Evaluate hybrid first-toucher prediction: SMART trajectories + playerToTouch.

Pipeline per clip:
1) Start from pre-pass context around kick frame.
2) Roll out SMART player trajectories with GT ball trajectory conditioning.
3) Build per-frame playerToTouch survival features from hybrid states.
4) Aggregate first-touch probabilities over time:
     P(i first) = sum_t S(t-1) * h_t * q_t(i)
   where h_t is hazard and q_t(i) is receiver prob at frame t.
5) Report receiver/top-k and team-success metrics.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from model import SurvivalTouchModel
from preprocess_sportec import (
    BALL_DIM,
    H,
    K,
    MAX_SEARCH,
    PLAYER_FEAT_DIM,
    PITCH_HALF_X,
    PITCH_HALF_Y,
    RECV_PROXIMITY,
    RECV_VEL_DELTA,
    TOUCH_SKIP,
    VEL_SCALE,
    Z_SCALE,
    detect_reception,
    get_ball_and_player_indices,
)


FPS = 25.0
SMART_DOWNSAMPLE_FACTOR = 2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--sportec-dir",
        type=str,
        default="/mnt/data/remains/opta2026/ballPlayerTrajModel/publicData/preprocessed/sportec_xyz_metres",
    )
    p.add_argument("--split", type=str, default="val")
    p.add_argument("--max-clips", type=int, default=0, help="0 = all clips")
    p.add_argument("--clip-stride", type=int, default=1)

    p.add_argument(
        "--ptt-checkpoint",
        type=str,
        default="/mnt/data/remains/opta2026/playerToTouch/checkpoints/sportec_survival_v2/best.pt",
    )
    p.add_argument(
        "--smart-dir",
        type=str,
        default="/mnt/data/remains/opta2026/trajModel_smart",
    )
    p.add_argument(
        "--smart-checkpoint",
        type=str,
        default="/mnt/data/remains/opta2026/trajModel_smart/checkpoints/smart_v1/best.pt",
    )
    p.add_argument("--smart-history", type=int, default=0, help="0 = use checkpoint history")
    p.add_argument("--smart-rollout", type=int, default=24)
    p.add_argument("--smart-top-k", type=int, default=0, help="0 = greedy argmax")
    p.add_argument("--smart-temp", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument(
        "--output-json",
        type=str,
        default="/mnt/data/remains/opta2026/playerToTouch/hybrid_eval_result.json",
    )
    p.add_argument("--no-cuda", action="store_true")
    return p.parse_args()


def _import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import module: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_manifest(path: Path) -> List[Dict[str, Any]]:
    out = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def identify_passer(features: np.ndarray, mask: np.ndarray, ball_idx: int, player_indices: np.ndarray, kick_frame: int):
    if not bool(mask[kick_frame, ball_idx]):
        return None
    vis = player_indices[mask[kick_frame, player_indices].astype(bool)]
    if len(vis) == 0:
        return None
    bx, by = features[kick_frame, ball_idx, 0], features[kick_frame, ball_idx, 1]
    d = np.sqrt((features[kick_frame, vis, 0] - bx) ** 2 + (features[kick_frame, vis, 1] - by) ** 2)
    return int(vis[np.argmin(d)])


def build_sample_with_knn(
    features: np.ndarray,
    mask: np.ndarray,
    ball_idx: int,
    player_indices: np.ndarray,
    frame: int,
    kick_frame: int,
    exclude_entity: Optional[int],
    k: int,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Build playerToTouch-style features and return KNN global entity indices."""
    if frame < H:
        return None
    if not bool(mask[frame, ball_idx]):
        return None

    bx = float(features[frame, ball_idx, 0])
    by = float(features[frame, ball_idx, 1])
    bvx = float(features[frame, ball_idx, 2])
    bvy = float(features[frame, ball_idx, 3])
    bz = float(features[frame, ball_idx, 4])
    bvz = float(features[frame, ball_idx, 5])

    if bool(mask[kick_frame, ball_idx]):
        anc_x = float(features[kick_frame, ball_idx, 0])
        anc_y = float(features[kick_frame, ball_idx, 1])
    else:
        anc_x, anc_y = bx, by

    ball_feats = np.array(
        [
            bx / PITCH_HALF_X,
            by / PITCH_HALF_Y,
            bvx / VEL_SCALE,
            bvy / VEL_SCALE,
            (anc_x - bx) / PITCH_HALF_X,
            (anc_y - by) / PITCH_HALF_Y,
            bz / Z_SCALE,
            bvz / VEL_SCALE,
        ],
        dtype=np.float32,
    )

    vis = player_indices[mask[frame, player_indices].astype(bool)]
    if exclude_entity is not None:
        vis = vis[vis != exclude_entity]
    if len(vis) == 0:
        return None

    d = np.sqrt((features[frame, vis, 0] - bx) ** 2 + (features[frame, vis, 1] - by) ** 2)
    knn = vis[np.argsort(d)[:k]]
    if len(knn) < k:
        knn = np.concatenate([knn, np.full(k - len(knn), knn[-1], dtype=knn.dtype)])

    player_feats = np.zeros((k, H * PLAYER_FEAT_DIM), dtype=np.float32)
    for i in range(k):
        ent = int(knn[i])
        last_valid = np.zeros(PLAYER_FEAT_DIM, dtype=np.float32)
        for h_i in range(H):
            t_h = frame - H + 1 + h_i
            off = h_i * PLAYER_FEAT_DIM
            if t_h < 0 or not bool(mask[t_h, ent]):
                player_feats[i, off : off + PLAYER_FEAT_DIM] = last_valid
                continue
            pp_x = float(features[t_h, ent, 0])
            pp_y = float(features[t_h, ent, 1])
            pv_x = float(features[t_h, ent, 2])
            pv_y = float(features[t_h, ent, 3])
            feat = np.array(
                [
                    (pp_x - bx) / PITCH_HALF_X,
                    (pp_y - by) / PITCH_HALF_Y,
                    pv_x / VEL_SCALE,
                    pv_y / VEL_SCALE,
                ],
                dtype=np.float32,
            )
            player_feats[i, off : off + PLAYER_FEAT_DIM] = feat
            last_valid = feat

    return ball_feats, player_feats, knn.astype(np.int64)


def load_ptt_model(ckpt_path: Path, device: torch.device) -> Tuple[SurvivalTouchModel, Dict[str, Any]]:
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    args = ckpt["args"]
    if args.get("model_type", "survival") != "survival":
        raise RuntimeError(f"Checkpoint is not survival model: {ckpt_path}")
    model = SurvivalTouchModel(
        ball_dim=BALL_DIM,
        player_hist_dim=H * PLAYER_FEAT_DIM,
        k=args["k"],
        d_model=args["d_model"],
        n_heads=args["n_heads"],
        n_layers=args["n_layers"],
        dropout=args["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, args


def load_smart_model(
    smart_dir: Path, smart_ckpt_path: Path, device: torch.device
) -> Tuple[Any, Any, Any, Dict[str, Any]]:
    tok_mod = _import_module(smart_dir / "tokenizer.py", "smart_tokenizer_mod")
    model_mod = _import_module(smart_dir / "model.py", "smart_model_mod")

    ckpt = torch.load(str(smart_ckpt_path), map_location=device, weights_only=False)
    ckpt_args = ckpt.get("args", {})

    vocab_dir = smart_dir / "vocabs"
    tokenizer = tok_mod.MotionTokenizer(vocab_dir / "player_vocab.npz", vocab_dir / "ball_vocab.npz")
    smart_model = model_mod.SMARTTransformer(
        d_model=ckpt_args.get("d_model", 256),
        n_heads=ckpt_args.get("n_heads", 8),
        n_layers=ckpt_args.get("n_layers", 6),
        dropout=0.0,
        use_rope=ckpt_args.get("use_rope", True),
        vocab_player=tokenizer.player_centroids.shape[0],
        vocab_ball=tokenizer.ball_centroids.shape[0],
        max_seq_len=ckpt_args.get("history", 8) + ckpt_args.get("rollout", 24),
    ).to(device)
    smart_model.load_state_dict(ckpt["model"])
    smart_model.eval()
    return tok_mod, model_mod, tokenizer, ckpt_args, smart_model


def smart_rollout_players_with_gt_ball(
    *,
    features: np.ndarray,
    mask: np.ndarray,
    entity_type: np.ndarray,
    kick_frame: int,
    player_indices: np.ndarray,
    ball_idx: int,
    tok_mod: Any,
    tokenizer: Any,
    smart_model: Any,
    device: torch.device,
    history_tokens: int,
    rollout_tokens: int,
    top_k: int,
    temperature: float,
    rng: np.random.Generator,
) -> Optional[Dict[str, Any]]:
    ds_feat = features[::SMART_DOWNSAMPLE_FACTOR].astype(np.float32)
    ds_mask = mask[::SMART_DOWNSAMPLE_FACTOR].astype(bool)

    steps_per_token = int(tok_mod.STEPS_PER_TOKEN)
    total_steps = history_tokens + rollout_tokens

    if len(player_indices) < 22:
        return None
    pidx = player_indices[:22]

    n_tokens_total = ds_feat.shape[0] // steps_per_token
    if n_tokens_total < total_steps + 1:
        return None

    kick_ds = kick_frame // SMART_DOWNSAMPLE_FACTOR
    kick_token = kick_ds // steps_per_token
    token_start = kick_token - history_tokens + 1
    token_end = token_start + total_steps
    if token_start < 0 or token_end > n_tokens_total:
        return None

    frame_start_ds = token_start * steps_per_token
    frame_end_ds = token_end * steps_per_token
    win_feat = ds_feat[frame_start_ds:frame_end_ds]
    win_mask = ds_mask[frame_start_ds:frame_end_ds]

    ball_xy = win_feat[:, ball_idx, :2]
    if win_feat.shape[-1] >= 6:
        ball_z = win_feat[:, ball_idx, 4:5]
    else:
        ball_z = np.zeros((win_feat.shape[0], 1), dtype=np.float32)
    ball_pos3d = np.concatenate([ball_xy, ball_z], axis=1)
    ball_disps = tok_mod.extract_ball_displacements(ball_pos3d, clamp=tok_mod.BALL_CLAMP)
    ball_abs = tok_mod.extract_absolute_positions(ball_xy, steps_per_token)
    if ball_disps.shape[0] < total_steps or ball_abs.shape[0] < total_steps:
        return None
    ball_tokens = tokenizer.encode_ball(ball_disps[:total_steps], noise_top_k=0)

    player_tokens = np.zeros((total_steps, 22), dtype=np.int64)
    player_abs = np.zeros((total_steps, 22, 2), dtype=np.float32)
    player_obs = np.zeros((total_steps, 22), dtype=bool)

    obs_idx = np.arange(total_steps) * steps_per_token
    obs_idx = np.minimum(obs_idx, win_mask.shape[0] - 1)
    for j, pi in enumerate(pidx):
        pxy = win_feat[:, pi, :2]
        pdisp = tok_mod.extract_player_displacements(pxy, clamp=tok_mod.PLAYER_CLAMP)
        pabs = tok_mod.extract_absolute_positions(pxy, steps_per_token)
        if pdisp.shape[0] < total_steps or pabs.shape[0] < total_steps:
            return None
        player_tokens[:, j] = tokenizer.encode_player(pdisp[:total_steps], noise_top_k=0)
        player_abs[:, j] = pabs[:total_steps]
        player_obs[:, j] = win_mask[obs_idx, pi]

    obs_mask = np.zeros((total_steps, 23), dtype=bool)
    obs_mask[:, 0] = win_mask[obs_idx, ball_idx]
    obs_mask[:, 1:] = player_obs

    entity_types = np.zeros((23,), dtype=np.int64)
    entity_types[0] = 2
    entity_types[1:] = entity_type[pidx].astype(np.int64)

    gt_player_tokens = torch.from_numpy(player_tokens).long().to(device)
    gt_ball_tokens = torch.from_numpy(ball_tokens).long().to(device)
    gt_player_pos = torch.from_numpy(player_abs).float().to(device)
    gt_ball_pos = torch.from_numpy(ball_abs[:total_steps]).float().to(device)
    ent_t = torch.from_numpy(entity_types).long().to(device).unsqueeze(0)
    obs_t = torch.from_numpy(obs_mask).bool().to(device)

    pred_tokens = gt_player_tokens[:history_tokens].clone()
    pred_pos = gt_player_pos[:history_tokens].clone()

    with torch.no_grad():
        for _ in range(rollout_tokens):
            cur_len = pred_tokens.shape[0]
            inp_player = pred_tokens.unsqueeze(0)
            inp_ball = gt_ball_tokens[:cur_len].unsqueeze(0)
            inp_player_pos = pred_pos.unsqueeze(0)
            inp_ball_pos = gt_ball_pos[:cur_len].unsqueeze(0)
            inp_obs = obs_t[:cur_len].unsqueeze(0)
            player_logits, _ = smart_model(
                inp_player, inp_ball, inp_player_pos, inp_ball_pos, ent_t, inp_obs, perm=None
            )
            logits = player_logits[:, -1, :, :]  # [1,22,V]
            if top_k > 0:
                logits = logits / max(temperature, 1e-3)
                tk_vals, tk_idx = logits.topk(top_k, dim=-1)
                masked = torch.full_like(logits, float("-inf"))
                masked.scatter_(-1, tk_idx, tk_vals)
                probs = torch.softmax(masked, dim=-1)
                sampled = torch.multinomial(probs.view(-1, probs.shape[-1]), 1).view(1, 22)[0]
            else:
                sampled = logits.argmax(dim=-1)[0]  # greedy

            pred_tokens = torch.cat([pred_tokens, sampled.unsqueeze(0)], dim=0)
            disp = tokenizer.decode_player(sampled.detach().cpu().numpy())  # [22,10]
            net = disp.reshape(22, steps_per_token, 2).sum(axis=1)
            nxt = pred_pos[-1].detach().cpu().numpy() + net
            pred_pos = torch.cat([pred_pos, torch.from_numpy(nxt).float().to(device).unsqueeze(0)], dim=0)

    pred_tokens_np = pred_tokens.detach().cpu().numpy()
    pred_pos_np = pred_pos.detach().cpu().numpy()  # [total_steps,22,2]

    ds_len = total_steps * steps_per_token
    pred_ds = np.zeros((ds_len, 22, 2), dtype=np.float32)
    for t in range(total_steps):
        base = pred_pos_np[t]
        pred_ds[t * steps_per_token] = base
        deltas = tokenizer.decode_player(pred_tokens_np[t]).reshape(22, steps_per_token, 2)
        cur = base.copy()
        for m in range(1, steps_per_token):
            cur = cur + deltas[:, m]
            pred_ds[t * steps_per_token + m] = cur

    raw_len = ds_len * SMART_DOWNSAMPLE_FACTOR
    pred_raw = np.zeros((raw_len, 22, 2), dtype=np.float32)
    for d in range(ds_len):
        pred_raw[2 * d] = pred_ds[d]
        if 2 * d + 1 < raw_len:
            if d + 1 < ds_len:
                pred_raw[2 * d + 1] = 0.5 * (pred_ds[d] + pred_ds[d + 1])
            else:
                pred_raw[2 * d + 1] = pred_ds[d]

    raw_start = frame_start_ds * SMART_DOWNSAMPLE_FACTOR
    raw_end = raw_start + raw_len
    return {
        "player_indices": pidx,
        "pred_raw_xy": pred_raw,
        "raw_start": raw_start,
        "raw_end": raw_end,
    }


def apply_predicted_players(features: np.ndarray, pred_info: Dict[str, Any]) -> Tuple[np.ndarray, int, int]:
    out = features.copy()
    T = out.shape[0]
    s = max(0, int(pred_info["raw_start"]))
    e = min(T, int(pred_info["raw_end"]))
    if e <= s:
        return out, s, e

    local_s = s - int(pred_info["raw_start"])
    local_e = local_s + (e - s)
    pred_xy = pred_info["pred_raw_xy"][local_s:local_e]  # [L,22,2]

    for j, pi in enumerate(pred_info["player_indices"]):
        pos = pred_xy[:, j, :]
        out[s:e, pi, 0:2] = pos
        vx = np.zeros((len(pos),), dtype=np.float32)
        vy = np.zeros((len(pos),), dtype=np.float32)
        if len(pos) >= 2:
            vx[1:] = (pos[1:, 0] - pos[:-1, 0]) * FPS
            vy[1:] = (pos[1:, 1] - pos[:-1, 1]) * FPS
            vx[0] = vx[1]
            vy[0] = vy[1]
        out[s:e, pi, 2] = vx
        out[s:e, pi, 3] = vy

    return out, s, e


def infer_first_toucher_from_hazard(
    *,
    features: np.ndarray,
    mask: np.ndarray,
    entity_type: np.ndarray,
    ball_idx: int,
    player_indices: np.ndarray,
    kick_frame: int,
    passer_entity: int,
    ptt_model: SurvivalTouchModel,
    device: torch.device,
    k: int,
    batch_size: int,
    eval_end_cap: int,
) -> Optional[Dict[str, Any]]:
    frame_start = kick_frame + TOUCH_SKIP
    frame_end = min(kick_frame + MAX_SEARCH, features.shape[0] - 1, eval_end_cap)
    if frame_end < frame_start:
        return None

    frames = []
    bf_list = []
    pf_list = []
    knn_list = []
    for t in range(max(frame_start, H), frame_end + 1):
        s = build_sample_with_knn(
            features=features,
            mask=mask,
            ball_idx=ball_idx,
            player_indices=player_indices,
            frame=t,
            kick_frame=kick_frame,
            exclude_entity=passer_entity,
            k=k,
        )
        if s is None:
            continue
        bf, pf, knn = s
        frames.append(t)
        bf_list.append(bf)
        pf_list.append(pf)
        knn_list.append(knn)

    if not frames:
        return None

    bf_np = np.stack(bf_list).astype(np.float32)
    pf_np = np.stack(pf_list).astype(np.float32)
    h_all = []
    q_all = []
    with torch.no_grad():
        for i in range(0, len(frames), batch_size):
            j = min(i + batch_size, len(frames))
            bf_t = torch.from_numpy(bf_np[i:j]).to(device)
            pf_t = torch.from_numpy(pf_np[i:j]).to(device)
            out = ptt_model(bf_t, pf_t)
            h = torch.sigmoid(out["hazard"]).squeeze(-1).cpu().numpy()
            q = F.softmax(out["player_logits"], dim=-1).cpu().numpy()
            h_all.append(h)
            q_all.append(q)
    hazards = np.concatenate(h_all)
    q_probs = np.concatenate(q_all)

    surv = 1.0
    p_entity: Dict[int, float] = {}
    p_touch_frame: List[Tuple[int, float]] = []
    for idx, t in enumerate(frames):
        h_t = float(np.clip(hazards[idx], 1e-6, 1.0 - 1e-6))
        w_t = surv * h_t
        p_touch_frame.append((t, w_t))
        knn = knn_list[idx]
        q_t = q_probs[idx]
        for j, ent in enumerate(knn):
            ent_i = int(ent)
            p_entity[ent_i] = p_entity.get(ent_i, 0.0) + w_t * float(q_t[j])
        surv *= (1.0 - h_t)

    if not p_entity:
        return None
    ranked = sorted(p_entity.items(), key=lambda kv: kv[1], reverse=True)
    pred_receiver = int(ranked[0][0])
    pred_touch_frame = int(max(p_touch_frame, key=lambda kv: kv[1])[0]) if p_touch_frame else -1
    return {
        "pred_receiver_entity": pred_receiver,
        "pred_receiver_top3": [int(x[0]) for x in ranked[:3]],
        "p_entity": p_entity,
        "p_no_touch": float(surv),
        "pred_touch_frame": pred_touch_frame,
    }


def safe_auc(y_true: List[int], y_score: List[float]) -> float:
    if len(y_true) == 0:
        return float("nan")
    if len(set(y_true)) < 2:
        return float("nan")
    try:
        from sklearn.metrics import roc_auc_score
    except Exception:
        return float("nan")
    return float(roc_auc_score(np.asarray(y_true), np.asarray(y_score)))


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    device = torch.device("cpu" if args.no_cuda or not torch.cuda.is_available() else "cuda")
    print(f"Device: {device}")

    ptt_model, ptt_args = load_ptt_model(Path(args.ptt_checkpoint), device)
    k = int(ptt_args["k"])
    print(f"Loaded PTT survival model: k={k}, run={ptt_args.get('run_name')}")

    smart_dir = Path(args.smart_dir)
    tok_mod, _, tokenizer, smart_args, smart_model = load_smart_model(
        smart_dir=smart_dir,
        smart_ckpt_path=Path(args.smart_checkpoint),
        device=device,
    )
    smart_history = args.smart_history if args.smart_history > 0 else int(smart_args.get("history", 8))
    smart_rollout = int(args.smart_rollout)
    print(
        f"Loaded SMART model: history={smart_history}, rollout={smart_rollout}, "
        f"top_k={args.smart_top_k}, temp={args.smart_temp}"
    )

    manifest = load_manifest(Path(args.sportec_dir) / args.split / "manifest.jsonl")
    print(f"Split {args.split}: {len(manifest)} clips in manifest")

    indices = list(range(0, len(manifest), max(1, args.clip_stride)))
    if args.max_clips > 0:
        indices = indices[: args.max_clips]

    n_total = 0
    n_skipped = 0
    n_eval = 0
    n_recv_top1 = 0
    n_recv_top3 = 0
    n_team_acc = 0
    success_true = []
    success_score = []
    clip_rows = []

    for idx in tqdm(indices, desc="hybrid-eval"):
        entry = manifest[idx]
        clip_path = entry["clip_path"]
        try:
            clip = np.load(clip_path)
        except Exception:
            n_skipped += 1
            continue
        n_total += 1

        features = clip["features"].astype(np.float32)  # [T,N,6]
        mask = clip["mask"].astype(bool)
        entity_type = clip["entity_type"].astype(np.int64)
        kick_frame = int(clip["kick_frame_local"]) if "kick_frame_local" in clip else int(entry["kick_frame"] - entry["start_frame"])
        if kick_frame < 0 or kick_frame >= features.shape[0]:
            n_skipped += 1
            continue

        ball_idx, player_indices = get_ball_and_player_indices(entity_type)
        if ball_idx is None or player_indices is None or len(player_indices) < 22:
            n_skipped += 1
            continue
        player_indices = np.asarray(player_indices, dtype=np.int64)

        passer_entity = identify_passer(features, mask, ball_idx, player_indices, kick_frame)
        if passer_entity is None:
            n_skipped += 1
            continue

        gt_recv = detect_reception(
            features=features,
            mask=mask,
            ball_idx=ball_idx,
            player_indices=player_indices,
            kick_frame=kick_frame,
            passer_entity=passer_entity,
            vel_delta=RECV_VEL_DELTA,
            proximity=RECV_PROXIMITY,
            max_search=MAX_SEARCH,
            min_gap=TOUCH_SKIP,
        )
        if gt_recv is None:
            n_skipped += 1
            continue
        gt_recv_frame, gt_recv_entity = int(gt_recv[0]), int(gt_recv[1])

        pred_info = smart_rollout_players_with_gt_ball(
            features=features,
            mask=mask,
            entity_type=entity_type,
            kick_frame=kick_frame,
            player_indices=player_indices,
            ball_idx=ball_idx,
            tok_mod=tok_mod,
            tokenizer=tokenizer,
            smart_model=smart_model,
            device=device,
            history_tokens=smart_history,
            rollout_tokens=smart_rollout,
            top_k=args.smart_top_k,
            temperature=args.smart_temp,
            rng=rng,
        )
        if pred_info is None:
            n_skipped += 1
            continue

        hybrid_feat, pred_start, pred_end = apply_predicted_players(features, pred_info)
        eval_result = infer_first_toucher_from_hazard(
            features=hybrid_feat,
            mask=mask,
            entity_type=entity_type,
            ball_idx=ball_idx,
            player_indices=player_indices,
            kick_frame=kick_frame,
            passer_entity=passer_entity,
            ptt_model=ptt_model,
            device=device,
            k=k,
            batch_size=args.batch_size,
            eval_end_cap=pred_end - 1,
        )
        if eval_result is None:
            n_skipped += 1
            continue

        pred_receiver = int(eval_result["pred_receiver_entity"])
        pred_top3 = set(eval_result["pred_receiver_top3"])

        passer_side = int(entity_type[passer_entity])
        true_side = int(entity_type[gt_recv_entity])
        pred_side = int(entity_type[pred_receiver])
        true_success = 1 if true_side == passer_side else 0
        p_success = float(
            sum(v for ent, v in eval_result["p_entity"].items() if int(entity_type[int(ent)]) == passer_side)
        )

        n_eval += 1
        n_recv_top1 += int(pred_receiver == gt_recv_entity)
        n_recv_top3 += int(gt_recv_entity in pred_top3)
        n_team_acc += int(pred_side == true_side)
        success_true.append(true_success)
        success_score.append(p_success)

        clip_rows.append(
            {
                "clip_path": clip_path,
                "kick_frame_local": kick_frame,
                "gt_recv_frame": gt_recv_frame,
                "gt_recv_entity": gt_recv_entity,
                "pred_recv_entity": pred_receiver,
                "pred_recv_top3": sorted(list(pred_top3)),
                "passer_entity": int(passer_entity),
                "true_success": int(true_success),
                "pred_success_prob": p_success,
                "pred_touch_frame": int(eval_result["pred_touch_frame"]),
                "p_no_touch": float(eval_result["p_no_touch"]),
            }
        )

    out = {
        "config": {
            "sportec_dir": args.sportec_dir,
            "split": args.split,
            "max_clips": args.max_clips,
            "clip_stride": args.clip_stride,
            "ptt_checkpoint": args.ptt_checkpoint,
            "smart_checkpoint": args.smart_checkpoint,
            "smart_history": smart_history,
            "smart_rollout": smart_rollout,
            "smart_top_k": args.smart_top_k,
            "smart_temp": args.smart_temp,
            "batch_size": args.batch_size,
        },
        "counts": {
            "n_manifest_considered": len(indices),
            "n_loaded": n_total,
            "n_skipped": n_skipped,
            "n_eval": n_eval,
        },
        "metrics": {
            "receiver_top1_acc": (n_recv_top1 / n_eval) if n_eval else float("nan"),
            "receiver_top3_acc": (n_recv_top3 / n_eval) if n_eval else float("nan"),
            "receiver_team_acc": (n_team_acc / n_eval) if n_eval else float("nan"),
            "pass_success_auroc": safe_auc(success_true, success_score),
        },
        "per_clip": clip_rows,
    }

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(out, f, indent=2)

    print("\n=== Hybrid Evaluation ===")
    print(f"Evaluated clips: {n_eval} (skipped={n_skipped}, loaded={n_total})")
    if n_eval > 0:
        print(f"Receiver Top-1 Acc:  {out['metrics']['receiver_top1_acc']:.4f}")
        print(f"Receiver Top-3 Acc:  {out['metrics']['receiver_top3_acc']:.4f}")
        print(f"Receiver Team Acc:   {out['metrics']['receiver_team_acc']:.4f}")
        auc = out["metrics"]["pass_success_auroc"]
        if math.isnan(auc):
            print("Pass Success AUROC: nan (single-class or sklearn unavailable)")
        else:
            print(f"Pass Success AUROC: {auc:.4f}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
