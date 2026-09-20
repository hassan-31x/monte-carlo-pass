#!/usr/bin/env python3
"""Render GT vs best-of-20 SMART prediction as a side-by-side PNG.

Uses 4 GPUs by default: splits 20 stochastic rollouts across devices.
Best sample is selected by lowest ADE (with FDE tiebreak), and minADE/minFDE
over all 20 samples are reported in the figure title.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import numpy as np
import torch

from dataset import N_PLAYERS, SMARTDataset, load_manifest
from model import SMARTTransformer
from tokenizer import MotionTokenizer


# Pitch geometry
PITCH_X, PITCH_Y = 105.0, 68.0
HALF_X, HALF_Y = PITCH_X / 2, PITCH_Y / 2

# Colors
HOME_COLOR = "#1f77b4"
AWAY_COLOR = "#d62728"
BALL_COLOR = "#ff7f0e"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default="checkpoints/smart_v1/best.pt")
    p.add_argument(
        "--preprocessed-dir",
        type=str,
        default="/mnt/data/remains/opta2026/ballPlayerTrajModel/publicData/preprocessed/sportec_xyz_metres",
    )
    p.add_argument("--vocab-dir", type=str, default=None)
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--window-idx", type=int, default=0)
    p.add_argument("--rollout-len", type=int, default=24)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--num-samples", type=int, default=20)
    p.add_argument("--num-gpus", type=int, default=4)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--output", type=str, default="test_clip_gt_vs_pred_bestof20.png")
    return p.parse_args()


def draw_pitch(ax):
    rect = Rectangle(
        (-HALF_X, -HALF_Y),
        PITCH_X,
        PITCH_Y,
        linewidth=2,
        edgecolor="white",
        facecolor="#2d6a4f",
    )
    ax.add_patch(rect)
    ax.axvline(0, color="white", linewidth=1, alpha=0.5)
    circ = Circle((0, 0), 9.15, color="white", fill=False, linewidth=1, alpha=0.5)
    ax.add_patch(circ)
    for sign in (-1, 1):
        x0 = sign * HALF_X - sign * 16.5
        box = Rectangle(
            (min(x0, sign * HALF_X), -20.16),
            16.5,
            40.32,
            linewidth=1,
            edgecolor="white",
            facecolor="none",
            alpha=0.5,
        )
        ax.add_patch(box)
    ax.set_xlim(-HALF_X - 2, HALF_X + 2)
    ax.set_ylim(-HALF_Y - 2, HALF_Y + 2)
    ax.set_aspect("equal")
    ax.set_facecolor("#2d6a4f")
    ax.axis("off")


def load_window(
    preprocessed_dir: Path,
    split: str,
    window_idx: int,
    tokenizer: MotionTokenizer,
    history: int,
    rollout: int,
) -> Dict[str, np.ndarray]:
    manifest_path = preprocessed_dir / split / "manifest.jsonl"
    entries = load_manifest(manifest_path)
    ds = SMARTDataset(
        entries=entries,
        tokenizer=tokenizer,
        history=history,
        rollout=rollout,
        window_stride=8,
        noise_top_k=0,
        augment_flip=False,
        seed=42,
    )
    if ds.num_windows == 0:
        raise RuntimeError(f"No windows found in split={split}")
    idx = min(max(window_idx, 0), ds.num_windows - 1)
    sample = ds[idx]
    out = {}
    for k, v in sample.items():
        out[k] = v.numpy()
    out["window_idx_used"] = idx
    out["num_windows"] = ds.num_windows
    return out


def sample_rollout_positions(
    model: SMARTTransformer,
    tokenizer: MotionTokenizer,
    sample: Dict[str, np.ndarray],
    history: int,
    rollout: int,
    temperature: float,
    top_k: int,
    seed: int,
    device: torch.device,
) -> np.ndarray:
    """Run one stochastic rollout (GT ball) and return predicted player positions [T,22,2]."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    gt_player_tokens = torch.from_numpy(sample["player_tokens"]).long().to(device)
    gt_ball_tokens = torch.from_numpy(sample["ball_tokens"]).long().to(device)
    gt_player_pos = torch.from_numpy(sample["player_pos"]).float().to(device)
    gt_ball_pos = torch.from_numpy(sample["ball_pos"]).float().to(device)
    entity_types = torch.from_numpy(sample["entity_types"]).long().to(device).unsqueeze(0)
    obs_mask = torch.from_numpy(sample["obs_mask"]).bool().to(device)

    T = gt_player_tokens.shape[0]
    total_steps = min(T, history + rollout)

    pred_player_tokens = gt_player_tokens[:history].clone()  # [H, 22]
    pred_player_pos = gt_player_pos[:history].clone()        # [H, 22, 2]

    model.eval()
    with torch.no_grad():
        for step in range(rollout):
            t_cur = history + step
            if t_cur >= total_steps:
                break

            cur_len = pred_player_tokens.shape[0]
            inp_player = pred_player_tokens.unsqueeze(0)          # [1, L, 22]
            inp_ball = gt_ball_tokens[:cur_len].unsqueeze(0)      # [1, L]
            inp_player_pos = pred_player_pos.unsqueeze(0)         # [1, L, 22, 2]
            inp_ball_pos = gt_ball_pos[:cur_len].unsqueeze(0)     # [1, L, 2]
            inp_obs = obs_mask[:cur_len].unsqueeze(0)             # [1, L, 23]

            player_logits, _ = model(
                inp_player,
                inp_ball,
                inp_player_pos,
                inp_ball_pos,
                entity_types,
                inp_obs,
                perm=None,
            )
            logits = player_logits[:, -1, :, :] / max(temperature, 1e-3)  # [1,22,V]
            if top_k > 0:
                top_vals, top_idx = logits.topk(top_k, dim=-1)
                masked = torch.full_like(logits, float("-inf"))
                masked.scatter_(-1, top_idx, top_vals)
                logits = masked
            probs = torch.softmax(logits, dim=-1)
            sampled = torch.multinomial(probs.view(-1, probs.shape[-1]), 1).view(1, N_PLAYERS)
            sampled_step = sampled[0]  # [22]

            pred_player_tokens = torch.cat([pred_player_tokens, sampled_step.unsqueeze(0)], dim=0)

            disp = tokenizer.decode_player(sampled_step.detach().cpu().numpy())  # [22, 10]
            net_disp = disp.reshape(N_PLAYERS, 5, 2).sum(axis=1)                 # [22, 2]
            prev = pred_player_pos[-1].detach().cpu().numpy()                    # [22, 2]
            next_pos = torch.from_numpy(prev + net_disp).float().to(device)      # [22, 2]
            pred_player_pos = torch.cat([pred_player_pos, next_pos.unsqueeze(0)], dim=0)

    # Build full-length [T,22,2], preserving GT outside rollout.
    gt_player_pos_np = sample["player_pos"].copy()
    pred_np = gt_player_pos_np.copy()
    n_valid = min(pred_player_pos.shape[0], pred_np.shape[0])
    pred_np[:n_valid] = pred_player_pos[:n_valid].detach().cpu().numpy()
    return pred_np


def compute_ade_fde(
    pred_player_pos: np.ndarray,
    gt_player_pos: np.ndarray,
    obs_mask: np.ndarray,
    history: int,
    rollout: int,
) -> Tuple[float, float]:
    end = min(gt_player_pos.shape[0], history + rollout)
    if end <= history:
        return float("inf"), float("inf")

    pred_roll = pred_player_pos[history:end]
    gt_roll = gt_player_pos[history:end]
    obs_roll = obs_mask[history:end, 1:]  # players only [R,22]

    d = np.linalg.norm(pred_roll - gt_roll, axis=-1)  # [R,22]
    if not np.any(obs_roll):
        return float("inf"), float("inf")

    ade = float(d[obs_roll].mean())

    final_t = end - 1
    final_obs = obs_mask[final_t, 1:]
    if np.any(final_obs):
        fde = float(d[-1, final_obs].mean())
    else:
        fde = float("inf")
    return ade, fde


def _worker(
    gpu_id: int,
    seeds: List[int],
    checkpoint_path: str,
    ckpt_args: Dict,
    tokenizer_paths: Tuple[str, str],
    sample: Dict[str, np.ndarray],
    history: int,
    rollout: int,
    temperature: float,
    top_k: int,
    queue: mp.Queue,
):
    try:
        if not seeds:
            queue.put({"gpu_id": gpu_id, "results": []})
            return

        torch.cuda.set_device(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")

        tokenizer = MotionTokenizer(tokenizer_paths[0], tokenizer_paths[1])
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

        model = SMARTTransformer(
            d_model=ckpt_args.get("d_model", 256),
            n_heads=ckpt_args.get("n_heads", 8),
            n_layers=ckpt_args.get("n_layers", 6),
            dropout=0.0,
            use_rope=ckpt_args.get("use_rope", True),
            vocab_player=tokenizer.player_centroids.shape[0],
            vocab_ball=tokenizer.ball_centroids.shape[0],
            max_seq_len=history + rollout,
        ).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()

        out = []
        for s in seeds:
            pred_pos = sample_rollout_positions(
                model=model,
                tokenizer=tokenizer,
                sample=sample,
                history=history,
                rollout=rollout,
                temperature=temperature,
                top_k=top_k,
                seed=s,
                device=device,
            )
            ade, fde = compute_ade_fde(
                pred_pos,
                sample["player_pos"],
                sample["obs_mask"],
                history,
                rollout,
            )
            out.append({"seed": s, "ade": ade, "fde": fde, "pred_player_pos": pred_pos})

        queue.put({"gpu_id": gpu_id, "results": out})
    except Exception as e:
        queue.put({"gpu_id": gpu_id, "error": repr(e)})


def render_png(
    output_path: Path,
    gt_player_pos: np.ndarray,
    gt_ball_pos: np.ndarray,
    pred_player_pos: np.ndarray,
    entity_types: np.ndarray,
    history: int,
    rollout: int,
    sample_ade: float,
    sample_fde: float,
    min_ade: float,
    min_fde: float,
    window_idx: int,
):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=200)
    fig.patch.set_facecolor("#111111")

    titles = ["Ground Truth (Test Clip)", "SMART Prediction (Best of 20)"]
    panels = [(gt_player_pos, gt_ball_pos), (pred_player_pos, gt_ball_pos)]

    for ax, title, (ppos, bpos) in zip(axes, titles, panels):
        draw_pitch(ax)

        hist_end = min(history, ppos.shape[0] - 1)
        pred_end = min(history + rollout - 1, ppos.shape[0] - 1)

        for j in range(ppos.shape[1]):
            et = int(entity_types[j + 1]) if j + 1 < len(entity_types) else 0
            color = HOME_COLOR if et == 0 else AWAY_COLOR

            # History trajectory
            if hist_end >= 1:
                ax.plot(
                    ppos[: hist_end + 1, j, 0],
                    ppos[: hist_end + 1, j, 1],
                    color=color,
                    linewidth=1.5,
                    alpha=0.9,
                )
            # Rollout trajectory
            if pred_end >= history:
                ax.plot(
                    ppos[history - 1 : pred_end + 1, j, 0],
                    ppos[history - 1 : pred_end + 1, j, 1],
                    color=color,
                    linewidth=2.0,
                    alpha=0.9,
                    linestyle="--",
                )
            ax.scatter(
                ppos[pred_end, j, 0],
                ppos[pred_end, j, 1],
                c=color,
                s=22,
                edgecolors="white",
                linewidths=0.4,
                zorder=5,
            )

        if hist_end >= 1:
            ax.plot(
                bpos[: hist_end + 1, 0],
                bpos[: hist_end + 1, 1],
                color=BALL_COLOR,
                linewidth=2.0,
                alpha=0.8,
            )
        if pred_end >= history:
            ax.plot(
                bpos[history - 1 : pred_end + 1, 0],
                bpos[history - 1 : pred_end + 1, 1],
                color=BALL_COLOR,
                linewidth=2.5,
                alpha=0.9,
                linestyle="--",
            )
        ax.scatter(
            bpos[pred_end, 0],
            bpos[pred_end, 1],
            c=BALL_COLOR,
            s=52,
            edgecolors="white",
            linewidths=0.8,
            zorder=8,
        )

        ax.set_title(title, color="white", fontsize=11, pad=8)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    checkpoint_path = str((script_dir / args.checkpoint).resolve() if not Path(args.checkpoint).is_absolute() else Path(args.checkpoint))
    preprocessed_dir = Path(args.preprocessed_dir)
    vocab_dir = Path(args.vocab_dir) if args.vocab_dir else script_dir / "vocabs"
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = script_dir / output_path

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this script.")
    available_gpus = torch.cuda.device_count()
    n_gpus = min(args.num_gpus, available_gpus)
    if n_gpus < 1:
        raise RuntimeError("No GPUs available.")
    print(f"Using {n_gpus} GPUs (available={available_gpus})")

    ckpt_cpu = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    ckpt_args = ckpt_cpu.get("args", {})
    history = int(ckpt_args.get("history", 8))
    rollout = int(args.rollout_len)

    tokenizer = MotionTokenizer(vocab_dir / "player_vocab.npz", vocab_dir / "ball_vocab.npz")
    sample = load_window(
        preprocessed_dir=preprocessed_dir,
        split=args.split,
        window_idx=args.window_idx,
        tokenizer=tokenizer,
        history=history,
        rollout=rollout,
    )
    window_idx_used = int(sample["window_idx_used"])
    num_windows = int(sample["num_windows"])
    print(f"Split={args.split} windows={num_windows}, using window_idx={window_idx_used}")

    seeds = [args.seed + i for i in range(args.num_samples)]
    seed_chunks = [seeds[i::n_gpus] for i in range(n_gpus)]
    print("Per-GPU sample counts:", [len(c) for c in seed_chunks])

    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    workers = []

    sample_worker = {k: v for k, v in sample.items() if isinstance(v, np.ndarray)}
    tokenizer_paths = (str(vocab_dir / "player_vocab.npz"), str(vocab_dir / "ball_vocab.npz"))

    for gpu_id in range(n_gpus):
        p = ctx.Process(
            target=_worker,
            args=(
                gpu_id,
                seed_chunks[gpu_id],
                checkpoint_path,
                ckpt_args,
                tokenizer_paths,
                sample_worker,
                history,
                rollout,
                args.temperature,
                args.top_k,
                queue,
            ),
        )
        p.start()
        workers.append(p)

    all_results = []
    for _ in workers:
        msg = queue.get()
        if "error" in msg:
            raise RuntimeError(f"Worker on GPU {msg['gpu_id']} failed: {msg['error']}")
        all_results.extend(msg["results"])

    for p in workers:
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"Worker exited with code {p.exitcode}")

    if not all_results:
        raise RuntimeError("No rollout results produced.")

    min_ade = min(r["ade"] for r in all_results)
    min_fde = min(r["fde"] for r in all_results)
    best = sorted(all_results, key=lambda x: (x["ade"], x["fde"]))[0]

    print(
        f"Selected seed={best['seed']} ADE={best['ade']:.4f} FDE={best['fde']:.4f} | "
        f"minADE@{args.num_samples}={min_ade:.4f} minFDE@{args.num_samples}={min_fde:.4f}"
    )

    render_png(
        output_path=output_path,
        gt_player_pos=sample["player_pos"],
        gt_ball_pos=sample["ball_pos"],
        pred_player_pos=best["pred_player_pos"],
        entity_types=sample["entity_types"],
        history=history,
        rollout=rollout,
        sample_ade=best["ade"],
        sample_fde=best["fde"],
        min_ade=min_ade,
        min_fde=min_fde,
        window_idx=window_idx_used,
    )
    print(f"Saved PNG: {output_path}")


if __name__ == "__main__":
    # Avoid tokenizer and CUDA fork issues.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
