#!/usr/bin/env python3
"""Visualize SMART model predictions as animated GIF.

AR rollout: ball tokens from GT or predicted, player tokens sampled from model.
Shows GT vs predicted side by side (2 or 3 panels).

Usage:
    python3 viz.py --checkpoint checkpoints/smart_v1/best.pt --window-idx 366 --output viz.gif
    python3 viz.py --checkpoint checkpoints/smart_v1/best.pt --window-idx 49 --predict-ball --output viz_genball.gif
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

from tokenizer import MotionTokenizer, STEPS_PER_TOKEN, PLAYER_CLAMP, BALL_CLAMP
from dataset import SMARTDataset, load_manifest, DOWNSAMPLE_FACTOR, N_PLAYERS, BALL_ENTITY_TYPE
from model import SMARTTransformer

# Pitch geometry
PITCH_X, PITCH_Y = 105.0, 68.0
HALF_X, HALF_Y = PITCH_X / 2, PITCH_Y / 2

# Colors
HOME_COLOR = "#1f77b4"
AWAY_COLOR = "#d62728"
BALL_COLOR = "#ff7f0e"
GT_ALPHA = 1.0
PRED_ALPHA = 0.85


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default="checkpoints/smart_v1/best.pt")
    p.add_argument("--preprocessed-dir", type=str,
                   default="/mnt/data/remains/opta2026/ballPlayerTrajModel/publicData/preprocessed/sportec_xyz_metres")
    p.add_argument("--vocab-dir", type=str, default=None)
    p.add_argument("--window-idx", type=int, default=366)
    p.add_argument("--split", type=str, default="val")
    p.add_argument("--rollout-len", type=int, default=24, help="Rollout token steps to predict.")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=10, help="Top-K sampling for player tokens.")
    p.add_argument("--predict-ball", action="store_true", help="Also predict ball tokens (adds 3rd panel).")
    p.add_argument("--ball-top-k", type=int, default=5, help="Top-K sampling for ball tokens.")
    p.add_argument("--ball-temperature", type=float, default=0.6, help="Temperature for ball sampling.")
    p.add_argument("--output", type=str, default="viz.gif")
    p.add_argument("--fps", type=int, default=6, help="GIF frames per second (token-level).")
    p.add_argument("--no-cuda", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def draw_pitch(ax):
    rect = Rectangle((-HALF_X, -HALF_Y), PITCH_X, PITCH_Y,
                      linewidth=2, edgecolor="white", facecolor="#2d6a4f")
    ax.add_patch(rect)
    ax.axvline(0, color="white", linewidth=1, alpha=0.5)
    circ = Circle((0, 0), 9.15, color="white", fill=False, linewidth=1, alpha=0.5)
    ax.add_patch(circ)
    for sign in (-1, 1):
        x0 = sign * HALF_X - sign * 16.5
        box = Rectangle((min(x0, sign * HALF_X), -20.16), 16.5, 40.32,
                         linewidth=1, edgecolor="white", facecolor="none", alpha=0.5)
        ax.add_patch(box)
    ax.set_xlim(-HALF_X - 3, HALF_X + 3)
    ax.set_ylim(-HALF_Y - 3, HALF_Y + 3)
    ax.set_aspect("equal")
    ax.set_facecolor("#2d6a4f")
    ax.axis("off")


def load_val_clip(preprocessed_dir, split, window_idx, tokenizer, history, rollout):
    """Load a clip and prepare GT data for visualization."""
    manifest_path = Path(preprocessed_dir) / split / "manifest.jsonl"
    entries = load_manifest(manifest_path)

    ds = SMARTDataset(
        entries=entries, tokenizer=tokenizer,
        history=history, rollout=rollout, window_stride=8,
        noise_top_k=0, augment_flip=False, seed=42,
    )
    print(f"Dataset has {ds.num_windows} windows, requesting idx {window_idx}")
    window_idx = min(window_idx, ds.num_windows - 1)
    return ds[window_idx]


def ar_rollout(model, sample, tokenizer, history, rollout, device, temperature, top_k):
    """Autoregressive rollout: GT ball, sampled players."""
    model.eval()

    player_tokens = sample["player_tokens"].unsqueeze(0).to(device)  # [1, T, 22]
    ball_tokens = sample["ball_tokens"].unsqueeze(0).to(device)      # [1, T]
    player_pos = sample["player_pos"].unsqueeze(0).to(device)        # [1, T, 22, 2]
    ball_pos = sample["ball_pos"].unsqueeze(0).to(device)            # [1, T, 2]
    entity_types = sample["entity_types"].unsqueeze(0).to(device)    # [1, 23]
    obs_mask = sample["obs_mask"].unsqueeze(0).to(device)            # [1, T, 23]

    T = player_tokens.shape[1]
    H = history

    # We'll build predicted tokens step by step
    pred_player_tokens = player_tokens[:, :H].clone()  # start with history

    with torch.no_grad():
        for step in range(rollout):
            t_cur = H + step
            if t_cur >= T:
                break

            # Build input: history + predicted so far + current GT ball
            cur_len = pred_player_tokens.shape[1]

            # Use all tokens up to current step
            inp_player = pred_player_tokens
            inp_ball = ball_tokens[:, :cur_len]
            inp_player_pos = player_pos[:, :cur_len]
            inp_ball_pos = ball_pos[:, :cur_len]
            inp_obs = obs_mask[:, :cur_len]

            # No permutation during inference
            player_logits, _ = model(
                inp_player, inp_ball, inp_player_pos, inp_ball_pos,
                entity_types, inp_obs, perm=None,
            )

            # Get logits at last position -> predict next token
            last_logits = player_logits[:, -1, :, :]  # [1, 22, vocab]

            # Top-K sampling with temperature
            logits_scaled = last_logits / max(temperature, 1e-3)

            if top_k > 0:
                topk_vals, topk_idx = logits_scaled.topk(top_k, dim=-1)
                mask = torch.full_like(logits_scaled, float("-inf"))
                mask.scatter_(-1, topk_idx, topk_vals)
                logits_scaled = mask

            probs = torch.softmax(logits_scaled, dim=-1)  # [1, 22, vocab]
            sampled = torch.multinomial(probs.view(-1, probs.shape[-1]), 1).view(1, 22)

            pred_player_tokens = torch.cat([pred_player_tokens, sampled.unsqueeze(1)], dim=1)

            # Update predicted positions using decoded displacements
            new_disps = tokenizer.decode_player(sampled.cpu().numpy())  # [1, 22, 10]
            # Sum 5-step displacements to get net displacement for this token
            new_disps = new_disps.reshape(1, 22, 5, 2)
            net_disp = new_disps.sum(axis=2)  # [1, 22, 2]

            prev_pos = player_pos[:, t_cur - 1, :, :].cpu().numpy()  # [1, 22, 2]
            new_pos = prev_pos + net_disp
            new_pos_t = torch.from_numpy(new_pos).to(device)

            # Extend player_pos for next step (needed for position embeddings)
            player_pos = torch.cat([player_pos, new_pos_t.unsqueeze(1)], dim=1)
            # Also extend obs_mask
            obs_mask = torch.cat([obs_mask, obs_mask[:, -1:]], dim=1)

    return pred_player_tokens[0].cpu().numpy()  # [H+R, 22]


def ar_rollout_full(model, sample, tokenizer, history, rollout, device,
                    temperature, top_k, ball_temperature, ball_top_k):
    """Autoregressive rollout: both ball and players sampled from model."""
    model.eval()

    player_tokens = sample["player_tokens"].unsqueeze(0).to(device)
    ball_tokens = sample["ball_tokens"].unsqueeze(0).to(device)
    player_pos = sample["player_pos"].unsqueeze(0).to(device).clone()
    ball_pos = sample["ball_pos"].unsqueeze(0).to(device).clone()
    entity_types = sample["entity_types"].unsqueeze(0).to(device)
    obs_mask = sample["obs_mask"].unsqueeze(0).to(device).clone()

    T = player_tokens.shape[1]
    H = history

    pred_player_tokens = player_tokens[:, :H].clone()
    pred_ball_tokens = ball_tokens[:, :H].clone()

    with torch.no_grad():
        for step in range(rollout):
            t_cur = H + step
            if t_cur >= T:
                break

            cur_len = pred_player_tokens.shape[1]

            player_logits, ball_logits = model(
                pred_player_tokens, pred_ball_tokens,
                player_pos[:, :cur_len], ball_pos[:, :cur_len],
                entity_types, obs_mask[:, :cur_len], perm=None,
            )

            # Sample player tokens
            p_logits = player_logits[:, -1, :, :] / max(temperature, 1e-3)
            if top_k > 0:
                topk_vals, topk_idx = p_logits.topk(top_k, dim=-1)
                p_mask = torch.full_like(p_logits, float("-inf"))
                p_mask.scatter_(-1, topk_idx, topk_vals)
                p_logits = p_mask
            p_probs = torch.softmax(p_logits, dim=-1)
            p_sampled = torch.multinomial(p_probs.view(-1, p_probs.shape[-1]), 1).view(1, 22)
            pred_player_tokens = torch.cat([pred_player_tokens, p_sampled.unsqueeze(1)], dim=1)

            # Sample ball tokens
            b_logits = ball_logits[:, -1, :] / max(ball_temperature, 1e-3)
            if ball_top_k > 0:
                topk_vals, topk_idx = b_logits.topk(ball_top_k, dim=-1)
                b_mask = torch.full_like(b_logits, float("-inf"))
                b_mask.scatter_(-1, topk_idx, topk_vals)
                b_logits = b_mask
            b_probs = torch.softmax(b_logits, dim=-1)
            b_sampled = torch.multinomial(b_probs, 1).view(1)  # [1]
            pred_ball_tokens = torch.cat([pred_ball_tokens, b_sampled.view(1, 1)], dim=1)

            # Update player positions
            new_p_disps = tokenizer.decode_player(p_sampled.cpu().numpy()).reshape(1, 22, 5, 2)
            net_p_disp = new_p_disps.sum(axis=2)
            prev_p_pos = player_pos[:, t_cur - 1, :, :].cpu().numpy()
            new_p_pos = torch.from_numpy(prev_p_pos + net_p_disp).to(device)
            player_pos = torch.cat([player_pos, new_p_pos.unsqueeze(1)], dim=1)

            # Update ball positions
            new_b_disp = tokenizer.decode_ball(b_sampled.cpu().numpy())  # [1, 15]
            new_b_disp = new_b_disp.reshape(5, 3).sum(axis=0)[:2]  # xy only
            prev_b_pos = ball_pos[:, t_cur - 1, :].cpu().numpy()[0]
            new_b_xy = prev_b_pos + new_b_disp
            new_b_pos = torch.from_numpy(new_b_xy).float().to(device).view(1, 1, 2)
            ball_pos = torch.cat([ball_pos, new_b_pos], dim=1)

            obs_mask = torch.cat([obs_mask, obs_mask[:, -1:]], dim=1)

    return (pred_player_tokens[0].cpu().numpy(),
            pred_ball_tokens[0].cpu().numpy())


def tokens_to_positions(player_tokens, ball_tokens, initial_player_pos, initial_ball_pos,
                        tokenizer, history):
    """Convert token sequences to position trajectories."""
    T, N = player_tokens.shape

    # Player positions
    player_pos = np.zeros((T, N, 2), dtype=np.float32)
    player_pos[0] = initial_player_pos

    for t in range(1, T):
        disps = tokenizer.decode_player(player_tokens[t:t+1])  # [1, 10]
        # Reshape to [5, 2] and sum for net displacement
        for j in range(N):
            disp = tokenizer.decode_player(np.array([player_tokens[t, j]]))  # [1, 10]
            disp = disp.reshape(5, 2).sum(axis=0)
            player_pos[t, j] = player_pos[t-1, j] + disp

    # Ball positions
    ball_pos = np.zeros((T, 2), dtype=np.float32)
    ball_pos[0] = initial_ball_pos
    for t in range(1, T):
        disp = tokenizer.decode_ball(np.array([ball_tokens[t]]))  # [1, 15]
        disp = disp.reshape(5, 3).sum(axis=0)[:2]  # xy only
        ball_pos[t] = ball_pos[t-1] + disp

    return player_pos, ball_pos


def make_gif(gt_player_pos, gt_ball_pos, panels, entity_types, history, output_path, fps):
    """Render multi-panel animation.

    panels: list of (player_pos, ball_pos, label) tuples
    """
    T = gt_player_pos.shape[0]
    N = gt_player_pos.shape[1]
    n_panels = 1 + len(panels)  # GT + prediction panels

    all_panels = [
        (gt_player_pos, gt_ball_pos, "Ground Truth"),
    ] + list(panels)

    frames = []
    for t in range(T):
        fig, axes = plt.subplots(1, n_panels, figsize=(8 * n_panels, 6))
        if n_panels == 1:
            axes = [axes]
        fig.patch.set_facecolor("#1a1a2e")

        for ax, ppos, bpos, label in zip(axes, *zip(*all_panels)):
            draw_pitch(ax)

            trail_start = max(0, t - 5)
            for j in range(N):
                et = int(entity_types[j + 1]) if j + 1 < len(entity_types) else 0
                color = HOME_COLOR if et == 0 else AWAY_COLOR
                if t > trail_start:
                    trail = ppos[trail_start:t+1, j]
                    ax.plot(trail[:, 0], trail[:, 1], color=color, alpha=0.3, linewidth=1)

            for j in range(N):
                et = int(entity_types[j + 1]) if j + 1 < len(entity_types) else 0
                color = HOME_COLOR if et == 0 else AWAY_COLOR
                ax.scatter(ppos[t, j, 0], ppos[t, j, 1], c=color, s=40,
                           zorder=5, edgecolors="white", linewidths=0.5)

            if t > trail_start:
                bt = bpos[trail_start:t+1]
                ax.plot(bt[:, 0], bt[:, 1], color=BALL_COLOR, alpha=0.5, linewidth=2)
            ax.scatter(bpos[t, 0], bpos[t, 1], c=BALL_COLOR, s=80,
                       zorder=8, edgecolors="white", linewidths=1)

            phase = "History" if t < history else "Rollout"
            step_in_phase = t if t < history else t - history
            token_time = t * 0.4
            ax.set_title(f"{label} | {phase} t={step_in_phase} ({token_time:.1f}s)",
                         color="white", fontsize=10, pad=4)

        if t == history:
            for ax in axes:
                ax.text(0, HALF_Y + 1.5, "ROLLOUT START",
                        ha="center", color="yellow", fontsize=8, fontweight="bold")

        plt.tight_layout()
        fig.canvas.draw()
        buf = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]
        frames.append(buf.copy())
        plt.close(fig)

    import imageio
    imageio.mimsave(output_path, frames, fps=fps, loop=0)
    print(f"Saved {len(frames)}-frame GIF to {output_path}")


def main():
    args = parse_args()
    device = torch.device("cpu") if args.no_cuda or not torch.cuda.is_available() else torch.device("cuda")

    script_dir = Path(__file__).parent
    vocab_dir = Path(args.vocab_dir) if args.vocab_dir else script_dir / "vocabs"

    # Load tokenizer
    tokenizer = MotionTokenizer(vocab_dir / "player_vocab.npz", vocab_dir / "ball_vocab.npz")

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    ckpt_args = ckpt.get("args", {})

    history = ckpt_args.get("history", 8)
    rollout = args.rollout_len
    total = history + rollout

    model = SMARTTransformer(
        d_model=ckpt_args.get("d_model", 256),
        n_heads=ckpt_args.get("n_heads", 8),
        n_layers=ckpt_args.get("n_layers", 6),
        dropout=0.0,  # no dropout at inference
        use_rope=ckpt_args.get("use_rope", True),
        vocab_player=tokenizer.player_centroids.shape[0],
        vocab_ball=tokenizer.ball_centroids.shape[0],
        max_seq_len=total,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    epoch = ckpt.get("epoch", "?")
    best_val = ckpt.get("best_val", "?")
    print(f"Loaded checkpoint: epoch={epoch}, best_val={best_val}")
    print(f"Model: {model.count_parameters():,} params")

    # Load data
    sample = load_val_clip(args.preprocessed_dir, args.split, args.window_idx,
                           tokenizer, history, rollout)

    gt_player_tokens = sample["player_tokens"].numpy()  # [T, 22]
    gt_ball_tokens = sample["ball_tokens"].numpy()       # [T]
    gt_player_pos = sample["player_pos"].numpy()         # [T, 22, 2]
    gt_ball_pos = sample["ball_pos"].numpy()              # [T, 2]
    entity_types = sample["entity_types"].numpy()         # [23]

    print(f"Window: {history} history + {rollout} rollout = {total} token steps")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # AR rollout with GT ball
    print("Running AR rollout (GT ball)...")
    pred_player_tokens = ar_rollout(
        model, sample, tokenizer, history, rollout, device,
        args.temperature, args.top_k,
    )

    pred_player_pos = gt_player_pos.copy()
    for t in range(history, min(pred_player_tokens.shape[0], total)):
        for j in range(N_PLAYERS):
            disp = tokenizer.decode_player(np.array([pred_player_tokens[t, j]]))
            disp = disp.reshape(5, 2).sum(axis=0)
            pred_player_pos[t, j] = pred_player_pos[t-1, j] + disp
    pred_ball_pos_gt = gt_ball_pos.copy()

    panels = [(pred_player_pos, pred_ball_pos_gt, "Pred (GT ball)")]

    # Full-generation rollout (predicted ball + players)
    if args.predict_ball:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        print("Running AR rollout (generated ball)...")
        gen_player_tokens, gen_ball_tokens = ar_rollout_full(
            model, sample, tokenizer, history, rollout, device,
            args.temperature, args.top_k,
            args.ball_temperature, args.ball_top_k,
        )

        gen_player_pos = gt_player_pos.copy()
        for t in range(history, min(gen_player_tokens.shape[0], total)):
            for j in range(N_PLAYERS):
                disp = tokenizer.decode_player(np.array([gen_player_tokens[t, j]]))
                disp = disp.reshape(5, 2).sum(axis=0)
                gen_player_pos[t, j] = gen_player_pos[t-1, j] + disp

        gen_ball_pos = gt_ball_pos.copy()
        for t in range(history, min(gen_ball_tokens.shape[0], total)):
            disp = tokenizer.decode_ball(np.array([gen_ball_tokens[t]]))
            disp = disp.reshape(5, 3).sum(axis=0)[:2]
            gen_ball_pos[t] = gen_ball_pos[t-1] + disp

        panels.append((gen_player_pos, gen_ball_pos, "Pred (gen ball)"))

    print("Rendering GIF...")
    make_gif(gt_player_pos, gt_ball_pos, panels,
             entity_types, history, args.output, args.fps)


if __name__ == "__main__":
    main()
