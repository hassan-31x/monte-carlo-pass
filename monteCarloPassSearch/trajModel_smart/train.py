#!/usr/bin/env python3
"""Training script for SMART trajectory model.

Teacher-forced cross-entropy classification with:
- Cosine LR schedule with warmup
- Label smoothing
- DDP multi-GPU support
- AMP fp16
- Checkpoint management with best/last tracking
- Top-1/Top-5 accuracy metrics

Usage:
    # Single GPU
    python3 train.py --run-name smart_v1

    # Multi-GPU
    CUDA_VISIBLE_DEVICES=0,1,2 torchrun --nproc_per_node=3 train.py --run-name smart_v1
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, DistributedSampler
from tqdm.auto import tqdm

from dataset import SMARTDataset, create_datasets
from model import SMARTTransformer
from tokenizer import MotionTokenizer


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train SMART trajectory model.")

    # Data
    p.add_argument("--preprocessed-dir", type=str,
                   default="/mnt/data/remains/opta2026/ballPlayerTrajModel/publicData/preprocessed/sportec_xyz_metres")
    p.add_argument("--vocab-dir", type=str, default=None,
                   help="Directory with player_vocab.npz and ball_vocab.npz. Default: vocabs/ in script dir.")
    p.add_argument("--save-dir", type=str, default=None,
                   help="Checkpoint save directory. Default: checkpoints/ in script dir.")
    p.add_argument("--run-name", type=str, default="smart_v1")

    # Model
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--n-layers", type=int, default=6)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--use-rope", action=argparse.BooleanOptionalAction, default=True)

    # Training
    p.add_argument("--history", type=int, default=8, help="History token steps.")
    p.add_argument("--rollout", type=int, default=24, help="Rollout token steps.")
    p.add_argument("--window-stride", type=int, default=8, help="Window stride in token steps.")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=48, help="Per-GPU batch size.")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--noise-top-k", type=int, default=3, help="Top-K nearest tokens for noise injection.")
    p.add_argument("--warmup-steps", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--patience", type=int, default=10, help="Early stopping patience (0=disabled).")

    # Infrastructure
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--clip-cache-size", type=int, default=8)
    p.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume from.")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------

def setup_dist():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank
    return 0, 1, 0


def cleanup_dist():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main(rank: int) -> bool:
    return rank == 0


def to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


# ---------------------------------------------------------------------------
# LR Schedule: cosine with linear warmup
# ---------------------------------------------------------------------------

def get_lr(step: int, warmup_steps: int, max_steps: int, base_lr: float) -> float:
    if step < warmup_steps:
        return base_lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def set_lr(optimizer, lr: float):
    for pg in optimizer.param_groups:
        pg["lr"] = lr


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_accuracy(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor, top_k: int = 5):
    """Compute top-1 and top-K accuracy.

    Args:
        logits: [*, vocab_size]
        targets: [*] int64
        mask: [*] bool - which positions to count

    Returns:
        (top1_acc, topk_acc, n_valid)
    """
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_targets = targets.reshape(-1)
    flat_mask = mask.reshape(-1)

    valid = flat_mask.sum().item()
    if valid == 0:
        return 0.0, 0.0, 0

    # Top-1
    preds = flat_logits.argmax(dim=-1)
    top1 = ((preds == flat_targets) & flat_mask).sum().item()

    # Top-K
    _, topk_preds = flat_logits.topk(top_k, dim=-1)
    topk_match = (topk_preds == flat_targets.unsqueeze(-1)).any(dim=-1)
    topk = (topk_match & flat_mask).sum().item()

    return top1 / valid, topk / valid, int(valid)


# ---------------------------------------------------------------------------
# Training / validation epoch
# ---------------------------------------------------------------------------

def run_epoch(
    model, loader, optimizer, device, scaler, args,
    epoch: int, split: str, show_progress: bool,
    global_step: int, max_steps: int,
) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train(mode=is_train)
    task_model = model.module if isinstance(model, DDP) else model

    H = args.history
    R = args.rollout

    total_loss = 0.0
    total_top1 = 0.0
    total_top5 = 0.0
    total_valid = 0
    n_batches = 0

    prog = tqdm(loader, desc=f"Epoch {epoch:03d} [{split}]", leave=False,
                disable=not show_progress, mininterval=0.5)

    for batch in prog:
        batch = to_device(batch, device)

        player_tokens = batch["player_tokens"]   # [B, T, 22]
        ball_tokens = batch["ball_tokens"]        # [B, T]
        player_pos = batch["player_pos"]          # [B, T, 22, 2]
        ball_pos = batch["ball_pos"]              # [B, T, 2]
        entity_types = batch["entity_types"]      # [B, 23]
        obs_mask = batch["obs_mask"]              # [B, T, 23]
        perm = batch["perm"]                      # [B, T, 22]

        B, T, N_players = player_tokens.shape

        # Update LR (cosine schedule)
        if is_train:
            lr = get_lr(global_step, args.warmup_steps, max_steps, args.lr)
            set_lr(optimizer, lr)

        with torch.amp.autocast("cuda", enabled=args.amp):
            player_logits, ball_logits = model(
                player_tokens, ball_tokens, player_pos, ball_pos,
                entity_types, obs_mask, perm,
            )
            # player_logits: [B, T, 22, vocab_player]
            # ball_logits:   [B, T, vocab_ball]

            # Next-token prediction: representation at t predicts token at t+1
            # Use tokens from history-1 to history-1+rollout as predictions
            # Target tokens are from history to history+rollout
            pred_player_logits = player_logits[:, H-1:H-1+R]   # [B, R, 22, V_p]
            pred_ball_logits = ball_logits[:, H-1:H-1+R]        # [B, R, V_b]

            target_player = player_tokens[:, H:H+R]             # [B, R, 22]
            target_ball = ball_tokens[:, H:H+R]                  # [B, R]

            # Masks for valid predictions
            player_valid = obs_mask[:, H:H+R, 1:]               # [B, R, 22]
            ball_valid = obs_mask[:, H:H+R, 0]                   # [B, R]

            # Player loss
            p_logits_flat = pred_player_logits.reshape(-1, task_model.vocab_player)
            p_targets_flat = target_player.reshape(-1)
            p_mask_flat = player_valid.reshape(-1)

            # Compute CE loss only on valid positions
            if p_mask_flat.any():
                p_loss_all = F.cross_entropy(
                    p_logits_flat, p_targets_flat,
                    label_smoothing=args.label_smoothing, reduction="none",
                )
                player_loss = (p_loss_all * p_mask_flat.float()).sum() / p_mask_flat.float().sum()
            else:
                player_loss = torch.tensor(0.0, device=device)

            # Ball loss (not used during ball-conditioned training, but compute for monitoring)
            if ball_valid.any():
                b_logits_flat = pred_ball_logits.reshape(-1, task_model.vocab_ball)
                b_targets_flat = target_ball.reshape(-1)
                b_mask_flat = ball_valid.reshape(-1)
                b_loss_all = F.cross_entropy(
                    b_logits_flat, b_targets_flat,
                    label_smoothing=args.label_smoothing, reduction="none",
                )
                ball_loss = (b_loss_all * b_mask_flat.float()).sum() / b_mask_flat.float().sum()
            else:
                ball_loss = torch.tensor(0.0, device=device)

            # Total loss: player + small ball loss (ball head must receive grads for DDP)
            loss = player_loss + 0.1 * ball_loss

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
            global_step += 1

        # Metrics
        with torch.no_grad():
            p_top1, p_top5, p_n = compute_accuracy(
                pred_player_logits, target_player, player_valid)

        total_loss += loss.item()
        total_top1 += p_top1 * p_n
        total_top5 += p_top5 * p_n
        total_valid += p_n
        n_batches += 1

        if show_progress:
            avg_loss = total_loss / n_batches
            avg_top1 = total_top1 / max(total_valid, 1) * 100
            avg_top5 = total_top5 / max(total_valid, 1) * 100
            prog.set_postfix(loss=f"{avg_loss:.4f}", top1=f"{avg_top1:.1f}%",
                             top5=f"{avg_top5:.1f}%",
                             lr=f"{lr:.2e}" if is_train else "—")

    metrics = {
        "loss": total_loss / max(n_batches, 1),
        "top1_acc": total_top1 / max(total_valid, 1),
        "top5_acc": total_top5 / max(total_valid, 1),
        "n_batches": n_batches,
        "global_step": global_step,
    }
    return metrics


# ---------------------------------------------------------------------------
# Checkpoint management
# ---------------------------------------------------------------------------

def save_checkpoint(path: Path, model, optimizer, scaler, epoch, best_val, args, global_step):
    task_model = model.module if isinstance(model, DDP) else model
    state = {
        "model": task_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "best_val": best_val,
        "global_step": global_step,
        "args": vars(args),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: Path, model, optimizer=None, scaler=None, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    task_model = model.module if isinstance(model, DDP) else model
    task_model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    return ckpt.get("epoch", 0), ckpt.get("best_val", float("inf")), ckpt.get("global_step", 0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    rank, world_size, local_rank = setup_dist()
    is_distributed = world_size > 1

    # Seeds
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    random.seed(args.seed + rank)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # Paths
    script_dir = Path(__file__).parent
    vocab_dir = Path(args.vocab_dir) if args.vocab_dir else script_dir / "vocabs"
    save_dir = Path(args.save_dir) if args.save_dir else script_dir / "checkpoints"
    run_dir = save_dir / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    if is_main(rank):
        print(f"SMART Trajectory Model Training")
        print(f"  Device: {device}, World size: {world_size}")
        print(f"  Run: {args.run_name}")
        print(f"  Vocab: {vocab_dir}")
        print(f"  Data: {args.preprocessed_dir}")

    # Load tokenizer
    tokenizer = MotionTokenizer(
        vocab_dir / "player_vocab.npz",
        vocab_dir / "ball_vocab.npz",
    )

    # Create datasets
    train_ds, val_ds = create_datasets(
        preprocessed_dir=args.preprocessed_dir,
        tokenizer=tokenizer,
        history=args.history,
        rollout=args.rollout,
        window_stride=args.window_stride,
        noise_top_k=args.noise_top_k,
        augment_flip=args.augment,
        clip_cache_size=args.clip_cache_size,
        seed=args.seed,
    )

    if is_main(rank):
        print(f"  Train windows: {train_ds.num_windows:,}")
        print(f"  Val windows: {val_ds.num_windows if val_ds else 0:,}")

    # Samplers
    train_sampler = DistributedSampler(train_ds, world_size, rank, shuffle=True) if is_distributed else None
    val_sampler = DistributedSampler(val_ds, world_size, rank, shuffle=False) if is_distributed and val_ds else None

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        sampler=train_sampler, shuffle=(train_sampler is None),
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = None
    if val_ds:
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size,
            sampler=val_sampler, shuffle=False,
            num_workers=args.num_workers, pin_memory=True, drop_last=False,
        )

    # Model
    model = SMARTTransformer(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
        use_rope=args.use_rope,
        vocab_player=tokenizer.player_centroids.shape[0],
        vocab_ball=tokenizer.ball_centroids.shape[0],
        max_seq_len=args.history + args.rollout,
    ).to(device)

    if is_main(rank):
        n_params = model.count_parameters()
        print(f"  Model: {n_params:,} parameters ({n_params/1e6:.1f}M)")

    if is_distributed:
        model = DDP(model, device_ids=[local_rank])

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # AMP
    scaler = torch.amp.GradScaler("cuda") if args.amp and device.type == "cuda" else None

    # Resume
    start_epoch = 0
    best_val = float("inf")
    global_step = 0
    if args.resume:
        if is_main(rank):
            print(f"  Resuming from {args.resume}")
        start_epoch, best_val, global_step = load_checkpoint(
            Path(args.resume), model, optimizer, scaler, device)
        start_epoch += 1  # resume from next epoch

    # Max steps for cosine schedule
    steps_per_epoch = len(train_loader)
    max_steps = steps_per_epoch * args.epochs

    if is_main(rank):
        print(f"  Steps/epoch: {steps_per_epoch}, Max steps: {max_steps}")
        print(f"  Warmup: {args.warmup_steps} steps")
        print(f"  History: {args.history}, Rollout: {args.rollout}")
        print(f"  Batch: {args.batch_size}/GPU x {world_size} GPUs = {args.batch_size * world_size} effective")
        print(f"  Label smoothing: {args.label_smoothing}")
        print(f"  Noise top-K: {args.noise_top_k}")
        # Save args
        with open(run_dir / "args.json", "w") as f:
            json.dump(vars(args), f, indent=2)

    patience_counter = 0

    for epoch in range(start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        t0 = time.time()

        # Train
        train_metrics = run_epoch(
            model, train_loader, optimizer, device, scaler, args,
            epoch, "train", show_progress=is_main(rank),
            global_step=global_step, max_steps=max_steps,
        )
        global_step = train_metrics["global_step"]

        # Val
        val_metrics = {}
        if val_loader:
            with torch.no_grad():
                val_metrics = run_epoch(
                    model, val_loader, None, device, None, args,
                    epoch, "val", show_progress=is_main(rank),
                    global_step=global_step, max_steps=max_steps,
                )

        elapsed = time.time() - t0

        if is_main(rank):
            train_loss = train_metrics["loss"]
            train_top1 = train_metrics["top1_acc"] * 100
            train_top5 = train_metrics["top5_acc"] * 100
            val_loss = val_metrics.get("loss", 0.0)
            val_top1 = val_metrics.get("top1_acc", 0.0) * 100
            val_top5 = val_metrics.get("top5_acc", 0.0) * 100

            print(f"Epoch {epoch:03d} ({elapsed:.0f}s) | "
                  f"Train: loss={train_loss:.4f} top1={train_top1:.1f}% top5={train_top5:.1f}% | "
                  f"Val: loss={val_loss:.4f} top1={val_top1:.1f}% top5={val_top5:.1f}%")

            # Save last checkpoint
            save_checkpoint(run_dir / "last.pt", model, optimizer, scaler,
                            epoch, best_val, args, global_step)

            # Save best checkpoint
            if val_metrics and val_metrics["loss"] < best_val:
                best_val = val_metrics["loss"]
                save_checkpoint(run_dir / "best.pt", model, optimizer, scaler,
                                epoch, best_val, args, global_step)
                print(f"  -> New best val loss: {best_val:.4f}")
                patience_counter = 0
            else:
                patience_counter += 1

            # Log to jsonl
            log_entry = {
                "epoch": epoch, "elapsed": elapsed,
                "train_loss": train_loss, "train_top1": train_top1, "train_top5": train_top5,
                "val_loss": val_loss, "val_top1": val_top1, "val_top5": val_top5,
                "lr": get_lr(global_step, args.warmup_steps, max_steps, args.lr),
                "best_val": best_val,
            }
            with open(run_dir / "log.jsonl", "a") as f:
                f.write(json.dumps(log_entry) + "\n")

            # Early stopping
            if args.patience > 0 and patience_counter >= args.patience:
                print(f"Early stopping: no improvement for {args.patience} epochs")
                break

    if is_main(rank):
        print(f"\nTraining complete. Best val loss: {best_val:.4f}")
        print(f"Checkpoints saved to: {run_dir}")

    cleanup_dist()


if __name__ == "__main__":
    main()
