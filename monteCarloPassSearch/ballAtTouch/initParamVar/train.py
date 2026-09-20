#!/usr/bin/env python3
"""Train ballAtTouch: predict outgoing ball velocity at moment of touch.

Uses Huber (smooth L1) loss on normalised [vx, vy, vz] components.
z-component (vz) will be near-zero for 2D tracking data but the model
can still learn it; simViz can override with a physics-based loft estimate.

Usage:
    # after preprocessing:
    python3 train.py --preprocessed-dir preprocessed --epochs 80
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from dataset import BallAtTouchDataset, create_datasets
from model import BallAtTouchModel, GaussianBallAtTouchModel


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    # Data
    p.add_argument("--preprocessed-dir", type=str,
                   default="/mnt/data/remains/opta2026/ballAtTouch/preprocessed")
    p.add_argument("--save-dir", type=str,
                   default="/mnt/data/remains/opta2026/ballAtTouch/checkpoints")
    p.add_argument("--run-name", type=str, default="run")
    # Model
    p.add_argument("--d-model",  type=int,   default=128)
    p.add_argument("--n-heads",  type=int,   default=4)
    p.add_argument("--n-layers", type=int,   default=4)
    p.add_argument("--dropout",  type=float, default=0.1)
    p.add_argument("--k-other",  type=int,   default=8)
    # Training
    p.add_argument("--epochs",       type=int,   default=80)
    p.add_argument("--batch-size",   type=int,   default=256)
    p.add_argument("--lr",           type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--huber-delta",  type=float, default=1.0,
                   help="Huber loss delta (in normalised vel units).")
    p.add_argument("--num-workers",  type=int,   default=4)
    p.add_argument("--resume",       type=str,   default=None)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--model-type",   type=str,   default="deterministic",
                   choices=["deterministic", "gaussian"],
                   help="deterministic = Huber loss, gaussian = NLL loss")
    p.add_argument("--augment", action="store_true",
                   help="Enable flip augmentation for training data")
    return p.parse_args()


# ── training helpers ──────────────────────────────────────────────────────────

def run_epoch(
    model: BallAtTouchModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    huber_delta: float,
    device: torch.device,
    amp_scaler: torch.cuda.amp.GradScaler,
    train: bool,
) -> Tuple[float, float]:
    """Returns (loss, mean_abs_error in normalised units)."""
    model.train(train)
    total_loss, total_mae, total_n = 0.0, 0.0, 0

    for bf, sf, of_, lv, vel_obs_mask in loader:
        bf           = bf.to(device,           non_blocking=True)
        sf           = sf.to(device,           non_blocking=True)
        of_          = of_.to(device,          non_blocking=True)
        lv           = lv.to(device,           non_blocking=True)
        vel_obs_mask = vel_obs_mask.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=amp_scaler.is_enabled()):
            pred = model(bf, sf, of_)              # [B, 3]
            # Masked Huber loss: only compute on observed velocity components.
            # obs_mask is [B, 3] bool; vz (index 2) is False for OPTA samples.
            obs_mask = vel_obs_mask.float()                          # [B, 3]
            sq_err   = F.huber_loss(pred, lv, delta=huber_delta,
                                    reduction="none")                # [B, 3]
            loss = (sq_err * obs_mask).sum(-1) / obs_mask.sum(-1).clamp(min=1)
            loss = loss.mean()

        if train:
            optimizer.zero_grad(set_to_none=True)
            amp_scaler.scale(loss).backward()
            amp_scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            amp_scaler.step(optimizer)
            amp_scaler.update()

        B = lv.size(0)
        with torch.no_grad():
            mae = (pred - lv).abs().mean().item()
        total_loss += loss.item() * B
        total_mae  += mae * B
        total_n    += B

    return total_loss / total_n, total_mae / total_n


def run_epoch_gaussian(
    model: GaussianBallAtTouchModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    amp_scaler: torch.cuda.amp.GradScaler,
    train: bool,
) -> Tuple[float, float, float]:
    """Returns (nll_loss, mae_in_norm_units, mean_sigma)."""
    model.train(train)
    total_nll, total_mae, total_sigma, total_n = 0.0, 0.0, 0.0, 0

    for bf, sf, of_, lv, vel_obs_mask in loader:
        bf           = bf.to(device,           non_blocking=True)
        sf           = sf.to(device,           non_blocking=True)
        of_          = of_.to(device,          non_blocking=True)
        lv           = lv.to(device,           non_blocking=True)
        vel_obs_mask = vel_obs_mask.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=amp_scaler.is_enabled()):
            out = model(bf, sf, of_)
            mu, log_sigma = out["mu"], out["log_sigma"]  # [B,3] each

            # Gaussian NLL: 0.5 * (2*log_sigma + (mu - target)^2 / sigma^2)
            # + const (0.5*log(2pi) dropped since it doesn't affect optimization)
            var = torch.exp(2.0 * log_sigma)  # sigma^2
            nll = 0.5 * (2.0 * log_sigma + (mu - lv) ** 2 / var)  # [B, 3]

            # Mask: only compute on observed components
            obs = vel_obs_mask.float()  # [B, 3]
            loss = (nll * obs).sum(-1) / obs.sum(-1).clamp(min=1)
            loss = loss.mean()

        if train:
            optimizer.zero_grad(set_to_none=True)
            amp_scaler.scale(loss).backward()
            amp_scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            amp_scaler.step(optimizer)
            amp_scaler.update()

        B = lv.size(0)
        with torch.no_grad():
            mae = (mu - lv).abs().mean().item()
            sigma_mean = torch.exp(log_sigma).mean().item()
        total_nll   += loss.item() * B
        total_mae   += mae * B
        total_sigma += sigma_mean * B
        total_n     += B

    return total_nll / total_n, total_mae / total_n, total_sigma / total_n


def save_checkpoint(path: str, epoch: int, model: nn.Module,
                    optimizer, best_val: float, args: argparse.Namespace) -> None:
    torch.save({
        "epoch": epoch, "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_val": best_val, "args": vars(args),
    }, path)


def load_checkpoint(path: str, model: nn.Module, optimizer=None):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    if optimizer and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt.get("epoch", 0), ckpt.get("best_val", float("inf"))


def _train_deterministic(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  Model type: deterministic")

    train_ds, val_ds = create_datasets(args.preprocessed_dir, augment=args.augment)
    print(f"  train={len(train_ds)}  val={len(val_ds)}  augment={args.augment}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers,
                              pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size * 2,
                              shuffle=False, num_workers=args.num_workers,
                              pin_memory=True)

    H = 8
    model = BallAtTouchModel(
        ball_dim=8, self_dim=H * 4, other_dim=H * 4,
        k_other=args.k_other, d_model=args.d_model,
        n_heads=args.n_heads, n_layers=args.n_layers, dropout=args.dropout,
    ).to(device)
    print(f"Model params: {model.n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.05)
    amp_scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    save_dir = Path(args.save_dir) / args.run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    start_epoch, best_val = 0, float("inf")
    if args.resume:
        start_epoch, best_val = load_checkpoint(args.resume, model, optimizer)
        print(f"Resumed from {args.resume}, epoch {start_epoch}, best_val={best_val:.4f}")

    history = []
    for epoch in range(start_epoch + 1, args.epochs + 1):
        tr_loss, tr_mae = run_epoch(model, train_loader, optimizer,
                                    args.huber_delta, device, amp_scaler, train=True)
        vl_loss, vl_mae = run_epoch(model, val_loader, None,
                                    args.huber_delta, device, amp_scaler, train=False)
        scheduler.step()

        is_best = vl_loss < best_val
        if is_best:
            best_val = vl_loss
            save_checkpoint(str(save_dir / "best.pt"), epoch, model, optimizer,
                            best_val, args)
        if epoch % 5 == 0 or epoch == 1:
            save_checkpoint(str(save_dir / f"epoch_{epoch:04d}.pt"), epoch,
                            model, optimizer, best_val, args)

        lr_now = scheduler.get_last_lr()[0]
        flag = " *" if is_best else ""
        print(f"[{epoch:4d}/{args.epochs}] "
              f"train loss={tr_loss:.4f} mae={tr_mae * 10:.2f}m/s  "
              f"val loss={vl_loss:.4f} mae={vl_mae * 10:.2f}m/s  "
              f"lr={lr_now:.2e}{flag}")
        history.append({"epoch": epoch, "train_loss": tr_loss, "train_mae": tr_mae,
                        "val_loss": vl_loss, "val_mae": vl_mae})

    with open(save_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining complete. Best val loss: {best_val:.4f}")
    print(f"Checkpoints in {save_dir}")


def _train_gaussian(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  Model type: gaussian")

    train_ds, val_ds = create_datasets(args.preprocessed_dir, augment=args.augment)
    print(f"  train={len(train_ds)}  val={len(val_ds)}  augment={args.augment}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers,
                              pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size * 2,
                              shuffle=False, num_workers=args.num_workers,
                              pin_memory=True)

    H = 8
    model = GaussianBallAtTouchModel(
        ball_dim=8, self_dim=H * 4, other_dim=H * 4,
        k_other=args.k_other, d_model=args.d_model,
        n_heads=args.n_heads, n_layers=args.n_layers, dropout=args.dropout,
    ).to(device)
    print(f"Model params: {model.n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.05)
    amp_scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    save_dir = Path(args.save_dir) / args.run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    start_epoch, best_val = 0, float("inf")
    if args.resume:
        start_epoch, best_val = load_checkpoint(args.resume, model, optimizer)
        print(f"Resumed from {args.resume}, epoch {start_epoch}, best_val={best_val:.4f}")

    history = []
    for epoch in range(start_epoch + 1, args.epochs + 1):
        tr_nll, tr_mae, tr_sig = run_epoch_gaussian(
            model, train_loader, optimizer, device, amp_scaler, train=True)
        vl_nll, vl_mae, vl_sig = run_epoch_gaussian(
            model, val_loader, None, device, amp_scaler, train=False)
        scheduler.step()

        is_best = vl_nll < best_val
        if is_best:
            best_val = vl_nll
            save_checkpoint(str(save_dir / "best.pt"), epoch, model, optimizer,
                            best_val, args)
        if epoch % 5 == 0 or epoch == 1:
            save_checkpoint(str(save_dir / f"epoch_{epoch:04d}.pt"), epoch,
                            model, optimizer, best_val, args)

        lr_now = scheduler.get_last_lr()[0]
        flag = " *" if is_best else ""
        print(f"[{epoch:4d}/{args.epochs}] "
              f"train nll={tr_nll:.4f} mae={tr_mae * 10:.2f}m/s σ={tr_sig:.3f}  "
              f"val nll={vl_nll:.4f} mae={vl_mae * 10:.2f}m/s σ={vl_sig:.3f}  "
              f"lr={lr_now:.2e}{flag}")
        history.append({"epoch": epoch,
                        "train_nll": tr_nll, "train_mae": tr_mae, "train_sigma": tr_sig,
                        "val_nll": vl_nll, "val_mae": vl_mae, "val_sigma": vl_sig})

    with open(save_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining complete. Best val NLL: {best_val:.4f}")
    print(f"Checkpoints in {save_dir}")


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("Loading datasets …")
    if args.model_type == "gaussian":
        _train_gaussian(args)
    else:
        _train_deterministic(args)


if __name__ == "__main__":
    main()
