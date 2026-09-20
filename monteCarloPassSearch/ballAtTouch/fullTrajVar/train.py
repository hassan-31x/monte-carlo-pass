#!/usr/bin/env python3
"""Train BAT full-trajectory prediction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import create_datasets, load_meta
from model import FullTrajectoryBallAtTouchModel


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--preprocessed-dir",
        type=str,
        default="/mnt/data/remains/opta2026/ballAtTouch/fullTrajVar/preprocessed_sportec",
    )
    p.add_argument(
        "--save-dir",
        type=str,
        default="/mnt/data/remains/opta2026/ballAtTouch/fullTrajVar/checkpoints",
    )
    p.add_argument("--run-name", type=str, default="run")
    p.add_argument("--d-model", type=int, default=192)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--k-other", type=int, default=8)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--huber-delta", type=float, default=1.0)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--augment", action="store_true")
    p.add_argument("--pos-loss-weight", type=float, default=1.0)
    p.add_argument("--vel-loss-weight", type=float, default=0.35)
    p.add_argument("--stop-loss-weight", type=float, default=0.20)
    p.add_argument("--smooth-loss-weight", type=float, default=0.05)
    p.add_argument("--ground-loss-weight", type=float, default=0.02)
    return p.parse_args()


def _trajectory_velocity(traj: torch.Tensor) -> torch.Tensor:
    vel = torch.zeros_like(traj)
    vel[:, 0] = traj[:, 0]
    vel[:, 1:] = traj[:, 1:] - traj[:, :-1]
    return vel


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (x * mask).sum() / mask.sum().clamp(min=1.0)


def run_epoch(
    *,
    model: FullTrajectoryBallAtTouchModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    amp_scaler: torch.cuda.amp.GradScaler,
    huber_delta: float,
    train: bool,
    args: argparse.Namespace,
) -> Tuple[float, float, float]:
    model.train(train)
    total_loss = 0.0
    total_pos_mae = 0.0
    total_stop_acc = 0.0
    total_n = 0

    for bf, sf, of_, traj_tgt, traj_mask, stop_idx in loader:
        bf = bf.to(device, non_blocking=True)
        sf = sf.to(device, non_blocking=True)
        of_ = of_.to(device, non_blocking=True)
        traj_tgt = traj_tgt.to(device, non_blocking=True)
        traj_mask = traj_mask.to(device, non_blocking=True)
        stop_idx = stop_idx.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, enabled=amp_scaler.is_enabled()):
            out = model(bf, sf, of_)
            traj_pred = out["traj"]
            stop_logits = out["stop_logits"]

            mask3 = traj_mask.unsqueeze(-1).float()
            pos_err = F.huber_loss(traj_pred, traj_tgt, reduction="none", delta=huber_delta)
            pos_loss = _masked_mean(pos_err, mask3)

            vel_pred = _trajectory_velocity(traj_pred)
            vel_tgt = _trajectory_velocity(traj_tgt)
            vel_err = F.huber_loss(vel_pred, vel_tgt, reduction="none", delta=huber_delta)
            vel_loss = _masked_mean(vel_err, mask3)

            stop_loss = F.cross_entropy(stop_logits, stop_idx)

            accel = traj_pred[:, 2:] - 2.0 * traj_pred[:, 1:-1] + traj_pred[:, :-2]
            accel_mask = (traj_mask[:, 2:] & traj_mask[:, 1:-1] & traj_mask[:, :-2]).unsqueeze(-1).float()
            smooth_loss = _masked_mean(accel.square(), accel_mask) if accel.shape[1] > 0 else traj_pred.new_zeros(())

            ground_penalty = _masked_mean(F.relu(-traj_pred[..., 2]), traj_mask.float())

            loss = (
                args.pos_loss_weight * pos_loss
                + args.vel_loss_weight * vel_loss
                + args.stop_loss_weight * stop_loss
                + args.smooth_loss_weight * smooth_loss
                + args.ground_loss_weight * ground_penalty
            )

        if train:
            optimizer.zero_grad(set_to_none=True)
            amp_scaler.scale(loss).backward()
            amp_scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            amp_scaler.step(optimizer)
            amp_scaler.update()

        batch_n = int(bf.shape[0])
        with torch.no_grad():
            pos_mae = _masked_mean((traj_pred - traj_tgt).abs(), mask3).item()
            stop_acc = (stop_logits.argmax(dim=-1) == stop_idx).float().mean().item()
        total_loss += float(loss.item()) * batch_n
        total_pos_mae += float(pos_mae) * batch_n
        total_stop_acc += float(stop_acc) * batch_n
        total_n += batch_n

    return total_loss / max(1, total_n), total_pos_mae / max(1, total_n), total_stop_acc / max(1, total_n)


def save_checkpoint(path: Path, epoch: int, model: nn.Module, optimizer, best_val: float, args: argparse.Namespace) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_val": best_val,
            "args": vars(args),
        },
        str(path),
    )


def load_checkpoint(path: str, model: nn.Module, optimizer=None):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt.get("epoch", 0), ckpt.get("best_val", float("inf"))


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    meta = load_meta(args.preprocessed_dir)
    max_traj_len = int(meta.get("max_traj_len", 160))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds, val_ds = create_datasets(args.preprocessed_dir, augment=args.augment)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = FullTrajectoryBallAtTouchModel(
        k_other=args.k_other,
        max_traj_len=max_traj_len,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.05)
    amp_scaler = torch.amp.GradScaler(device.type, enabled=(device.type == "cuda"))

    save_dir = Path(args.save_dir) / args.run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    start_epoch, best_val = 0, float("inf")
    if args.resume:
        start_epoch, best_val = load_checkpoint(args.resume, model, optimizer)

    history = []
    for epoch in range(start_epoch + 1, args.epochs + 1):
        tr_loss, tr_mae, tr_stop = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            amp_scaler=amp_scaler,
            huber_delta=args.huber_delta,
            train=True,
            args=args,
        )
        vl_loss, vl_mae, vl_stop = run_epoch(
            model=model,
            loader=val_loader,
            optimizer=None,
            device=device,
            amp_scaler=amp_scaler,
            huber_delta=args.huber_delta,
            train=False,
            args=args,
        )
        scheduler.step()

        is_best = vl_loss < best_val
        if is_best:
            best_val = vl_loss
            save_checkpoint(save_dir / "best.pt", epoch, model, optimizer, best_val, args)
        if epoch % 5 == 0 or epoch == 1:
            save_checkpoint(save_dir / f"epoch_{epoch:04d}.pt", epoch, model, optimizer, best_val, args)

        lr_now = scheduler.get_last_lr()[0]
        print(
            f"[{epoch:4d}/{args.epochs}] "
            f"train loss={tr_loss:.4f} pos_mae={tr_mae:.4f} stop_acc={tr_stop:.3f}  "
            f"val loss={vl_loss:.4f} pos_mae={vl_mae:.4f} stop_acc={vl_stop:.3f}  "
            f"lr={lr_now:.2e}{' *' if is_best else ''}"
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": tr_loss,
                "train_pos_mae": tr_mae,
                "train_stop_acc": tr_stop,
                "val_loss": vl_loss,
                "val_pos_mae": vl_mae,
                "val_stop_acc": vl_stop,
            }
        )

    with open(save_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()
