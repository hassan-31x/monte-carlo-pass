#!/usr/bin/env python3
"""Train playerToTouch: who touches ball next (cross-entropy classifier).

Usage:
    # after preprocessing:
    python3 train.py --preprocessed-dir preprocessed --epochs 50

The label space is [0..K]: index of touching player among K nearest, or K = no touch.
Class imbalance is handled by preprocess downsampling + optional loss weighting here.
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

from dataset import TouchDataset, create_datasets, SurvivalTouchDataset, create_survival_datasets
from model import TouchPredictorModel, SurvivalTouchModel


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    # Data
    p.add_argument("--preprocessed-dir", type=str,
                   default="/mnt/data/remains/opta2026/playerToTouch/preprocessed")
    p.add_argument("--save-dir", type=str,
                   default="/mnt/data/remains/opta2026/playerToTouch/checkpoints")
    p.add_argument("--run-name", type=str, default="run")
    # Model
    p.add_argument("--d-model",  type=int, default=128)
    p.add_argument("--n-heads",  type=int, default=4)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--dropout",  type=float, default=0.1)
    p.add_argument("--k",        type=int, default=8)
    # Training
    p.add_argument("--epochs",     type=int,   default=60)
    p.add_argument("--batch-size", type=int,   default=512)
    p.add_argument("--lr",         type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--label-smoothing", type=float, default=0.0,
                   help="Label smoothing factor for CE loss (e.g. 0.05)")
    p.add_argument("--warmup-epochs", type=int, default=0,
                   help="Linear warmup epochs before cosine annealing")
    p.add_argument("--augment", action="store_true",
                   help="Enable flip augmentation for training data")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    # Survival model
    p.add_argument("--model-type", choices=["classifier", "survival"], default="classifier",
                   help="'classifier' = old CE model, 'survival' = hazard+player model")
    p.add_argument("--player-loss-weight", type=float, default=5.0,
                   help="Weight for player CE loss relative to hazard BCE (survival mode)")
    return p.parse_args()


# ── training helpers ──────────────────────────────────────────────────────────

def run_epoch(
    model: TouchPredictorModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    criterion: nn.CrossEntropyLoss,
    device: torch.device,
    amp_scaler: torch.cuda.amp.GradScaler,
    train: bool,
) -> Tuple[float, float]:
    """Run one epoch. Returns (loss, top1_accuracy)."""
    model.train(train)
    total_loss, total_correct, total_n = 0.0, 0, 0

    for bf, pf, lb in loader:
        bf = bf.to(device, non_blocking=True)
        pf = pf.to(device, non_blocking=True)
        lb = lb.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=amp_scaler.is_enabled()):
            logits = model(bf, pf)           # [B, K+1]
            loss   = criterion(logits, lb)

        if train:
            optimizer.zero_grad(set_to_none=True)
            amp_scaler.scale(loss).backward()
            amp_scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            amp_scaler.step(optimizer)
            amp_scaler.update()

        B = lb.size(0)
        total_loss    += loss.item() * B
        total_correct += (logits.argmax(1) == lb).sum().item()
        total_n       += B

    return total_loss / total_n, total_correct / total_n


def run_epoch_survival(
    model: SurvivalTouchModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    amp_scaler: torch.cuda.amp.GradScaler,
    train: bool,
    player_loss_weight: float = 5.0,
) -> dict:
    """Run one survival epoch. Returns dict of metrics."""
    model.train(train)
    total_hazard_loss = 0.0
    total_player_loss = 0.0
    total_n = 0
    total_events = 0
    total_player_correct = 0

    # For AUROC computation
    all_hazard_probs = []
    all_is_event = []
    # For mean hazard on event/non-event
    sum_h_event = 0.0
    sum_h_nonevent = 0.0
    n_event = 0
    n_nonevent = 0

    for bf, pf, ie, pl in loader:
        bf = bf.to(device, non_blocking=True)       # [B, 6]
        pf = pf.to(device, non_blocking=True)       # [B, K, 32]
        ie = ie.to(device, non_blocking=True)        # [B] float (0/1)
        pl = pl.to(device, non_blocking=True)        # [B] long (-1 or 0..K-1)

        B = bf.size(0)

        with torch.cuda.amp.autocast(enabled=amp_scaler.is_enabled()):
            out = model(bf, pf)
            hazard_logit = out["hazard"]              # [B, 1]
            player_logits = out["player_logits"]      # [B, K]

            # Hazard loss: BCE on all frames
            hazard_loss = F.binary_cross_entropy_with_logits(
                hazard_logit.squeeze(-1), ie, reduction="mean"
            )

            # Player loss: CE only on event frames
            event_mask = ie > 0.5                      # [B] bool
            n_ev = event_mask.sum().item()
            if n_ev > 0:
                player_loss = F.cross_entropy(
                    player_logits[event_mask], pl[event_mask], reduction="mean"
                )
            else:
                player_loss = torch.tensor(0.0, device=device)

            loss = hazard_loss + player_loss_weight * player_loss

        if train:
            optimizer.zero_grad(set_to_none=True)
            amp_scaler.scale(loss).backward()
            amp_scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            amp_scaler.step(optimizer)
            amp_scaler.update()

        total_hazard_loss += hazard_loss.item() * B
        total_player_loss += player_loss.item() * n_ev if n_ev > 0 else 0.0
        total_n += B
        total_events += n_ev

        # Player accuracy on events
        if n_ev > 0:
            preds = player_logits[event_mask].argmax(1)
            total_player_correct += (preds == pl[event_mask]).sum().item()

        # Hazard probs for AUROC
        with torch.no_grad():
            h_prob = torch.sigmoid(hazard_logit.squeeze(-1))  # [B]
            all_hazard_probs.append(h_prob.cpu())
            all_is_event.append(ie.cpu())

            ev = event_mask
            nev = ~event_mask
            if ev.any():
                sum_h_event += h_prob[ev].sum().item()
                n_event += ev.sum().item()
            if nev.any():
                sum_h_nonevent += h_prob[nev].sum().item()
                n_nonevent += nev.sum().item()

    # Compute AUROC
    auroc = -1.0
    try:
        from sklearn.metrics import roc_auc_score
        all_h = torch.cat(all_hazard_probs).numpy()
        all_e = torch.cat(all_is_event).numpy()
        if len(np.unique(all_e)) > 1:
            auroc = float(roc_auc_score(all_e, all_h))
    except ImportError:
        pass

    return {
        "hazard_loss": total_hazard_loss / max(total_n, 1),
        "player_loss": total_player_loss / max(total_events, 1),
        "player_acc":  total_player_correct / max(total_events, 1),
        "auroc":       auroc,
        "h_event_mean":    sum_h_event / max(n_event, 1),
        "h_nonevent_mean": sum_h_nonevent / max(n_nonevent, 1),
        "n_events":        total_events,
    }


def save_checkpoint(path: str, epoch: int, model: nn.Module,
                    optimizer: torch.optim.Optimizer, best_val: float,
                    args: argparse.Namespace) -> None:
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_val": best_val,
        "args": vars(args),
    }, path)


def load_checkpoint(path: str, model: nn.Module,
                    optimizer: torch.optim.Optimizer | None):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt.get("epoch", 0), ckpt.get("best_val", float("inf"))


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model type: {args.model_type}")

    # Save dir
    save_dir = Path(args.save_dir) / args.run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    amp_scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    if args.model_type == "survival":
        _train_survival(args, device, save_dir, amp_scaler)
    else:
        _train_classifier(args, device, save_dir, amp_scaler)


def _train_classifier(args, device, save_dir, amp_scaler):
    """Original classifier training loop."""
    print("Loading datasets …")
    train_ds, val_ds = create_datasets(args.preprocessed_dir, k=args.k,
                                        augment=args.augment)
    print(f"  train={len(train_ds)}  val={len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers,
                              pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size * 2,
                              shuffle=False, num_workers=args.num_workers,
                              pin_memory=True)

    model = TouchPredictorModel(
        ball_dim=8, player_hist_dim=32, k=args.k,
        d_model=args.d_model, n_heads=args.n_heads,
        n_layers=args.n_layers, dropout=args.dropout,
    ).to(device)
    print(f"Model params: {model.n_params:,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=args.lr, weight_decay=args.weight_decay)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs - args.warmup_epochs),
        eta_min=args.lr * 0.05
    )
    if args.warmup_epochs > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, total_iters=args.warmup_epochs
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine],
            milestones=[args.warmup_epochs]
        )
    else:
        scheduler = cosine

    start_epoch, best_val_loss = 0, float("inf")
    if args.resume:
        start_epoch, best_val_loss = load_checkpoint(args.resume, model, optimizer)
        print(f"Resumed from {args.resume}, epoch {start_epoch}, best_val={best_val_loss:.4f}")

    history = []
    for epoch in range(start_epoch + 1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, optimizer, criterion,
                                    device, amp_scaler, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, None, criterion,
                                      device, amp_scaler, train=False)
        scheduler.step()

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            save_checkpoint(str(save_dir / "best.pt"), epoch, model, optimizer,
                            best_val_loss, args)

        if epoch % 5 == 0 or epoch == 1:
            save_checkpoint(str(save_dir / f"epoch_{epoch:04d}.pt"), epoch,
                            model, optimizer, best_val_loss, args)

        lr_now = scheduler.get_last_lr()[0]
        flag = " *" if is_best else ""
        print(f"[{epoch:4d}/{args.epochs}] "
              f"train loss={tr_loss:.4f} acc={tr_acc:.3f}  "
              f"val loss={val_loss:.4f} acc={val_acc:.3f}  "
              f"lr={lr_now:.2e}{flag}")

        history.append({
            "epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
            "val_loss": val_loss, "val_acc": val_acc,
        })

    with open(save_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoints in {save_dir}")


def _train_survival(args, device, save_dir, amp_scaler):
    """Survival model training loop."""
    print("Loading survival datasets …")
    train_ds, val_ds = create_survival_datasets(args.preprocessed_dir, k=args.k,
                                                 augment=args.augment)
    print(f"  train={len(train_ds)} ({train_ds.n_events} events, rate={train_ds.event_rate:.4f})")
    print(f"  val={len(val_ds)} ({val_ds.n_events} events, rate={val_ds.event_rate:.4f})")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers,
                              pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size * 2,
                              shuffle=False, num_workers=args.num_workers,
                              pin_memory=True)

    model = SurvivalTouchModel(
        ball_dim=8, player_hist_dim=32, k=args.k,
        d_model=args.d_model, n_heads=args.n_heads,
        n_layers=args.n_layers, dropout=args.dropout,
    ).to(device)
    print(f"Model params: {model.n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=args.lr, weight_decay=args.weight_decay)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs - args.warmup_epochs),
        eta_min=args.lr * 0.05
    )
    if args.warmup_epochs > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, total_iters=args.warmup_epochs
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine],
            milestones=[args.warmup_epochs]
        )
    else:
        scheduler = cosine

    start_epoch, best_val_loss = 0, float("inf")
    if args.resume:
        start_epoch, best_val_loss = load_checkpoint(args.resume, model, optimizer)
        print(f"Resumed from {args.resume}, epoch {start_epoch}, best_val={best_val_loss:.4f}")

    history = []
    for epoch in range(start_epoch + 1, args.epochs + 1):
        tr = run_epoch_survival(model, train_loader, optimizer, device,
                                amp_scaler, train=True,
                                player_loss_weight=args.player_loss_weight)
        vl = run_epoch_survival(model, val_loader, None, device,
                                amp_scaler, train=False,
                                player_loss_weight=args.player_loss_weight)
        scheduler.step()

        # Total val loss for best-model selection
        val_total = vl["hazard_loss"] + args.player_loss_weight * vl["player_loss"]
        is_best = val_total < best_val_loss
        if is_best:
            best_val_loss = val_total
            save_checkpoint(str(save_dir / "best.pt"), epoch, model, optimizer,
                            best_val_loss, args)

        if epoch % 5 == 0 or epoch == 1:
            save_checkpoint(str(save_dir / f"epoch_{epoch:04d}.pt"), epoch,
                            model, optimizer, best_val_loss, args)

        lr_now = scheduler.get_last_lr()[0]
        flag = " *" if is_best else ""
        print(f"[{epoch:4d}/{args.epochs}] "
              f"tr h_loss={tr['hazard_loss']:.4f} p_loss={tr['player_loss']:.4f} "
              f"p_acc={tr['player_acc']:.3f} auroc={tr['auroc']:.3f}  "
              f"vl h_loss={vl['hazard_loss']:.4f} p_loss={vl['player_loss']:.4f} "
              f"p_acc={vl['player_acc']:.3f} auroc={vl['auroc']:.3f}  "
              f"h_ev={vl['h_event_mean']:.4f} h_ne={vl['h_nonevent_mean']:.4f} "
              f"lr={lr_now:.2e}{flag}")

        history.append({
            "epoch": epoch,
            "train_hazard_loss": tr["hazard_loss"],
            "train_player_loss": tr["player_loss"],
            "train_player_acc":  tr["player_acc"],
            "train_auroc":       tr["auroc"],
            "val_hazard_loss":   vl["hazard_loss"],
            "val_player_loss":   vl["player_loss"],
            "val_player_acc":    vl["player_acc"],
            "val_auroc":         vl["auroc"],
            "val_h_event_mean":  vl["h_event_mean"],
            "val_h_nonevent_mean": vl["h_nonevent_mean"],
        })

    with open(save_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining complete. Best val total loss: {best_val_loss:.4f}")
    print(f"Checkpoints in {save_dir}")


if __name__ == "__main__":
    main()
