#!/usr/bin/env python3
"""Train xG transformer model."""

import argparse
import json
import os
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from xg_model import xGTransformer


class xGDataset(Dataset):
    def __init__(self, npz_path: str, augment: bool = False):
        data = np.load(npz_path)
        self.X = torch.from_numpy(data["X"].astype(np.float32))
        self.y = torch.from_numpy(data["y"].astype(np.float32))
        self.xg_soft = torch.from_numpy(data["xg_soft"].astype(np.float32))
        self.mask = torch.from_numpy(data["feat_mask"].astype(np.float32))
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        mask = self.mask[idx]
        if self.augment:
            noise = torch.randn_like(x) * 0.05
            x = x + noise * mask
        return x, mask, self.y[idx], self.xg_soft[idx]


def train_epoch(model, loader, optimizer, device, soft_weight):
    model.train()
    total_loss = 0.0
    total_n = 0

    bce = nn.BCEWithLogitsLoss()

    for X, mask, y, xg_soft in loader:
        X, mask, y = X.to(device), mask.to(device), y.to(device)
        xg_soft = xg_soft.to(device)

        logits = model(X, mask).squeeze(-1)  # [B]

        # Hard label loss (unweighted BCE for calibration)
        loss_hard = bce(logits, y)

        # Soft label loss (distillation from statsbomb_xg)
        has_xg = xg_soft >= 0
        loss_soft = torch.tensor(0.0, device=device)
        if has_xg.any():
            loss_soft = nn.functional.mse_loss(
                torch.sigmoid(logits[has_xg]), xg_soft[has_xg]
            )

        loss = loss_hard + soft_weight * loss_soft

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * len(y)
        total_n += len(y)

    return total_loss / max(total_n, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_probs = []
    all_labels = []
    all_xg_soft = []

    for X, mask, y, xg_soft in loader:
        X, mask = X.to(device), mask.to(device)
        logits = model(X, mask).squeeze(-1)
        probs = torch.sigmoid(logits)
        all_probs.append(probs.cpu())
        all_labels.append(y)
        all_xg_soft.append(xg_soft)

    all_probs = torch.cat(all_probs)
    all_labels = torch.cat(all_labels)
    all_xg_soft = torch.cat(all_xg_soft)

    # Brier score (primary metric — measures calibration)
    brier = ((all_probs - all_labels) ** 2).mean().item()

    # Log loss
    eps = 1e-7
    log_loss = -torch.mean(
        all_labels * torch.log(all_probs + eps) + (1 - all_labels) * torch.log(1 - all_probs + eps)
    ).item()

    # MSE vs statsbomb_xg (distillation quality)
    has_xg = all_xg_soft >= 0
    xg_mse = 0.0
    if has_xg.any():
        xg_mse = ((all_probs[has_xg] - all_xg_soft[has_xg]) ** 2).mean().item()

    return {
        "brier": brier,
        "log_loss": log_loss,
        "xg_mse": xg_mse,
        "mean_pred": all_probs.mean().item(),
        "mean_label": all_labels.mean().item(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="/mnt/data/remains/opta2026/expectedThreat/xg_data")
    parser.add_argument("--checkpoint-dir", default="/mnt/data/remains/opta2026/expectedThreat/checkpoints/xg")
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--soft-weight", type=float, default=2.0,
                        help="Weight for MSE loss vs statsbomb_xg soft labels")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    with open(os.path.join(args.data_dir, "meta.json")) as f:
        meta = json.load(f)
    n_features = meta["n_features"]
    print(f"Features: {n_features}, names: {meta['feature_names']}")

    train_ds = xGDataset(os.path.join(args.data_dir, "train.npz"), augment=True)
    val_ds = xGDataset(os.path.join(args.data_dir, "val.npz"), augment=False)

    n_pos = float(train_ds.y.sum())
    n_xg = float((train_ds.xg_soft >= 0).sum())
    print(f"Train: {len(train_ds)} shots, {int(n_pos)} goals ({100*n_pos/len(train_ds):.1f}%), "
          f"{int(n_xg)} with soft xG")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = xGTransformer(
        n_features=n_features,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} params, soft_weight={args.soft_weight}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val_brier = float("inf")
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device, args.soft_weight)
        val = evaluate(model, val_loader, device)
        scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:3d} | "
            f"train={train_loss:.4f} | "
            f"brier={val['brier']:.4f} log_loss={val['log_loss']:.4f} "
            f"xg_mse={val['xg_mse']:.4f} "
            f"mean_pred={val['mean_pred']:.4f} mean_label={val['mean_label']:.4f} | "
            f"lr={lr:.1e}"
        )

        if val["brier"] < best_val_brier:
            best_val_brier = val["brier"]
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_metrics": val,
                "args": vars(args),
                "meta": meta,
            }, ckpt_dir / "best.pt")
            print(f"  -> New best brier! Saved.")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # Test eval
    test_path = os.path.join(args.data_dir, "test.npz")
    if os.path.exists(test_path):
        test_ds = xGDataset(test_path, augment=False)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
        ckpt = torch.load(ckpt_dir / "best.pt", map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        test = evaluate(model, test_loader, device)
        print(f"\nTest: brier={test['brier']:.4f} log_loss={test['log_loss']:.4f} "
              f"xg_mse={test['xg_mse']:.4f} mean_pred={test['mean_pred']:.4f}")

    print("Done!")


if __name__ == "__main__":
    main()
