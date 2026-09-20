#!/usr/bin/env python3
"""Train Expected Threat (xT) transformer model on Sportec tracking data."""

import argparse
import json
import os
import math
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from xt_model import xTTransformer


class xTDataset(Dataset):
    """Load xT windows from preprocessed chunks with optional oversampling of positive windows."""

    def __init__(self, data_dir: str, augment: bool = False, oversample_positive: int = 0):
        self.augment = augment
        manifest_path = os.path.join(data_dir, "manifest.json")
        with open(manifest_path) as f:
            manifest = json.load(f)

        # Pre-calculate total size, then allocate and fill
        total = sum(e["n_windows"] for e in manifest)
        first_chunk = np.load(manifest[0]["path"])
        W, N, F = first_chunk["features"].shape[1:]
        self.entity_type = torch.from_numpy(first_chunk["entity_type"].astype(np.int64))
        L = first_chunk["labels"].shape[1]

        feat_buf = np.empty((total, W, N, F), dtype=np.float32)
        mask_buf = np.empty((total, W, N), dtype=np.float32)
        label_buf = np.empty((total, L), dtype=np.float32)

        offset = 0
        for entry in manifest:
            chunk = np.load(entry["path"])
            n = chunk["features"].shape[0]
            feat_buf[offset:offset + n] = chunk["features"]
            mask_buf[offset:offset + n] = chunk["mask"]
            label_buf[offset:offset + n] = chunk["labels"]
            offset += n

        # Oversample positive windows (those with shots)
        if oversample_positive > 0:
            has_shot = (label_buf[:, 3] > 0.5) | (label_buf[:, 4] > 0.5)
            pos_idx = np.where(has_shot)[0]
            if len(pos_idx) > 0:
                repeat_idx = np.tile(pos_idx, oversample_positive)
                feat_buf = np.concatenate([feat_buf, feat_buf[repeat_idx]], axis=0)
                mask_buf = np.concatenate([mask_buf, mask_buf[repeat_idx]], axis=0)
                label_buf = np.concatenate([label_buf, label_buf[repeat_idx]], axis=0)

        self.features = torch.from_numpy(feat_buf)
        self.masks = torch.from_numpy(mask_buf)
        self.labels = torch.from_numpy(label_buf)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feat = self.features[idx]  # [T, N, 6]
        mask = self.masks[idx]  # [T, N]
        labels = self.labels[idx]  # [6]

        if self.augment:
            # Random horizontal flip (negate x coordinates and vx)
            if torch.rand(1).item() < 0.5:
                feat = feat.clone()
                feat[:, :, 0] = -feat[:, :, 0]  # x
                feat[:, :, 2] = -feat[:, :, 2]  # vx
            # Random vertical flip (negate y coordinates and vy)
            if torch.rand(1).item() < 0.5:
                feat = feat.clone()
                feat[:, :, 1] = -feat[:, :, 1]  # y
                feat[:, :, 3] = -feat[:, :, 3]  # vy

        return feat, mask, self.entity_type, labels


def compute_loss(outputs, labels, device, shot_pos_weight=10.0):
    """
    Multi-task loss for xT model.

    Labels: [xt_value, home_threat, away_threat, has_shot_home, has_shot_away, next_xg]
    """
    has_shot_home = labels[:, 3]  # binary
    has_shot_away = labels[:, 4]  # binary
    home_threat_label = labels[:, 1]  # xG * discount
    away_threat_label = labels[:, 2]

    # Shot probability loss (BCE with positive class weighting)
    pw = torch.tensor([shot_pos_weight], device=device)
    bce = nn.BCEWithLogitsLoss(pos_weight=pw)
    loss_shot_home = bce(outputs["shot_home_logit"], has_shot_home)
    loss_shot_away = bce(outputs["shot_away_logit"], has_shot_away)

    # Threat regression loss (MSE on P(shot)*xG)
    loss_home_threat = nn.functional.mse_loss(outputs["home_threat"], home_threat_label)
    loss_away_threat = nn.functional.mse_loss(outputs["away_threat"], away_threat_label)

    # xG prediction loss (only on windows with shots)
    loss_xg = torch.tensor(0.0, device=device)
    n_xg = 0
    if has_shot_home.sum() > 0:
        # For home shots, xG label ≈ home_threat_label / P(shot)
        # But simpler: use the raw xG from next_xg when shot is by home
        home_mask = has_shot_home > 0.5
        if home_mask.any():
            xg_target = labels[home_mask, 5]  # next_xg
            loss_xg = loss_xg + nn.functional.mse_loss(outputs["xg_home"][home_mask], xg_target)
            n_xg += 1
    if has_shot_away.sum() > 0:
        away_mask = has_shot_away > 0.5
        if away_mask.any():
            xg_target = labels[away_mask, 5]
            loss_xg = loss_xg + nn.functional.mse_loss(outputs["xg_away"][away_mask], xg_target)
            n_xg += 1
    if n_xg > 0:
        loss_xg = loss_xg / n_xg

    # Total loss with weights
    total = 1.0 * (loss_shot_home + loss_shot_away) + 2.0 * (loss_home_threat + loss_away_threat) + 1.0 * loss_xg

    return {
        "total": total,
        "shot_home": loss_shot_home.item(),
        "shot_away": loss_shot_away.item(),
        "threat_home": loss_home_threat.item(),
        "threat_away": loss_away_threat.item(),
        "xg": loss_xg.item(),
    }


def train_epoch(model, loader, optimizer, scaler, device, use_amp=True, shot_pos_weight=10.0):
    model.train()
    total_loss = 0.0
    loss_components = {"shot_home": 0, "shot_away": 0, "threat_home": 0, "threat_away": 0, "xg": 0}
    n_batches = 0

    for features, mask, entity_type, labels in loader:
        features = features.to(device)
        mask = mask.to(device)
        entity_type = entity_type[0].to(device)  # same for all in batch
        labels = labels.to(device)

        with torch.amp.autocast("cuda", enabled=use_amp):
            outputs = model(features, mask, entity_type)
            losses = compute_loss(outputs, labels, device, shot_pos_weight=shot_pos_weight)
            loss = losses["total"]

        optimizer.zero_grad()
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item()
        for k in loss_components:
            loss_components[k] += losses[k]
        n_batches += 1

    avg = {k: v / max(n_batches, 1) for k, v in loss_components.items()}
    return total_loss / max(n_batches, 1), avg


@torch.no_grad()
def evaluate(model, loader, device, use_amp=True, shot_pos_weight=10.0):
    model.eval()
    total_loss = 0.0
    loss_components = {"shot_home": 0, "shot_away": 0, "threat_home": 0, "threat_away": 0, "xg": 0}
    n_batches = 0

    all_xt_pred = []
    all_xt_label = []
    all_shot_home_pred = []
    all_shot_home_label = []
    all_shot_away_pred = []
    all_shot_away_label = []

    for features, mask, entity_type, labels in loader:
        features = features.to(device)
        mask = mask.to(device)
        entity_type = entity_type[0].to(device)
        labels = labels.to(device)

        with torch.amp.autocast("cuda", enabled=use_amp):
            outputs = model(features, mask, entity_type)
            losses = compute_loss(outputs, labels, device, shot_pos_weight=shot_pos_weight)

        total_loss += losses["total"].item()
        for k in loss_components:
            loss_components[k] += losses[k]
        n_batches += 1

        all_xt_pred.append(outputs["xt_value"].cpu())
        all_xt_label.append(labels[:, 0].cpu())
        all_shot_home_pred.append(outputs["p_shot_home"].cpu())
        all_shot_home_label.append(labels[:, 3].cpu())
        all_shot_away_pred.append(outputs["p_shot_away"].cpu())
        all_shot_away_label.append(labels[:, 4].cpu())

    avg_components = {k: v / max(n_batches, 1) for k, v in loss_components.items()}

    xt_pred = torch.cat(all_xt_pred)
    xt_label = torch.cat(all_xt_label)
    shot_home_pred = torch.cat(all_shot_home_pred)
    shot_home_label = torch.cat(all_shot_home_label)
    shot_away_pred = torch.cat(all_shot_away_pred)
    shot_away_label = torch.cat(all_shot_away_label)

    # Metrics
    xt_mse = ((xt_pred - xt_label) ** 2).mean().item()
    xt_corr = float(torch.corrcoef(torch.stack([xt_pred, xt_label]))[0, 1].item()) if len(xt_pred) > 1 else 0

    # Shot prediction accuracy
    shot_home_acc = ((shot_home_pred > 0.5).float() == shot_home_label).float().mean().item()
    shot_away_acc = ((shot_away_pred > 0.5).float() == shot_away_label).float().mean().item()

    return {
        "loss": total_loss / max(n_batches, 1),
        "components": avg_components,
        "xt_mse": xt_mse,
        "xt_corr": xt_corr,
        "shot_home_acc": shot_home_acc,
        "shot_away_acc": shot_away_acc,
        "mean_xt_pred": xt_pred.mean().item(),
        "mean_xt_label": xt_label.mean().item(),
        "shot_home_rate": shot_home_label.mean().item(),
        "shot_away_rate": shot_away_label.mean().item(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="/mnt/data/remains/opta2026/expectedThreat/xt_data")
    parser.add_argument("--checkpoint-dir", default="/mnt/data/remains/opta2026/expectedThreat/checkpoints/xt")
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--oversample", type=int, default=0, help="Repeat positive windows N times")
    parser.add_argument("--shot-pos-weight", type=float, default=10.0, help="BCE pos_weight for shot prediction")
    parser.add_argument("--run-name", type=str, default="", help="Appended to checkpoint dir")
    args = parser.parse_args()
    if args.run_name:
        args.checkpoint_dir = args.checkpoint_dir.rstrip("/") + "_" + args.run_name

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cpu" if args.no_cuda or not torch.cuda.is_available() else "cuda")
    use_amp = device.type == "cuda"
    print(f"Device: {device}, AMP: {use_amp}")

    # Load metadata
    with open(os.path.join(args.data_dir, "meta.json")) as f:
        meta = json.load(f)
    window_size = meta["window_size"]
    feat_dim = meta["feat_dim"]
    n_entities = meta["n_entities"]
    print(f"Window: {window_size} frames, {n_entities} entities, {feat_dim}D features")

    # Datasets
    train_dir = os.path.join(args.data_dir, "train")
    val_dir = os.path.join(args.data_dir, "val")

    train_ds = xTDataset(train_dir, augment=True, oversample_positive=args.oversample)
    val_ds = xTDataset(val_dir, augment=False)
    print(f"Train: {len(train_ds)} windows, Val: {len(val_ds)} windows")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=2, pin_memory=(device.type == "cuda"), drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=2, pin_memory=(device.type == "cuda"),
    )

    # Model
    model = xTTransformer(
        feat_dim=feat_dim,
        n_entities=n_entities,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
        max_seq_len=window_size,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # Training
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    patience_counter = 0
    spw = args.shot_pos_weight

    for epoch in range(1, args.epochs + 1):
        train_loss, train_comp = train_epoch(model, train_loader, optimizer, scaler, device, use_amp, spw)
        val_metrics = evaluate(model, val_loader, device, use_amp, spw)
        scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:3d} | "
            f"train={train_loss:.5f} | "
            f"val={val_metrics['loss']:.5f} "
            f"xt_mse={val_metrics['xt_mse']:.6f} "
            f"xt_corr={val_metrics['xt_corr']:.3f} "
            f"shot_h_acc={val_metrics['shot_home_acc']:.3f} "
            f"shot_a_acc={val_metrics['shot_away_acc']:.3f} | "
            f"lr={lr:.1e}"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_metrics,
                "args": vars(args),
                "meta": meta,
            }, ckpt_dir / "best.pt")
            print(f"  -> Best val loss! Saved.")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # Test evaluation
    test_dir = os.path.join(args.data_dir, "test")
    if os.path.exists(os.path.join(test_dir, "manifest.json")):
        test_ds = xTDataset(test_dir, augment=False)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
        ckpt = torch.load(ckpt_dir / "best.pt", map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        test_metrics = evaluate(model, test_loader, device, use_amp, spw)
        print(f"\nTest: loss={test_metrics['loss']:.5f} xt_mse={test_metrics['xt_mse']:.6f} "
              f"xt_corr={test_metrics['xt_corr']:.3f}")

    print("Done!")


if __name__ == "__main__":
    main()
