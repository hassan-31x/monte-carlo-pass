#!/usr/bin/env python3
"""Train Possession Value (PV) transformer model on Sportec tracking data.

PV ≈ P(goal within 10s) = P(shot) × E[xG|shot]
Trained with BCE using soft labels (xG values), which naturally teaches
the model to predict the product P(shot) × E[xG|shot].
"""

import argparse
import json
import os
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from pv_model import PossessionValueTransformer


class PVDataset(Dataset):
    def __init__(self, data_dir: str, augment: bool = False):
        self.augment = augment
        manifest_path = os.path.join(data_dir, "manifest.json")
        with open(manifest_path) as f:
            manifest = json.load(f)

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

        self.features = torch.from_numpy(feat_buf)
        self.masks = torch.from_numpy(mask_buf)
        self.labels = torch.from_numpy(label_buf)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feat = self.features[idx]
        mask = self.masks[idx]
        labels = self.labels[idx]

        if self.augment:
            if torch.rand(1).item() < 0.5:
                feat = feat.clone()
                feat[:, :, 0] = -feat[:, :, 0]
                feat[:, :, 2] = -feat[:, :, 2]
            if torch.rand(1).item() < 0.5:
                feat = feat.clone()
                feat[:, :, 1] = -feat[:, :, 1]
                feat[:, :, 3] = -feat[:, :, 3]

        return feat, mask, self.entity_type, labels


def compute_loss(outputs, labels, device):
    """
    BCE loss with soft targets (xG values).
    Labels: [home_xg_target, away_xg_target, has_shot_home, has_shot_away]
    """
    home_target = labels[:, 0]  # xG if shot, 0 otherwise
    away_target = labels[:, 1]

    loss_home = nn.functional.binary_cross_entropy_with_logits(
        outputs["logit_home"], home_target
    )
    loss_away = nn.functional.binary_cross_entropy_with_logits(
        outputs["logit_away"], away_target
    )

    return {
        "total": loss_home + loss_away,
        "home": loss_home.item(),
        "away": loss_away.item(),
    }


def train_epoch(model, loader, optimizer, scaler, device, use_amp=True):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for features, mask, entity_type, labels in loader:
        features = features.to(device)
        mask = mask.to(device)
        entity_type = entity_type[0].to(device)
        labels = labels.to(device)

        with torch.amp.autocast("cuda", enabled=use_amp):
            outputs = model(features, mask, entity_type)
            losses = compute_loss(outputs, labels, device)
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
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader, device, use_amp=True):
    model.eval()
    total_loss = 0.0
    n_batches = 0

    all_pv_home = []
    all_pv_away = []
    all_target_home = []
    all_target_away = []
    all_has_shot_home = []
    all_has_shot_away = []

    for features, mask, entity_type, labels in loader:
        features = features.to(device)
        mask = mask.to(device)
        entity_type = entity_type[0].to(device)
        labels = labels.to(device)

        with torch.amp.autocast("cuda", enabled=use_amp):
            outputs = model(features, mask, entity_type)
            losses = compute_loss(outputs, labels, device)

        total_loss += losses["total"].item()
        n_batches += 1

        all_pv_home.append(outputs["pv_home"].cpu())
        all_pv_away.append(outputs["pv_away"].cpu())
        all_target_home.append(labels[:, 0].cpu())
        all_target_away.append(labels[:, 1].cpu())
        all_has_shot_home.append(labels[:, 2].cpu())
        all_has_shot_away.append(labels[:, 3].cpu())

    pv_home = torch.cat(all_pv_home)
    pv_away = torch.cat(all_pv_away)
    target_home = torch.cat(all_target_home)
    target_away = torch.cat(all_target_away)
    has_shot_home = torch.cat(all_has_shot_home)
    has_shot_away = torch.cat(all_has_shot_away)

    # Mean PV on shot vs no-shot windows (separation quality)
    shot_mask_h = has_shot_home > 0.5
    shot_mask_a = has_shot_away > 0.5
    noshot_mask_h = ~shot_mask_h
    noshot_mask_a = ~shot_mask_a

    pv_h_shot = pv_home[shot_mask_h].mean().item() if shot_mask_h.any() else 0
    pv_h_noshot = pv_home[noshot_mask_h].mean().item() if noshot_mask_h.any() else 0
    pv_a_shot = pv_away[shot_mask_a].mean().item() if shot_mask_a.any() else 0
    pv_a_noshot = pv_away[noshot_mask_a].mean().item() if noshot_mask_a.any() else 0

    # Correlation between predicted PV and target xG
    pv_all = torch.cat([pv_home, pv_away])
    target_all = torch.cat([target_home, target_away])
    corr = 0.0
    if len(pv_all) > 1 and target_all.std() > 1e-8:
        corr = float(torch.corrcoef(torch.stack([pv_all, target_all]))[0, 1].item())

    return {
        "loss": total_loss / max(n_batches, 1),
        "corr": corr,
        "pv_h_shot": pv_h_shot,
        "pv_h_noshot": pv_h_noshot,
        "pv_a_shot": pv_a_shot,
        "pv_a_noshot": pv_a_noshot,
        "mean_pv_home": pv_home.mean().item(),
        "mean_pv_away": pv_away.mean().item(),
        "separation_h": pv_h_shot / max(pv_h_noshot, 1e-8),
        "separation_a": pv_a_shot / max(pv_a_noshot, 1e-8),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="/mnt/data/remains/opta2026/expectedThreat/pv_data")
    parser.add_argument("--checkpoint-dir", default="/mnt/data/remains/opta2026/expectedThreat/checkpoints/pv")
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--run-name", type=str, default="")
    args = parser.parse_args()
    if args.run_name:
        args.checkpoint_dir = args.checkpoint_dir.rstrip("/") + "_" + args.run_name

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cpu" if args.no_cuda or not torch.cuda.is_available() else "cuda")
    use_amp = device.type == "cuda"
    print(f"Device: {device}, AMP: {use_amp}")

    with open(os.path.join(args.data_dir, "meta.json")) as f:
        meta = json.load(f)
    window_size = meta["window_size"]
    feat_dim = meta["feat_dim"]
    n_entities = meta["n_entities"]
    print(f"Window: {window_size} frames, {n_entities} entities, {feat_dim}D features")

    train_dir = os.path.join(args.data_dir, "train")
    val_dir = os.path.join(args.data_dir, "val")

    train_ds = PVDataset(train_dir, augment=True)
    val_ds = PVDataset(val_dir, augment=False)
    print(f"Train: {len(train_ds)} windows, Val: {len(val_ds)} windows")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=2, pin_memory=(device.type == "cuda"), drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=2, pin_memory=(device.type == "cuda"),
    )

    model = PossessionValueTransformer(
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

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    patience_counter = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, scaler, device, use_amp)
        val = evaluate(model, val_loader, device, use_amp)
        scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:3d} | "
            f"train={train_loss:.5f} | "
            f"val={val['loss']:.5f} corr={val['corr']:.3f} "
            f"h_shot={val['pv_h_shot']:.4f} h_no={val['pv_h_noshot']:.4f} "
            f"a_shot={val['pv_a_shot']:.4f} a_no={val['pv_a_noshot']:.4f} "
            f"sep_h={val['separation_h']:.1f}x sep_a={val['separation_a']:.1f}x | "
            f"lr={lr:.1e}"
        )

        epoch_record = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val["loss"]),
            "val_corr": float(val["corr"]),
            "val_mean_pv_home": float(val["mean_pv_home"]),
            "val_mean_pv_away": float(val["mean_pv_away"]),
            "lr": float(lr),
        }
        history.append(epoch_record)
        with (ckpt_dir / "history.json").open("w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "val_metrics": val,
            "args": vars(args),
            "meta": meta,
        }, ckpt_dir / "latest.pt")

        if val["loss"] < best_val_loss:
            best_val_loss = val["loss"]
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_metrics": val,
                "args": vars(args),
                "meta": meta,
            }, ckpt_dir / "best.pt")
            print(f"  -> Best val loss! Saved.")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # Test eval
    test_dir = os.path.join(args.data_dir, "test")
    if os.path.exists(os.path.join(test_dir, "manifest.json")):
        test_ds = PVDataset(test_dir, augment=False)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
        ckpt = torch.load(ckpt_dir / "best.pt", map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        test = evaluate(model, test_loader, device, use_amp)
        print(f"\nTest: loss={test['loss']:.5f} corr={test['corr']:.3f} "
              f"sep_h={test['separation_h']:.1f}x sep_a={test['separation_a']:.1f}x")

    print("Done!")


if __name__ == "__main__":
    main()
