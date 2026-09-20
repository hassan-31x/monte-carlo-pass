#!/usr/bin/env python3
"""Evaluate PV model vs EPV grid baseline and visualize examples.

EPV grid: 32 rows (y) × 50 cols (x), ball-location-only baseline.
PV model: tracking-based transformer, uses all 23 entities.

Comparison: for each test window, look up EPV grid value at ball position
and compare with PV model prediction.
"""

import argparse
import json
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle
from pathlib import Path
from pv_model import PossessionValueTransformer


def load_epv_grid(path: str):
    """Load EPV grid (32×50) and return as numpy array."""
    grid = np.loadtxt(path, delimiter=",")
    assert grid.shape == (32, 50), f"Expected 32×50, got {grid.shape}"
    return grid


def ball_pos_to_grid_idx(ball_x, ball_y, grid_rows=32, grid_cols=50,
                         pitch_x=105.0, pitch_y=68.0):
    """Convert ball position (centered coords) to EPV grid indices.

    Ball coords: x ∈ [-52.5, 52.5], y ∈ [-34, 34]
    Grid: row 0 = bottom (y=-34), row 31 = top (y=34)
           col 0 = left (x=-52.5), col 49 = right (x=52.5)
    """
    # Normalize to [0, 1]
    nx = (ball_x + pitch_x / 2) / pitch_x  # 0=left, 1=right
    ny = (ball_y + pitch_y / 2) / pitch_y  # 0=bottom, 1=top

    col = np.clip((nx * grid_cols).astype(int), 0, grid_cols - 1)
    row = np.clip((ny * grid_rows).astype(int), 0, grid_rows - 1)
    return row, col


def infer_attack_direction(features, masks):
    """Infer home attacking direction from player positions.

    Compares mean x of home (0-10) vs away (11-21) players.
    Teams are generally positioned closer to the goal they defend.
    If home_mean_x < away_mean_x, home defends left → attacks right (+x).

    Args:
        features: [N, T, 23, 6] or [T, 23, 6]
        masks: [N, T, 23] or [T, 23]

    Returns:
        home_attacks_right: [N] bool array (or scalar)
            True if home attacks toward +x, False if toward -x.
    """
    single = features.ndim == 3
    if single:
        features = features[np.newaxis]
        masks = masks[np.newaxis]

    # Average across all frames in the window for robustness
    # Home players: indices 0-10
    home_mask = masks[:, :, :11]  # [N, T, 11]
    home_x = features[:, :, :11, 0]  # [N, T, 11]
    home_sum = (home_x * home_mask).sum(axis=(1, 2))
    home_count = home_mask.sum(axis=(1, 2)).clip(1)
    home_mean_x = home_sum / home_count

    # Away players: indices 11-21
    away_mask = masks[:, :, 11:22]  # [N, T, 11]
    away_x = features[:, :, 11:22, 0]  # [N, T, 11]
    away_sum = (away_x * away_mask).sum(axis=(1, 2))
    away_count = away_mask.sum(axis=(1, 2)).clip(1)
    away_mean_x = away_sum / away_count

    # If home mean x < away mean x → home defends left → attacks right
    home_attacks_right = home_mean_x < away_mean_x

    if single:
        return home_attacks_right[0]
    return home_attacks_right


def lookup_epv_directed(epv_grid, ball_x, ball_y, attacks_right):
    """Look up EPV for a team given their attacking direction.

    EPV grid col 0 = own goal, col 49 = opponent goal (attacking left to right).
    We orient the ball x-coordinate so that the attacking direction maps to
    increasing column index.

    Args:
        epv_grid: [32, 50] array
        ball_x: [N] array, centered coords
        ball_y: [N] array, centered coords
        attacks_right: [N] bool, True if attacking toward +x

    Returns:
        epv: [N] array of EPV values
    """
    attacks_right = np.asarray(attacks_right)
    ball_x = np.asarray(ball_x, dtype=np.float64)
    ball_y = np.asarray(ball_y, dtype=np.float64)

    # If attacking right, ball x maps directly (more positive = closer to goal)
    # If attacking left, flip x so that more negative = closer to goal
    oriented_x = np.where(attacks_right, ball_x, -ball_x)

    row, col = ball_pos_to_grid_idx(oriented_x, ball_y)
    return epv_grid[row, col]


def draw_pitch(ax, pitch_x=105, pitch_y=68):
    """Draw a football pitch on the given axes."""
    ax.set_xlim(-pitch_x/2 - 3, pitch_x/2 + 3)
    ax.set_ylim(-pitch_y/2 - 3, pitch_y/2 + 3)
    ax.set_aspect("equal")
    ax.set_facecolor("#2d5a27")

    # Pitch outline
    ax.plot([-pitch_x/2, pitch_x/2, pitch_x/2, -pitch_x/2, -pitch_x/2],
            [-pitch_y/2, -pitch_y/2, pitch_y/2, pitch_y/2, -pitch_y/2],
            "white", lw=1.5)
    # Center line
    ax.axvline(0, color="white", lw=1, ls="-", ymin=0.044, ymax=0.956)
    # Center circle
    ax.add_patch(Circle((0, 0), 9.15, fill=False, ec="white", lw=1))
    # Penalty areas
    for sign in [-1, 1]:
        x0 = sign * pitch_x/2
        # 16.5m box
        ax.plot([x0, x0 - sign*16.5, x0 - sign*16.5, x0],
                [-20.16, -20.16, 20.16, 20.16], "white", lw=1)
        # 5.5m box
        ax.plot([x0, x0 - sign*5.5, x0 - sign*5.5, x0],
                [-9.16, -9.16, 9.16, 9.16], "white", lw=1)
        # Penalty spot
        ax.plot(x0 - sign*11, 0, "wo", ms=3)
        # Goal
        ax.plot([x0, x0], [-3.66, 3.66], "white", lw=3)

    ax.set_xticks([])
    ax.set_yticks([])


def visualize_epv_grid(epv_grid, out_path):
    """Plot the EPV grid as a heatmap on the pitch."""
    fig, axes = plt.subplots(1, 2, figsize=(20, 7))

    for idx, (ax, title, grid) in enumerate(zip(
        axes,
        ["EPV Grid: Home (attacking right)", "EPV Grid: Away (attacking left)"],
        [epv_grid, epv_grid[:, ::-1]]
    )):
        draw_pitch(ax)
        # Overlay grid as heatmap
        extent = [-52.5, 52.5, -34, 34]
        im = ax.imshow(grid, extent=extent, origin="lower", alpha=0.7,
                       cmap="RdYlGn", vmin=0, vmax=0.05, aspect="auto",
                       zorder=2)
        ax.set_title(title, fontsize=13, color="white")
        plt.colorbar(im, ax=ax, label="EPV", shrink=0.7)

    fig.suptitle("EPV Grid Baseline (location-only)", fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#1a1a1a")
    plt.close()
    print(f"Saved EPV grid visualization: {out_path}")


def visualize_pv_examples(model, chunks, entity_type, epv_grid, device,
                          out_path, n_examples=8):
    """Show PV model predictions vs EPV baseline for specific windows."""
    model.eval()

    # Find windows with shots and without
    all_features = []
    all_masks = []
    all_labels = []
    for chunk_data in chunks:
        all_features.append(chunk_data["features"])
        all_masks.append(chunk_data["mask"])
        all_labels.append(chunk_data["labels"])

    features = np.concatenate(all_features)
    masks = np.concatenate(all_masks)
    labels = np.concatenate(all_labels)

    # Pick examples: half with shots, half without
    has_shot = (labels[:, 2] > 0.5) | (labels[:, 3] > 0.5)
    shot_idx = np.where(has_shot)[0]
    noshot_idx = np.where(~has_shot)[0]

    np.random.seed(42)
    n_shot = min(n_examples // 2, len(shot_idx))
    n_noshot = n_examples - n_shot
    chosen_shot = np.random.choice(shot_idx, n_shot, replace=False)
    chosen_noshot = np.random.choice(noshot_idx, n_noshot, replace=False)
    chosen = np.concatenate([chosen_shot, chosen_noshot])

    fig, axes = plt.subplots(2, n_examples // 2, figsize=(5 * (n_examples // 2), 10))
    axes = axes.flatten()

    et_tensor = torch.from_numpy(entity_type.astype(np.int64)).to(device)

    for i, idx in enumerate(chosen):
        feat = torch.from_numpy(features[idx:idx+1]).to(device)
        mask = torch.from_numpy(masks[idx:idx+1].astype(np.float32)).to(device)

        with torch.no_grad(), torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            out = model(feat, mask, et_tensor)

        pv_home = out["pv_home"].item()
        pv_away = out["pv_away"].item()

        # Ball position at last frame
        last_frame = features[idx, -1]  # [23, 6]
        ball_mask_last = masks[idx, -1, 22]
        ball_x = last_frame[22, 0]
        ball_y = last_frame[22, 1]

        # Infer attacking direction
        home_right = infer_attack_direction(features[idx], masks[idx])

        # EPV lookup with correct orientation
        epv_home = lookup_epv_directed(epv_grid, ball_x, ball_y, home_right)
        epv_away = lookup_epv_directed(epv_grid, ball_x, ball_y, not home_right)

        # Draw pitch
        ax = axes[i]
        draw_pitch(ax)

        # Plot players at last frame, with arrows showing attack direction
        home_color = "#ff4444"
        away_color = "#4444ff"
        for j in range(11):
            if masks[idx, -1, j] > 0.5:
                ax.plot(last_frame[j, 0], last_frame[j, 1], "o",
                       color=home_color, ms=8, mec="white", mew=0.5, zorder=5)
        for j in range(11, 22):
            if masks[idx, -1, j] > 0.5:
                ax.plot(last_frame[j, 0], last_frame[j, 1], "o",
                       color=away_color, ms=8, mec="white", mew=0.5, zorder=5)

        # Ball
        if ball_mask_last > 0.5:
            ax.plot(ball_x, ball_y, "o", color="yellow", ms=10, mec="black",
                   mew=1.5, zorder=6)

        # Attack direction arrow
        arrow_y = 31
        h_dir = ">" if home_right else "<"
        a_dir = "<" if home_right else ">"
        ax.text(-48, arrow_y, f"H{h_dir}", color=home_color, fontsize=8,
                fontweight="bold", va="center", zorder=7)
        ax.text(38, arrow_y, f"A{a_dir}", color=away_color, fontsize=8,
                fontweight="bold", va="center", zorder=7)

        # Labels
        lab = labels[idx]
        shot_str = ""
        if lab[2] > 0.5:
            shot_str += f"H shot xG={lab[0]:.3f} "
        if lab[3] > 0.5:
            shot_str += f"A shot xG={lab[1]:.3f}"
        if not shot_str:
            shot_str = "No shot"

        ax.set_title(
            f"PV: H={pv_home:.4f} A={pv_away:.4f}\n"
            f"EPV: H={epv_home:.4f} A={epv_away:.4f}\n"
            f"{shot_str}",
            fontsize=9, color="white"
        )

    fig.suptitle("PV Model vs EPV Grid — Test Examples", fontsize=14, color="white")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#1a1a1a")
    plt.close()
    print(f"Saved PV examples: {out_path}")


def quantitative_comparison(model, chunks, entity_type, epv_grid, device):
    """Compare PV model vs EPV grid on all test data."""
    model.eval()

    all_pv_home = []
    all_pv_away = []
    all_epv_home = []
    all_epv_away = []
    all_target_home = []
    all_target_away = []
    all_has_shot_home = []
    all_has_shot_away = []

    et_tensor = torch.from_numpy(entity_type.astype(np.int64)).to(device)

    for chunk_data in chunks:
        features = chunk_data["features"]
        masks = chunk_data["mask"]
        labels = chunk_data["labels"]

        # Ball position at last frame of each window
        ball_x = features[:, -1, 22, 0]
        ball_y = features[:, -1, 22, 1]

        # Infer attacking direction per window
        home_right = infer_attack_direction(features, masks)

        epv_h = lookup_epv_directed(epv_grid, ball_x, ball_y, home_right)
        epv_a = lookup_epv_directed(epv_grid, ball_x, ball_y, ~home_right)

        # Run model in batches
        n = len(features)
        batch_size = 256
        pv_h_list = []
        pv_a_list = []

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            feat = torch.from_numpy(features[start:end]).to(device)
            mask = torch.from_numpy(masks[start:end].astype(np.float32)).to(device)

            with torch.no_grad(), torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                out = model(feat, mask, et_tensor)

            pv_h_list.append(out["pv_home"].cpu().numpy())
            pv_a_list.append(out["pv_away"].cpu().numpy())

        pv_h = np.concatenate(pv_h_list)
        pv_a = np.concatenate(pv_a_list)

        all_pv_home.append(pv_h)
        all_pv_away.append(pv_a)
        all_epv_home.append(epv_h)
        all_epv_away.append(epv_a)
        all_target_home.append(labels[:, 0])
        all_target_away.append(labels[:, 1])
        all_has_shot_home.append(labels[:, 2])
        all_has_shot_away.append(labels[:, 3])

    pv_home = np.concatenate(all_pv_home)
    pv_away = np.concatenate(all_pv_away)
    epv_home = np.concatenate(all_epv_home)
    epv_away = np.concatenate(all_epv_away)
    target_home = np.concatenate(all_target_home)
    target_away = np.concatenate(all_target_away)
    has_shot_home = np.concatenate(all_has_shot_home)
    has_shot_away = np.concatenate(all_has_shot_away)

    # Combine both teams
    pv_all = np.concatenate([pv_home, pv_away])
    epv_all = np.concatenate([epv_home, epv_away])
    target_all = np.concatenate([target_home, target_away])
    has_shot_all = np.concatenate([has_shot_home, has_shot_away])

    shot_mask = has_shot_all > 0.5
    noshot_mask = ~shot_mask

    print("\n" + "=" * 70)
    print("QUANTITATIVE COMPARISON: PV Model vs EPV Grid (test set)")
    print("=" * 70)
    print(f"Total windows: {len(pv_home)}")
    print(f"Windows with shots: {shot_mask.sum()} ({100*shot_mask.mean():.1f}%)")

    # Brier score (MSE between prediction and target xG)
    pv_brier = ((pv_all - target_all) ** 2).mean()
    epv_brier = ((epv_all - target_all) ** 2).mean()
    print(f"\nBrier Score (lower is better):")
    print(f"  PV model: {pv_brier:.6f}")
    print(f"  EPV grid: {epv_brier:.6f}")
    print(f"  Improvement: {100*(epv_brier - pv_brier)/epv_brier:.1f}%")

    # Separation: mean prediction on shot vs no-shot windows
    pv_shot_mean = pv_all[shot_mask].mean()
    pv_noshot_mean = pv_all[noshot_mask].mean()
    epv_shot_mean = epv_all[shot_mask].mean()
    epv_noshot_mean = epv_all[noshot_mask].mean()

    print(f"\nSeparation (shot vs no-shot mean prediction):")
    print(f"  PV model:  shot={pv_shot_mean:.5f}  no-shot={pv_noshot_mean:.5f}  "
          f"ratio={pv_shot_mean/max(pv_noshot_mean, 1e-8):.1f}x")
    print(f"  EPV grid:  shot={epv_shot_mean:.5f}  no-shot={epv_noshot_mean:.5f}  "
          f"ratio={epv_shot_mean/max(epv_noshot_mean, 1e-8):.1f}x")

    # Correlation with target xG
    pv_corr = np.corrcoef(pv_all, target_all)[0, 1] if target_all.std() > 1e-8 else 0
    epv_corr = np.corrcoef(epv_all, target_all)[0, 1] if target_all.std() > 1e-8 else 0
    print(f"\nCorrelation with target xG:")
    print(f"  PV model: {pv_corr:.4f}")
    print(f"  EPV grid: {epv_corr:.4f}")

    # Mean predictions
    print(f"\nMean predictions:")
    print(f"  PV model:  home={pv_home.mean():.5f}  away={pv_away.mean():.5f}")
    print(f"  EPV grid:  home={epv_home.mean():.5f}  away={epv_away.mean():.5f}")
    print(f"  Targets:   home={target_home.mean():.5f}  away={target_away.mean():.5f}")

    # On shot windows only: correlation with xG
    if shot_mask.sum() > 10:
        pv_shot_corr = np.corrcoef(pv_all[shot_mask], target_all[shot_mask])[0, 1]
        epv_shot_corr = np.corrcoef(epv_all[shot_mask], target_all[shot_mask])[0, 1]
        print(f"\nCorrelation with xG (shot windows only):")
        print(f"  PV model: {pv_shot_corr:.4f}")
        print(f"  EPV grid: {epv_shot_corr:.4f}")

    print("=" * 70)

    return {
        "pv_all": pv_all, "epv_all": epv_all, "target_all": target_all,
        "has_shot_all": has_shot_all,
        "pv_home": pv_home, "pv_away": pv_away,
        "epv_home": epv_home, "epv_away": epv_away,
    }


def plot_comparison(results, out_path):
    """Plot comparison charts."""
    pv_all = results["pv_all"]
    epv_all = results["epv_all"]
    target_all = results["target_all"]
    has_shot = results["has_shot_all"] > 0.5

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. Histogram of predictions
    ax = axes[0, 0]
    ax.hist(pv_all, bins=100, alpha=0.7, label="PV model", color="#4CAF50", density=True)
    ax.hist(epv_all, bins=100, alpha=0.7, label="EPV grid", color="#2196F3", density=True)
    ax.set_xlabel("Predicted value")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of Predictions")
    ax.legend()
    ax.set_xlim(0, 0.1)

    # 2. Predictions on shot vs no-shot windows
    ax = axes[0, 1]
    data = [
        pv_all[has_shot], pv_all[~has_shot],
        epv_all[has_shot], epv_all[~has_shot],
    ]
    positions = [1, 2, 4, 5]
    bp = ax.boxplot(data, positions=positions, widths=0.6, patch_artist=True,
                    showfliers=False)
    colors = ["#4CAF50", "#81C784", "#2196F3", "#64B5F6"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
    ax.set_xticks([1.5, 4.5])
    ax.set_xticklabels(["PV Model", "EPV Grid"])
    ax.legend([bp["boxes"][0], bp["boxes"][1]],
              ["Shot windows", "No-shot windows"], loc="upper left")
    ax.set_ylabel("Predicted value")
    ax.set_title("Shot vs No-Shot Separation")

    # 3. PV model vs EPV grid scatter (on shot windows)
    ax = axes[1, 0]
    ax.scatter(epv_all[has_shot], pv_all[has_shot], s=10, alpha=0.4,
              color="#FF5722", label=f"Shot (n={has_shot.sum()})")
    ax.scatter(epv_all[~has_shot][::50], pv_all[~has_shot][::50], s=3, alpha=0.2,
              color="#9E9E9E", label=f"No-shot (1/50 shown)")
    lim = max(pv_all.max(), epv_all.max()) * 1.1
    ax.plot([0, lim], [0, lim], "k--", alpha=0.3, label="y=x")
    ax.set_xlabel("EPV Grid")
    ax.set_ylabel("PV Model")
    ax.set_title("PV Model vs EPV Grid")
    ax.legend(fontsize=8)

    # 4. Calibration: bin by prediction, show actual positive rate
    ax = axes[1, 1]
    for pred, label, color in [
        (pv_all, "PV model", "#4CAF50"),
        (epv_all, "EPV grid", "#2196F3"),
    ]:
        # Bin predictions
        bins = np.linspace(0, max(pred.max(), 0.05), 20)
        bin_idx = np.digitize(pred, bins) - 1
        bin_centers = []
        bin_actuals = []
        for b in range(len(bins) - 1):
            mask = bin_idx == b
            if mask.sum() > 50:
                bin_centers.append((bins[b] + bins[b+1]) / 2)
                bin_actuals.append(target_all[mask].mean())
        ax.plot(bin_centers, bin_actuals, "o-", color=color, label=label, ms=5)

    lim = 0.05
    ax.plot([0, lim], [0, lim], "k--", alpha=0.3, label="Perfect calibration")
    ax.set_xlabel("Predicted PV")
    ax.set_ylabel("Actual mean xG target")
    ax.set_title("Calibration")
    ax.legend(fontsize=8)

    plt.suptitle("PV Model vs EPV Grid Baseline", fontsize=15)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved comparison plot: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/pv_v1/best.pt")
    parser.add_argument("--epv-grid", default="EPV_grid.csv")
    parser.add_argument("--data-dir", default="pv_data")
    parser.add_argument("--out-dir", default="eval_pv_output")
    parser.add_argument("--no-cuda", action="store_true")
    args = parser.parse_args()

    device = torch.device("cpu" if args.no_cuda or not torch.cuda.is_available() else "cuda")
    print(f"Device: {device}")

    # Load EPV grid
    epv_grid = load_epv_grid(args.epv_grid)
    print(f"EPV grid: {epv_grid.shape}, range [{epv_grid.min():.4f}, {epv_grid.max():.4f}]")

    # Load model
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    meta = ckpt["meta"]
    model_args = ckpt["args"]

    model = PossessionValueTransformer(
        feat_dim=meta["feat_dim"],
        n_entities=meta["n_entities"],
        d_model=model_args["d_model"],
        n_heads=model_args["n_heads"],
        n_layers=model_args["n_layers"],
        dropout=0.0,  # No dropout at inference
        max_seq_len=meta["window_size"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded PV model from epoch {ckpt['epoch']}")

    # Load test data
    test_dir = os.path.join(args.data_dir, "test")
    with open(os.path.join(test_dir, "manifest.json")) as f:
        manifest = json.load(f)

    chunks = []
    entity_type = None
    for entry in manifest:
        data = np.load(entry["path"])
        chunks.append({
            "features": data["features"],
            "mask": data["mask"],
            "labels": data["labels"],
        })
        if entity_type is None:
            entity_type = data["entity_type"]

    total_windows = sum(c["features"].shape[0] for c in chunks)
    print(f"Test data: {total_windows} windows from {len(chunks)} chunks")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Visualize EPV grid
    visualize_epv_grid(epv_grid, out_dir / "epv_grid.png")

    # 2. Visualize PV examples
    visualize_pv_examples(
        model, chunks, entity_type, epv_grid, device,
        out_dir / "pv_examples.png", n_examples=8
    )

    # 3. Quantitative comparison
    results = quantitative_comparison(model, chunks, entity_type, epv_grid, device)

    # 4. Plot comparison charts
    plot_comparison(results, out_dir / "comparison.png")

    print(f"\nAll outputs saved to {out_dir}/")


if __name__ == "__main__":
    main()
