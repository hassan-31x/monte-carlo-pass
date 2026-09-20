#!/usr/bin/env python3
"""Create compact MCPS training and counterfactual diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def simulate_ball(start: np.ndarray, row: pd.Series, n_frames: int = 129) -> np.ndarray:
    pos = start.astype(np.float64).copy()
    vel = np.array([row.v0x, row.v0y, row.v0z], dtype=np.float64)
    spin = float(row.spin_scalar)
    result = np.zeros((n_frames, 3), dtype=np.float32)
    for index in range(n_frames):
        result[index] = pos
        speed = float(np.linalg.norm(vel[:2]))
        if speed > 1e-6 and abs(spin) > 1e-8:
            perpendicular = np.array([-vel[1], vel[0]]) / speed
            vel[:2] += perpendicular * (spin * 0.14 * speed) / 25.0
        vel *= 1.0 - 0.00648
        vel[2] -= 9.81 / 25.0
        pos += vel / 25.0
        if pos[2] < 0.0:
            pos[2] = -pos[2] * 0.109
            vel[2] = -vel[2] * 0.109
            vel[:2] *= 1.0 - 0.02731
    return result


def plot_ball_trajectories(sample: pd.DataFrame, out: Path) -> None:
    eligible = sample[sample["variant_group"].isin(["observed", "local", "global"])]
    if eligible.empty:
        return
    clip_path = Path(str(eligible.iloc[0]["clip_path"]))
    if not clip_path.exists():
        return
    with np.load(clip_path) as clip:
        features = np.asarray(clip["features"])
        entity_type = np.asarray(clip["entity_type"])
    ball_idx = int(np.flatnonzero(entity_type == 2)[0])
    kick = int(eligible.iloc[0]["kick_frame_local_refined"])
    start = np.array([features[kick, ball_idx, 0], features[kick, ball_idx, 1], features[kick, ball_idx, 4]])
    observed = eligible[eligible["variant_group"] == "observed"]
    others = eligible[eligible["variant_group"] != "observed"].groupby("variant_group", group_keys=False).head(8)
    selected = pd.concat([others, observed.head(1)])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    recorded = features[kick : min(len(features), kick + 129), ball_idx]
    axes[0].plot(recorded[:, 0], recorded[:, 1], color="black", lw=2.2, linestyle="--", label="recorded")
    axes[1].plot(np.arange(len(recorded)) / 25.0, recorded[:, 4], color="black", lw=2.2, linestyle="--")
    for _, row in selected.iterrows():
        trajectory = simulate_ball(start, row)
        color = {"observed": "#dc2626", "local": "#2563eb", "global": "#f59e0b"}[row.variant_group]
        width = 2.5 if row.variant_group == "observed" else 0.9
        alpha = 1.0 if row.variant_group == "observed" else 0.55
        axes[0].plot(trajectory[:, 0], trajectory[:, 1], color=color, lw=width, alpha=alpha)
        axes[1].plot(np.arange(len(trajectory)) / 25.0, trajectory[:, 2], color=color, lw=width, alpha=alpha)
    axes[0].set(xlim=(-52.5, 52.5), ylim=(-34, 34), aspect="equal", xlabel="x (m)", ylabel="y (m)", title="Candidate ball flights")
    axes[0].legend()
    axes[1].set(xlabel="seconds after kick", ylabel="height (m)", title="Vertical flight profiles")
    fig.tight_layout()
    fig.savefig(out / "ball_trajectories.png", dpi=160)
    plt.close(fig)


def plot_histories(work: Path, out: Path) -> None:
    histories = {
        "SMART": work / "checkpoints" / "smart" / "log.jsonl",
        "Player-to-Touch": work / "checkpoints" / "touch" / "history.json",
        "Ball-at-Touch": work / "checkpoints" / "bat" / "history.json",
        "Possession Value": work / "checkpoints" / "pv" / "history.json",
    }
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (name, path) in zip(axes.flat, histories.items()):
        if not path.exists():
            ax.text(0.5, 0.5, "log unavailable", ha="center")
            ax.set_title(name)
            continue
        if path.suffix == ".jsonl":
            rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        else:
            rows = json.loads(path.read_text())
        frame = pd.DataFrame(rows)
        for key in frame.columns:
            if "loss" in key and pd.api.types.is_numeric_dtype(frame[key]):
                ax.plot(frame.get("epoch", np.arange(len(frame))), frame[key], label=key)
        ax.set_title(name)
        ax.set_xlabel("epoch")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "training_curves.png", dpi=160)
    plt.close(fig)


def plot_search(csv_path: Path, out: Path) -> None:
    if not csv_path.exists():
        return
    data = pd.read_csv(csv_path)
    fit = data.drop_duplicates("clip_idx").copy()
    for column in ("observed_fit_rmse_xy", "observed_fit_rmse_z"):
        fit[column] = pd.to_numeric(fit[column], errors="coerce")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(fit["observed_fit_rmse_xy"], fit["observed_fit_rmse_z"],
                    c=np.where(fit["status_code"].eq("rejected_fit"), "#dc2626", "#16a34a"))
    axes[0].axvline(0.03, color="gray", linestyle="--", linewidth=1)
    axes[0].axhline(0.10, color="gray", linestyle="--", linewidth=1)
    axes[0].set(xlabel="ball-fit XY RMSE (m)", ylabel="ball-fit Z RMSE (m)", title="Kick-fit acceptance")
    counts = fit["status_code"].fillna("unknown").value_counts()
    axes[1].barh(counts.index.astype(str), counts.values, color="#64748b")
    axes[1].set(xlabel="passes", title="Search terminal status")
    fig.tight_layout()
    fig.savefig(out / "fit_diagnostics.png", dpi=160)
    plt.close(fig)

    value_col = "pv_added_vs_prepass" if "pv_added_vs_prepass" in data else "pv_net"
    valid = data[np.isfinite(pd.to_numeric(data[value_col], errors="coerce"))].copy()
    if valid.empty:
        return
    valid[value_col] = pd.to_numeric(valid[value_col])
    clip = valid["clip_idx"].iloc[0]
    sample = valid[valid["clip_idx"] == clip]
    plot_ball_trajectories(sample, out)
    observed = sample[sample["variant_group"] == "observed"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for group, color in (("local", "#2563eb"), ("global", "#f59e0b")):
        values = sample.loc[sample["variant_group"] == group, value_col]
        if len(values):
            axes[0].hist(values, bins=min(12, len(values)), alpha=0.55, color=color, label=group)
    if not observed.empty:
        axes[0].axvline(observed[value_col].iloc[0], color="#dc2626", linestyle="--", label="observed")
    axes[0].set_title(f"Counterfactual value distribution, clip {clip}")
    axes[0].set_xlabel("gained possession value")
    axes[0].legend()
    colors = sample["variant_group"].map({"observed": "#dc2626", "local": "#2563eb", "global": "#f59e0b"})
    axes[1].scatter(sample["v0x"], sample["v0y"], c=colors, s=35, alpha=0.8)
    axes[1].set_title("Sampled initial horizontal velocities")
    axes[1].set_xlabel("v0x (m/s)")
    axes[1].set_ylabel("v0y (m/s)")
    fig.tight_layout()
    fig.savefig(out / "counterfactual_summary.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", required=True, type=Path)
    args = parser.parse_args()
    out = args.work_dir / "figures"
    out.mkdir(parents=True, exist_ok=True)
    plot_histories(args.work_dir, out)
    plot_search(args.work_dir / "search" / "variants.csv", out)
    print(f"Figures written to {out}")


if __name__ == "__main__":
    main()
