#!/usr/bin/env python3
"""Build SMART motion token vocabularies from training data.

Extracts displacement sequences from all training clips, clusters with
MiniBatchKMeans, and saves vocabularies to vocabs/ directory.

Usage:
    python3 build_vocab.py [--preprocessed-dir PATH] [--output-dir vocabs]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans

from tokenizer import (
    BALL_CLAMP,
    BALL_DIM,
    BALL_VOCAB_SIZE,
    PLAYER_CLAMP,
    PLAYER_DIM,
    PLAYER_VOCAB_SIZE,
    STEPS_PER_TOKEN,
    extract_ball_displacements,
    extract_player_displacements,
)


DOWNSAMPLE_FACTOR = 2  # 25Hz -> 12.5Hz


def parse_args():
    p = argparse.ArgumentParser(description="Build SMART motion token vocabularies.")
    p.add_argument("--preprocessed-dir", type=str,
                   default="/mnt/data/remains/opta2026/ballPlayerTrajModel/publicData/preprocessed/sportec_xyz_metres",
                   help="Preprocessed data directory (Sportec).")
    p.add_argument("--output-dir", type=str, default=None,
                   help="Output directory for vocab files. Default: vocabs/ in script dir.")
    p.add_argument("--player-vocab-size", type=int, default=PLAYER_VOCAB_SIZE)
    p.add_argument("--ball-vocab-size", type=int, default=BALL_VOCAB_SIZE)
    p.add_argument("--batch-size", type=int, default=4096, help="MiniBatchKMeans batch size.")
    p.add_argument("--max-iter", type=int, default=200, help="KMeans max iterations.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_manifest(preprocessed_dir: Path):
    manifest_path = preprocessed_dir / "train" / "manifest.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No manifest at {manifest_path}")
    entries = []
    with open(manifest_path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def extract_all_displacements(entries, preprocessed_dir: Path):
    """Extract player and ball displacement sequences from all training clips."""
    all_player = []
    all_ball = []
    n_clips = 0

    for entry in entries:
        clip_path = entry["clip_path"]
        try:
            clip = np.load(clip_path)
        except Exception as e:
            print(f"  Warning: failed to load {clip_path}: {e}")
            continue

        features = clip["features"]  # [T, N, F]
        entity_type = clip["entity_type"]  # [N]
        mask = clip.get("mask", np.ones(features.shape[:2], dtype=bool))
        T, N, F = features.shape

        # Downsample: take every 2nd frame (25Hz -> 12.5Hz)
        features = features[::DOWNSAMPLE_FACTOR]
        mask = mask[::DOWNSAMPLE_FACTOR]
        T_ds = features.shape[0]

        if T_ds < STEPS_PER_TOKEN:
            continue

        # Ball: entity_type == 2
        ball_idxs = np.where(entity_type == 2)[0]
        if len(ball_idxs) > 0:
            bi = int(ball_idxs[0])
            # Ball position: [x, y, z] from features layout [x, y, vx, vy, z, vz]
            ball_xy = features[:, bi, :2]  # [T_ds, 2]
            if F >= 6:
                ball_z = features[:, bi, 4:5]  # [T_ds, 1]
            else:
                ball_z = np.zeros((T_ds, 1), dtype=np.float32)
            ball_pos = np.concatenate([ball_xy, ball_z], axis=1)  # [T_ds, 3]
            ball_disps = extract_ball_displacements(ball_pos, clamp=BALL_CLAMP)
            if ball_disps.shape[0] > 0:
                all_ball.append(ball_disps)

        # Players: entity_type 0 or 1
        player_idxs = np.where((entity_type == 0) | (entity_type == 1))[0]
        for pi in player_idxs:
            if not mask[:, pi].any():
                continue
            player_xy = features[:, pi, :2]  # [T_ds, 2]
            player_disps = extract_player_displacements(player_xy, clamp=PLAYER_CLAMP)
            if player_disps.shape[0] > 0:
                all_player.append(player_disps)

        n_clips += 1
        if n_clips % 500 == 0:
            n_p = sum(d.shape[0] for d in all_player)
            n_b = sum(d.shape[0] for d in all_ball)
            print(f"  Processed {n_clips}/{len(entries)} clips, {n_p:,} player tokens, {n_b:,} ball tokens")

    player_all = np.concatenate(all_player, axis=0) if all_player else np.zeros((0, PLAYER_DIM), np.float32)
    ball_all = np.concatenate(all_ball, axis=0) if all_ball else np.zeros((0, BALL_DIM), np.float32)
    return player_all, ball_all


def build_vocab(data, n_clusters, batch_size, max_iter, seed, name):
    """Standardize + MiniBatchKMeans clustering."""
    print(f"\n=== Building {name} vocab: {data.shape[0]:,} samples -> {n_clusters} clusters ===")

    # Compute per-dim std for standardization
    std = data.std(axis=0).astype(np.float32)
    std = np.maximum(std, 1e-6)
    data_std = data / std

    print(f"  Per-dim std range: [{std.min():.4f}, {std.max():.4f}]")

    t0 = time.time()
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        init="k-means++",
        batch_size=batch_size,
        max_iter=max_iter,
        random_state=seed,
        verbose=1,
    )
    kmeans.fit(data_std)
    elapsed = time.time() - t0
    print(f"  KMeans converged in {elapsed:.1f}s, inertia={kmeans.inertia_:.2f}")

    # Un-standardize centroids back to original space
    centroids = (kmeans.cluster_centers_ * std).astype(np.float32)

    # Compute quantization error stats
    labels = kmeans.predict(data_std)
    reconstructed = centroids[labels]
    errors = np.linalg.norm(data - reconstructed, axis=1)
    print(f"  Quantization error: mean={errors.mean():.4f}, median={np.median(errors):.4f}, "
          f"p95={np.percentile(errors, 95):.4f}, max={errors.max():.4f}")

    return centroids, std


def main():
    args = parse_args()
    preprocessed_dir = Path(args.preprocessed_dir)

    if args.output_dir is None:
        output_dir = Path(__file__).parent / "vocabs"
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Preprocessed dir: {preprocessed_dir}")
    print(f"Output dir: {output_dir}")

    # Load manifest
    entries = load_manifest(preprocessed_dir)
    print(f"Found {len(entries)} training clips")

    # Extract displacements
    print("\nExtracting displacement sequences...")
    t0 = time.time()
    player_data, ball_data = extract_all_displacements(entries, preprocessed_dir)
    print(f"\nExtraction done in {time.time() - t0:.1f}s")
    print(f"  Player: {player_data.shape[0]:,} tokens, shape {player_data.shape}")
    print(f"  Ball:   {ball_data.shape[0]:,} tokens, shape {ball_data.shape}")

    if player_data.shape[0] < args.player_vocab_size:
        print(f"WARNING: Only {player_data.shape[0]} player tokens for {args.player_vocab_size} clusters!")
    if ball_data.shape[0] < args.ball_vocab_size:
        print(f"WARNING: Only {ball_data.shape[0]} ball tokens for {args.ball_vocab_size} clusters!")

    # Build vocabularies
    player_centroids, player_std = build_vocab(
        player_data, args.player_vocab_size, args.batch_size, args.max_iter, args.seed, "player")
    ball_centroids, ball_std = build_vocab(
        ball_data, args.ball_vocab_size, args.batch_size, args.max_iter, args.seed, "ball")

    # Save
    player_path = output_dir / "player_vocab.npz"
    ball_path = output_dir / "ball_vocab.npz"
    np.savez(player_path, centroids=player_centroids, std=player_std)
    np.savez(ball_path, centroids=ball_centroids, std=ball_std)
    print(f"\nSaved player vocab: {player_path} ({player_centroids.shape})")
    print(f"Saved ball vocab:   {ball_path} ({ball_centroids.shape})")

    # Summary
    print("\n=== Summary ===")
    print(f"Player vocab: {args.player_vocab_size} tokens, {PLAYER_DIM}D")
    print(f"Ball vocab:   {args.ball_vocab_size} tokens, {BALL_DIM}D")
    print("Done!")


if __name__ == "__main__":
    main()
