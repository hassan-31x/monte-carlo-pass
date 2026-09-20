#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/data/remains/opta2026/ballPlayerTrajModel"
PUB="$ROOT/publicData"
LOG_DIR="$ROOT/preprocess_logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/full_preprocess_stop_$(date +%Y%m%d_%H%M%S).log"

exec > >(tee -a "$LOG_FILE") 2>&1

echo "[$(date -Is)] Starting full preprocessing workflow (stop-event paradigm)."
echo "[$(date -Is)] Log file: $LOG_FILE"

cd "$ROOT"

echo "[$(date -Is)] Step 1/4: Current dataset preprocessing (all 0..99)."
python3 preprocess.py \
  --data-dir /mnt/data/remains \
  --passes-dir /mnt/data/remains/eventData/passes \
  --out-dir /mnt/data/remains/opta2026/ballPlayerTrajModel/preprocessed \
  --file-start 0 \
  --file-end 99 \
  --history-frames 512 \
  --future-frames 1024 \
  --allow-noncontiguous-windows \
  --rebuild

echo "[$(date -Is)] Step 2/4: Metrica XY preprocessing."
cd "$PUB"
python3 kloppy_to_preprocessed.py \
  --provider metrica \
  --ball-mode xy \
  --out-dir /mnt/data/remains/opta2026/ballPlayerTrajModel/publicData/preprocessed/metrica_xy \
  --history-frames 512 \
  --future-frames 1024 \
  --seed 42 \
  --rebuild

echo "[$(date -Is)] Step 3/4: Sportec XYZ preprocessing (7 open Bundesliga matches)."
python3 kloppy_to_preprocessed.py \
  --provider sportec \
  --ball-mode xyz \
  --out-dir /mnt/data/remains/opta2026/ballPlayerTrajModel/publicData/preprocessed/sportec_xyz \
  --history-frames 512 \
  --future-frames 1024 \
  --seed 42 \
  --rebuild

echo "[$(date -Is)] Step 4/4: Build merged stage dataset (train=all sources, val/test=sportec_xyz)."
python3 make_stage_datasets.py \
  --current-preprocessed /mnt/data/remains/opta2026/ballPlayerTrajModel/preprocessed \
  --metrica-preprocessed /mnt/data/remains/opta2026/ballPlayerTrajModel/publicData/preprocessed/metrica_xy \
  --sportec-xyz-preprocessed /mnt/data/remains/opta2026/ballPlayerTrajModel/publicData/preprocessed/sportec_xyz \
  --out-root /mnt/data/remains/opta2026/ballPlayerTrajModel/publicData/stages \
  --out-name all_from_start_xyz_eval_stop_v1 \
  --no-eager-convert

echo "[$(date -Is)] Full preprocessing workflow completed successfully."
