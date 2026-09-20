# publicData

Utilities for collecting public tracking data via `kloppy` and converting it to the same preprocessed clip format used by `ballPlayerTrajModel`.

## What this folder provides

- `kloppy_to_preprocessed.py`
  - Loads open tracking from `kloppy` (`metrica` or `sportec`)
  - Standardizes coordinates to centered meters on a `105 x 68` pitch
  - Builds fixed-length clips in `train/`, `val/`, `test/` with manifests compatible with `dataset.py`
  - Supports:
    - `--ball-mode xy`: features `[x, y, vx, vy]`
    - `--ball-mode xyz`: features `[x, y, vx, vy, z, vz]` (ball z/vz channels only)

- `make_stage_datasets.py`
  - Builds a single merged dataset (default):
    - Train: current Opta + Metrica + Sportec XYZ (and optional Sportec XY)
    - Val/Test: Sportec XYZ only
  - Automatically zero-pads feature channels so mixed xy/xyz sources can be trained together.

## Coordinate standardization

- Metrica input: normalized `[0,1]` with origin at top-left.
  - Converted as:
    - `x_m = (x - 0.5) * 105`
    - `y_m = (0.5 - y) * 68`
- Sportec input: centered metric coordinates.
  - Kept as-is.

## Example workflow

1. Build Metrica xy preprocessed set

```bash
python kloppy_to_preprocessed.py \
  --provider metrica \
  --match-ids 1,2,3 \
  --ball-mode xy \
  --out-dir /mnt/data/remains/opta2026/ballPlayerTrajModel/publicData/preprocessed/metrica_xy
```

2. Build Sportec xy preprocessed set

```bash
python kloppy_to_preprocessed.py \
  --provider sportec \
  --ball-mode xy \
  --out-dir /mnt/data/remains/opta2026/ballPlayerTrajModel/publicData/preprocessed/sportec_xy
```

3. Build Sportec xyz preprocessed set

```bash
python kloppy_to_preprocessed.py \
  --provider sportec \
  --ball-mode xyz \
  --out-dir /mnt/data/remains/opta2026/ballPlayerTrajModel/publicData/preprocessed/sportec_xyz
```

4. Build merged training dir (single-stage default)

```bash
python make_stage_datasets.py \
  --current-preprocessed /mnt/data/remains/opta2026/ballPlayerTrajModel/preprocessed \
  --metrica-preprocessed /mnt/data/remains/opta2026/ballPlayerTrajModel/publicData/preprocessed/metrica_xy \
  --sportec-xyz-preprocessed /mnt/data/remains/opta2026/ballPlayerTrajModel/publicData/preprocessed/sportec_xyz \
  --out-root /mnt/data/remains/opta2026/ballPlayerTrajModel/publicData/stages
```

5. Train once from the merged dataset (reports val/test each epoch)

```bash
python ../train.py \
  --preprocessed-dir /mnt/data/remains/opta2026/ballPlayerTrajModel/publicData/stages/all_from_start_xyz_eval \
  --save-dir /mnt/data/remains/opta2026/ballPlayerTrajModel/checkpoints_all_from_start
```

Notes:
- `train.py` now supports optional test reporting from `test/manifest.jsonl` each epoch.
- If inputs come from mixed xy/xyz sources, `make_stage_datasets.py` pads non-target feature dims with zeros.
