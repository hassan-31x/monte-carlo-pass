# ballAtTouch/fullTrajVar

Replaces the old single-step outgoing-velocity target with a fixed 3-second
future ball trajectory target from the current touch.

Files:
- `preprocess_sportec.py`: builds touch-context samples with masked 75-frame future trajectories
- `dataset.py`: loads `traj_pos`, `traj_mask`, `stop_idx`
- `model.py`: transformer encoder with trajectory and stop heads
- `train.py`: masked trajectory training loop with position/velocity/stop/smoothness losses

Smoke-verified locally on March 9, 2026 with:

```bash
python3 /mnt/data/remains/opta2026/ballAtTouch/fullTrajVar/train.py \
  --preprocessed-dir /mnt/data/tmp/bat_fulltraj_smoke \
  --save-dir /mnt/data/tmp/bat_fulltraj_ckpt \
  --run-name smoke \
  --epochs 1 \
  --batch-size 1 \
  --num-workers 0
```
