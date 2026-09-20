# Checkpoints

Checkpoint binaries are excluded from Git. This file records the canonical local source paths and the rationale for each selected model.

## Canonical Models

| Component | Local source path | Selection basis | Google Drive link |
| --- | --- | --- | --- |
| `ballPlayerTrajModel` | `/mnt/data/remains/opta2026/ballPlayerTrajModel/checkpoints/all_from_start_xyz_eval_stop_v1_4gpu/best.pt` | Best logged `best_val=0.166603` in `train_allxyz_stopv1_4gpu_oobstill05_20260225_163157.log` | `TODO` |
| `trajModel_smart` | `/mnt/data/remains/opta2026/trajModel_smart/checkpoints/smart_v1/best.pt` | Best `val_loss=4.888073` at epoch 39 in `trajModel_smart/checkpoints/smart_v1/log.jsonl` | `TODO` |
| `playerToTouch` | `/mnt/data/remains/opta2026/playerToTouch/checkpoints/sportec_survival_v2/best.pt` | Best Sportec survival `val_hazard_loss=0.039044` and used in `hybrid_eval_result_val.json` | `TODO` |
| `ballAtTouch` | `/mnt/data/remains/opta2026/ballAtTouch/initParamVar/checkpoints/sportec_gaussian_recv_v5/best.pt` | Best receive-conditioned Gaussian checkpoint in local history (`val_nll=-0.900268`, `val_mae=0.202671`) | `TODO` |
| `possessionValue` | `/mnt/data/remains/opta2026/possessionValue/checkpoints/pv_v1/best.pt` | Current production default used by pass search | `TODO` |
| `setPiecePV` | `/mnt/data/remains/opta2026/possessionValue/checkpoints/set_piece_pv_v1/model.json` | Current production default used by pass search | `TODO` |

## Notes

- `trajModel_allData` is intentionally not part of this repo and should not be uploaded alongside these checkpoints.
- `trajModel_smart/vocabs/player_vocab.npz` and `trajModel_smart/vocabs/ball_vocab.npz` stay in Git because they are small runtime assets.
- Replace each `TODO` with the final shareable Google Drive URL after upload.

