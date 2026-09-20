# monteCarloPassSearch (CVPR 2026 CVSports Workshop)

Paper link: `https://openaccess.thecvf.com/content/CVPR2026W/CVsports/papers/Kang_Monte_Carlo_Pass_Search_Using_Trajectory_Generation_for_3D_Counterfactual_CVPRW_2026_paper.pdf`

`monteCarloPassSearch` is a research codebase for counterfactual soccer pass evaluation built around linked models for player rollout, next-touch prediction, post-touch ball continuation, and possession value estimation.

This repository uses data derived from the public dataset described in:

- Bassek, M., Rein, R., Weber, H. and Memmert, D. "An integrated dataset of spatiotemporal and event data in elite soccer." *Scientific Data* 12, 195 (2025). DOI: `10.1038/s41597-025-04505-y`
- Paper: `https://www.nature.com/articles/s41597-025-04505-y`

That dataset combines synchronized event and tracking data from seven German Bundesliga and 2. Bundesliga matches, making it a useful public benchmark for pass evaluation, possession value modeling, and trajectory forecasting research.

## Repository Scope

This repo keeps the code for:

- `monteCarloPassSearch/initParamVar`: Monte Carlo pass search and ranking pipeline
- `ballPlayerTrajModel`: clip preprocessing and trajectory-model training code
- `trajModel_smart`: SMART player rollout model and tokenization
- `playerToTouch`: next-touch prediction models
- `ballAtTouch`: post-touch ball continuation models
- `possessionValue`: open-play PV, xG, xT, and set-piece PV models

It intentionally excludes large artifacts such as checkpoints, preprocessed datasets, logs, generated figures, and older experimental branches.

## Main Pipeline

The core workflow in `monteCarloPassSearch/initParamVar` is:

1. Load a preprocessed pass clip built from synchronized tracking and event data.
2. Infer the observed pass parameters and sample local/global counterfactual variants.
3. Roll players forward with `trajModel_smart`.
4. Predict the next toucher with `playerToTouch`.
5. Continue the ball after touch with `ballAtTouch`.
6. Score the resulting state with `possessionValue`.

The main entrypoint is:

```bash
python3 monteCarloPassSearch/initParamVar/pass_mc_runner.py \
  --output-csv /tmp/pass_mc_smoke.csv \
  --max-clips 2 \
  --variants-local 8 \
  --variants-global 8 \
  --gpus 0 \
  --workers-per-gpu 1
```

Passer ranking from an existing Monte Carlo output:

```bash
python3 monteCarloPassSearch/initParamVar/rank_passers.py \
  --input-csv /tmp/pass_mc_smoke.csv \
  --output-csv /tmp/passer_rankings.csv \
  --output-csv-local /tmp/passer_rankings_local.csv \
  --output-csv-global /tmp/passer_rankings_global.csv \
  --clip-output-csv /tmp/pass_clip_percentiles.csv
```

Pre-pass PV backfill:

```bash
python3 monteCarloPassSearch/initParamVar/backfill_prepass_pv.py \
  --input-csv /tmp/pass_mc_smoke.csv \
  --output-csv /tmp/pass_mc_smoke_with_prepass.csv
```

## Notes

- The repo assumes you have already prepared local training or evaluation inputs derived from the public source dataset.
- Canonical checkpoint selections are listed in `CHECKPOINTS.md`; the binaries themselves are not tracked in Git.
- Small SMART vocabulary assets in `trajModel_smart/vocabs` are kept because they are required at runtime.
