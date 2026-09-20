# monteCarloPassSearch/initParamVar

`initParamVar` is the main Monte Carlo pass-search pipeline in this repository.

It evaluates observed passes and sampled counterfactual variants using:

- `trajModel_smart` for player rollout
- `playerToTouch` for next-touch prediction
- `ballAtTouch` for post-touch ball continuation
- `possessionValue` for open-play and set-piece value scoring

The intended input is preprocessed clip data derived from the public integrated event-plus-tracking dataset described in the Scientific Data paper linked in the root README.

## Output

The runner writes one CSV row per pass variant, including:

- observed/local/global variant labels
- inferred release parameters
- predicted toucher and touch frame
- out-of-play or restart classification
- possession value outputs
- per-clip metadata for ranking and audit

## Default Runtime Inputs

- Manifest: `ballPlayerTrajModel/publicData/preprocessed/sportec_xyz/test/manifest.jsonl`
- SMART checkpoint: `trajModel_smart/checkpoints/smart_v1/best.pt`
- SMART vocab dir: `trajModel_smart/vocabs`
- PTT checkpoint: `playerToTouch/checkpoints/sportec_survival_v2/best.pt`
- BAT checkpoint: `ballAtTouch/initParamVar/checkpoints/sportec_gaussian_recv_v5/best.pt`
- PV checkpoint: `possessionValue/checkpoints/pv_v1/best.pt`
- Set-piece PV model: `possessionValue/checkpoints/set_piece_pv_v1/model.json`

## Example

Smoke run:

```bash
python3 monteCarloPassSearch/initParamVar/pass_mc_runner.py \
  --output-csv /tmp/pass_mc_smoke.csv \
  --max-clips 2 \
  --variants-local 8 \
  --variants-global 8 \
  --gpus 0 \
  --workers-per-gpu 1
```

Passer ranking:

```bash
python3 monteCarloPassSearch/initParamVar/rank_passers.py \
  --input-csv /tmp/pass_mc_smoke.csv \
  --output-csv /tmp/passer_rankings.csv \
  --output-csv-local /tmp/passer_rankings_local.csv \
  --output-csv-global /tmp/passer_rankings_global.csv \
  --clip-output-csv /tmp/pass_clip_percentiles.csv
```

See `CHECKPOINTS.md` for the canonical checkpoint inventory.
