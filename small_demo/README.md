# Small-data Monte Carlo Pass Search

This directory turns the supplied MCPS research code into a single-GPU,
one-match experiment. It preserves the paper architectures and limits data,
epochs, and counterfactual samples.

References:

- MCPS paper: <https://arxiv.org/abs/2606.11120>
- IDSSE data paper: <https://doi.org/10.1038/s41597-025-04505-y>
- SMART paper: <https://arxiv.org/abs/2405.15677>

The IDSSE files are CC BY 4.0. Publications and shared artifacts must credit
the Deutsche Fußball Liga and cite the IDSSE paper.

## Colab

Open `mcps_small_colab.ipynb`, set `REPO_URL` to the Git URL containing this
implementation, select a GPU runtime, and run all cells. The default run uses
the first 30,000 tracking frames (about 20 minutes) from match `J03WN1`,
downloads about 392 MB, and targets a 1–2 hour Colab session.
Runtime depends on the GPU assigned by Colab and is recorded rather than
assumed.

## Command line

```bash
python -m pip install -r small_demo/requirements-colab.txt
python small_demo/run_pipeline.py all
```

Individual stages can be run separately:

```bash
python small_demo/run_pipeline.py prepare
python small_demo/run_pipeline.py train
python small_demo/run_pipeline.py search
python small_demo/run_pipeline.py report
```

Outputs are written to `artifacts/mcps_small/`:

- `logs/`: persistent console logs for each stage
- `checkpoints/`: best checkpoints for SMART, touch, BAT, PV, and restart PV
- `search/variants.csv`: one row per observed/local/global pass execution
- `search/clip_percentiles.csv`: per-pass execution-surplus results
- `figures/`: learning curves and counterfactual diagnostics
- `compute_report.json`: measured stage durations and GPU snapshots

The run is a pipeline and compute demonstration. A single-match subset is too
small for reliable player rankings or calibrated tactical conclusions.
