# experiments

Experiments combine one prepared dataset from `datas/` with one model from `models/`.

Experiment files should describe the combination, parameters, backtest period, and output location. They should not contain raw data preparation logic or reusable model logic.

## Intended Pattern

```text
experiments/
  <experiment_id>/
    config.yml
    run.py
    README.md
```

## Experiment ID Convention

Use IDs that make the dataset and model clear.

Examples:

- `lapd_legacy_2010_2024_pooled_etas_weekly`
- `lapd_legacy_2010_2024_transformer_weekly`

## Required Experiment Metadata

- dataset ID
- model ID
- training/history window
- forecast horizon
- backtest period
- top-k or area budget
- planned metric list, if a metric experiment is being run
- random seed, if applicable
- output directory under `results/`

## Current Runners

- `run_lapd_full_etas.py`: current pooled 21-area LAPD weekly ETAS runner. It writes one GIF plus replay tables for the full 2010-2024 legacy dataset.
- `run_lapd_transformer_v1.py`: current PyTorch Transformer v1 runner. It trains on 2010-2019 weekly cell sequences and forecasts 2020-2024 weekly LAPD risk.
- `animate_adaptive_forecast.py`: earlier GIF helper for daily adaptive ETAS animations on prepared small datasets.
- `animate_etas_v2_forecast.py`: earlier marked ETAS v2 GIF runner for the three-division exploratory setup.

Metric runners are intentionally absent for now. Metric design will be added later when the evaluation criteria are clearer.

Current animation runs also write replay artifacts under each `results/<run_id>/tables/` directory:

- `animation_config.yml`
- `forecast_frames.csv`
- `forecast_cell_risk.parquet`
- `forecast_top_cells.csv`
- `forecast_group_risk.csv` for the current pooled LAPD run
- `training_summary.yml` and `model/model_state.pt` for Transformer v1 runs
- `forecast_cell_crime_risk.parquet` for earlier marked ETAS v2 runs
- `observed_events.parquet`
