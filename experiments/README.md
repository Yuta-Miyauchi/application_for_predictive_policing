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

- `lapd_foothill_burglary_adaptive_etas_2012_trial`
- `chicago_districtXX_burglary_adaptive_etas_external`
- `philadelphia_burglary_hunchlab_like_history_only`

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

- `animate_adaptive_forecast.py`: GIF animation of smoothed forecast heatmaps and fading red event bursts.
- `run_all_gif_experiments.py`: batch runner that creates the same GIF and replay tables for all currently prepared datasets.

Metric runners are intentionally absent for now. Metric design will be added later when the evaluation criteria are clearer.

Current animation runs also write replay artifacts under each `results/<run_id>/tables/` directory:

- `animation_config.yml`
- `forecast_frames.csv`
- `forecast_cell_risk.parquet`
- `forecast_top_cells.csv`
- `observed_events.parquet`
