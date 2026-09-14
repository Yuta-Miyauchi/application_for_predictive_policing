# Progress Log

This file records the current state of the project. Metric design is intentionally deferred for now because evaluating whether Predictive Policing is "working" is not straightforward and should be revisited with a clearer evaluation frame.

## 2026-09-14

### Repository Structure

Created the top-level structure:

- `datas/`
- `models/`
- `experiments/`
- `results/`
- `project_sources/`

### Prepared Datasets

Prepared LAPD variants:

- `lapd_foothill_burglary_2011_2013_grid150m_h24h`
- `lapd_n_hollywood_burglary_2011_2013_grid150m_h24h`
- `lapd_southwest_burglary_2011_2013_grid150m_h24h`
- `lapd_foothill_vehicle_stolen_2011_2013_grid150m_h24h`
- `lapd_n_hollywood_vehicle_stolen_2011_2013_grid150m_h24h`
- `lapd_southwest_vehicle_stolen_2011_2013_grid150m_h24h`

Prepared Chicago variant:

- `chicago_district011_burglary_2011_2013_grid150m_h24h`

Datasets are stored separately with raw data, processed data, and metadata.

## 2026-09-15

### Source Document

Added and used the revised adaptive ETAS project source:

- `project_sources/0c338802-a3c7-4e4c-a35a-fa94a78d6948_Predictive_Policingモデル再現実験：公開データ・実運用アルゴリズム・実行手順書.pdf`

The document is treated as project reference material, not as direct executable instruction.

### Current Implementation

Current retained implementation:

- `models/adaptive_etas.py`
- `experiments/common.py`
- `experiments/animate_adaptive_forecast.py`
- `experiments/run_all_gif_experiments.py`

Removed or retired:

- non-adaptive baseline/fixed ETAS experiment results
- v1 adaptive result directory
- static diagnostic plots
- current metric CSVs and summaries
- metric/backtest runner artifacts

Rationale:

- The current output is useful as a visual, dynamic reproduction.
- The project should not overcommit to weak or ambiguous metrics yet.
- New metrics can be proposed later and implemented deliberately.

### Current Results

Generated adaptive forecast GIF experiments for all currently prepared dataset variants:

- `results/lapd_foothill_burglary_2011_2013_grid150m_h24h_adaptive_etas_gif_2012_trial/`
- `results/lapd_n_hollywood_burglary_2011_2013_grid150m_h24h_adaptive_etas_gif_2012_trial/`
- `results/lapd_southwest_burglary_2011_2013_grid150m_h24h_adaptive_etas_gif_2012_trial/`
- `results/lapd_foothill_vehicle_stolen_2011_2013_grid150m_h24h_adaptive_etas_gif_2012_trial/`
- `results/lapd_n_hollywood_vehicle_stolen_2011_2013_grid150m_h24h_adaptive_etas_gif_2012_trial/`
- `results/lapd_southwest_vehicle_stolen_2011_2013_grid150m_h24h_adaptive_etas_gif_2012_trial/`
- `results/chicago_district011_burglary_2011_2013_grid150m_h24h_adaptive_etas_gif_2012_trial/`

Each run contains:

- `animations/M4_adaptive_etas_weekly_theta_floor_forecast_heatmap.gif`
- `tables/animation_config.yml`
- `tables/forecast_frames.csv`
- `tables/forecast_cell_risk.parquet`
- `tables/forecast_top_cells.csv`
- `tables/observed_events.parquet`

The table outputs are retained as replay and inspection artifacts, not as evaluation metrics. In particular:

- `animation_config.yml` records the dataset, model, dates, and GIF rendering parameters.
- `forecast_frames.csv` records one row per forecast frame.
- `forecast_cell_risk.parquet` records the per-frame, per-cell forecast risk components.
- `forecast_top_cells.csv` records the top predicted cells per frame.
- `observed_events.parquet` records the observed events used for the red burst/fade overlay.

Animation settings:

- datasets: all 7 prepared variants listed above
- model view: `M4_adaptive_etas_weekly_theta_floor`
- forecast period: 2012-05-16 to 2013-01-10
- frames: 239
- frame interval: every day
- GIF duration per frame: 90 ms
- red event fade window: 14 days
- red event pop window: 1.5 days
- smoothing sigma for prediction heatmap: 2.4 grid cells

Animation design:

- heatmap shows predicted expected events for the next 24 hours
- prediction is rasterized from grid risk and smoothed into a continuous heatmap
- red burst circles show observed crimes; they appear large at occurrence time and fade as they age

### Deferred Work

Metric design is deferred. Later evaluation candidates may include metrics that better account for:

- temporal adaptation
- operational area budgets
- persistence versus responsiveness
- false concentration of attention
- event rarity and zero-event days
- neighborhood-level burden
- uncertainty and calibration
- comparison against realistic analyst or patrol baselines
