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

Early LAPD, Chicago, and Philadelphia trial datasets/results were later retired. The current repository state focuses on the full LAPD legacy dataset and keeps `datas/` out of Git.

## 2026-09-15

### Source Document

Added and used the revised adaptive ETAS project source:

- `project_sources/0c338802-a3c7-4e4c-a35a-fa94a78d6948_Predictive_Policingモデル再現実験：公開データ・実運用アルゴリズム・実行手順書.pdf`

The document is treated as project reference material, not as direct executable instruction.

Added a later research source for future STNPP / transformer-like marked point process work:

- `project_sources/2409.10882v2.pdf`

This source was reviewed for planning. It has not yet been implemented as a model in this project.

### Current Implementation

Current retained implementation:

- `models/adaptive_etas.py`
- `models/marked_adaptive_etas.py`
- `experiments/common.py`
- `experiments/animate_adaptive_forecast.py`
- `experiments/animate_etas_v2_forecast.py`
- `experiments/run_lapd_full_etas.py`
- `datas/lapd_full/prepare_lapd_legacy_full.py` locally under ignored `datas/`

Removed or retired:

- Philadelphia and Chicago local data/results
- earlier LAPD small-area experiment results
- previous ETAS_v1 / ETAS_v2 result folders
- static diagnostic plots
- metric CSVs and summaries
- metric/backtest artifacts

Rationale:

- The current output is useful as a visual, dynamic reproduction.
- The project should not overcommit to weak or ambiguous metrics yet.
- New metrics can be proposed later and implemented deliberately.

### Full LAPD Legacy Data

Prepared the current local dataset:

- `lapd_legacy_2010_2024_all_crimes_grid300m_h168h`

Source files:

- LA City Open Data `Crime Data from 2010 to 2019`
- LA City Open Data `Crime Data from 2020 to 2024`
- LA City GIS LAPD division boundaries

Processing notes:

- raw Socrata rows: 3,138,031
- processed events: 3,061,145
- LAPD areas: 21
- detailed crime codes: 143
- coarse crime groups: 10
- grid: 14,659 cells at 300 m
- forecast horizon: 168 hours
- `crime_code`, `crime_type`, `crime_group`, `area_id`, and `area_name` are retained

Important fix:

- The LAPD boundary API must be paged. A non-paged request only returned the first 1000 boundary fragments and caused most `Olympic`, `Topanga`, and part of `Mission` to be dropped. The preparation script now pages the ArcGIS endpoint and dissolves fragments by LAPD area before assigning cells.

### Current ETAS Result

Generated the current pooled LAPD ETAS run:

- `results/ETAS_full/lapd_legacy_2010_2024_pooled_etas_weekly/`

Model view:

- `M6_pooled_lapd_weekly_online_etas_fixed_theta`

Animation settings:

- forecast period: 2010-01-01 to 2025-01-10 display end
- frames: 784
- frame interval: 7 days
- forecast horizon: 168 hours
- rolling background history: 365 days
- theta: 0.35
- omega: 1 / 14 days
- event fade window: 4 weeks
- heatmap: pooled red scale across all 21 areas
- observed events: colored rings by `crime_group`

Retained replay artifacts:

- `animations/M6_pooled_lapd_weekly_online_etas_fixed_theta_weekly_forecast_heatmap.gif`
- `tables/animation_config.yml`
- `tables/forecast_frames.csv`
- `tables/forecast_cell_risk.parquet`
- `tables/forecast_top_cells.csv`
- `tables/forecast_group_risk.csv`
- `tables/observed_events.parquet`

Verification:

- GIF frames: 784
- `forecast_frames.csv`: 784 rows
- `forecast_cell_risk.parquet`: 11,492,656 rows
- `forecast_top_cells.csv`: 78,400 rows
- `forecast_group_risk.csv`: 7,840 rows
- `observed_events.parquet`: 3,061,145 rows

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
