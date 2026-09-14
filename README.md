# Predictive Policing Demo

Predictive Policing の場所型犯罪予測モデルを、公開データで再現・比較するための実験リポジトリです。

このリポジトリでは、データセット準備、アルゴリズム実装、実験実行、結果保存を分離します。特定のデータセットやモデルに閉じず、研究で見つけた複数のアルゴリズムを複数の公開データセットで検証できる構成を目指します。

## Directory Layout

```text
datas/
  Prepared datasets and dataset-specific metadata.
models/
  Algorithm implementations independent from a specific dataset.
experiments/
  Experiment configs and scripts that combine one dataset with one model.
results/
  Experiment outputs. The current retained visual result is a GIF animation, with replay tables kept for reproducibility.
README.md
  Repository overview.
PROGRESS.md
  Running log of prepared datasets, implemented models, experiments, and results.
```

## Basic Policy

- `datas` contains dataset variants separately. LAPD variants, Chicago variants, Philadelphia variants, and future datasets should not be merged into one master table by default.
- `models` contains reusable algorithm implementations. Model code should not assume a specific city or source dataset.
- `experiments` selects a dataset and a model, then runs a reproducible evaluation.
- `results` stores outputs by run ID. Current visual artifacts are intentionally GIF-focused; replay parameters and forecast tables are retained.
- `PROGRESS.md` is updated whenever a dataset, model, experiment, or result is added.

## Initial Scope

The first planned reproduction target is a location-based, PredPol-like ETAS model based on public algorithmic descriptions, not a full reproduction of any commercial product.

The initial dataset candidates are:

- LAPD public crime data, with multiple variants by division, crime type, period, grid size, and forecast horizon.
- Chicago public crime data for external reproduction.
- Philadelphia public crime incidents for a later HunchLab-like experiment.

This project does not create individual risk scores and does not claim that model accuracy alone evaluates policing effectiveness or social impact.

## Current Status

Prepared dataset variants:

- 6 LAPD variants:
  - Foothill / N Hollywood / Southwest
  - BURGLARY / VEHICLE - STOLEN
  - 2011-01-01 to 2013-01-11
  - 150 m grid, 24 hour forecast horizon
- 1 Chicago variant:
  - District 011 / BURGLARY
  - 2011-01-01 to 2013-01-11
  - 150 m grid, 24 hour forecast horizon

Implemented model utilities:

- `adaptive_etas`

Current experiment outputs:

- Adaptive forecast GIFs have been generated for all 7 prepared dataset variants.
- Each run keeps the GIF plus replay tables needed to reproduce the animation and inspect the forecast values.
- Metric design is deferred until a clearer evaluation strategy is chosen.

## Reproduction Commands

Create the environment:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Prepare LAPD variants:

```bash
.venv/bin/python datas/lapd/prepare_lapd_variants.py --overwrite
```

Prepare the first Chicago variant:

```bash
.venv/bin/python datas/chicago/prepare_chicago_variants.py --overwrite
```

Create adaptive forecast GIFs for all prepared datasets:

```bash
.venv/bin/python -u experiments/run_all_gif_experiments.py
```

Create one adaptive forecast GIF:

```bash
.venv/bin/python -u experiments/animate_adaptive_forecast.py \
  --dataset-id lapd_foothill_burglary_2011_2013_grid150m_h24h \
  --run-id lapd_foothill_burglary_2011_2013_grid150m_h24h_adaptive_etas_gif_2012_trial \
  --model-id M4_adaptive_etas_weekly_theta_floor \
  --start 2012-05-16 \
  --end 2013-01-10 \
  --history-days 365 \
  --horizon-hours 24 \
  --top-k 20 \
  --refit-days 7 \
  --fixed-theta 0.35 \
  --fixed-omega 0.07142857142857142 \
  --theta-floor 0.05 \
  --fade-days 14 \
  --pop-days 1.5 \
  --smooth-sigma 2.4 \
  --frame-step 1 \
  --duration-ms 90 \
  --dpi 90
```

Each run writes:

- `animations/M4_adaptive_etas_weekly_theta_floor_forecast_heatmap.gif`
- `tables/animation_config.yml`
- `tables/forecast_frames.csv`
- `tables/forecast_cell_risk.parquet`
- `tables/forecast_top_cells.csv`
- `tables/observed_events.parquet`

## Data Sources

- LA City Open Data: `https://data.lacity.org/resource/63jg-8b9z.json`
- LA City LAPD division boundaries: `https://maps.lacity.org/arcgis/rest/services/Mapping/Boundaries/MapServer/11`
- City of Chicago crimes: `https://data.cityofchicago.org/resource/ijzp-q8t2.json`
- City of Chicago police district boundaries: `https://data.cityofchicago.org/resource/9vmg-9p8p.geojson`
