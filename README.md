# Predictive Policing Demo

Predictive Policing の場所型犯罪予測モデルを、公開データで再現・比較するための実験リポジトリです。

このリポジトリでは、データセット準備、アルゴリズム実装、実験実行、結果保存を分離します。特定のデータセットやモデルに閉じず、研究で見つけた複数のアルゴリズムを複数の公開データセットで検証できる構成を目指します。

## Directory Layout

```text
datas/
  Local prepared datasets and dataset-specific metadata. This folder is git-ignored.
models/
  Algorithm implementations independent from a specific dataset.
experiments/
  Experiment configs and scripts that combine one dataset with one model.
results/
  Local experiment outputs with GIF animations and replay tables.
README.md
  Repository overview.
PROGRESS.md
  Running log of prepared datasets, implemented models, experiments, and results.
```

## Basic Policy

- `datas` contains local dataset variants and is excluded from Git. Raw and processed data should stay local unless a separate data-versioning policy is chosen.
- `models` contains reusable algorithm implementations. Model code should not assume a specific city or source dataset.
- `experiments` selects a dataset and a model, then runs a reproducible evaluation.
- `results` stores outputs by run ID and version. Current visual artifacts are intentionally GIF-focused; replay parameters and forecast tables are retained.
- `PROGRESS.md` is updated whenever a dataset, model, experiment, or result is added.

## Initial Scope

The first planned reproduction target is a location-based, PredPol-like ETAS model based on public algorithmic descriptions, not a full reproduction of any commercial product.

The current dataset focus is LAPD legacy public crime data. Chicago and Philadelphia were removed from the local workspace for now.

This project does not create individual risk scores and does not claim that model accuracy alone evaluates policing effectiveness or social impact.

## Current Status

Prepared local dataset target:

- `lapd_legacy_2010_2024_all_crimes_grid300m_h168h`
  - combines LA City Open Data `Crime Data from 2010 to 2019` and `Crime Data from 2020 to 2024`
  - contains 3,061,145 processed events after deduplication, datetime/coordinate cleaning, and LAPD boundary assignment
  - includes all 21 LAPD areas, 143 detailed crime codes, and 10 coarse crime groups
  - preserves `crime_code`, `crime_type`, `crime_group`, and `area_id`
  - assigns events to 14,659 cells on a 300 m grid over paged and dissolved LAPD division boundaries
  - uses a 168 hour forecast horizon for weekly ETAS visualization

Implemented model utilities:

- `adaptive_etas`
- `marked_adaptive_etas`

Current experiment outputs:

- Previous ETAS results were removed.
- The current ETAS rerun uses a pooled 21-area LAPD legacy dataset and weekly forecast frames.
- Each run keeps the GIF plus replay tables needed to reproduce the animation and inspect forecast values.
- Metric design is deferred until a clearer evaluation strategy is chosen.

## Local Dataset Policy

`datas/` is ignored by Git. The local preparation scripts under `datas/` do the following:

- download the two LAPD legacy Socrata datasets in paged chunks
- preserve raw chunk parquet files locally
- clean occurrence dates, 24 hour occurrence times, coordinates, area IDs, and crime codes
- remove duplicate `DR_NO` rows, invalid datetimes, invalid `(0, 0)` or out-of-bounds coordinates, and points outside LAPD division boundaries
- fetch LAPD division boundaries from LA City GIS with ArcGIS paging, then dissolve the boundary fragments into LAPD areas
- create a 300 m grid clipped to the union of LAPD division boundaries
- spatially assign each event to a grid cell
- write processed events, grid, boundaries, metadata, and cleaning summaries

The processed data is intentionally not committed to GitHub.

## Reproduction Commands

Create the environment:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Prepare the full local LAPD legacy dataset with the local-only script under ignored `datas/`:

```bash
.venv/bin/python datas/lapd_full/prepare_lapd_legacy_full.py --overwrite
```

Run the current weekly ETAS experiment:

```bash
.venv/bin/python -u experiments/run_lapd_full_etas.py
```

The run writes local outputs under:

```text
results/ETAS_full/lapd_legacy_2010_2024_pooled_etas_weekly/
```

The ETAS run writes:

- `animations/M6_pooled_lapd_weekly_online_etas_fixed_theta_weekly_forecast_heatmap.gif`
- `tables/animation_config.yml`
- `tables/forecast_frames.csv`
- `tables/forecast_cell_risk.parquet`
- `tables/forecast_top_cells.csv`
- `tables/forecast_group_risk.csv`
- `tables/observed_events.parquet`

## Data Sources

- LA City Crime Data from 2010 to 2019: `https://data.lacity.org/resource/63jg-8b9z.json`
- LA City Crime Data from 2020 to 2024: `https://data.lacity.org/resource/2nrs-mtv8.json`
- LA City LAPD division boundaries: `https://maps.lacity.org/arcgis/rest/services/Mapping/Boundaries/MapServer/11`
