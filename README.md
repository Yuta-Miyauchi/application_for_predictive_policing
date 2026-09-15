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
  - remains the source dataset for experiment-specific derived grids and target-crime subsets

Current experiment-derived local dataset:

- `lapd_legacy_2010_2024_mohler_target_crimes_grid150m_h24h`
  - derived from the full LAPD legacy dataset under ignored `datas/`
  - keeps Mohler et al.'s Los Angeles target-crime families: burglary, car theft, and theft from vehicle
  - contains 919,284 target-crime events
  - assigns target-crime events to 150 m cells
  - includes 56,927 cells across all 21 LAPD areas
  - keeps `target_crime`, original LAPD crime labels, and cell-based LAPD area assignment
  - uses a 24 hour forecast horizon for the current ETAS and STNPP-GAT GIFs

Implemented model utilities:

- `adaptive_etas`
- `marked_adaptive_etas`
- `stnpp_gat`
- `temporal_attention_transformer`（旧v1スキャフォールド、現行結果では未使用）

Current experiment outputs:

- The current ETAS rerun is a Mohler-style target-crime experiment using 150 m cells, 365 day history, 24 hour horizon, and division-wise ETAS parameter refits.
- The current Transformer/STNPP rerun replaces the earlier weekly `TransformerEncoder` baseline with an STNPP-GAT-style marked point process over crime-type x LAPD-area marks.
- Both current runs forecast 2020-2024 target-crime risk and publish one all-area GIF each.
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

Run the current Mohler-style ETAS experiment:

```bash
.venv/bin/python -u experiments/run_lapd_mohler_etas.py
```

Run the current STNPP-GAT-style experiment:

```bash
.venv/bin/python -u experiments/run_lapd_stnpp_gat.py
```

The run writes local outputs under:

```text
results/ETAS_mohler/lapd_legacy_2020_2024_mohler_etas_target_150m/
```

The ETAS run writes:

- `animations/M7_mohler_style_lapd_etas_24h_forecast_heatmap.gif`
- `tables/animation_config.yml`
- `tables/forecast_frames.csv`
- `tables/forecast_cell_risk.parquet`
- `tables/forecast_top_cells.csv`
- `tables/etas_area_parameters.csv`
- `tables/observed_events.parquet`

The STNPP-GAT run writes local outputs under:

```text
results/STNPP_GAT/lapd_legacy_2020_2024_stnpp_gat_target_150m/
```

The STNPP-GAT run writes:

- `animations/T2_stnpp_gat_marked_point_process_24h_forecast_heatmap.gif`
- `model/model_state.pt`
- `tables/animation_config.yml`
- `tables/training_summary.yml`
- `tables/forecast_frames.csv`
- `tables/forecast_cell_risk.parquet`
- `tables/forecast_top_cells.csv`
- `tables/learned_mark_transition.csv`
- `tables/observed_events.parquet`

## Data Sources

- LA City Crime Data from 2010 to 2019: `https://data.lacity.org/resource/63jg-8b9z.json`
- LA City Crime Data from 2020 to 2024: `https://data.lacity.org/resource/2nrs-mtv8.json`
- LA City LAPD division boundaries: `https://maps.lacity.org/arcgis/rest/services/Mapping/Boundaries/MapServer/11`
