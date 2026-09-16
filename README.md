# Predictive Policing Demo

Predictive Policing の既存研究を公開データで再現しつつ、STNPP-GATをベースにした独自モデルを改良していくための実験リポジトリです。

このリポジトリでは、先行研究の再現モデルと、自分たちの改良モデルを分けて管理します。詳細な研究ログ、データ仕様、モデル仕様、実験条件、限界は [PROGRESS.md](PROGRESS.md) に統合しています。

## Directory Layout

```text
datas/
  Local prepared datasets. This folder is git-ignored.
metrics/
  Evaluation scripts. Outputs are saved under each model's results/metrics/.
project_sources/
  Papers and reference documents.
our_model/
  Root-level project-owned model versions.
  vol1/
    STNPP-GAT baseline.
  vol2/
    STNPP-GAT with ETAS-calibrated area-wise triggering.
our_experiment/
  Root-level project-owned experiments and selected result GIFs.
  common/
    Shared LAPD target-crime data and GIF helpers.
  vol1/
    Experiment for our_model/vol1.
  vol2/
    Experiment for our_model/vol2.
ref_models/
  Reproductions of prior research.
  models/etas/
    ETAS reference implementation.
  experiments/etas/
    Mohler-style ETAS reproduction and selected result GIF.
PROGRESS.md
README.md
```

## Current Data

Local datasets are not committed to Git.

- Source dataset: `lapd_legacy_2010_2024_all_crimes_grid300m_h168h`
- Current experiment dataset: `lapd_legacy_2010_2024_mohler_target_crimes_grid150m_h24h`
- Target crimes: `BURGLARY`, `CAR_THEFT`, `THEFT_FROM_VEHICLE`
- Target-crime events: 919,284
- Grid: 150 m, 56,927 cells, all 21 LAPD areas

## Current Models

- `ref_models/models/etas`: Mohler-style ETAS reference model.
- `our_model/vol1`: STNPP-GAT mark-interaction point-process baseline.
- `our_model/vol2`: vol1 plus ETAS-style area-wise rolling `theta/omega` calibration for self-excitation.

## Reproduction Commands

Create the environment:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Prepare the local LAPD source dataset:

```bash
.venv/bin/python datas/lapd_full/prepare_lapd_legacy_full.py --overwrite
```

Run the reference ETAS experiment:

```bash
.venv/bin/python -u ref_models/experiments/etas/run_lapd_mohler_etas.py
```

Run our STNPP-GAT vol1 experiment:

```bash
.venv/bin/python -u our_experiment/vol1/run_lapd_stnpp_gat.py
```

Run our ETAS-enhanced STNPP-GAT vol2 experiment:

```bash
.venv/bin/python -u our_experiment/vol2/run_lapd_etas_enhanced_stnpp_gat.py
```

Evaluate the current experiment outputs:

```bash
.venv/bin/python -u metrics/evaluate_current_results.py
```

Metric tables and plots are written into each model's result directory:

- `ref_models/experiments/etas/results/metrics/`
- `our_experiment/vol1/results/metrics/`
- `our_experiment/vol2/results/metrics/`

## Published GIFs

- `ref_models/experiments/etas/results/animations/M7_mohler_style_lapd_etas_24h_forecast_heatmap.gif`
- `our_experiment/vol1/results/animations/T2_stnpp_gat_marked_point_process_24h_forecast_heatmap.gif`
- `our_experiment/vol2/results/animations/O2_etas_enhanced_stnpp_gat_24h_forecast_heatmap.gif`

Replay tables and model states are retained locally but ignored by Git.

## Data Sources

- LA City Crime Data from 2010 to 2019: `https://data.lacity.org/resource/63jg-8b9z.json`
- LA City Crime Data from 2020 to 2024: `https://data.lacity.org/resource/2nrs-mtv8.json`
- LA City LAPD division boundaries: `https://maps.lacity.org/arcgis/rest/services/Mapping/Boundaries/MapServer/11`
