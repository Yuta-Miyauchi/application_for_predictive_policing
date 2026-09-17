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
  vol3/
    Multiscale STNPP-GAT with uncertainty-weighted STMGNN-ZINB calibration.
  vol4/
    Multiscale STNPP-GAT with HCL-regularized ZINB calibration.
our_experiment/
  Root-level project-owned experiments and selected result GIFs.
  common/
    Shared LAPD target-crime data and GIF helpers.
  vol1/
    Experiment for our_model/vol1.
  vol2/
    Experiment for our_model/vol2.
  vol3/
    Experiment for our_model/vol3.
  vol4/
    Experiment for our_model/vol4.
ref_models/
  Reproductions of prior research.
  models/etas/
    ETAS reference implementation.
  models/stmgnn_zinb/
    STMGNN-ZINB uncertainty-aware graph model.
  models/hcl/
    Hawkes-enhanced spatial-temporal hypergraph contrastive model.
  experiments/etas/
    Mohler-style ETAS reproduction and selected result GIF.
  experiments/stmgnn_zinb/
    Paper-constrained STMGNN-ZINB experiment, metrics, and selected GIF.
  experiments/hcl/
    Paper-constrained HCL quantity experiment, metrics, and selected GIF.
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
- STMGNN-ZINB/HCL derived dataset: daily counts on 205 intersecting 3 km cells

## Current Models

- `ref_models/models/etas`: Mohler-style ETAS reference model.
- `ref_models/models/stmgnn_zinb`: DGCN + MTCN model with a zero-inflated negative binomial output.
- `ref_models/models/hcl`: Hypergraph encoding with HCL's Hawkes and correlation losses.
- `our_model/vol1`: STNPP-GAT mark-interaction point-process baseline.
- `our_model/vol2`: vol1 plus ETAS-style area-wise rolling `theta/omega` calibration for self-excitation.
- `our_model/vol3`: vol2 fine-grid allocation plus DGCN/MTCN ZINB count distributions on 3 km cells, fused according to predictive uncertainty.
- `our_model/vol4`: vol3's fine branch and fusion with an HCL-regularized 3 km ZINB count model.

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

Run the STMGNN-ZINB experiment:

```bash
.venv/bin/python -u ref_models/experiments/stmgnn_zinb/run_lapd_stmgnn_zinb.py
```

This command derives the git-ignored 3 km daily dataset, trains with the paper's
chronological 7:1 split and 30-day validation tail, evaluates ZINB uncertainty,
and creates the LAPD-wide GIF. Neural hyperparameters omitted by the paper are
recorded in `training_summary.yml` and [PROGRESS.md](PROGRESS.md).

Run the HCL quantity-prediction experiment:

```bash
.venv/bin/python -u ref_models/experiments/hcl/run_lapd_hcl.py
```

This uses the same 3 km daily LAPD tensor, a 30-day rolling history, and the
paper's HCL coefficients selected for NYC. The HCL-specific equations follow
the paper; omitted backbone details and the CPU-oriented training subsampling
are recorded in `training_summary.yml` and [PROGRESS.md](PROGRESS.md).

Run our STNPP-GAT vol1 experiment:

```bash
.venv/bin/python -u our_experiment/vol1/run_lapd_stnpp_gat.py
```

Run our ETAS-enhanced STNPP-GAT vol2 experiment:

```bash
.venv/bin/python -u our_experiment/vol2/run_lapd_etas_enhanced_stnpp_gat.py
```

Run our uncertainty-calibrated multiscale vol3 experiment:

```bash
.venv/bin/python -u our_experiment/vol3/run_lapd_stnpp_gat_zinb.py
```

Vol3 trains the 3 km daily ZINB branch on 2010-2019 data, retains vol2's
adaptive GAT/ETAS allocation at 150 m, and forecasts 2020 through November
2024. To regenerate allocation and the GIF from saved local weights without
retraining, add `--reuse-model-state`.

Run our HCL-regularized multiscale vol4 experiment:

```bash
.venv/bin/python -u our_experiment/vol4/run_lapd_stnpp_gat_hcl_zinb.py
```

Vol4 applies HCL's type, cardinal-neighbor, and representation-level Hawkes
correlations only to the 3 km ZINB branch. The 150 m GAT/ETAS allocation and
vol3's uncertainty-weighted fusion remain intact. Add `--reuse-model-state`
to regenerate the replay and GIF from the saved local weights.

Evaluate the current experiment outputs:

```bash
.venv/bin/python -u metrics/evaluate_current_results.py
```

Use `--model our_vol3` to evaluate only vol3.

Metric tables and plots are written into each model's result directory:

- `ref_models/experiments/etas/results/metrics/`
- `ref_models/experiments/stmgnn_zinb/results/metrics/` (written by its experiment script)
- `ref_models/experiments/hcl/results/metrics/` (written by its experiment script)
- `our_experiment/vol1/results/metrics/`
- `our_experiment/vol2/results/metrics/`
- `our_experiment/vol3/results/metrics/`
- `our_experiment/vol4/results/metrics/`

## Published GIFs

- `ref_models/experiments/etas/results/animations/M7_mohler_style_lapd_etas_24h_forecast_heatmap.gif`
- `ref_models/experiments/stmgnn_zinb/results/animations/U1_stmgnn_zinb_lapd_24h_forecast_heatmap.gif`
- `ref_models/experiments/hcl/results/animations/H1_hcl_lapd_quantity_24h_forecast_heatmap.gif`
- `our_experiment/vol1/results/animations/T2_stnpp_gat_marked_point_process_24h_forecast_heatmap.gif`
- `our_experiment/vol2/results/animations/O2_etas_enhanced_stnpp_gat_24h_forecast_heatmap.gif`
- `our_experiment/vol3/results/animations/O3_stnpp_gat_zinb_multiscale_24h_forecast_heatmap.gif`
- `our_experiment/vol4/results/animations/O4_stnpp_gat_hcl_zinb_multiscale_24h_forecast_heatmap.gif`

Replay tables and model states are retained locally but ignored by Git.

## Data Sources

- LA City Crime Data from 2010 to 2019: `https://data.lacity.org/resource/63jg-8b9z.json`
- LA City Crime Data from 2020 to 2024: `https://data.lacity.org/resource/2nrs-mtv8.json`
- LA City LAPD division boundaries: `https://maps.lacity.org/arcgis/rest/services/Mapping/Boundaries/MapServer/11`
