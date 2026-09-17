"""Run a paper-constrained HCL quantity-prediction experiment on LAPD data.

The experiment follows Liang et al. (AAAI 2024) for the 3 km grid, daily
multivariate crime tensor, 7:1 chronological split, 30-day validation tail,
representation-level Hawkes enhancement, and two correlation losses. The
paper's HCL implementation is unavailable and some backbone details are not
specified, so the experiment uses the compact structural hypergraph backbone
implemented in ``ref_models.models.hcl`` and records every substituted choice.
"""

from __future__ import annotations

import argparse
import copy
import math
import os
import random
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from our_experiment.common.lapd_target_common import (
    TARGET_CRIME_ORDER,
    load_target_dataset,
    observed_window_counts,
    save_forecast_animation,
    write_forecast_tables,
)
from ref_models.experiments.stmgnn_zinb.run_lapd_stmgnn_zinb import (
    DATASET_ID,
    DailyCrimeTensor,
    build_coarse_dataset,
    make_daily_tensor,
    split_indices,
    standardize_features,
)
from ref_models.models.hcl import (
    HCLConfig,
    HawkesContrastiveHypergraph,
    hcl_quantity_loss,
)


MODEL_ID = "H1_hcl_lapd_quantity"
RESULT_DIR = ROOT / "ref_models" / "experiments" / "hcl" / "results"


class HCLDailyWindowDataset(Dataset):
    def __init__(
        self,
        features: np.ndarray,
        counts: np.ndarray,
        target_indices: np.ndarray,
        input_steps: int,
    ):
        self.features = features
        self.counts = counts
        self.target_indices = np.asarray(target_indices, dtype=np.int64)
        self.input_steps = int(input_steps)

    def __len__(self) -> int:
        return len(self.target_indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        target = int(self.target_indices[index])
        history_slice = slice(target - self.input_steps, target)
        features = self.features[history_slice].transpose(1, 0, 2)
        raw_history = self.counts[history_slice].transpose(1, 0, 2)
        target_counts = self.counts[target]
        return (
            torch.from_numpy(np.ascontiguousarray(features, dtype=np.float32)),
            torch.from_numpy(np.ascontiguousarray(raw_history, dtype=np.float32)),
            torch.from_numpy(np.ascontiguousarray(target_counts, dtype=np.float32)),
        )


def build_cardinal_graph(
    grid: gpd.GeoDataFrame,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    by_position = {
        (int(row), int(col)): idx
        for idx, row, col in grid[["row", "col"]].itertuples(index=True, name=None)
    }
    source: list[int] = []
    target: list[int] = []
    neighbor_center: list[int] = []
    neighbor_node: list[int] = []
    cardinal_offsets = ((-1, 0), (1, 0), (0, -1), (0, 1))
    for center, row, col in grid[["row", "col"]].itertuples(index=True, name=None):
        source.append(int(center))
        target.append(int(center))
        for delta_row, delta_col in cardinal_offsets:
            neighbor = by_position.get((int(row) + delta_row, int(col) + delta_col))
            if neighbor is None:
                continue
            source.append(int(neighbor))
            target.append(int(center))
            neighbor_center.append(int(center))
            neighbor_node.append(int(neighbor))

    source_array = np.asarray(source, dtype=np.int64)
    target_array = np.asarray(target, dtype=np.int64)
    degree = np.bincount(target_array, minlength=len(grid)).astype(np.float32)
    edge_weight = 1.0 / degree[target_array]
    return (
        torch.from_numpy(np.vstack([source_array, target_array])),
        torch.from_numpy(edge_weight),
        torch.tensor(neighbor_center, dtype=torch.long),
        torch.tensor(neighbor_node, dtype=torch.long),
        {
            "nodes": int(len(grid)),
            "backbone_edges_including_self_loops": int(len(source_array)),
            "directed_cardinal_neighbor_pairs": int(len(neighbor_center)),
            "minimum_hyperedge_degree": int(degree.min()),
            "maximum_hyperedge_degree": int(degree.max()),
            "neighbor_definition": "four cardinal neighbors: N +/- 1 and N +/- n_cols",
        },
    )


def category_coefficients(
    counts: np.ndarray,
    training_day_stop: int,
    neighbor_center: torch.Tensor,
    neighbor_node: torch.Tensor,
) -> np.ndarray:
    presence = counts[:training_day_stop] > 0
    neighbor_presence = np.zeros_like(presence, dtype=bool)
    centers = neighbor_center.numpy()
    neighbors = neighbor_node.numpy()
    for center, neighbor in zip(centers, neighbors, strict=True):
        neighbor_presence[:, center] |= presence[:, neighbor]
    numerator = np.sum(presence & neighbor_presence, axis=(0, 1), dtype=np.float64)
    denominator = np.sum(presence, axis=(0, 1), dtype=np.float64)
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float64),
        where=denominator > 0,
    ).astype(np.float32)


def make_loader(
    features: np.ndarray,
    counts: np.ndarray,
    target_indices: np.ndarray,
    args: argparse.Namespace,
    shuffle: bool,
) -> DataLoader:
    dataset = HCLDailyWindowDataset(
        features=features,
        counts=counts,
        target_indices=target_indices,
        input_steps=args.input_days,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=0,
        generator=torch.Generator().manual_seed(args.seed) if shuffle else None,
    )


def loader_losses(
    model: HawkesContrastiveHypergraph,
    loader: DataLoader,
    neighbor_center: torch.Tensor,
    neighbor_node: torch.Tensor,
    category_coefficient: torch.Tensor,
    args: argparse.Namespace,
) -> dict[str, float]:
    model.eval()
    totals = {name: 0.0 for name in ("total", "task", "type", "neighbor")}
    samples = 0
    device = next(model.parameters()).device
    with torch.no_grad():
        for features, raw_history, target in loader:
            features = features.to(device)
            raw_history = raw_history.to(device)
            target = target.to(device)
            loss = hcl_quantity_loss(
                model(features),
                target,
                raw_history,
                neighbor_center,
                neighbor_node,
                category_coefficient,
                lambda_type=args.lambda_type,
                lambda_neighbor=args.lambda_neighbor,
                contrast_steps=args.contrast_steps,
            )
            batch = int(features.shape[0])
            totals["total"] += float(loss.total.cpu()) * batch
            totals["task"] += float(loss.task.cpu()) * batch
            totals["type"] += float(loss.type_contrast.cpu()) * batch
            totals["neighbor"] += float(loss.neighbor_contrast.cpu()) * batch
            samples += batch
    return {name: value / max(samples, 1) for name, value in totals.items()}


def train_model(
    config: HCLConfig,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    neighbor_center_cpu: torch.Tensor,
    neighbor_node_cpu: torch.Tensor,
    category_coefficient_array: np.ndarray,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    args: argparse.Namespace,
) -> tuple[HawkesContrastiveHypergraph, pd.DataFrame, dict]:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.torch_threads)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = HawkesContrastiveHypergraph(config, edge_index, edge_weight).to(device)
    neighbor_center = neighbor_center_cpu.to(device)
    neighbor_node = neighbor_node_cpu.to(device)
    category_coefficient = torch.from_numpy(category_coefficient_array).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    best_state = copy.deepcopy(model.state_dict())
    best_validation = math.inf
    best_epoch = 0
    stale_epochs = 0
    history: list[dict] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        totals = {name: 0.0 for name in ("total", "task", "type", "neighbor")}
        samples = 0
        for features, raw_history, target in train_loader:
            features = features.to(device)
            raw_history = raw_history.to(device)
            target = target.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = hcl_quantity_loss(
                model(features),
                target,
                raw_history,
                neighbor_center,
                neighbor_node,
                category_coefficient,
                lambda_type=args.lambda_type,
                lambda_neighbor=args.lambda_neighbor,
                contrast_steps=args.contrast_steps,
            )
            if not torch.isfinite(loss.total):
                raise RuntimeError(f"Non-finite HCL loss at epoch {epoch}")
            loss.total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            batch = int(features.shape[0])
            totals["total"] += float(loss.total.detach().cpu()) * batch
            totals["task"] += float(loss.task.detach().cpu()) * batch
            totals["type"] += float(loss.type_contrast.detach().cpu()) * batch
            totals["neighbor"] += float(loss.neighbor_contrast.detach().cpu()) * batch
            samples += batch

        train_values = {name: value / max(samples, 1) for name, value in totals.items()}
        validation_values = loader_losses(
            model,
            validation_loader,
            neighbor_center,
            neighbor_node,
            category_coefficient,
            args,
        )
        history.append(
            {
                "epoch": epoch,
                **{f"train_{name}": value for name, value in train_values.items()},
                **{f"validation_{name}": value for name, value in validation_values.items()},
            }
        )
        improved = validation_values["total"] < best_validation - args.min_delta
        if improved:
            best_validation = validation_values["total"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        if epoch == 1 or epoch % args.log_every == 0 or improved:
            print(
                f"HCL epoch {epoch}/{args.epochs}: "
                f"train={train_values['total']:.6f} "
                f"val={validation_values['total']:.6f} "
                f"task={validation_values['task']:.6f} "
                f"type={validation_values['type']:.6f} "
                f"neighbor={validation_values['neighbor']:.6f}",
                flush=True,
            )
        if stale_epochs >= args.patience:
            print(f"Early stopping at epoch {epoch}; best epoch was {best_epoch}", flush=True)
            break

    model.load_state_dict(best_state)
    summary = {
        "device": str(device),
        "torch_version": str(torch.__version__),
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "epochs_requested": int(args.epochs),
        "epochs_completed": int(len(history)),
        "best_epoch": int(best_epoch),
        "best_validation_total_loss": float(best_validation),
        "early_stopping_patience": int(args.patience),
    }
    return model, pd.DataFrame(history), summary


def predict(
    model: HawkesContrastiveHypergraph,
    loader: DataLoader,
) -> tuple[np.ndarray, np.ndarray]:
    device = next(model.parameters()).device
    observed: list[np.ndarray] = []
    predicted: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for features, _, target in loader:
            output = model(features.to(device))
            observed.append(target.numpy())
            predicted.append(output.prediction.cpu().numpy())
    return np.concatenate(observed), np.concatenate(predicted)


def evaluate_predictions(
    observed: np.ndarray,
    predicted: np.ndarray,
    crimes: pd.Index,
    dates: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    scopes: list[tuple[str, np.ndarray | None]] = [("ALL", None)] + [
        (str(crime), np.asarray([index], dtype=np.int64))
        for index, crime in enumerate(crimes)
    ]
    for label, crime_indices in scopes:
        y = observed if crime_indices is None else observed[..., crime_indices]
        y_hat = predicted if crime_indices is None else predicted[..., crime_indices]
        positive = y > 0
        metrics = {
            "mae": float(np.mean(np.abs(y - y_hat))),
            "mape_positive": (
                float(np.mean(np.abs(y[positive] - y_hat[positive]) / y[positive]))
                if positive.any()
                else np.nan
            ),
            "rmse": float(np.sqrt(np.mean(np.square(y - y_hat)))),
            "mean_observed_count": float(np.mean(y)),
            "mean_predicted_count": float(np.mean(y_hat)),
        }
        for metric, value in metrics.items():
            rows.append({"scope": label, "metric": metric, "value": value})

    daily_rows = []
    for index, date in enumerate(dates):
        y = observed[index]
        y_hat = predicted[index]
        positive = y > 0
        daily_rows.append(
            {
                "forecast_date": date.date().isoformat(),
                "mae": float(np.mean(np.abs(y - y_hat))),
                "mape_positive": (
                    float(np.mean(np.abs(y[positive] - y_hat[positive]) / y[positive]))
                    if positive.any()
                    else np.nan
                ),
                "observed_events": float(y.sum()),
                "predicted_events": float(y_hat.sum()),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(daily_rows)


def save_prediction_table(
    output_path: Path,
    observed: np.ndarray,
    predicted: np.ndarray,
    dates: pd.DatetimeIndex,
    cells: pd.Index,
    crimes: pd.Index,
) -> None:
    n_days, n_cells, n_crimes = predicted.shape
    frame = pd.DataFrame(
        {
            "forecast_date": np.repeat(
                dates.to_numpy(dtype="datetime64[D]"), n_cells * n_crimes
            ),
            "cell_id": np.tile(np.repeat(cells.to_numpy(), n_crimes), n_days),
            "target_crime": np.tile(crimes.to_numpy(), n_days * n_cells),
            "observed_count": observed.reshape(-1),
            "predicted_count": predicted.reshape(-1),
        }
    )
    frame.to_parquet(output_path, index=False)


def save_metric_plots(
    metric_dir: Path,
    summary: pd.DataFrame,
    daily: pd.DataFrame,
    training: pd.DataFrame,
) -> None:
    plot_dir = metric_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    axes[0].plot(training["epoch"], training["train_total"], label="Train total")
    axes[0].plot(
        training["epoch"], training["validation_total"], label="Validation total"
    )
    axes[0].set_title("HCL total objective")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    for column, label in (
        ("validation_task", "Task MSE"),
        ("validation_type", "Type alignment"),
        ("validation_neighbor", "Neighbor alignment"),
    ):
        axes[1].plot(training["epoch"], training[column], label=label)
    axes[1].set_title("Validation objective components")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    fig.savefig(plot_dir / "training_history.png", dpi=170)
    plt.close(fig)

    crime_summary = summary[summary["scope"] != "ALL"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for axis, metric, title, color in (
        (axes[0], "mae", "MAE by crime", "#2b8cbe"),
        (axes[1], "mape_positive", "MAPE on positive counts", "#e34a33"),
    ):
        values = crime_summary[crime_summary["metric"] == metric]
        axis.bar(values["scope"], values["value"], color=color)
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=20)
        axis.grid(axis="y", alpha=0.25)
    fig.savefig(plot_dir / "paper_metrics_by_crime.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, constrained_layout=True)
    dates = pd.to_datetime(daily["forecast_date"])
    axes[0].plot(dates, daily["observed_events"], label="Observed", color="#252525")
    axes[0].plot(
        dates, daily["predicted_events"], label="Predicted", color="#de2d26"
    )
    axes[0].set_ylabel("Daily events")
    axes[0].set_title("HCL rolling one-day-ahead forecast")
    axes[0].grid(alpha=0.2)
    axes[0].legend()
    axes[1].plot(dates, daily["mae"], color="#2b8cbe")
    axes[1].set_ylabel("Daily MAE")
    axes[1].grid(alpha=0.2)
    fig.savefig(plot_dir / "daily_forecast_diagnostics.png", dpi=170)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell-size-m", type=int, default=3000)
    parser.add_argument("--input-days", type=int, default=30)
    parser.add_argument("--validation-days", type=int, default=30)
    parser.add_argument("--data-end-exclusive", default="2024-12-01")
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--hypergraph-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--hawkes-scope", type=int, default=3)
    parser.add_argument("--hawkes-delta", type=float, default=0.01)
    parser.add_argument("--lambda-type", type=float, default=0.15)
    parser.add_argument("--lambda-neighbor", type=float, default=0.15)
    parser.add_argument("--contrast-steps", type=int, default=1)
    parser.add_argument("--train-stride", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--frame-days", type=int, default=14)
    parser.add_argument("--horizon-hours", type=int, default=24)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--fade-days", type=int, default=28)
    parser.add_argument("--pop-days", type=float, default=3.0)
    parser.add_argument("--smooth-sigma", type=float, default=0.75)
    parser.add_argument("--vmax-quantile", type=float, default=0.985)
    parser.add_argument("--heatmap-gamma", type=float, default=0.48)
    parser.add_argument("--duration-ms", type=int, default=115)
    parser.add_argument("--dpi", type=int, default=82)
    parser.add_argument("--event-location-mode", choices=("event", "cell"), default="event")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--device")
    parser.add_argument("--overwrite-derived-data", action="store_true")
    parser.add_argument("--skip-gif", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.cell_size_m != 3000:
        raise ValueError("The HCL paper experiment uses 3000 m cells")
    if args.train_stride < 1:
        raise ValueError("train_stride must be positive")
    args.history_days = args.input_days

    base = load_target_dataset(overwrite=False, cell_size_m=150)
    dataset = build_coarse_dataset(base, args.cell_size_m, args.overwrite_derived_data)
    daily: DailyCrimeTensor = make_daily_tensor(dataset, args.data_end_exclusive)
    train_all, validation_indices, test_indices, split_index = split_indices(
        n_days=len(daily.dates),
        input_steps=args.input_days,
        output_steps=1,
        validation_days=args.validation_days,
    )
    train_indices = train_all[:: args.train_stride]
    validation_start = split_index - args.validation_days
    features, feature_mean, feature_std = standardize_features(
        daily.counts, training_day_stop=validation_start
    )
    train_loader = make_loader(features, daily.counts, train_indices, args, shuffle=True)
    validation_loader = make_loader(
        features, daily.counts, validation_indices, args, shuffle=False
    )
    test_loader = make_loader(features, daily.counts, test_indices, args, shuffle=False)
    (
        edge_index,
        edge_weight,
        neighbor_center,
        neighbor_node,
        graph_summary,
    ) = build_cardinal_graph(dataset.grid)
    coefficient = category_coefficients(
        daily.counts,
        training_day_stop=validation_start,
        neighbor_center=neighbor_center,
        neighbor_node=neighbor_node,
    )

    config = HCLConfig(
        n_nodes=len(daily.cells),
        n_crimes=len(daily.crimes),
        input_steps=args.input_days,
        hidden_dim=args.hidden_dim,
        hypergraph_layers=args.hypergraph_layers,
        dropout=args.dropout,
        hawkes_scope=args.hawkes_scope,
        hawkes_delta=args.hawkes_delta,
    )
    model, training_history, training_summary = train_model(
        config,
        edge_index,
        edge_weight,
        neighbor_center,
        neighbor_node,
        coefficient,
        train_loader,
        validation_loader,
        args,
    )
    observed, predicted = predict(model, test_loader)
    test_dates = daily.dates[test_indices]
    summary_metrics, daily_metrics = evaluate_predictions(
        observed, predicted, daily.crimes, test_dates
    )

    metric_dir = RESULT_DIR / "metrics"
    table_dir = RESULT_DIR / "tables"
    model_dir = RESULT_DIR / "model"
    metric_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    summary_metrics.to_csv(metric_dir / "summary_metrics.csv", index=False)
    daily_metrics.to_csv(metric_dir / "daily_metrics.csv", index=False)
    training_history.to_csv(metric_dir / "training_history.csv", index=False)
    save_metric_plots(metric_dir, summary_metrics, daily_metrics, training_history)
    save_prediction_table(
        table_dir / "daily_predictions.parquet",
        observed,
        predicted,
        test_dates,
        daily.cells,
        daily.crimes,
    )

    frame_offsets = np.arange(0, len(test_indices), args.frame_days, dtype=np.int64)
    frame_risk = predicted[frame_offsets].sum(axis=-1).astype(np.float32)
    records: list[dict] = []
    for frame_index, prediction_offset in enumerate(frame_offsets):
        cutoff = test_dates[int(prediction_offset)]
        display_time = cutoff + pd.Timedelta(days=1)
        records.append(
            {
                "frame_index": frame_index,
                "forecast_date": cutoff,
                "display_time": display_time,
                "actual_events_in_window": observed_window_counts(
                    dataset.events, cutoff, display_time
                ),
                "predicted_events": float(frame_risk[frame_index].sum()),
            }
        )

    gif_path = RESULT_DIR / "animations" / f"{MODEL_ID}_24h_forecast_heatmap.gif"
    if not args.skip_gif:
        save_forecast_animation(
            out_path=gif_path,
            dataset=dataset,
            records=records,
            risk_matrix=frame_risk,
            args=args,
            title="LAPD Hawkes-enhanced Hypergraph Contrastive Learning",
            note=(
                "Heatmap: predicted next-day crime quantity. "
                "Colored rings: observed target-crime events."
            ),
        )

    split_summary = {
        "full_start": daily.dates[0].date().isoformat(),
        "full_end": daily.dates[-1].date().isoformat(),
        "train_target_start": daily.dates[train_all[0]].date().isoformat(),
        "train_target_end": daily.dates[validation_indices[0] - 1].date().isoformat(),
        "validation_start": daily.dates[validation_indices[0]].date().isoformat(),
        "validation_end": daily.dates[validation_indices[-1]].date().isoformat(),
        "test_start": test_dates[0].date().isoformat(),
        "test_end": test_dates[-1].date().isoformat(),
        "available_train_daily_targets": int(len(train_all)),
        "used_train_targets": int(len(train_indices)),
        "train_stride_days": int(args.train_stride),
        "validation_samples": int(len(validation_indices)),
        "test_samples": int(len(test_indices)),
    }
    write_forecast_tables(
        result_dir=RESULT_DIR,
        model_id=MODEL_ID,
        dataset=dataset,
        records=records,
        cells=daily.cells,
        risk_matrix=frame_risk,
        args=args,
        gif_path=gif_path,
        extra_config={
            "paper": "Liang et al. (2024), AAAI, DOI 10.1609/aaai.v38i8.28719",
            "task": "crime quantity prediction",
            "reproduction_status": (
                "paper-constrained HCL equations with a compact structural hypergraph "
                "backbone; exact HCL author code and full backbone settings unavailable"
            ),
            "split": split_summary,
            "model_config": asdict(config),
            "experiment_defined_output_head": (
                "attention pooling over all Hawkes-enhanced days concatenated with the "
                "last-day representation, followed by a Softplus count head"
            ),
            "graph_summary": graph_summary,
            "category_coefficients": {
                str(crime): float(value)
                for crime, value in zip(daily.crimes, coefficient, strict=True)
            },
            "training_summary": training_summary,
            "lambda_type": args.lambda_type,
            "lambda_neighbor": args.lambda_neighbor,
            "contrast_steps_per_window": args.contrast_steps,
            "data_end_exclusive": args.data_end_exclusive,
            "sequential_forecast": (
                "Each daily test prediction uses the immediately preceding 30 observed days; "
                "GIF frames sample the rolling forecasts every frame_days days."
            ),
        },
    )

    torch.save(
        {
            "model_id": MODEL_ID,
            "state_dict": model.state_dict(),
            "config": asdict(config),
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "cells": daily.cells.tolist(),
            "crimes": daily.crimes.tolist(),
            "category_coefficients": coefficient,
            "split": split_summary,
            "training_summary": training_summary,
        },
        model_dir / "model_state.pt",
    )
    with (table_dir / "training_summary.yml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            {
                "model_id": MODEL_ID,
                "dataset_id": DATASET_ID,
                "paper_specified": {
                    "cell_size_m": args.cell_size_m,
                    "temporal_resolution": "daily",
                    "split_ratio": "7:1 chronological",
                    "validation_days": args.validation_days,
                    "task_loss": "regression loss; ST-HSL official code uses MSE",
                    "lambda_type": args.lambda_type,
                    "lambda_neighbor": args.lambda_neighbor,
                    "hawkes_scope": args.hawkes_scope,
                    "hawkes_delta": args.hawkes_delta,
                    "neighbor_definition": "four cardinal 1-hop grid neighbors",
                },
                "experiment_defined": {
                    "backbone": "compact structural crime-type/cardinal-neighborhood hypergraph",
                    "output_head": (
                        "temporal attention context plus last-day representation, "
                        "with Softplus nonnegative output"
                    ),
                    "input_days": args.input_days,
                    "hidden_dim": args.hidden_dim,
                    "hypergraph_layers": args.hypergraph_layers,
                    "dropout": args.dropout,
                    "contrast_steps_per_window": args.contrast_steps,
                    "train_stride_days": args.train_stride,
                    "batch_size": args.batch_size,
                    "epochs": args.epochs,
                    "optimizer": "Adam",
                    "learning_rate": args.learning_rate,
                    "weight_decay": args.weight_decay,
                    "seed": args.seed,
                    "number_of_runs": 1,
                    "data_end_exclusive": args.data_end_exclusive,
                },
                "split": split_summary,
                "graph": graph_summary,
                "category_coefficients": {
                    str(crime): float(value)
                    for crime, value in zip(daily.crimes, coefficient, strict=True)
                },
                "training": training_summary,
            },
            handle,
            sort_keys=False,
            allow_unicode=True,
        )
    print(f"Wrote {RESULT_DIR}", flush=True)


if __name__ == "__main__":
    main()
