"""Run a paper-constrained ST-MoGE experiment on LAPD crime data.

The experiment implements Wu et al.'s category-specific and universal graph
experts, attentive spatial gates, regional predictors, CECL, and HALR. The
paper's official code is unavailable. Operational choices that are absent from
the paper or adapted for CPU execution are recorded with every result.
"""

from __future__ import annotations

import argparse
import copy
import math
import os
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, Dataset

from our_experiment.common.lapd_target_common import (
    load_target_dataset,
    observed_window_counts,
    save_forecast_animation,
    write_forecast_tables,
)
from ref_models.experiments.stmgnn_zinb.run_lapd_stmgnn_zinb import (
    DATASET_ID,
    DailyCrimeTensor,
    build_coarse_dataset,
    build_graph,
    make_daily_tensor,
    standardize_features,
)
from ref_models.models.st_moge import (
    STMOGEConfig,
    SpatialTemporalMixtureOfGraphExperts,
    cecl_loss,
    hierarchical_weighted_mse,
)


MODEL_ID = "G1_st_moge_lapd_quantity"
RESULT_DIR = ROOT / "ref_models" / "experiments" / "st_moge" / "results"


class STMoGEDailyDataset(Dataset):
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

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        target = int(self.target_indices[index])
        history = self.features[target - self.input_steps : target].transpose(1, 0, 2)
        return (
            torch.from_numpy(np.ascontiguousarray(history, dtype=np.float32)),
            torch.from_numpy(np.ascontiguousarray(self.counts[target], dtype=np.float32)),
        )


def subset_daily_tensor(
    daily: DailyCrimeTensor,
    start: str,
    end_exclusive: str,
) -> DailyCrimeTensor:
    mask = (daily.dates >= pd.Timestamp(start)) & (
        daily.dates < pd.Timestamp(end_exclusive)
    )
    if int(mask.sum()) <= 10:
        raise ValueError("Requested ST-MoGE period is too short")
    return DailyCrimeTensor(
        dates=daily.dates[mask],
        cells=daily.cells,
        crimes=daily.crimes,
        counts=daily.counts[mask],
    )


def split_811(
    n_days: int,
    input_steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    train_stop = int(math.floor(n_days * 0.8))
    validation_stop = int(math.floor(n_days * 0.9))
    valid = np.arange(input_steps, n_days, dtype=np.int64)
    train = valid[valid < train_stop]
    validation = valid[(valid >= train_stop) & (valid < validation_stop)]
    test = valid[valid >= validation_stop]
    if not len(train) or not len(validation) or not len(test):
        raise ValueError("The chronological 8:1:1 split produced an empty partition")
    return train, validation, test, train_stop, validation_stop


def make_loader(
    features: np.ndarray,
    counts: np.ndarray,
    target_indices: np.ndarray,
    batch_size: int,
    input_steps: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    return DataLoader(
        STMoGEDailyDataset(features, counts, target_indices, input_steps),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        generator=torch.Generator().manual_seed(seed) if shuffle else None,
    )


def update_region_clusters(
    model: SpatialTemporalMixtureOfGraphExperts,
    previous_centers: list[np.ndarray] | None,
    seed: int,
) -> list[np.ndarray]:
    embeddings = model.category_node_embeddings().detach().cpu().numpy()
    assignments = []
    centers = []
    for crime, values in enumerate(embeddings):
        if previous_centers is None:
            initial: str | np.ndarray = "k-means++"
            n_init = 10
        else:
            initial = previous_centers[crime]
            n_init = 1
        fitted = KMeans(
            n_clusters=model.config.n_clusters,
            init=initial,
            n_init=n_init,
            random_state=seed,
        ).fit(values)
        assignments.append(fitted.labels_.astype(np.int64))
        centers.append(fitted.cluster_centers_.astype(np.float32))
    model.set_cluster_assignments(torch.from_numpy(np.stack(assignments)))
    return centers


def halr_weights(
    category_history: list[np.ndarray],
    cluster_history: list[np.ndarray],
    temperature: float,
    n_crimes: int,
    n_clusters: int,
) -> tuple[np.ndarray, np.ndarray]:
    if len(category_history) < 2:
        return (
            np.full(n_crimes, 1.0 / n_crimes, dtype=np.float32),
            np.full((n_crimes, n_clusters), 1.0 / n_clusters, dtype=np.float32),
        )
    epsilon = 1e-8
    category_rate = category_history[-1] / np.maximum(category_history[-2], epsilon)
    category_logits = category_rate / temperature
    category_logits -= category_logits.max()
    category_weight = np.exp(category_logits)
    category_weight /= category_weight.sum()

    cluster_rate = cluster_history[-1] / np.maximum(cluster_history[-2], epsilon)
    cluster_logits = cluster_rate / temperature
    cluster_logits -= cluster_logits.max(axis=1, keepdims=True)
    cluster_weight = np.exp(cluster_logits)
    cluster_weight /= cluster_weight.sum(axis=1, keepdims=True)
    return category_weight.astype(np.float32), cluster_weight.astype(np.float32)


def validation_metrics(
    model: SpatialTemporalMixtureOfGraphExperts,
    loader: DataLoader,
) -> tuple[float, float]:
    device = next(model.parameters()).device
    absolute_error = 0.0
    squared_error = 0.0
    observations = 0
    model.eval()
    with torch.no_grad():
        for features, target in loader:
            target = target.to(device)
            prediction = model(
                features.to(device), compute_contrastive=False
            ).prediction
            absolute_error += float(torch.sum(torch.abs(prediction - target)).cpu())
            squared_error += float(torch.sum((prediction - target).square()).cpu())
            observations += int(target.numel())
    return absolute_error / observations, squared_error / observations


def train_model(
    model: SpatialTemporalMixtureOfGraphExperts,
    features: np.ndarray,
    counts: np.ndarray,
    train_indices: np.ndarray,
    validation_loader: DataLoader,
    args: argparse.Namespace,
) -> tuple[SpatialTemporalMixtureOfGraphExperts, pd.DataFrame, dict]:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.torch_threads)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.minimum_learning_rate,
    )
    training_model: torch.nn.Module = model
    compiled = False
    if args.compile_model:
        try:
            training_model = torch.compile(model, mode="reduce-overhead")
            compiled = True
        except Exception as error:
            print(f"torch.compile unavailable; continuing eagerly: {error}", flush=True)

    best_state = copy.deepcopy(model.state_dict())
    best_validation_mae = math.inf
    best_epoch = 0
    centers: list[np.ndarray] | None = None
    category_history: list[np.ndarray] = []
    cluster_history: list[np.ndarray] = []
    visited_targets: set[int] = set()
    rows: list[dict] = []
    checkpoint_dir = RESULT_DIR / "model"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        epoch_started = time.perf_counter()
        if epoch <= args.cluster_refresh_epochs:
            centers = update_region_clusters(model, centers, args.seed)
        category_weight_np, cluster_weight_np = halr_weights(
            category_history,
            cluster_history,
            args.halr_temperature,
            model.config.n_crimes,
            model.config.n_clusters,
        )
        category_weight = torch.from_numpy(category_weight_np).to(device)
        cluster_weight = torch.from_numpy(cluster_weight_np).to(device)

        offset = (epoch - 1) % args.train_stride
        epoch_indices = train_indices[offset:: args.train_stride]
        visited_targets.update(int(value) for value in epoch_indices)
        train_loader = make_loader(
            features,
            counts,
            epoch_indices,
            args.batch_size,
            args.input_days,
            shuffle=True,
            seed=args.seed + epoch,
        )
        model.train()
        totals = {name: 0.0 for name in ("total", "prediction", "cecl_specific", "cecl_universal")}
        category_total = np.zeros(model.config.n_crimes, dtype=np.float64)
        cluster_total = np.zeros(
            (model.config.n_crimes, model.config.n_clusters), dtype=np.float64
        )
        samples = 0
        for batch_features, target in train_loader:
            batch_features = batch_features.to(device)
            target = target.to(device)
            optimizer.zero_grad(set_to_none=True)
            output = training_model(batch_features, compute_contrastive=True)
            prediction_loss, batch_cluster_loss = hierarchical_weighted_mse(
                output.prediction,
                target,
                model.cluster_assignments,
                category_weight,
                cluster_weight,
            )
            specific_loss, universal_loss = cecl_loss(output, args.cecl_temperature)
            total = (
                args.lambda_prediction * prediction_loss
                + args.lambda_contrastive * (specific_loss + universal_loss)
            )
            if not torch.isfinite(total):
                raise RuntimeError(f"Non-finite ST-MoGE loss at epoch {epoch}")
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            batch = int(batch_features.shape[0])
            totals["total"] += float(total.detach().cpu()) * batch
            totals["prediction"] += float(prediction_loss.detach().cpu()) * batch
            totals["cecl_specific"] += float(specific_loss.detach().cpu()) * batch
            totals["cecl_universal"] += float(universal_loss.detach().cpu()) * batch
            category_total += (
                (output.prediction.detach() - target).square().mean(dim=(0, 1)).cpu().numpy()
                * batch
            )
            cluster_total += batch_cluster_loss.detach().cpu().numpy() * batch
            samples += batch

        category_epoch = (category_total / max(samples, 1)).astype(np.float32)
        cluster_epoch = (cluster_total / max(samples, 1)).astype(np.float32)
        category_history.append(category_epoch)
        cluster_history.append(cluster_epoch)
        train_values = {name: value / max(samples, 1) for name, value in totals.items()}

        should_validate = (
            epoch == 1
            or epoch == args.epochs
            or epoch % args.validation_every == 0
        )
        if should_validate:
            validation_mae, validation_mse = validation_metrics(model, validation_loader)
            if validation_mae < best_validation_mae:
                best_validation_mae = validation_mae
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                torch.save(
                    {
                        "epoch": epoch,
                        "state_dict": best_state,
                        "validation_mae": best_validation_mae,
                    },
                    checkpoint_dir / "best_training_checkpoint.pt",
                )
        else:
            validation_mae = math.nan
            validation_mse = math.nan

        row = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "cyclic_offset": offset,
            "used_train_targets": len(epoch_indices),
            **{f"train_{name}": value for name, value in train_values.items()},
            "validation_mae": validation_mae,
            "validation_mse": validation_mse,
            "elapsed_seconds": time.perf_counter() - epoch_started,
        }
        for crime in range(model.config.n_crimes):
            row[f"halr_category_weight_{crime}"] = float(category_weight_np[crime])
            for cluster in range(model.config.n_clusters):
                row[f"halr_cluster_weight_{crime}_{cluster}"] = float(
                    cluster_weight_np[crime, cluster]
                )
        rows.append(row)
        if epoch == 1 or epoch % args.log_every == 0 or should_validate:
            validation_text = (
                f" val_mae={validation_mae:.6f}" if should_validate else ""
            )
            print(
                f"ST-MoGE epoch {epoch}/{args.epochs}: "
                f"total={train_values['total']:.6f} "
                f"prediction={train_values['prediction']:.6f} "
                f"cecl={train_values['cecl_specific'] + train_values['cecl_universal']:.6f}"
                f"{validation_text} elapsed={row['elapsed_seconds']:.1f}s",
                flush=True,
            )
        scheduler.step()

    model.load_state_dict(best_state)
    summary = {
        "device": str(device),
        "torch_version": str(torch.__version__),
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "epochs_requested": int(args.epochs),
        "epochs_completed": int(len(rows)),
        "best_epoch": int(best_epoch),
        "best_validation_mae": float(best_validation_mae),
        "distinct_train_targets_visited": int(len(visited_targets)),
        "available_train_targets": int(len(train_indices)),
        "torch_compile": compiled,
    }
    return model, pd.DataFrame(rows), summary


def predict(
    model: SpatialTemporalMixtureOfGraphExperts,
    loader: DataLoader,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    device = next(model.parameters()).device
    observed = []
    predicted = []
    gates = []
    model.eval()
    with torch.no_grad():
        for features, target in loader:
            output = model(features.to(device), compute_contrastive=False)
            observed.append(target.numpy())
            predicted.append(output.prediction.cpu().numpy())
            gates.append(output.gate.cpu().numpy())
    return np.concatenate(observed), np.concatenate(predicted), np.concatenate(gates)


def evaluate_predictions(
    observed: np.ndarray,
    predicted: np.ndarray,
    crimes: pd.Index,
    dates: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    scopes: list[tuple[str, np.ndarray | None]] = [("ALL", None)] + [
        (str(crime), np.asarray([index], dtype=np.int64))
        for index, crime in enumerate(crimes)
    ]
    for label, indices in scopes:
        actual = observed if indices is None else observed[..., indices]
        forecast = predicted if indices is None else predicted[..., indices]
        positive = actual > 0
        values = {
            "mae": float(np.mean(np.abs(actual - forecast))),
            "mape_positive": (
                float(np.mean(np.abs(actual[positive] - forecast[positive]) / actual[positive]))
                if positive.any()
                else np.nan
            ),
            "rmse": float(np.sqrt(np.mean((actual - forecast) ** 2))),
            "mean_observed_count": float(np.mean(actual)),
            "mean_predicted_count": float(np.mean(forecast)),
        }
        rows.extend(
            {"scope": label, "metric": metric, "value": value}
            for metric, value in values.items()
        )

    daily_rows = []
    for index, date in enumerate(dates):
        actual = observed[index]
        forecast = predicted[index]
        positive = actual > 0
        daily_rows.append(
            {
                "forecast_date": date.date().isoformat(),
                "mae": float(np.mean(np.abs(actual - forecast))),
                "mape_positive": (
                    float(
                        np.mean(
                            np.abs(actual[positive] - forecast[positive]) / actual[positive]
                        )
                    )
                    if positive.any()
                    else np.nan
                ),
                "observed_events": float(actual.sum()),
                "predicted_events": float(forecast.sum()),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(daily_rows)


def frequency_quantile_metrics(
    observed: np.ndarray,
    predicted: np.ndarray,
    training_counts: np.ndarray,
    crimes: pd.Index,
) -> pd.DataFrame:
    frequency = training_counts.sum(axis=0)
    rows = []
    for crime, crime_name in enumerate(crimes):
        order = np.argsort(frequency[:, crime], kind="stable")
        rank = np.empty(len(order), dtype=np.int64)
        rank[order] = np.arange(len(order))
        groups = np.minimum(rank * 4 // len(order), 3)
        for group in range(4):
            mask = groups == group
            actual = observed[:, mask, crime]
            forecast = predicted[:, mask, crime]
            positive = actual > 0
            rows.append(
                {
                    "target_crime": str(crime_name),
                    "frequency_quantile": f"Q{group + 1}",
                    "cell_count": int(mask.sum()),
                    "training_events_min": float(frequency[mask, crime].min()),
                    "training_events_max": float(frequency[mask, crime].max()),
                    "mae": float(np.mean(np.abs(actual - forecast))),
                    "mape_positive": (
                        float(
                            np.mean(
                                np.abs(actual[positive] - forecast[positive])
                                / actual[positive]
                            )
                        )
                        if positive.any()
                        else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def save_prediction_table(
    path: Path,
    observed: np.ndarray,
    predicted: np.ndarray,
    gates: np.ndarray,
    dates: pd.DatetimeIndex,
    cells: pd.Index,
    crimes: pd.Index,
) -> None:
    n_days, n_cells, n_crimes = predicted.shape
    pd.DataFrame(
        {
            "forecast_date": np.repeat(
                dates.to_numpy(dtype="datetime64[D]"), n_cells * n_crimes
            ),
            "cell_id": np.tile(np.repeat(cells.to_numpy(), n_crimes), n_days),
            "target_crime": np.tile(crimes.to_numpy(), n_days * n_cells),
            "observed_count": observed.reshape(-1),
            "predicted_count": predicted.reshape(-1),
            "specific_gate_weight": gates.reshape(-1),
        }
    ).to_parquet(path, index=False)


def save_plots(
    metric_dir: Path,
    summary: pd.DataFrame,
    daily: pd.DataFrame,
    frequency: pd.DataFrame,
    training: pd.DataFrame,
    gate_summary: pd.DataFrame,
) -> None:
    plot_dir = metric_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    axes[0].plot(training["epoch"], training["train_total"], label="Total")
    axes[0].plot(training["epoch"], training["train_prediction"], label="HALR prediction")
    axes[0].set_title("ST-MoGE training objective")
    axes[0].set_xlabel("Epoch")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    valid = training.dropna(subset=["validation_mae"])
    axes[1].plot(valid["epoch"], valid["validation_mae"], marker="o", color="#2b8cbe")
    axes[1].set_title("Validation MAE")
    axes[1].set_xlabel("Epoch")
    axes[1].grid(alpha=0.25)
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
    axes[0].plot(dates, daily["predicted_events"], label="Predicted", color="#de2d26")
    axes[0].set_ylabel("Daily events")
    axes[0].set_title("ST-MoGE rolling one-day-ahead forecast")
    axes[0].grid(alpha=0.2)
    axes[0].legend()
    axes[1].plot(dates, daily["mae"], color="#2b8cbe")
    axes[1].set_ylabel("Daily MAE")
    axes[1].grid(alpha=0.2)
    fig.savefig(plot_dir / "daily_forecast_diagnostics.png", dpi=170)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
    for crime, values in frequency.groupby("target_crime", sort=False):
        axis.plot(values["frequency_quantile"], values["mae"], marker="o", label=crime)
    axis.set_title("MAE by training-frequency quartile")
    axis.set_xlabel("Region frequency quartile (Q1 is sparsest)")
    axis.set_ylabel("MAE")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.savefig(plot_dir / "frequency_quantile_mae.png", dpi=170)
    plt.close(fig)

    pivot = gate_summary.pivot(index="target_crime", columns="cluster", values="mean_specific_gate")
    fig, axis = plt.subplots(figsize=(8, 3.8), constrained_layout=True)
    image = axis.imshow(pivot.to_numpy(), aspect="auto", vmin=0, vmax=1, cmap="viridis")
    axis.set_yticks(np.arange(len(pivot.index)), pivot.index)
    axis.set_xticks(np.arange(len(pivot.columns)), [f"Cluster {value}" for value in pivot.columns])
    axis.set_title("Mean category-specific gate weight")
    fig.colorbar(image, ax=axis, label="Specific expert weight")
    fig.savefig(plot_dir / "gate_weights_by_cluster.png", dpi=170)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell-size-m", type=int, default=3000)
    parser.add_argument("--data-start", default="2020-07-31")
    parser.add_argument("--data-end-exclusive", default="2022-12-01")
    parser.add_argument("--input-days", type=int, default=7)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--node-embedding-dim", type=int, default=16)
    parser.add_argument("--st-blocks", type=int, default=3)
    parser.add_argument("--spatial-layers", type=int, default=2)
    parser.add_argument("--temporal-layers", type=int, default=3)
    parser.add_argument("--temporal-kernel-size", type=int, default=3)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--clusters", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--cecl-temperature", type=float, default=0.05)
    parser.add_argument("--halr-temperature", type=float, default=1.0)
    parser.add_argument("--lambda-prediction", type=float, default=0.80)
    parser.add_argument("--lambda-contrastive", type=float, default=0.20)
    parser.add_argument("--train-stride", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--minimum-learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--validation-every", type=int, default=5)
    parser.add_argument("--cluster-refresh-epochs", type=int, default=2)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--frame-days", type=int, default=7)
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
    parser.add_argument("--torch-threads", type=int, default=8)
    parser.add_argument(
        "--compile-model",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--device")
    parser.add_argument("--overwrite-derived-data", action="store_true")
    parser.add_argument("--skip-gif", action="store_true")
    parser.add_argument("--reuse-model-state", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.cell_size_m != 3000:
        raise ValueError("ST-MoGE uses the paper-aligned 3000 m grid")
    if args.input_days != 7:
        raise ValueError("The ST-MoGE paper uses seven historical days")
    if args.horizon_hours != 24:
        raise ValueError("This experiment supports the paper's next-day horizon")
    if args.train_stride < 1:
        raise ValueError("train_stride must be positive")
    if args.cluster_refresh_epochs < 1:
        raise ValueError("cluster_refresh_epochs must be positive")
    if not math.isclose(
        args.lambda_prediction + args.lambda_contrastive, 1.0, abs_tol=1e-6
    ):
        raise ValueError("lambda_prediction and lambda_contrastive must sum to one")
    args.history_days = args.input_days
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    base = load_target_dataset(overwrite=False, cell_size_m=150)
    dataset = build_coarse_dataset(base, args.cell_size_m, args.overwrite_derived_data)
    daily = subset_daily_tensor(
        make_daily_tensor(dataset, args.data_end_exclusive),
        args.data_start,
        args.data_end_exclusive,
    )
    train_indices, validation_indices, test_indices, train_stop, validation_stop = split_811(
        len(daily.dates), args.input_days
    )
    edge_index, edge_weight, graph_summary = build_graph(dataset.grid)
    config = STMOGEConfig(
        n_nodes=len(daily.cells),
        n_crimes=len(daily.crimes),
        input_steps=args.input_days,
        hidden_dim=args.hidden_dim,
        node_embedding_dim=args.node_embedding_dim,
        st_blocks=args.st_blocks,
        spatial_layers=args.spatial_layers,
        temporal_layers=args.temporal_layers,
        temporal_kernel_size=args.temporal_kernel_size,
        attention_heads=args.attention_heads,
        n_clusters=args.clusters,
        dropout=args.dropout,
    )
    model_state_path = RESULT_DIR / "model" / "model_state.pt"
    checkpoint = None
    if args.reuse_model_state:
        if not model_state_path.exists():
            raise FileNotFoundError(f"Cannot reuse missing state: {model_state_path}")
        checkpoint = torch.load(model_state_path, map_location="cpu", weights_only=False)
        config = STMOGEConfig(**checkpoint["config"])
        feature_mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32)
        feature_std = np.asarray(checkpoint["feature_std"], dtype=np.float32)
        features = (
            (np.log1p(daily.counts).astype(np.float32) - feature_mean.reshape(1, 1, -1))
            / feature_std.reshape(1, 1, -1)
        ).astype(np.float32)
    else:
        features, feature_mean, feature_std = standardize_features(
            daily.counts, training_day_stop=train_stop
        )

    validation_loader = make_loader(
        features,
        daily.counts,
        validation_indices,
        args.batch_size,
        args.input_days,
        shuffle=False,
        seed=args.seed,
    )
    test_loader = make_loader(
        features,
        daily.counts,
        test_indices,
        args.batch_size,
        args.input_days,
        shuffle=False,
        seed=args.seed,
    )
    model = SpatialTemporalMixtureOfGraphExperts(config, edge_index, edge_weight)
    if checkpoint is None:
        model, training_history, training_summary = train_model(
            model,
            features,
            daily.counts,
            train_indices,
            validation_loader,
            args,
        )
    else:
        device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        model = model.to(device)
        model.load_state_dict(checkpoint["state_dict"])
        training_summary = checkpoint["training_summary"]
        training_history = pd.read_csv(RESULT_DIR / "metrics" / "training_history.csv")
        print(f"Reused ST-MoGE state from {model_state_path}", flush=True)

    observed, predicted, gates = predict(model, test_loader)
    test_dates = daily.dates[test_indices]
    summary_metrics, daily_metrics = evaluate_predictions(
        observed, predicted, daily.crimes, test_dates
    )
    frequency_metrics = frequency_quantile_metrics(
        observed,
        predicted,
        daily.counts[:train_stop],
        daily.crimes,
    )

    assignments = model.cluster_assignments.detach().cpu().numpy()
    gate_mean = gates.mean(axis=0)
    frequency = daily.counts[:train_stop].sum(axis=0)
    cluster_rows = []
    gate_rows = []
    for crime, crime_name in enumerate(daily.crimes):
        for cell, cell_id in enumerate(daily.cells):
            cluster_rows.append(
                {
                    "cell_id": str(cell_id),
                    "target_crime": str(crime_name),
                    "cluster": int(assignments[crime, cell]),
                    "training_events": float(frequency[cell, crime]),
                    "mean_specific_gate": float(gate_mean[cell, crime]),
                }
            )
        for cluster in range(config.n_clusters):
            mask = assignments[crime] == cluster
            gate_rows.append(
                {
                    "target_crime": str(crime_name),
                    "cluster": cluster,
                    "cell_count": int(mask.sum()),
                    "mean_specific_gate": float(gate_mean[mask, crime].mean()),
                    "mean_training_events": float(frequency[mask, crime].mean()),
                }
            )
    cluster_frame = pd.DataFrame(cluster_rows)
    gate_summary = pd.DataFrame(gate_rows)

    metric_dir = RESULT_DIR / "metrics"
    table_dir = RESULT_DIR / "tables"
    model_dir = RESULT_DIR / "model"
    metric_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    summary_metrics.to_csv(metric_dir / "summary_metrics.csv", index=False)
    daily_metrics.to_csv(metric_dir / "daily_metrics.csv", index=False)
    frequency_metrics.to_csv(metric_dir / "frequency_quantile_metrics.csv", index=False)
    training_history.to_csv(metric_dir / "training_history.csv", index=False)
    gate_summary.to_csv(metric_dir / "gate_summary.csv", index=False)
    save_plots(
        metric_dir,
        summary_metrics,
        daily_metrics,
        frequency_metrics,
        training_history,
        gate_summary,
    )
    save_prediction_table(
        table_dir / "daily_predictions.parquet",
        observed,
        predicted,
        gates,
        test_dates,
        daily.cells,
        daily.crimes,
    )
    cluster_frame.to_csv(table_dir / "region_clusters_and_gates.csv", index=False)

    frame_offsets = np.arange(0, len(test_indices), args.frame_days, dtype=np.int64)
    frame_risk = predicted[frame_offsets].sum(axis=-1).astype(np.float32)
    records = []
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
            title="LAPD Spatial-Temporal Mixture-of-Graph-Experts",
            note=(
                "Heatmap: predicted next-day multi-type crime quantity. "
                "Colored rings: observed target-crime events."
            ),
        )

    split_summary = {
        "full_start": daily.dates[0].date().isoformat(),
        "full_end": daily.dates[-1].date().isoformat(),
        "train_target_start": daily.dates[train_indices[0]].date().isoformat(),
        "train_target_end": daily.dates[train_indices[-1]].date().isoformat(),
        "validation_start": daily.dates[validation_indices[0]].date().isoformat(),
        "validation_end": daily.dates[validation_indices[-1]].date().isoformat(),
        "test_start": test_dates[0].date().isoformat(),
        "test_end": test_dates[-1].date().isoformat(),
        "train_targets": int(len(train_indices)),
        "validation_targets": int(len(validation_indices)),
        "test_targets": int(len(test_indices)),
        "ratio": "8:1:1 chronological",
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
            "paper": (
                "Wu et al. (2026), Spatial-Temporal Mixture-of-Graph-Experts "
                "for Multi-Type Crime Prediction, DOI 10.1007/s11390-025-5404-1"
            ),
            "task": "next-day multi-type crime quantity prediction",
            "reproduction_status": (
                "paper-constrained implementation; author code is unavailable and "
                "explicit experiment-defined choices are recorded in training_summary.yml"
            ),
            "split": split_summary,
            "model_config": asdict(config),
            "graph_summary": graph_summary,
            "training_summary": training_summary,
            "sequential_forecast": (
                "Each test prediction uses only the preceding seven observed days; "
                "the GIF samples those rolling forecasts every frame_days days."
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
                "paper": {
                    "doi": "10.1007/s11390-025-5404-1",
                    "official_code_available": False,
                    "specified": {
                        "cell_graph": (
                            "geographical-proximity adjacency plus learned "
                            "adaptive adjacency"
                        ),
                        "temporal_resolution": "daily",
                        "input_days": 7,
                        "split": "8:1:1 chronological",
                        "hidden_dim": 32,
                        "node_embedding_dim": 16,
                        "st_blocks": 3,
                        "spatial_layers_nyc": 2,
                        "temporal_layers": 3,
                        "temporal_kernel_size": 3,
                        "regional_clusters": 4,
                        "halr_temperature": 1.0,
                        "cecl_temperature": 0.05,
                        "optimizer": "Adam",
                        "initial_learning_rate": 0.01,
                        "batch_size": 64,
                        "epochs": 50,
                    },
                },
                "experiment_defined": {
                    "lapd_period": f"{args.data_start} to {args.data_end_exclusive} exclusive",
                    "target_crimes": daily.crimes.tolist(),
                    "prior_graph": "row-normalized 8-neighbor 3 km grid with self-loops",
                    "attention_heads": args.attention_heads,
                    "dropout": args.dropout,
                    "lambda_prediction": args.lambda_prediction,
                    "lambda_contrastive": args.lambda_contrastive,
                    "weight_decay": args.weight_decay,
                    "learning_rate_schedule": (
                        f"cosine annealing to {args.minimum_learning_rate}"
                    ),
                    "cecl_stabilization": (
                        "positive-inclusive InfoNCE denominator; the printed paper equations "
                        "show only negative terms in their denominators"
                    ),
                    "auxiliary_views": (
                        "one shared universal-expert corrupted representation with two "
                        "independent representation-level dropout views"
                    ),
                    "cluster_update": (
                        f"K-means on each category expert's learned E1 embedding for the first "
                        f"{args.cluster_refresh_epochs} epochs; the second fit follows one warm-up "
                        "epoch and is then fixed for stable CPU training"
                    ),
                    "halr_update": (
                        "category and cluster loss-reduction weights updated once per epoch"
                    ),
                    "cpu_training": (
                        f"cyclic stride {args.train_stride}; epoch e uses targets at offset "
                        "(e-1) mod stride so all dates are visited across cycles"
                    ),
                    "torch_compile": args.compile_model,
                    "mape_zero_handling": "MAPE is computed only where observed count is positive",
                    "seed": args.seed,
                    "number_of_runs": 1,
                },
                "model_config": asdict(config),
                "split": split_summary,
                "graph": graph_summary,
                "training": training_summary,
            },
            handle,
            sort_keys=False,
            allow_unicode=True,
        )
    print(f"Wrote {RESULT_DIR}", flush=True)


if __name__ == "__main__":
    main()
