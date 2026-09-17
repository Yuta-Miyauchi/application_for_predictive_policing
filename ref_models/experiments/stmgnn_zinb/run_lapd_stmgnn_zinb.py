"""Run a paper-constrained STMGNN-ZINB experiment on LAPD crime data.

The paper specifies 3 km spatial grids, daily multivariate crime counts, a
7:1 chronological train/test split, a 30-day validation tail, DGCN and MTCN
branches, Hadamard fusion, and direct ZINB negative log likelihood. It does not
publish code or most neural-network hyperparameters. All experiment-defined
choices are exposed as command-line arguments and recorded with the outputs.
"""

from __future__ import annotations

import argparse
import copy
import math
import os
import random
import sys
from dataclasses import asdict, dataclass
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
from scipy.special import gammaln
from scipy.stats import nbinom
from shapely.geometry import box
from shapely.prepared import prep
from torch.utils.data import DataLoader, Dataset

from our_experiment.common.lapd_target_common import (
    SOURCE_DATASET_ID,
    TARGET_CRIME_ORDER,
    TARGET_DIR,
    TargetDataset,
    load_target_dataset,
    observed_window_counts,
    save_forecast_animation,
    write_forecast_tables,
)
from ref_models.models.stmgnn_zinb import (
    STMGNNZINB,
    STMGNNZINBConfig,
    zinb_nll,
)


MODEL_ID = "U1_stmgnn_zinb_lapd"
DATASET_ID = "lapd_legacy_2010_2024_stmgnn_target_crimes_grid3000m_daily"
DATASET_DIR = TARGET_DIR.parent / DATASET_ID
RESULT_DIR = ROOT / "ref_models" / "experiments" / "stmgnn_zinb" / "results"


@dataclass(frozen=True)
class DailyCrimeTensor:
    dates: pd.DatetimeIndex
    cells: pd.Index
    crimes: pd.Index
    counts: np.ndarray


class DailyWindowDataset(Dataset):
    def __init__(
        self,
        features: np.ndarray,
        counts: np.ndarray,
        target_indices: np.ndarray,
        input_steps: int,
        output_steps: int,
    ):
        self.features = features
        self.counts = counts
        self.target_indices = np.asarray(target_indices, dtype=np.int64)
        self.input_steps = int(input_steps)
        self.output_steps = int(output_steps)

    def __len__(self) -> int:
        return len(self.target_indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        target = int(self.target_indices[index])
        x = self.features[target - self.input_steps : target].transpose(1, 0, 2)
        y = self.counts[target : target + self.output_steps].transpose(1, 0, 2)
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(
            np.ascontiguousarray(y, dtype=np.float32)
        )


def _cell_id(row: int, col: int, cell_size_m: int) -> str:
    return f"g{cell_size_m}_r{row:04d}_c{col:04d}"


def build_coarse_dataset(
    base: TargetDataset,
    cell_size_m: int,
    overwrite: bool,
) -> TargetDataset:
    metadata_path = DATASET_DIR / "metadata.yml"
    events_path = DATASET_DIR / "events.parquet"
    grid_path = DATASET_DIR / "grid.parquet"
    boundaries_path = DATASET_DIR / "division_boundaries.parquet"
    if not overwrite and all(
        path.exists() for path in (metadata_path, events_path, grid_path, boundaries_path)
    ):
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = yaml.safe_load(handle)
        events = pd.read_parquet(events_path)
        events["occurred_at"] = pd.to_datetime(events["occurred_at"])
        events["target_crime"] = pd.Categorical(
            events["target_crime"].astype(str), categories=TARGET_CRIME_ORDER, ordered=True
        )
        grid = gpd.read_parquet(grid_path)
        boundaries = gpd.read_parquet(boundaries_path)
        return TargetDataset(events=events, grid=grid, boundaries=boundaries, metadata=metadata)

    boundaries = base.boundaries.copy()
    minx, miny, maxx, maxy = boundaries.total_bounds
    origin_x = math.floor(minx / cell_size_m) * cell_size_m
    origin_y = math.floor(miny / cell_size_m) * cell_size_m
    row_start = int(math.floor((miny - origin_y) / cell_size_m))
    row_stop = int(math.ceil((maxy - origin_y) / cell_size_m))
    col_start = int(math.floor((minx - origin_x) / cell_size_m))
    col_stop = int(math.ceil((maxx - origin_x) / cell_size_m))
    city = boundaries.geometry.union_all()
    prepared_city = prep(city)

    records: list[dict] = []
    for row in range(row_start, row_stop):
        for col in range(col_start, col_stop):
            square = box(
                origin_x + col * cell_size_m,
                origin_y + row * cell_size_m,
                origin_x + (col + 1) * cell_size_m,
                origin_y + (row + 1) * cell_size_m,
            )
            if not prepared_city.intersects(square):
                continue
            clipped = city.intersection(square)
            if clipped.is_empty or clipped.area <= 0:
                continue
            overlap = boundaries.geometry.intersection(square).area
            dominant = boundaries.loc[overlap.idxmax()]
            records.append(
                {
                    "cell_id": _cell_id(row, col, cell_size_m),
                    "row": row,
                    "col": col,
                    "area_m2": float(clipped.area),
                    "area_id": str(dominant["area_id"]).zfill(2),
                    "area_name": str(dominant["area_name"]),
                    "geometry": clipped,
                }
            )
    grid = gpd.GeoDataFrame(records, geometry="geometry", crs=boundaries.crs)
    grid = grid.sort_values(["row", "col"]).reset_index(drop=True)

    events = base.events.copy()
    events = events.rename(columns={"area_id": "fine_area_id", "area_name": "fine_area_name"})
    events["row"] = np.floor(
        (events["y"].to_numpy(dtype=float) - origin_y) / cell_size_m
    ).astype(np.int32)
    events["col"] = np.floor(
        (events["x"].to_numpy(dtype=float) - origin_x) / cell_size_m
    ).astype(np.int32)
    events["cell_id"] = [
        _cell_id(int(row), int(col), cell_size_m)
        for row, col in zip(events["row"], events["col"], strict=False)
    ]
    cell_attributes = grid[["cell_id", "area_id", "area_name"]]
    events = events.drop(columns=["area_id", "area_name"], errors="ignore")
    events = events.merge(cell_attributes, on="cell_id", how="inner")
    events["target_crime"] = pd.Categorical(
        events["target_crime"].astype(str), categories=TARGET_CRIME_ORDER, ordered=True
    )
    events = events.sort_values("occurred_at").reset_index(drop=True)

    metadata = {
        "dataset_id": DATASET_ID,
        "source_dataset_id": SOURCE_DATASET_ID,
        "derived_from": base.metadata.get("dataset_id"),
        "target_crimes": TARGET_CRIME_ORDER,
        "temporal_resolution": "1 day",
        "spatial": {
            "projected_crs": str(boundaries.crs),
            "cell_size_m": int(cell_size_m),
            "origin_x": float(origin_x),
            "origin_y": float(origin_y),
            "graph": "8-neighbor grid with self-loops, row-normalized",
        },
        "counts": {
            "events": int(len(events)),
            "cells": int(len(grid)),
            "areas": int(boundaries["area_id"].nunique()),
            "target_crimes": len(TARGET_CRIME_ORDER),
        },
        "paper_alignment": {
            "paper": "Wang et al. (2024), Uncertainty-Aware Crime Prediction With STMGNNs",
            "specified": [
                "3 km x 3 km grid",
                "daily crime counts",
                "multivariate crime channels",
                "7:1 chronological train/test split",
                "last 30 training days used for validation",
            ],
        },
    }
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    events.to_parquet(events_path, index=False)
    grid.to_parquet(grid_path, index=False)
    boundaries.to_parquet(boundaries_path, index=False)
    with metadata_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(metadata, handle, sort_keys=False, allow_unicode=True)
    return TargetDataset(events=events, grid=grid, boundaries=boundaries, metadata=metadata)


def make_daily_tensor(
    dataset: TargetDataset,
    data_end_exclusive: str | None,
) -> DailyCrimeTensor:
    events = dataset.events
    if data_end_exclusive:
        events = events[events["occurred_at"] < pd.Timestamp(data_end_exclusive)]
    if events.empty:
        raise ValueError("No events remain inside the requested data period")
    dates = pd.date_range(
        events["occurred_at"].min().floor("D"),
        events["occurred_at"].max().floor("D"),
        freq="D",
    )
    cells = pd.Index(dataset.grid["cell_id"].astype(str), name="cell_id")
    crimes = pd.Index(TARGET_CRIME_ORDER, name="target_crime")
    cell_position = pd.Series(np.arange(len(cells), dtype=np.int32), index=cells)
    crime_position = pd.Series(np.arange(len(crimes), dtype=np.int16), index=crimes)

    event_days = events["occurred_at"].dt.floor("D")
    day_index = (event_days - dates[0]).dt.days.to_numpy(dtype=np.int32)
    cell_index = cell_position.reindex(events["cell_id"].astype(str)).to_numpy(
        dtype=np.int32
    )
    crime_index = crime_position.reindex(
        events["target_crime"].astype(str)
    ).to_numpy(dtype=np.int16)
    counts = np.zeros((len(dates), len(cells), len(crimes)), dtype=np.float32)
    np.add.at(counts, (day_index, cell_index, crime_index), 1.0)
    return DailyCrimeTensor(dates=dates, cells=cells, crimes=crimes, counts=counts)


def build_graph(grid: gpd.GeoDataFrame) -> tuple[torch.Tensor, torch.Tensor, dict]:
    by_position = {
        (int(row), int(col)): idx
        for idx, row, col in grid[["row", "col"]].itertuples(index=True, name=None)
    }
    source: list[int] = []
    target: list[int] = []
    for target_idx, row, col in grid[["row", "col"]].itertuples(index=True, name=None):
        for delta_row in (-1, 0, 1):
            for delta_col in (-1, 0, 1):
                source_idx = by_position.get((int(row) + delta_row, int(col) + delta_col))
                if source_idx is not None:
                    source.append(int(source_idx))
                    target.append(int(target_idx))
    source_array = np.asarray(source, dtype=np.int64)
    target_array = np.asarray(target, dtype=np.int64)
    degree = np.bincount(target_array, minlength=len(grid)).astype(np.float32)
    weight = 1.0 / degree[target_array]
    edge_index = torch.from_numpy(np.vstack([source_array, target_array]))
    edge_weight = torch.from_numpy(weight)
    return edge_index, edge_weight, {
        "nodes": int(len(grid)),
        "directed_edges_including_self_loops": int(len(source_array)),
        "minimum_in_degree": int(degree.min()),
        "maximum_in_degree": int(degree.max()),
    }


def split_indices(
    n_days: int,
    input_steps: int,
    output_steps: int,
    validation_days: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    split = int(math.floor(n_days * 7.0 / 8.0))
    validation_start = split - validation_days
    valid_targets = np.arange(input_steps, n_days - output_steps + 1, dtype=np.int64)
    train = valid_targets[valid_targets < validation_start]
    validation = valid_targets[
        (valid_targets >= validation_start) & (valid_targets < split)
    ]
    test = valid_targets[valid_targets >= split]
    if not len(train) or not len(validation) or not len(test):
        raise ValueError("The chronological split produced an empty partition")
    return train, validation, test, split


def standardize_features(
    counts: np.ndarray,
    training_day_stop: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    transformed = np.log1p(counts).astype(np.float32)
    training_values = transformed[:training_day_stop]
    mean = training_values.mean(axis=(0, 1), keepdims=True)
    std = training_values.std(axis=(0, 1), keepdims=True)
    std = np.maximum(std, 1e-5)
    return ((transformed - mean) / std).astype(np.float32), mean.reshape(-1), std.reshape(-1)


def make_loader(
    features: np.ndarray,
    counts: np.ndarray,
    target_indices: np.ndarray,
    args: argparse.Namespace,
    shuffle: bool,
) -> DataLoader:
    dataset = DailyWindowDataset(
        features=features,
        counts=counts,
        target_indices=target_indices,
        input_steps=args.input_days,
        output_steps=args.output_days,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=0,
        generator=torch.Generator().manual_seed(args.seed) if shuffle else None,
    )


def mean_loader_loss(
    model: STMGNNZINB,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    total = 0.0
    observations = 0
    with torch.no_grad():
        for features, target in loader:
            features = features.to(device)
            target = target.to(device)
            output = model(features)
            loss = zinb_nll(target, output, reduction="sum")
            total += float(loss.cpu())
            observations += int(target.numel())
    return total / max(observations, 1)


def train_model(
    config: STMGNNZINBConfig,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    args: argparse.Namespace,
) -> tuple[STMGNNZINB, pd.DataFrame, dict]:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.torch_threads)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = STMGNNZINB(config, edge_index=edge_index, edge_weight=edge_weight).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    best_state = copy.deepcopy(model.state_dict())
    best_validation = math.inf
    best_epoch = 0
    stale_epochs = 0
    history: list[dict] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_total = 0.0
        train_observations = 0
        for features, target in train_loader:
            features = features.to(device)
            target = target.to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(features)
            loss = zinb_nll(target, output)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            train_total += float(loss.detach().cpu()) * int(target.numel())
            train_observations += int(target.numel())

        train_loss = train_total / max(train_observations, 1)
        validation_loss = mean_loader_loss(model, validation_loader, device)
        history.append(
            {
                "epoch": epoch,
                "train_nll": train_loss,
                "validation_nll": validation_loss,
            }
        )
        improved = validation_loss < best_validation - args.min_delta
        if improved:
            best_validation = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        if epoch == 1 or epoch % args.log_every == 0 or improved:
            print(
                f"STMGNN-ZINB epoch {epoch}/{args.epochs}: "
                f"train_nll={train_loss:.6f} val_nll={validation_loss:.6f}"
            )
        if stale_epochs >= args.patience:
            print(f"Early stopping at epoch {epoch}; best epoch was {best_epoch}")
            break

    model.load_state_dict(best_state)
    summary = {
        "device": str(device),
        "torch_version": str(torch.__version__),
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "epochs_requested": int(args.epochs),
        "epochs_completed": int(len(history)),
        "best_epoch": int(best_epoch),
        "best_validation_nll": float(best_validation),
        "early_stopping_patience": int(args.patience),
    }
    return model, pd.DataFrame(history), summary


def predict(
    model: STMGNNZINB,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, np.ndarray]:
    values: dict[str, list[np.ndarray]] = {
        "observed": [],
        "mean": [],
        "variance": [],
        "pi": [],
        "p": [],
        "r": [],
    }
    model.eval()
    with torch.no_grad():
        for features, target in loader:
            output = model(features.to(device))
            values["observed"].append(target.numpy())
            for name in ("mean", "variance", "pi", "p", "r"):
                values[name].append(getattr(output, name).cpu().numpy())
    return {name: np.concatenate(parts, axis=0) for name, parts in values.items()}


def zinb_quantile(
    quantile: float,
    pi: np.ndarray,
    p: np.ndarray,
    r: np.ndarray,
) -> np.ndarray:
    adjusted = (quantile - pi) / np.maximum(1.0 - pi, 1e-12)
    result = np.zeros_like(pi, dtype=np.float32)
    positive = adjusted > 0
    if positive.any():
        q = np.clip(adjusted[positive], 1e-12, 1.0 - 1e-12)
        result[positive] = nbinom.ppf(q, r[positive], 1.0 - p[positive]).astype(np.float32)
    return result


def binary_f1(observed: np.ndarray, predicted: np.ndarray) -> float:
    actual_positive = observed > 0
    predicted_positive = predicted > 0
    true_positive = int(np.sum(actual_positive & predicted_positive))
    false_positive = int(np.sum(~actual_positive & predicted_positive))
    false_negative = int(np.sum(actual_positive & ~predicted_positive))
    denominator = 2 * true_positive + false_positive + false_negative
    return float(2 * true_positive / denominator) if denominator else 0.0


def distribution_kl(
    observed: np.ndarray,
    pi: np.ndarray,
    p: np.ndarray,
    r: np.ndarray,
) -> float:
    observed_int = observed.astype(np.int64).ravel()
    cutoff = max(10, min(int(observed_int.max(initial=0)), 100))
    empirical = np.bincount(np.minimum(observed_int, cutoff), minlength=cutoff + 1).astype(float)
    empirical /= max(empirical.sum(), 1.0)

    pi_flat = pi.ravel().astype(np.float64)
    p_flat = p.ravel().astype(np.float64)
    r_flat = r.ravel().astype(np.float64)
    predicted = np.zeros(cutoff + 1, dtype=np.float64)
    for count in range(cutoff):
        nb_log_probability = (
            gammaln(count + r_flat)
            - gammaln(r_flat)
            - gammaln(count + 1.0)
            + count * np.log(p_flat)
            + r_flat * np.log1p(-p_flat)
        )
        probability = (1.0 - pi_flat) * np.exp(nb_log_probability)
        if count == 0:
            probability += pi_flat
        predicted[count] = probability.mean()
    predicted[-1] = max(1.0 - predicted[:-1].sum(), 1e-12)
    predicted = np.clip(predicted, 1e-12, None)
    predicted /= predicted.sum()
    positive = empirical > 0
    return float(np.sum(empirical[positive] * np.log(empirical[positive] / predicted[positive])))


def evaluate_predictions(
    predictions: dict[str, np.ndarray],
    crimes: pd.Index,
    dates: pd.DatetimeIndex,
    low_quantile: float,
    high_quantile: float,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    observed = predictions["observed"]
    mean = predictions["mean"]
    pi = predictions["pi"]
    p = predictions["p"]
    r = predictions["r"]
    lower = zinb_quantile(low_quantile, pi, p, r)
    upper = zinb_quantile(high_quantile, pi, p, r)
    rounded = np.rint(mean).clip(min=0)
    total_zero_probability = pi + (1.0 - pi) * np.power(1.0 - p, r)

    rows: list[dict] = []
    scopes: list[tuple[str, np.ndarray | None]] = [("ALL", None)] + [
        (str(crime), np.asarray([index], dtype=np.int64))
        for index, crime in enumerate(crimes)
    ]
    for label, crime_indices in scopes:
        if crime_indices is None:
            y = observed
            mu = mean
            lo = lower
            hi = upper
            pi_scope = pi
            p_scope = p
            r_scope = r
            zero_probability = total_zero_probability
            pred = rounded
        else:
            y = observed[..., crime_indices]
            mu = mean[..., crime_indices]
            lo = lower[..., crime_indices]
            hi = upper[..., crime_indices]
            pi_scope = pi[..., crime_indices]
            p_scope = p[..., crime_indices]
            r_scope = r[..., crime_indices]
            zero_probability = total_zero_probability[..., crime_indices]
            pred = rounded[..., crime_indices]
        actual_zero = y == 0
        predicted_zero = pred == 0
        nll = -(
            np.where(
                actual_zero,
                np.log(pi_scope + (1.0 - pi_scope) * np.power(1.0 - p_scope, r_scope)),
                np.log1p(-pi_scope)
                + gammaln(y + r_scope)
                - gammaln(r_scope)
                - gammaln(y + 1.0)
                + y * np.log(p_scope)
                + r_scope * np.log1p(-p_scope),
            )
        ).mean()
        true_zero_rate = float(np.mean(predicted_zero[actual_zero])) if actual_zero.any() else np.nan
        metrics = {
            "mae": float(np.mean(np.abs(y - mu))),
            "picp": float(np.mean((y >= lo) & (y <= hi))),
            "mpiw": float(np.mean(hi - lo)),
            "occurrence_f1": binary_f1(y, pred),
            "true_zero_rate": true_zero_rate,
            "actual_zero_rate": float(np.mean(actual_zero)),
            "predicted_zero_rate": float(np.mean(predicted_zero)),
            "mean_extra_zero_probability_pi": float(np.mean(pi_scope)),
            "mean_total_zero_probability": float(np.mean(zero_probability)),
            "mean_observed_count": float(np.mean(y)),
            "mean_predicted_count": float(np.mean(mu)),
            "zinb_nll_per_observation": float(nll),
            "distribution_kl": distribution_kl(y, pi_scope, p_scope, r_scope),
        }
        for metric, value in metrics.items():
            rows.append(
                {
                    "scope": label,
                    "metric": metric,
                    "value": value,
                    "low_quantile": low_quantile,
                    "high_quantile": high_quantile,
                }
            )

    daily_rows: list[dict] = []
    for day_index, date in enumerate(dates):
        y = observed[day_index]
        mu = mean[day_index]
        lo = lower[day_index]
        hi = upper[day_index]
        pred = rounded[day_index]
        actual_zero = y == 0
        daily_rows.append(
            {
                "forecast_date": date.date().isoformat(),
                "mae": float(np.mean(np.abs(y - mu))),
                "picp": float(np.mean((y >= lo) & (y <= hi))),
                "mpiw": float(np.mean(hi - lo)),
                "occurrence_f1": binary_f1(y, pred),
                "true_zero_rate": (
                    float(np.mean((pred == 0)[actual_zero])) if actual_zero.any() else np.nan
                ),
                "observed_events": float(y.sum()),
                "predicted_events": float(mu.sum()),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(daily_rows), lower, upper


def save_prediction_table(
    output_path: Path,
    predictions: dict[str, np.ndarray],
    lower: np.ndarray,
    upper: np.ndarray,
    dates: pd.DatetimeIndex,
    cells: pd.Index,
    crimes: pd.Index,
) -> None:
    n_days, n_cells, output_steps, n_crimes = predictions["mean"].shape
    if output_steps != 1:
        raise NotImplementedError("Long-form output currently expects one-day forecasts")
    date_values = np.repeat(dates.to_numpy(dtype="datetime64[D]"), n_cells * n_crimes)
    cell_values = np.tile(np.repeat(cells.to_numpy(), n_crimes), n_days)
    crime_values = np.tile(crimes.to_numpy(), n_days * n_cells)
    frame = pd.DataFrame(
        {
            "forecast_date": date_values,
            "cell_id": cell_values,
            "target_crime": crime_values,
            "observed_count": predictions["observed"][:, :, 0, :].reshape(-1),
            "predicted_mean": predictions["mean"][:, :, 0, :].reshape(-1),
            "lower_10": lower[:, :, 0, :].reshape(-1),
            "upper_90": upper[:, :, 0, :].reshape(-1),
            "pi": predictions["pi"][:, :, 0, :].reshape(-1),
            "p": predictions["p"][:, :, 0, :].reshape(-1),
            "r": predictions["r"][:, :, 0, :].reshape(-1),
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

    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    ax.plot(training["epoch"], training["train_nll"], label="Train NLL")
    ax.plot(training["epoch"], training["validation_nll"], label="Validation NLL")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("NLL per observation")
    ax.set_title("STMGNN-ZINB training history")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(plot_dir / "training_history.png", dpi=170)
    plt.close(fig)

    crime_summary = summary[summary["scope"] != "ALL"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for metric, color in (("mae", "#2b8cbe"), ("mpiw", "#e34a33")):
        values = crime_summary[crime_summary["metric"] == metric]
        axes[0].plot(values["scope"], values["value"], marker="o", label=metric.upper(), color=color)
    axes[0].set_title("Point error and interval width")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].legend()
    for metric, color in (
        ("picp", "#756bb1"),
        ("occurrence_f1", "#31a354"),
        ("true_zero_rate", "#636363"),
    ):
        values = crime_summary[crime_summary["metric"] == metric]
        axes[1].plot(values["scope"], values["value"], marker="o", label=metric, color=color)
    axes[1].axhline(0.8, color="#999999", linestyle="--", linewidth=1, label="nominal 10%-90%")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("Coverage and discrete metrics")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].legend(fontsize=8)
    fig.savefig(plot_dir / "paper_metrics_by_crime.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, constrained_layout=True)
    dates = pd.to_datetime(daily["forecast_date"])
    axes[0].plot(dates, daily["observed_events"], label="Observed", color="#252525")
    axes[0].plot(dates, daily["predicted_events"], label="Predicted mean", color="#de2d26")
    axes[0].set_ylabel("Daily events")
    axes[0].set_title("Daily test-set predictions")
    axes[0].grid(alpha=0.2)
    axes[0].legend()
    axes[1].plot(dates, daily["picp"], label="PICP", color="#756bb1")
    axes[1].axhline(0.8, color="#999999", linestyle="--", linewidth=1, label="nominal 0.80")
    axes[1].set_ylabel("Coverage")
    axes[1].set_ylim(0, 1.02)
    axes[1].grid(alpha=0.2)
    axes[1].legend()
    fig.savefig(plot_dir / "daily_forecast_diagnostics.png", dpi=170)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell-size-m", type=int, default=3000)
    parser.add_argument("--input-days", type=int, default=28)
    parser.add_argument("--output-days", type=int, default=1)
    parser.add_argument("--validation-days", type=int, default=30)
    parser.add_argument(
        "--data-end-exclusive",
        default="2024-12-01",
        help="Exclude the right-censored December 2024 tail by default.",
    )
    parser.add_argument("--hidden-dim", type=int, default=24)
    parser.add_argument("--graph-layers", type=int, default=1)
    parser.add_argument("--temporal-hidden-steps", type=int, default=14)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--low-quantile", type=float, default=0.10)
    parser.add_argument("--high-quantile", type=float, default=0.90)
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
        raise ValueError("The paper-constrained experiment requires 3000 m cells")
    if args.output_days != 1:
        raise ValueError("This experiment currently reports one-day-ahead forecasts only")
    args.history_days = args.input_days

    base = load_target_dataset(overwrite=False, cell_size_m=150)
    dataset = build_coarse_dataset(base, args.cell_size_m, args.overwrite_derived_data)
    daily = make_daily_tensor(dataset, args.data_end_exclusive)
    train_indices, validation_indices, test_indices, split_index = split_indices(
        n_days=len(daily.dates),
        input_steps=args.input_days,
        output_steps=args.output_days,
        validation_days=args.validation_days,
    )
    validation_start = split_index - args.validation_days
    features, feature_mean, feature_std = standardize_features(
        daily.counts, training_day_stop=validation_start
    )
    train_loader = make_loader(features, daily.counts, train_indices, args, shuffle=True)
    validation_loader = make_loader(
        features, daily.counts, validation_indices, args, shuffle=False
    )
    test_loader = make_loader(features, daily.counts, test_indices, args, shuffle=False)
    edge_index, edge_weight, graph_summary = build_graph(dataset.grid)

    config = STMGNNZINBConfig(
        n_nodes=len(daily.cells),
        n_crimes=len(daily.crimes),
        input_steps=args.input_days,
        output_steps=args.output_days,
        hidden_dim=args.hidden_dim,
        graph_layers=args.graph_layers,
        temporal_hidden_steps=args.temporal_hidden_steps,
        dropout=args.dropout,
    )
    model, training_history, training_summary = train_model(
        config=config,
        edge_index=edge_index,
        edge_weight=edge_weight,
        train_loader=train_loader,
        validation_loader=validation_loader,
        args=args,
    )
    device = next(model.parameters()).device
    predictions = predict(model, test_loader, device)
    test_dates = daily.dates[test_indices]
    summary_metrics, daily_metrics, lower, upper = evaluate_predictions(
        predictions=predictions,
        crimes=daily.crimes,
        dates=test_dates,
        low_quantile=args.low_quantile,
        high_quantile=args.high_quantile,
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
        table_dir / "zinb_daily_predictions.parquet",
        predictions,
        lower,
        upper,
        test_dates,
        daily.cells,
        daily.crimes,
    )

    frame_offsets = np.arange(0, len(test_indices), args.frame_days, dtype=np.int64)
    frame_risk = predictions["mean"][frame_offsets, :, 0, :].sum(axis=-1).astype(np.float32)
    records: list[dict] = []
    for frame_index, prediction_offset in enumerate(frame_offsets):
        cutoff = test_dates[int(prediction_offset)]
        display_time = cutoff + pd.Timedelta(days=args.output_days)
        records.append(
            {
                "frame_index": frame_index,
                "forecast_date": cutoff,
                "display_time": display_time,
                "actual_events_in_window": observed_window_counts(
                    dataset.events, cutoff, display_time
                ),
                "predicted_events": float(frame_risk[frame_index].sum()),
                "mean_pi": float(predictions["pi"][prediction_offset].mean()),
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
            title="LAPD STMGNN-ZINB",
            note=(
                "Heatmap: predicted 24h mean count from the ZINB distribution. "
                "Colored rings: observed target-crime events."
            ),
        )

    split_summary = {
        "full_start": daily.dates[0].date().isoformat(),
        "full_end": daily.dates[-1].date().isoformat(),
        "train_input_start": daily.dates[0].date().isoformat(),
        "train_target_start": daily.dates[train_indices[0]].date().isoformat(),
        "train_target_end": daily.dates[validation_indices[0] - 1].date().isoformat(),
        "validation_start": daily.dates[validation_indices[0]].date().isoformat(),
        "validation_end": daily.dates[validation_indices[-1]].date().isoformat(),
        "test_start": test_dates[0].date().isoformat(),
        "test_end": test_dates[-1].date().isoformat(),
        "train_samples": int(len(train_indices)),
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
            "paper": "Wang et al. (2024), arXiv:2408.04193v1",
            "reproduction_status": "paper-constrained; author code and neural hyperparameters unavailable",
            "split": split_summary,
            "model_config": asdict(config),
            "graph_summary": graph_summary,
            "training_summary": training_summary,
            "low_quantile": args.low_quantile,
            "high_quantile": args.high_quantile,
            "interval_nominal_coverage": args.high_quantile - args.low_quantile,
            "data_end_exclusive": args.data_end_exclusive,
            "data_cutoff_reason": (
                "December 2024 becomes sharply right-censored after mid-month; the experiment "
                "uses the last complete month through 2024-11-30."
            ),
            "sequential_forecast": (
                "Each daily test prediction uses the immediately preceding observed input window; "
                "GIF frames sample those rolling one-day-ahead predictions every frame_days days."
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
                "paper_specified": {
                    "cell_size_m": args.cell_size_m,
                    "temporal_resolution": "daily",
                    "split_ratio": "7:1 chronological",
                    "validation_days": args.validation_days,
                    "distribution": "zero-inflated negative binomial",
                    "loss": "direct negative log likelihood",
                    "fusion": "Hadamard product",
                },
                "experiment_defined": {
                    "input_days": args.input_days,
                    "output_days": args.output_days,
                    "hidden_dim": args.hidden_dim,
                    "graph_layers": args.graph_layers,
                    "temporal_hidden_steps": args.temporal_hidden_steps,
                    "dropout": args.dropout,
                    "batch_size": args.batch_size,
                    "epochs": args.epochs,
                    "optimizer": "Adam",
                    "learning_rate": args.learning_rate,
                    "weight_decay": args.weight_decay,
                    "seed": args.seed,
                    "data_end_exclusive": args.data_end_exclusive,
                },
                "split": split_summary,
                "graph": graph_summary,
                "training": training_summary,
            },
            handle,
            sort_keys=False,
            allow_unicode=True,
        )
    print(f"Wrote {RESULT_DIR}")


if __name__ == "__main__":
    main()
