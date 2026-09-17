"""Run vol5: STNPP-GAT with an ST-MoGE-enhanced HCL-ZINB branch.

Vol5 preserves vol4's 150 m GAT/ETAS allocation and inverse-variance fusion.
At 3 km resolution, HCL becomes the universal expert, category-specific graph
experts provide distinct crime patterns, attentive gates fuse both sources,
regional predictors parameterize ZINB distributions, and HALR balances the
category/region likelihood. CECL is deliberately omitted because its expert
separation objective can oppose HCL's crime-type representation alignment.
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

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader

from our_experiment.common.lapd_target_common import (
    load_target_dataset,
    save_forecast_animation,
    write_forecast_tables,
)
from our_experiment.vol1.run_lapd_stnpp_gat import (
    build_mark_indices,
    empirical_transition_matrix,
    event_arrays,
    make_mark_features,
    train_mark_gat,
)
from our_experiment.vol3.run_lapd_stnpp_gat_zinb import (
    chronological_indices,
    fine_to_coarse_indices,
    replay_vol3,
)
from our_experiment.vol4.run_lapd_stnpp_gat_hcl_zinb import (
    HCLZINBWindowDataset,
)
from our_model.vol1.stnpp_gat import MarkGraphAttention, STNPPGATConfig
from our_model.vol5 import (
    MoGEHCLZINBConfig,
    MoGEHCLZINBCountModel,
    moge_hcl_zinb_loss,
)
from ref_models.experiments.hcl.run_lapd_hcl import (
    build_cardinal_graph,
    category_coefficients,
)
from ref_models.experiments.st_moge.run_lapd_st_moge import (
    frequency_quantile_metrics,
)
from ref_models.experiments.stmgnn_zinb.run_lapd_stmgnn_zinb import (
    build_coarse_dataset,
    evaluate_predictions,
    make_daily_tensor,
    standardize_features,
)


MODEL_ID = "O5_stnpp_gat_moge_hcl_zinb_multiscale"


def make_window_loader(
    features: np.ndarray,
    counts: np.ndarray,
    target_indices: np.ndarray,
    args: argparse.Namespace,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    return DataLoader(
        HCLZINBWindowDataset(
            features=features,
            counts=counts,
            target_indices=target_indices,
            input_steps=args.input_days,
        ),
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=0,
        generator=torch.Generator().manual_seed(seed) if shuffle else None,
    )


def update_region_clusters(
    model: MoGEHCLZINBCountModel,
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
    category_rate = category_history[-1] / np.maximum(
        category_history[-2], epsilon
    )
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


def mean_loader_losses(
    model: MoGEHCLZINBCountModel,
    loader: DataLoader,
    neighbor_center: torch.Tensor,
    neighbor_node: torch.Tensor,
    category_coefficient: torch.Tensor,
    category_weight: torch.Tensor,
    cluster_weight: torch.Tensor,
    args: argparse.Namespace,
) -> dict[str, float]:
    model.eval()
    totals = {
        name: 0.0
        for name in (
            "total",
            "weighted_nll",
            "unweighted_nll",
            "type",
            "neighbor",
        )
    }
    samples = 0
    device = next(model.parameters()).device
    with torch.no_grad():
        for features, raw_history, target in loader:
            features = features.to(device)
            raw_history = raw_history.to(device)
            target = target.to(device)
            loss = moge_hcl_zinb_loss(
                model(features),
                target,
                raw_history,
                neighbor_center,
                neighbor_node,
                category_coefficient,
                model.cluster_assignments,
                category_weight,
                cluster_weight,
                lambda_type=args.lambda_type,
                lambda_neighbor=args.lambda_neighbor,
                contrast_steps=args.contrast_steps,
            )
            batch = int(features.shape[0])
            comparable_total = (
                loss.unweighted_nll
                + args.lambda_type * loss.type_contrast
                + args.lambda_neighbor * loss.neighbor_contrast
            )
            values = {
                "total": comparable_total,
                "weighted_nll": loss.weighted_nll,
                "unweighted_nll": loss.unweighted_nll,
                "type": loss.type_contrast,
                "neighbor": loss.neighbor_contrast,
            }
            for name, value in values.items():
                totals[name] += float(value.cpu()) * batch
            samples += batch
    return {name: value / max(samples, 1) for name, value in totals.items()}


def train_count_model(
    config: MoGEHCLZINBConfig,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    neighbor_center_cpu: torch.Tensor,
    neighbor_node_cpu: torch.Tensor,
    category_coefficient_array: np.ndarray,
    features: np.ndarray,
    counts: np.ndarray,
    train_indices: np.ndarray,
    validation_loader: DataLoader,
    args: argparse.Namespace,
) -> tuple[MoGEHCLZINBCountModel, pd.DataFrame, dict]:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.torch_threads)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = MoGEHCLZINBCountModel(config, edge_index, edge_weight).to(device)
    neighbor_center = neighbor_center_cpu.to(device)
    neighbor_node = neighbor_node_cpu.to(device)
    category_coefficient = torch.from_numpy(category_coefficient_array).to(device)
    optimizer = torch.optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    best_state = copy.deepcopy(model.state_dict())
    best_validation_nll = math.inf
    best_epoch = 0
    stale_epochs = 0
    centers: list[np.ndarray] | None = None
    category_history: list[np.ndarray] = []
    cluster_history: list[np.ndarray] = []
    visited_targets: set[int] = set()
    rows: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        if epoch <= args.cluster_refresh_epochs:
            centers = update_region_clusters(model, centers, args.seed)
        category_weight_np, cluster_weight_np = halr_weights(
            category_history,
            cluster_history,
            args.halr_temperature,
            config.n_crimes,
            config.n_clusters,
        )
        category_weight = torch.from_numpy(category_weight_np).to(device)
        cluster_weight = torch.from_numpy(cluster_weight_np).to(device)

        offset = (epoch - 1) % args.train_stride
        epoch_indices = train_indices[offset:: args.train_stride]
        visited_targets.update(int(value) for value in epoch_indices)
        train_loader = make_window_loader(
            features,
            counts,
            epoch_indices,
            args,
            shuffle=True,
            seed=args.seed + epoch,
        )
        model.train()
        totals = {
            name: 0.0
            for name in (
                "total",
                "weighted_nll",
                "unweighted_nll",
                "type",
                "neighbor",
            )
        }
        cluster_total = np.zeros(
            (config.n_crimes, config.n_clusters),
            dtype=np.float64,
        )
        samples = 0
        for features_batch, raw_history, target in train_loader:
            features_batch = features_batch.to(device)
            raw_history = raw_history.to(device)
            target = target.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = moge_hcl_zinb_loss(
                model(features_batch),
                target,
                raw_history,
                neighbor_center,
                neighbor_node,
                category_coefficient,
                model.cluster_assignments,
                category_weight,
                cluster_weight,
                lambda_type=args.lambda_type,
                lambda_neighbor=args.lambda_neighbor,
                contrast_steps=args.contrast_steps,
            )
            if not torch.isfinite(loss.total):
                raise RuntimeError(f"Non-finite vol5 loss at epoch {epoch}")
            loss.total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            batch = int(features_batch.shape[0])
            values = {
                "total": loss.total,
                "weighted_nll": loss.weighted_nll,
                "unweighted_nll": loss.unweighted_nll,
                "type": loss.type_contrast,
                "neighbor": loss.neighbor_contrast,
            }
            for name, value in values.items():
                totals[name] += float(value.detach().cpu()) * batch
            cluster_total += loss.cluster_nll.detach().cpu().numpy() * batch
            samples += batch

        cluster_epoch = (cluster_total / max(samples, 1)).astype(np.float32)
        category_epoch = cluster_epoch.mean(axis=1).astype(np.float32)
        category_history.append(category_epoch)
        cluster_history.append(cluster_epoch)
        train_values = {name: value / max(samples, 1) for name, value in totals.items()}
        validation_values = mean_loader_losses(
            model,
            validation_loader,
            neighbor_center,
            neighbor_node,
            category_coefficient,
            category_weight,
            cluster_weight,
            args,
        )
        improved = (
            validation_values["unweighted_nll"]
            < best_validation_nll - args.minimum_delta
        )
        if improved:
            best_validation_nll = validation_values["unweighted_nll"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1

        row = {
            "epoch": epoch,
            "cyclic_offset": offset,
            "used_train_targets": len(epoch_indices),
            **{f"train_{name}": value for name, value in train_values.items()},
            **{
                f"validation_{name}": value
                for name, value in validation_values.items()
            },
            "elapsed_seconds": time.perf_counter() - started,
        }
        for crime in range(config.n_crimes):
            row[f"halr_category_weight_{crime}"] = float(category_weight_np[crime])
            for cluster in range(config.n_clusters):
                row[f"halr_cluster_weight_{crime}_{cluster}"] = float(
                    cluster_weight_np[crime, cluster]
                )
        rows.append(row)
        if epoch == 1 or epoch % args.log_every == 0 or improved:
            print(
                f"Vol5 epoch {epoch}/{args.epochs}: "
                f"train={train_values['total']:.6f} "
                f"val_nll={validation_values['unweighted_nll']:.6f} "
                f"type={validation_values['type']:.6f} "
                f"neighbor={validation_values['neighbor']:.6f} "
                f"elapsed={row['elapsed_seconds']:.1f}s",
                flush=True,
            )
        if stale_epochs >= args.patience:
            print(
                f"Vol5 early stopping at epoch {epoch}; best epoch was {best_epoch}",
                flush=True,
            )
            break

    model.load_state_dict(best_state)
    summary = {
        "device": str(device),
        "torch_version": str(torch.__version__),
        "trainable_parameter_count": int(
            sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            )
        ),
        "epochs_requested": int(args.epochs),
        "epochs_completed": int(len(rows)),
        "best_epoch": int(best_epoch),
        "best_validation_unweighted_nll": float(best_validation_nll),
        "early_stopping_patience": int(args.patience),
        "distinct_train_targets_visited": int(len(visited_targets)),
        "available_train_targets": int(len(train_indices)),
    }
    return model, pd.DataFrame(rows), summary


def predict_count_distributions(
    model: MoGEHCLZINBCountModel,
    loader: DataLoader,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    device = next(model.parameters()).device
    values: dict[str, list[np.ndarray]] = {
        name: [] for name in ("observed", "mean", "variance", "pi", "p", "r")
    }
    gates = []
    model.eval()
    with torch.no_grad():
        for features, _, target in loader:
            output = model(features.to(device))
            values["observed"].append(target.numpy())
            for name in ("mean", "variance", "pi", "p", "r"):
                values[name].append(
                    getattr(output.distribution, name).cpu().numpy()
                )
            gates.append(output.gate.cpu().numpy())
    predictions = {
        name: np.concatenate(parts, axis=0) for name, parts in values.items()
    }
    return predictions, np.concatenate(gates, axis=0)


def save_distribution_table(
    path: Path,
    predictions: dict[str, np.ndarray],
    gates: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    dates: pd.DatetimeIndex,
    cells: pd.Index,
    crimes: pd.Index,
) -> None:
    n_days, n_cells, output_steps, n_crimes = predictions["mean"].shape
    if output_steps != 1:
        raise NotImplementedError("Vol5 stores one-day forecasts only")
    pd.DataFrame(
        {
            "forecast_date": np.repeat(
                dates.to_numpy(dtype="datetime64[D]"), n_cells * n_crimes
            ),
            "cell_id": np.tile(np.repeat(cells.to_numpy(), n_crimes), n_days),
            "target_crime": np.tile(crimes.to_numpy(), n_days * n_cells),
            "observed_count": predictions["observed"][:, :, 0, :].reshape(-1),
            "predicted_mean": predictions["mean"][:, :, 0, :].reshape(-1),
            "lower_10": lower[:, :, 0, :].reshape(-1),
            "upper_90": upper[:, :, 0, :].reshape(-1),
            "pi": predictions["pi"][:, :, 0, :].reshape(-1),
            "p": predictions["p"][:, :, 0, :].reshape(-1),
            "r": predictions["r"][:, :, 0, :].reshape(-1),
            "specific_gate_weight": gates.reshape(-1),
        }
    ).to_parquet(path, index=False)


def gate_tables(
    model: MoGEHCLZINBCountModel,
    gates: np.ndarray,
    training_counts: np.ndarray,
    cells: pd.Index,
    crimes: pd.Index,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    assignments = model.cluster_assignments.detach().cpu().numpy()
    mean_gate = gates.mean(axis=0)
    frequency = training_counts.sum(axis=0)
    detail_rows = []
    summary_rows = []
    for crime, crime_name in enumerate(crimes):
        for node, cell_id in enumerate(cells):
            detail_rows.append(
                {
                    "cell_id": cell_id,
                    "target_crime": str(crime_name),
                    "cluster": int(assignments[crime, node]),
                    "mean_specific_gate": float(mean_gate[node, crime]),
                    "training_events": float(frequency[node, crime]),
                }
            )
        for cluster in range(model.config.n_clusters):
            mask = assignments[crime] == cluster
            summary_rows.append(
                {
                    "target_crime": str(crime_name),
                    "cluster": cluster,
                    "cell_count": int(mask.sum()),
                    "mean_specific_gate": float(mean_gate[mask, crime].mean()),
                    "mean_training_events": float(frequency[mask, crime].mean()),
                }
            )
    return pd.DataFrame(detail_rows), pd.DataFrame(summary_rows)


def save_plots(
    metric_dir: Path,
    summary: pd.DataFrame,
    daily: pd.DataFrame,
    training: pd.DataFrame,
    frequency: pd.DataFrame,
    gates: pd.DataFrame,
) -> None:
    plot_dir = metric_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    axes[0].plot(training["epoch"], training["train_total"], label="Train total")
    axes[0].plot(
        training["epoch"],
        training["validation_total"],
        label="Validation total",
    )
    axes[0].set_title("Vol5 MoGE-HCL-ZINB objective")
    axes[0].set_xlabel("Epoch")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    for column, label in (
        ("validation_unweighted_nll", "ZINB NLL"),
        ("validation_type", "Type alignment"),
        ("validation_neighbor", "Neighbor alignment"),
    ):
        axes[1].plot(training["epoch"], training[column], label=label)
    axes[1].set_title("Validation components")
    axes[1].set_xlabel("Epoch")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    fig.savefig(plot_dir / "moge_hcl_zinb_training_history.png", dpi=170)
    plt.close(fig)

    crime_summary = summary[summary["scope"] != "ALL"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for metric, color in (("mae", "#2b8cbe"), ("mpiw", "#e34a33")):
        values = crime_summary[crime_summary["metric"] == metric]
        axes[0].plot(
            values["scope"],
            values["value"],
            marker="o",
            label=metric.upper(),
            color=color,
        )
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
        axes[1].plot(
            values["scope"],
            values["value"],
            marker="o",
            label=metric,
            color=color,
        )
    axes[1].axhline(0.8, color="#999999", linestyle="--", linewidth=1)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("Coverage and discrete metrics")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].legend(fontsize=8)
    fig.savefig(plot_dir / "moge_hcl_zinb_metrics_by_crime.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, constrained_layout=True)
    dates = pd.to_datetime(daily["forecast_date"])
    axes[0].plot(dates, daily["observed_events"], label="Observed", color="#252525")
    axes[0].plot(
        dates,
        daily["predicted_events"],
        label="Predicted mean",
        color="#de2d26",
    )
    axes[0].set_ylabel("Daily events")
    axes[0].set_title("Vol5 coarse daily predictions")
    axes[0].grid(alpha=0.2)
    axes[0].legend()
    axes[1].plot(dates, daily["picp"], label="PICP", color="#756bb1")
    axes[1].axhline(0.8, color="#999999", linestyle="--", linewidth=1)
    axes[1].set_ylabel("Coverage")
    axes[1].set_ylim(0, 1.02)
    axes[1].grid(alpha=0.2)
    axes[1].legend()
    fig.savefig(plot_dir / "moge_hcl_zinb_daily_diagnostics.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for crime, values in frequency.groupby("target_crime", sort=False):
        axes[0].plot(
            values["frequency_quantile"],
            values["mae"],
            marker="o",
            label=crime,
        )
    axes[0].set_title("MAE by training-frequency quartile")
    axes[0].set_ylabel("MAE")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    pivot = gates.pivot(
        index="target_crime",
        columns="cluster",
        values="mean_specific_gate",
    )
    image = axes[1].imshow(
        pivot.to_numpy(),
        aspect="auto",
        vmin=0,
        vmax=1,
        cmap="viridis",
    )
    axes[1].set_yticks(np.arange(len(pivot.index)), pivot.index)
    axes[1].set_xticks(
        np.arange(len(pivot.columns)),
        [f"Cluster {value}" for value in pivot.columns],
    )
    axes[1].set_title("Mean category-specific gate")
    fig.colorbar(image, ax=axes[1], label="Specific expert weight")
    fig.savefig(plot_dir / "moge_regional_diagnostics.png", dpi=170)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-start", default="2010-01-01")
    parser.add_argument("--train-end", default="2020-01-01")
    parser.add_argument("--data-end-exclusive", default="2024-12-01")
    parser.add_argument("--validation-days", type=int, default=30)
    parser.add_argument("--cell-size-m", type=int, default=3000)
    parser.add_argument("--input-days", type=int, default=28)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--hypergraph-layers", type=int, default=1)
    parser.add_argument("--hcl-dropout", type=float, default=0.20)
    parser.add_argument("--hawkes-scope", type=int, default=3)
    parser.add_argument("--hawkes-delta", type=float, default=0.01)
    parser.add_argument("--lambda-type", type=float, default=0.15)
    parser.add_argument("--lambda-neighbor", type=float, default=0.15)
    parser.add_argument("--contrast-steps", type=int, default=1)
    parser.add_argument("--specific-st-blocks", type=int, default=1)
    parser.add_argument("--specific-spatial-layers", type=int, default=1)
    parser.add_argument("--specific-temporal-layers", type=int, default=2)
    parser.add_argument("--temporal-kernel-size", type=int, default=3)
    parser.add_argument("--node-embedding-dim", type=int, default=16)
    parser.add_argument("--moge-attention-heads", type=int, default=4)
    parser.add_argument("--clusters", type=int, default=4)
    parser.add_argument("--specific-dropout", type=float, default=0.10)
    parser.add_argument("--halr-temperature", type=float, default=1.0)
    parser.add_argument("--cluster-refresh-epochs", type=int, default=2)
    parser.add_argument("--train-stride", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--minimum-delta", type=float, default=1e-5)
    parser.add_argument("--minimum-fusion-variance", type=float, default=1e-4)
    parser.add_argument("--allocation-epsilon", type=float, default=1e-8)
    parser.add_argument("--history-days", type=int, default=365)
    parser.add_argument("--horizon-hours", type=int, default=24)
    parser.add_argument("--frame-days", type=int, default=14)
    parser.add_argument("--refit-days", type=int, default=28)
    parser.add_argument("--lookback-days", type=float, default=30.0)
    parser.add_argument("--initial-theta", type=float, default=0.35)
    parser.add_argument("--initial-omega", type=float, default=1.0 / 14.0)
    parser.add_argument("--theta-min", type=float, default=0.01)
    parser.add_argument("--background-alpha", type=float, default=1e-3)
    parser.add_argument("--transition-smoothing", type=float, default=0.25)
    parser.add_argument("--gat-hidden-dim", type=int, default=64)
    parser.add_argument("--attention-heads", type=int, default=8)
    parser.add_argument("--gat-dropout", type=float, default=0.05)
    parser.add_argument("--gat-epochs", type=int, default=1500)
    parser.add_argument("--gat-learning-rate", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--device")
    parser.add_argument("--torch-threads", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--fade-days", type=int, default=28)
    parser.add_argument("--pop-days", type=float, default=3.0)
    parser.add_argument("--smooth-sigma", type=float, default=2.1)
    parser.add_argument("--vmax-quantile", type=float, default=0.985)
    parser.add_argument("--heatmap-gamma", type=float, default=0.42)
    parser.add_argument("--duration-ms", type=int, default=105)
    parser.add_argument("--dpi", type=int, default=82)
    parser.add_argument("--event-location-mode", choices=("event", "cell"), default="event")
    parser.add_argument("--low-quantile", type=float, default=0.10)
    parser.add_argument("--high-quantile", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--reuse-vol4-gat-state",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--vol4-state-path",
        type=Path,
        default=ROOT / "our_experiment/vol4/results/model/model_state.pt",
    )
    parser.add_argument("--overwrite-derived-data", action="store_true")
    parser.add_argument("--skip-gif", action="store_true")
    parser.add_argument("--reuse-model-state", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.cell_size_m != 3000:
        raise ValueError("Vol5 requires the 3000 m coarse grid")
    if args.horizon_hours != 24:
        raise ValueError("Vol5 supports a 24-hour forecast horizon")
    if args.train_stride < 1:
        raise ValueError("train_stride must be positive")
    if args.hidden_dim % args.moge_attention_heads:
        raise ValueError("hidden_dim must be divisible by moge_attention_heads")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    result_dir = ROOT / "our_experiment" / "vol5" / "results"

    fine = load_target_dataset(overwrite=args.overwrite_derived_data, cell_size_m=150)
    coarse = build_coarse_dataset(fine, args.cell_size_m, args.overwrite_derived_data)
    daily = make_daily_tensor(coarse, args.data_end_exclusive)
    train_all, validation_indices, test_indices, validation_start_index = (
        chronological_indices(
            dates=daily.dates,
            input_steps=args.input_days,
            output_steps=1,
            train_start=pd.Timestamp(args.train_start),
            train_end=pd.Timestamp(args.train_end),
            validation_days=args.validation_days,
        )
    )
    features, feature_mean, feature_std = standardize_features(
        daily.counts,
        training_day_stop=validation_start_index,
    )
    validation_loader = make_window_loader(
        features,
        daily.counts,
        validation_indices,
        args,
        shuffle=False,
        seed=args.seed,
    )
    test_loader = make_window_loader(
        features,
        daily.counts,
        test_indices,
        args,
        shuffle=False,
        seed=args.seed,
    )
    (
        edge_index,
        edge_weight,
        neighbor_center,
        neighbor_node,
        graph_summary,
    ) = build_cardinal_graph(coarse.grid)
    coefficient = category_coefficients(
        daily.counts,
        training_day_stop=validation_start_index,
        neighbor_center=neighbor_center,
        neighbor_node=neighbor_node,
    )
    config = MoGEHCLZINBConfig(
        n_nodes=len(daily.cells),
        n_crimes=len(daily.crimes),
        input_steps=args.input_days,
        hidden_dim=args.hidden_dim,
        hypergraph_layers=args.hypergraph_layers,
        hcl_dropout=args.hcl_dropout,
        hawkes_scope=args.hawkes_scope,
        hawkes_delta=args.hawkes_delta,
        node_embedding_dim=args.node_embedding_dim,
        specific_st_blocks=args.specific_st_blocks,
        specific_spatial_layers=args.specific_spatial_layers,
        specific_temporal_layers=args.specific_temporal_layers,
        temporal_kernel_size=args.temporal_kernel_size,
        attention_heads=args.moge_attention_heads,
        n_clusters=args.clusters,
        specific_dropout=args.specific_dropout,
    )

    model_state_path = result_dir / "model" / "model_state.pt"
    checkpoint = None
    if args.reuse_model_state:
        if not model_state_path.exists():
            raise FileNotFoundError(f"Cannot reuse missing state: {model_state_path}")
        checkpoint = torch.load(model_state_path, map_location="cpu", weights_only=False)
        config = MoGEHCLZINBConfig(**checkpoint["count_model_config"])
        device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        count_model = MoGEHCLZINBCountModel(config, edge_index, edge_weight).to(device)
        count_model.load_state_dict(checkpoint["count_model_state_dict"])
        count_train_summary = checkpoint["count_training_summary"]
        count_history = pd.read_csv(
            result_dir / "metrics" / "moge_hcl_zinb_training_history.csv"
        )
        print(f"Reused vol5 model state from {model_state_path}", flush=True)
    else:
        count_model, count_history, count_train_summary = train_count_model(
            config,
            edge_index,
            edge_weight,
            neighbor_center,
            neighbor_node,
            coefficient,
            features,
            daily.counts,
            train_all,
            validation_loader,
            args,
        )

    coarse_predictions, coarse_gates = predict_count_distributions(
        count_model,
        test_loader,
    )
    test_dates = daily.dates[test_indices]
    coarse_summary, coarse_daily, lower, upper = evaluate_predictions(
        coarse_predictions,
        daily.crimes,
        test_dates,
        args.low_quantile,
        args.high_quantile,
    )
    frequency_metrics = frequency_quantile_metrics(
        coarse_predictions["observed"][:, :, 0, :],
        coarse_predictions["mean"][:, :, 0, :],
        daily.counts[:validation_start_index],
        daily.crimes,
    )
    gate_detail, gate_summary = gate_tables(
        count_model,
        coarse_gates,
        daily.counts[:validation_start_index],
        daily.cells,
        daily.crimes,
    )

    crimes, areas, crime_pos, area_pos, cell_area_index = build_mark_indices(fine)
    events = event_arrays(fine, crime_pos, area_pos)
    mark_features = make_mark_features(crimes, areas, fine)
    default_gat_config = STNPPGATConfig(
        n_crimes=len(crimes),
        n_areas=len(areas),
        hidden_dim=args.gat_hidden_dim,
        attention_heads=args.attention_heads,
        dropout=args.gat_dropout,
    )
    empirical, source_weight, transition_summary = empirical_transition_matrix(
        events=events,
        n_marks=default_gat_config.n_marks,
        omega=args.initial_omega,
        lookback_days=args.lookback_days,
        train_start=pd.Timestamp(args.train_start),
        train_end=pd.Timestamp(args.train_end),
        smoothing=args.transition_smoothing,
    )
    gat_source = "trained for vol5"
    if checkpoint is not None:
        gat_config = STNPPGATConfig(**checkpoint["gat_config"])
        gat_model = MarkGraphAttention(gat_config).to(next(count_model.parameters()).device)
        gat_model.load_state_dict(checkpoint["gat_state_dict"])
        gat_train_summary = checkpoint["gat_training_summary"]
        gat_source = checkpoint.get("gat_source", "vol5 checkpoint")
    elif args.reuse_vol4_gat_state:
        if not args.vol4_state_path.exists():
            raise FileNotFoundError(f"Missing vol4 GAT state: {args.vol4_state_path}")
        vol4_checkpoint = torch.load(
            args.vol4_state_path,
            map_location="cpu",
            weights_only=False,
        )
        gat_config = STNPPGATConfig(**vol4_checkpoint["gat_config"])
        gat_model = MarkGraphAttention(gat_config).to(next(count_model.parameters()).device)
        gat_model.load_state_dict(vol4_checkpoint["gat_state_dict"])
        gat_train_summary = vol4_checkpoint["gat_training_summary"]
        gat_source = str(args.vol4_state_path.relative_to(ROOT))
    else:
        gat_config = default_gat_config
        gat_args = argparse.Namespace(**vars(args))
        gat_args.epochs = args.gat_epochs
        gat_args.learning_rate = args.gat_learning_rate
        gat_model, gat_train_summary, _ = train_mark_gat(
            mark_features,
            empirical,
            source_weight,
            gat_config,
            gat_args,
        )
    gat_model.eval()
    with torch.no_grad():
        transition = (
            gat_model(
                torch.from_numpy(mark_features).to(next(gat_model.parameters()).device)
            )
            .cpu()
            .numpy()
            .astype(np.float32)
        )

    mapping = fine_to_coarse_indices(fine, coarse, args.cell_size_m)
    cells, records, risk_matrix, area_params, coarse_frame_predictions = replay_vol3(
        fine=fine,
        coarse=coarse,
        events=events,
        transition=transition,
        crime_pos=crime_pos,
        area_pos=area_pos,
        cell_area_index=cell_area_index,
        fine_to_coarse=mapping,
        coarse_predictions=coarse_predictions,
        test_dates=test_dates,
        args=args,
        run_label="Vol5",
    )

    gif_path = result_dir / "animations" / f"{MODEL_ID}_24h_forecast_heatmap.gif"
    if not args.skip_gif:
        gif_path = save_forecast_animation(
            out_path=gif_path,
            dataset=fine,
            records=records,
            risk_matrix=risk_matrix,
            args=args,
            title="LAPD STNPP-GAT-MoGE-HCL-ZINB vol5",
            note=(
                "Heatmap: MoGE-HCL-ZINB uncertainty-fused 24h mean. "
                "Colored rings: observed target-crime events."
            ),
        )

    split_summary = {
        "full_start": daily.dates[0].date().isoformat(),
        "full_end": daily.dates[-1].date().isoformat(),
        "train_target_start": daily.dates[train_all[0]].date().isoformat(),
        "train_target_end": daily.dates[train_all[-1]].date().isoformat(),
        "validation_start": daily.dates[validation_indices[0]].date().isoformat(),
        "validation_end": daily.dates[validation_indices[-1]].date().isoformat(),
        "test_start": test_dates[0].date().isoformat(),
        "test_end": test_dates[-1].date().isoformat(),
        "available_train_samples": int(len(train_all)),
        "train_stride_days": int(args.train_stride),
        "validation_samples": int(len(validation_indices)),
        "test_samples": int(len(test_indices)),
        "gif_frames": int(len(records)),
    }
    write_forecast_tables(
        result_dir=result_dir,
        model_id=MODEL_ID,
        dataset=fine,
        records=records,
        cells=cells,
        risk_matrix=risk_matrix,
        args=args,
        gif_path=gif_path,
        extra_config={
            "model_development": (
                "Vol5 uses vol4 HCL as the universal coarse expert and adds "
                "category-specific graph experts, attentive gates, regional ZINB "
                "heads, and HALR. Fine GAT/ETAS and inverse-variance fusion remain."
            ),
            "cecl_decision": (
                "Not adopted: CECL expert separation can oppose HCL type alignment; "
                "vol5 isolates the MoE, gate, regional predictor, and HALR changes."
            ),
            "coarse_dataset_id": coarse.metadata["dataset_id"],
            "coarse_cell_size_m": args.cell_size_m,
            "fine_cell_size_m": fine.metadata["spatial"]["cell_size_m"],
            "split": split_summary,
            "count_model_config": asdict(config),
            "graph_summary": graph_summary,
            "count_training_summary": count_train_summary,
            "gat_training_summary": gat_train_summary,
            "gat_source": gat_source,
            "transition_summary": transition_summary,
            "lambda_type": args.lambda_type,
            "lambda_neighbor": args.lambda_neighbor,
            "halr_temperature": args.halr_temperature,
            "uncertainty_scope": (
                "ZINB uncertainty is defined at 3 km x crime x day; the 150 m "
                "surface receives the fused predictive mean only."
            ),
        },
    )

    metric_dir = result_dir / "metrics"
    table_dir = result_dir / "tables"
    model_dir = result_dir / "model"
    metric_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    coarse_summary.to_csv(
        metric_dir / "moge_hcl_zinb_summary_metrics.csv", index=False
    )
    coarse_daily.to_csv(
        metric_dir / "moge_hcl_zinb_daily_metrics.csv", index=False
    )
    count_history.to_csv(
        metric_dir / "moge_hcl_zinb_training_history.csv", index=False
    )
    frequency_metrics.to_csv(
        metric_dir / "frequency_quantile_metrics.csv", index=False
    )
    gate_summary.to_csv(metric_dir / "gate_summary.csv", index=False)
    save_plots(
        metric_dir,
        coarse_summary,
        coarse_daily,
        count_history,
        frequency_metrics,
        gate_summary,
    )
    save_distribution_table(
        table_dir / "coarse_moge_hcl_zinb_daily_predictions.parquet",
        coarse_predictions,
        coarse_gates,
        lower,
        upper,
        test_dates,
        daily.cells,
        daily.crimes,
    )
    gate_detail.to_csv(table_dir / "region_clusters_and_gates.csv", index=False)
    area_params.to_csv(table_dir / "area_etas_parameters.csv", index=False)
    coarse_frame_predictions.to_parquet(
        table_dir / "coarse_moge_hcl_zinb_frame_predictions.parquet",
        index=False,
    )
    pd.DataFrame(
        transition,
        columns=[f"source_{index}" for index in range(transition.shape[1])],
    ).to_csv(table_dir / "learned_mark_transition.csv", index_label="target_mark")
    torch.save(
        {
            "model_id": MODEL_ID,
            "count_model_state_dict": count_model.state_dict(),
            "count_model_config": asdict(config),
            "gat_state_dict": gat_model.state_dict(),
            "gat_config": asdict(gat_config),
            "gat_source": gat_source,
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "coarse_cells": daily.cells.tolist(),
            "crimes": daily.crimes.tolist(),
            "category_coefficients": coefficient,
            "split": split_summary,
            "count_training_summary": count_train_summary,
            "gat_training_summary": gat_train_summary,
        },
        model_dir / "model_state.pt",
    )
    with (table_dir / "training_summary.yml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            {
                "model_id": MODEL_ID,
                "split": split_summary,
                "count_model_config": asdict(config),
                "graph": graph_summary,
                "category_coefficients": {
                    str(crime): float(value)
                    for crime, value in zip(daily.crimes, coefficient, strict=True)
                },
                "objective": {
                    "task": "HALR-weighted regional ZINB negative log likelihood",
                    "checkpoint_metric": "unweighted validation ZINB NLL",
                    "lambda_type": args.lambda_type,
                    "lambda_neighbor": args.lambda_neighbor,
                    "halr_temperature": args.halr_temperature,
                    "cecl": "not used because it can conflict with HCL type alignment",
                },
                "experiment_defined": {
                    "cyclic_train_stride_days": args.train_stride,
                    "batch_size": args.batch_size,
                    "epochs": args.epochs,
                    "optimizer": "Adam",
                    "learning_rate": args.learning_rate,
                    "weight_decay": args.weight_decay,
                    "cluster_refresh_epochs": args.cluster_refresh_epochs,
                    "seed": args.seed,
                    "number_of_runs": 1,
                },
                "count_training": count_train_summary,
                "gat_training": gat_train_summary,
                "gat_source": gat_source,
                "transition": transition_summary,
            },
            handle,
            sort_keys=False,
            allow_unicode=True,
        )
    print(f"Wrote {result_dir}", flush=True)


if __name__ == "__main__":
    main()
