"""Run vol4: STNPP-GAT with an HCL-regularized coarse ZINB branch.

Vol4 preserves vol3's event-level GAT/ETAS allocation at 150 m and its
uncertainty-weighted coarse-to-fine fusion. It replaces the coarse STMGNN
calibrator with a compact HCL encoder trained by direct ZINB likelihood plus
crime-type and cardinal-neighbor alignment losses. HCL's representation-level
Hawkes enhancement is used only in the coarse branch so that it does not
double-count the fine branch's event-level ETAS triggering.
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

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

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
from our_model.vol1.stnpp_gat import MarkGraphAttention, STNPPGATConfig
from our_model.vol4 import HCLZINBConfig, HCLZINBCountModel, hcl_zinb_loss
from ref_models.experiments.hcl.run_lapd_hcl import (
    build_cardinal_graph,
    category_coefficients,
)
from ref_models.experiments.stmgnn_zinb.run_lapd_stmgnn_zinb import (
    build_coarse_dataset,
    evaluate_predictions,
    make_daily_tensor,
    save_prediction_table,
    standardize_features,
)


MODEL_ID = "O4_stnpp_gat_hcl_zinb_multiscale"


class HCLZINBWindowDataset(Dataset):
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
        target_counts = self.counts[target : target + 1].transpose(1, 0, 2)
        return (
            torch.from_numpy(np.ascontiguousarray(features, dtype=np.float32)),
            torch.from_numpy(np.ascontiguousarray(raw_history, dtype=np.float32)),
            torch.from_numpy(np.ascontiguousarray(target_counts, dtype=np.float32)),
        )


def make_hcl_loader(
    features: np.ndarray,
    counts: np.ndarray,
    target_indices: np.ndarray,
    args: argparse.Namespace,
    shuffle: bool,
) -> DataLoader:
    dataset = HCLZINBWindowDataset(
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


def mean_loader_losses(
    model: HCLZINBCountModel,
    loader: DataLoader,
    neighbor_center: torch.Tensor,
    neighbor_node: torch.Tensor,
    category_coefficient: torch.Tensor,
    args: argparse.Namespace,
) -> dict[str, float]:
    model.eval()
    totals = {name: 0.0 for name in ("total", "nll", "type", "neighbor")}
    samples = 0
    device = next(model.parameters()).device
    with torch.no_grad():
        for features, raw_history, target in loader:
            features = features.to(device)
            raw_history = raw_history.to(device)
            target = target.to(device)
            loss = hcl_zinb_loss(
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
            totals["nll"] += float(loss.nll.cpu()) * batch
            totals["type"] += float(loss.type_contrast.cpu()) * batch
            totals["neighbor"] += float(loss.neighbor_contrast.cpu()) * batch
            samples += batch
    return {name: value / max(samples, 1) for name, value in totals.items()}


def train_count_calibrator(
    config: HCLZINBConfig,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    neighbor_center_cpu: torch.Tensor,
    neighbor_node_cpu: torch.Tensor,
    category_coefficient_array: np.ndarray,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    args: argparse.Namespace,
) -> tuple[HCLZINBCountModel, pd.DataFrame, dict]:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.torch_threads)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = HCLZINBCountModel(config, edge_index, edge_weight).to(device)
    neighbor_center = neighbor_center_cpu.to(device)
    neighbor_node = neighbor_node_cpu.to(device)
    category_coefficient = torch.from_numpy(category_coefficient_array).to(device)
    optimizer = torch.optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.hcl_learning_rate,
        weight_decay=args.hcl_weight_decay,
    )

    best_state = copy.deepcopy(model.state_dict())
    best_validation = math.inf
    best_epoch = 0
    stale_epochs = 0
    rows: list[dict] = []
    for epoch in range(1, args.hcl_epochs + 1):
        model.train()
        totals = {name: 0.0 for name in ("total", "nll", "type", "neighbor")}
        samples = 0
        for features, raw_history, target in train_loader:
            features = features.to(device)
            raw_history = raw_history.to(device)
            target = target.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = hcl_zinb_loss(
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
                raise RuntimeError(f"Non-finite HCL-ZINB loss at epoch {epoch}")
            loss.total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.hcl_grad_clip)
            optimizer.step()
            batch = int(features.shape[0])
            totals["total"] += float(loss.total.detach().cpu()) * batch
            totals["nll"] += float(loss.nll.detach().cpu()) * batch
            totals["type"] += float(loss.type_contrast.detach().cpu()) * batch
            totals["neighbor"] += float(loss.neighbor_contrast.detach().cpu()) * batch
            samples += batch

        train_values = {name: value / max(samples, 1) for name, value in totals.items()}
        validation_values = mean_loader_losses(
            model,
            validation_loader,
            neighbor_center,
            neighbor_node,
            category_coefficient,
            args,
        )
        rows.append(
            {
                "epoch": epoch,
                **{f"train_{name}": value for name, value in train_values.items()},
                **{f"validation_{name}": value for name, value in validation_values.items()},
            }
        )
        improved = validation_values["total"] < best_validation - args.hcl_min_delta
        if improved:
            best_validation = validation_values["total"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        if epoch == 1 or epoch % args.log_every == 0 or improved:
            print(
                f"Vol4 HCL-ZINB epoch {epoch}/{args.hcl_epochs}: "
                f"train={train_values['total']:.6f} "
                f"val={validation_values['total']:.6f} "
                f"nll={validation_values['nll']:.6f} "
                f"type={validation_values['type']:.6f} "
                f"neighbor={validation_values['neighbor']:.6f}",
                flush=True,
            )
        if stale_epochs >= args.hcl_patience:
            print(
                f"Vol4 HCL-ZINB early stopping at epoch {epoch}; "
                f"best epoch was {best_epoch}",
                flush=True,
            )
            break

    model.load_state_dict(best_state)
    summary = {
        "device": str(device),
        "torch_version": str(torch.__version__),
        "trainable_parameter_count": int(
            sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        ),
        "epochs_requested": int(args.hcl_epochs),
        "epochs_completed": int(len(rows)),
        "best_epoch": int(best_epoch),
        "best_validation_total_loss": float(best_validation),
        "early_stopping_patience": int(args.hcl_patience),
    }
    return model, pd.DataFrame(rows), summary


def predict_count_distributions(
    model: HCLZINBCountModel,
    loader: DataLoader,
) -> dict[str, np.ndarray]:
    device = next(model.parameters()).device
    values: dict[str, list[np.ndarray]] = {
        name: [] for name in ("observed", "mean", "variance", "pi", "p", "r")
    }
    model.eval()
    with torch.no_grad():
        for features, _, target in loader:
            distribution = model(features.to(device)).distribution
            values["observed"].append(target.numpy())
            for name in ("mean", "variance", "pi", "p", "r"):
                values[name].append(getattr(distribution, name).cpu().numpy())
    return {name: np.concatenate(parts, axis=0) for name, parts in values.items()}


def save_hcl_zinb_plots(
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
    axes[0].set_title("Vol4 HCL-ZINB objective")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    for column, label in (
        ("validation_nll", "ZINB NLL"),
        ("validation_type", "Type alignment"),
        ("validation_neighbor", "Neighbor alignment"),
    ):
        axes[1].plot(training["epoch"], training[column], label=label)
    axes[1].set_title("Validation components")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    fig.savefig(plot_dir / "hcl_zinb_training_history.png", dpi=170)
    plt.close(fig)

    crime_summary = summary[summary["scope"] != "ALL"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for metric, color in (("mae", "#2b8cbe"), ("mpiw", "#e34a33")):
        values = crime_summary[crime_summary["metric"] == metric]
        axes[0].plot(
            values["scope"], values["value"], marker="o", label=metric.upper(), color=color
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
        axes[1].plot(values["scope"], values["value"], marker="o", label=metric, color=color)
    axes[1].axhline(0.8, color="#999999", linestyle="--", linewidth=1)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("Coverage and discrete metrics")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].legend(fontsize=8)
    fig.savefig(plot_dir / "hcl_zinb_paper_metrics_by_crime.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, constrained_layout=True)
    dates = pd.to_datetime(daily["forecast_date"])
    axes[0].plot(dates, daily["observed_events"], label="Observed", color="#252525")
    axes[0].plot(
        dates, daily["predicted_events"], label="Predicted mean", color="#de2d26"
    )
    axes[0].set_ylabel("Daily events")
    axes[0].set_title("Vol4 coarse daily predictions")
    axes[0].grid(alpha=0.2)
    axes[0].legend()
    axes[1].plot(dates, daily["picp"], label="PICP", color="#756bb1")
    axes[1].axhline(0.8, color="#999999", linestyle="--", linewidth=1)
    axes[1].set_ylabel("Coverage")
    axes[1].set_ylim(0, 1.02)
    axes[1].grid(alpha=0.2)
    axes[1].legend()
    fig.savefig(plot_dir / "hcl_zinb_daily_forecast_diagnostics.png", dpi=170)
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
    parser.add_argument("--train-stride", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hcl-epochs", type=int, default=25)
    parser.add_argument("--hcl-learning-rate", type=float, default=1e-3)
    parser.add_argument("--hcl-weight-decay", type=float, default=1e-4)
    parser.add_argument("--hcl-grad-clip", type=float, default=5.0)
    parser.add_argument("--hcl-patience", type=int, default=6)
    parser.add_argument("--hcl-min-delta", type=float, default=1e-5)
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
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--device")
    parser.add_argument("--torch-threads", type=int, default=2)
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
    parser.add_argument("--overwrite-derived-data", action="store_true")
    parser.add_argument("--skip-gif", action="store_true")
    parser.add_argument("--reuse-model-state", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.cell_size_m != 3000:
        raise ValueError("Vol4 requires the paper-aligned 3000 m coarse grid")
    if args.horizon_hours != 24:
        raise ValueError("Vol4 currently supports a 24-hour forecast horizon")
    if args.train_stride < 1:
        raise ValueError("train_stride must be positive")
    result_dir = ROOT / "our_experiment" / "vol4" / "results"

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
    train_indices = train_all[:: args.train_stride]
    features, feature_mean, feature_std = standardize_features(
        daily.counts,
        training_day_stop=validation_start_index,
    )
    train_loader = make_hcl_loader(
        features, daily.counts, train_indices, args, shuffle=True
    )
    validation_loader = make_hcl_loader(
        features, daily.counts, validation_indices, args, shuffle=False
    )
    test_loader = make_hcl_loader(
        features, daily.counts, test_indices, args, shuffle=False
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
    config = HCLZINBConfig(
        n_nodes=len(daily.cells),
        n_crimes=len(daily.crimes),
        input_steps=args.input_days,
        hidden_dim=args.hidden_dim,
        hypergraph_layers=args.hypergraph_layers,
        dropout=args.hcl_dropout,
        hawkes_scope=args.hawkes_scope,
        hawkes_delta=args.hawkes_delta,
    )

    model_state_path = result_dir / "model" / "model_state.pt"
    checkpoint = None
    if args.reuse_model_state:
        if not model_state_path.exists():
            raise FileNotFoundError(f"Cannot reuse missing model state: {model_state_path}")
        checkpoint = torch.load(model_state_path, map_location="cpu", weights_only=False)
        device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        count_model = HCLZINBCountModel(config, edge_index, edge_weight).to(device)
        count_model.load_state_dict(checkpoint["hcl_zinb_state_dict"])
        hcl_train_summary = checkpoint["hcl_zinb_training_summary"]
        hcl_history = pd.read_csv(result_dir / "metrics" / "hcl_zinb_training_history.csv")
        print(f"Reused HCL-ZINB and GAT states from {model_state_path}", flush=True)
    else:
        count_model, hcl_history, hcl_train_summary = train_count_calibrator(
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

    coarse_predictions = predict_count_distributions(count_model, test_loader)
    test_dates = daily.dates[test_indices]
    coarse_summary, coarse_daily, lower, upper = evaluate_predictions(
        coarse_predictions,
        daily.crimes,
        test_dates,
        args.low_quantile,
        args.high_quantile,
    )

    crimes, areas, crime_pos, area_pos, cell_area_index = build_mark_indices(fine)
    events = event_arrays(fine, crime_pos, area_pos)
    mark_features = make_mark_features(crimes, areas, fine)
    gat_config = STNPPGATConfig(
        n_crimes=len(crimes),
        n_areas=len(areas),
        hidden_dim=args.gat_hidden_dim,
        attention_heads=args.attention_heads,
        dropout=args.gat_dropout,
    )
    empirical, source_weight, transition_summary = empirical_transition_matrix(
        events=events,
        n_marks=gat_config.n_marks,
        omega=args.initial_omega,
        lookback_days=args.lookback_days,
        train_start=pd.Timestamp(args.train_start),
        train_end=pd.Timestamp(args.train_end),
        smoothing=args.transition_smoothing,
    )
    if checkpoint is not None:
        device = next(count_model.parameters()).device
        gat_model = MarkGraphAttention(gat_config).to(device)
        gat_model.load_state_dict(checkpoint["gat_state_dict"])
        gat_model.eval()
        with torch.no_grad():
            transition = (
                gat_model(torch.from_numpy(mark_features).to(device))
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
        gat_train_summary = checkpoint["gat_training_summary"]
    else:
        gat_args = argparse.Namespace(**vars(args))
        gat_args.epochs = args.gat_epochs
        gat_args.learning_rate = args.gat_learning_rate
        gat_model, gat_train_summary, transition = train_mark_gat(
            mark_features,
            empirical,
            source_weight,
            gat_config,
            gat_args,
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
        run_label="Vol4",
    )

    gif_path = result_dir / "animations" / f"{MODEL_ID}_24h_forecast_heatmap.gif"
    if not args.skip_gif:
        gif_path = save_forecast_animation(
            out_path=gif_path,
            dataset=fine,
            records=records,
            risk_matrix=risk_matrix,
            args=args,
            title="LAPD STNPP-GAT-HCL-ZINB vol4",
            note=(
                "Heatmap: HCL-ZINB uncertainty-fused 24h mean. "
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
        "used_train_samples": int(len(train_indices)),
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
                "Vol4 replaces vol3's coarse DGCN/MTCN branch with an HCL-regularized "
                "ZINB distribution model. Fine GAT/ETAS allocation and inverse-variance "
                "coarse-to-fine fusion are retained."
            ),
            "hcl_integration_boundary": (
                "HCL correlations are applied only at 3 km daily resolution; event-level "
                "ETAS remains the sole Hawkes-style mechanism on the 150 m branch."
            ),
            "coarse_dataset_id": coarse.metadata["dataset_id"],
            "coarse_cell_size_m": args.cell_size_m,
            "fine_cell_size_m": fine.metadata["spatial"]["cell_size_m"],
            "split": split_summary,
            "count_model_config": asdict(config),
            "graph_summary": graph_summary,
            "category_coefficients": {
                str(crime): float(value)
                for crime, value in zip(daily.crimes, coefficient, strict=True)
            },
            "hcl_zinb_training_summary": hcl_train_summary,
            "gat_training_summary": gat_train_summary,
            "transition_summary": transition_summary,
            "lambda_type": args.lambda_type,
            "lambda_neighbor": args.lambda_neighbor,
            "contrast_steps_per_window": args.contrast_steps,
            "uncertainty_scope": (
                "HCL-ZINB uncertainty is defined at 3 km x crime x day resolution; "
                "the 150 m surface receives the fused predictive mean only."
            ),
        },
    )

    metric_dir = result_dir / "metrics"
    table_dir = result_dir / "tables"
    model_dir = result_dir / "model"
    metric_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    coarse_summary.to_csv(metric_dir / "hcl_zinb_summary_metrics.csv", index=False)
    coarse_daily.to_csv(metric_dir / "hcl_zinb_daily_metrics.csv", index=False)
    hcl_history.to_csv(metric_dir / "hcl_zinb_training_history.csv", index=False)
    save_hcl_zinb_plots(metric_dir, coarse_summary, coarse_daily, hcl_history)
    save_prediction_table(
        table_dir / "coarse_hcl_zinb_daily_predictions.parquet",
        coarse_predictions,
        lower,
        upper,
        test_dates,
        daily.cells,
        daily.crimes,
    )
    area_params.to_csv(table_dir / "area_etas_parameters.csv", index=False)
    coarse_frame_predictions.to_parquet(
        table_dir / "coarse_hcl_zinb_frame_predictions.parquet", index=False
    )
    pd.DataFrame(
        transition,
        columns=[f"source_{index}" for index in range(transition.shape[1])],
    ).to_csv(table_dir / "learned_mark_transition.csv", index_label="target_mark")
    torch.save(
        {
            "model_id": MODEL_ID,
            "hcl_zinb_state_dict": count_model.state_dict(),
            "hcl_zinb_config": asdict(config),
            "gat_state_dict": gat_model.state_dict(),
            "gat_config": asdict(gat_config),
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "coarse_cells": daily.cells.tolist(),
            "crimes": daily.crimes.tolist(),
            "category_coefficients": coefficient,
            "split": split_summary,
            "hcl_zinb_training_summary": hcl_train_summary,
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
                "hcl_objective": {
                    "task": "direct ZINB negative log likelihood",
                    "lambda_type": args.lambda_type,
                    "lambda_neighbor": args.lambda_neighbor,
                    "contrast_steps_per_window": args.contrast_steps,
                },
                "experiment_defined": {
                    "train_stride_days": args.train_stride,
                    "batch_size": args.batch_size,
                    "epochs": args.hcl_epochs,
                    "optimizer": "Adam",
                    "learning_rate": args.hcl_learning_rate,
                    "weight_decay": args.hcl_weight_decay,
                    "seed": args.seed,
                },
                "hcl_zinb_training": hcl_train_summary,
                "gat_training": gat_train_summary,
                "transition": transition_summary,
            },
            handle,
            sort_keys=False,
            allow_unicode=True,
        )
    print(f"Wrote {result_dir}", flush=True)


if __name__ == "__main__":
    main()
