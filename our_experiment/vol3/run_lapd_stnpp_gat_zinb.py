"""Run vol3: STNPP-GAT with coarse STMGNN-ZINB count calibration.

The adaptive GAT/ETAS component produces 150 m crime-specific allocation risk.
A DGCN/MTCN ZINB model independently predicts daily crime-count distributions
on 3 km cells. Vol3 rescales the fine allocation inside each coarse cell and
crime channel to an inverse-variance fusion of both means, retaining the
point-process ranking while adding multivariate temporal, spatial diffusion,
and count uncertainty information. The fusion weight decreases where ZINB
predictive variance is high.
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

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from our_experiment.common.lapd_target_common import (
    TargetDataset,
    load_target_dataset,
    observed_window_counts,
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
from our_experiment.vol2.run_lapd_etas_enhanced_stnpp_gat import (
    fit_area_etas_params,
    forecast_risk_for_cutoff,
)
from our_model.vol1.stnpp_gat import MarkGraphAttention, STNPPGATConfig
from our_model.vol2.etas_enhanced_stnpp_gat import AreaETASState
from our_model.vol3.stnpp_gat_zinb import (
    MultiscaleSTNPPGATZINB,
    MultiscaleSTNPPGATZINBConfig,
    allocate_coarse_counts_to_fine_cells,
    uncertainty_weighted_count_fusion,
)
from ref_models.experiments.stmgnn_zinb.run_lapd_stmgnn_zinb import (
    build_coarse_dataset,
    build_graph,
    evaluate_predictions,
    make_daily_tensor,
    make_loader,
    save_metric_plots,
    save_prediction_table,
    standardize_features,
)
from ref_models.models.etas.adaptive_etas import ETASParameters
from ref_models.models.stmgnn_zinb import zinb_nll


MODEL_ID = "O3_stnpp_gat_zinb_multiscale"


def chronological_indices(
    dates: pd.DatetimeIndex,
    input_steps: int,
    output_steps: int,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    validation_days: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    validation_start = train_end - pd.Timedelta(days=validation_days)
    valid = np.arange(input_steps, len(dates) - output_steps + 1, dtype=np.int64)
    target_dates = dates[valid]
    train = valid[(target_dates >= train_start) & (target_dates < validation_start)]
    validation = valid[(target_dates >= validation_start) & (target_dates < train_end)]
    test = valid[target_dates >= train_end]
    if not len(train) or not len(validation) or not len(test):
        raise ValueError("The requested dates produced an empty train, validation, or test split")
    validation_start_index = int(dates.searchsorted(validation_start))
    return train, validation, test, validation_start_index


def mean_loader_loss(
    model: MultiscaleSTNPPGATZINB,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    total = 0.0
    observations = 0
    with torch.no_grad():
        for features, target in loader:
            target = target.to(device)
            loss = zinb_nll(target, model(features.to(device)), reduction="sum")
            total += float(loss.cpu())
            observations += int(target.numel())
    return total / max(observations, 1)


def train_count_calibrator(
    config: MultiscaleSTNPPGATZINBConfig,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    args: argparse.Namespace,
) -> tuple[MultiscaleSTNPPGATZINB, pd.DataFrame, dict]:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.torch_threads)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = MultiscaleSTNPPGATZINB(config, edge_index, edge_weight).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.zinb_learning_rate,
        weight_decay=args.zinb_weight_decay,
    )

    best_state = copy.deepcopy(model.state_dict())
    best_validation = math.inf
    best_epoch = 0
    stale_epochs = 0
    rows: list[dict] = []
    for epoch in range(1, args.zinb_epochs + 1):
        model.train()
        total = 0.0
        observations = 0
        for features, target in train_loader:
            features = features.to(device)
            target = target.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = zinb_nll(target, model(features))
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite ZINB loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.zinb_grad_clip)
            optimizer.step()
            total += float(loss.detach().cpu()) * int(target.numel())
            observations += int(target.numel())

        train_loss = total / max(observations, 1)
        validation_loss = mean_loader_loss(model, validation_loader, device)
        rows.append(
            {
                "epoch": epoch,
                "train_nll": train_loss,
                "validation_nll": validation_loss,
            }
        )
        improved = validation_loss < best_validation - args.zinb_min_delta
        if improved:
            best_validation = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        if epoch == 1 or epoch % args.log_every == 0 or improved:
            print(
                f"Vol3 ZINB epoch {epoch}/{args.zinb_epochs}: "
                f"train_nll={train_loss:.6f} val_nll={validation_loss:.6f}"
            )
        if stale_epochs >= args.zinb_patience:
            print(f"Vol3 ZINB early stopping at epoch {epoch}; best epoch was {best_epoch}")
            break

    model.load_state_dict(best_state)
    summary = {
        "device": str(device),
        "torch_version": str(torch.__version__),
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "epochs_requested": int(args.zinb_epochs),
        "epochs_completed": len(rows),
        "best_epoch": int(best_epoch),
        "best_validation_nll": float(best_validation),
        "early_stopping_patience": int(args.zinb_patience),
    }
    return model, pd.DataFrame(rows), summary


def predict_count_distributions(
    model: MultiscaleSTNPPGATZINB,
    loader: DataLoader,
) -> dict[str, np.ndarray]:
    device = next(model.parameters()).device
    values: dict[str, list[np.ndarray]] = {
        name: [] for name in ("observed", "mean", "variance", "pi", "p", "r")
    }
    model.eval()
    with torch.no_grad():
        for features, target in loader:
            output = model(features.to(device))
            values["observed"].append(target.numpy())
            for name in ("mean", "variance", "pi", "p", "r"):
                values[name].append(getattr(output, name).cpu().numpy())
    return {name: np.concatenate(parts, axis=0) for name, parts in values.items()}


def fine_to_coarse_indices(
    fine: TargetDataset,
    coarse: TargetDataset,
    cell_size_m: int,
) -> np.ndarray:
    origin_x = float(coarse.metadata["spatial"]["origin_x"])
    origin_y = float(coarse.metadata["spatial"]["origin_y"])
    centroids = fine.grid.geometry.centroid
    rows = np.floor((centroids.y.to_numpy(dtype=float) - origin_y) / cell_size_m).astype(int)
    cols = np.floor((centroids.x.to_numpy(dtype=float) - origin_x) / cell_size_m).astype(int)
    coarse_ids = [
        f"g{cell_size_m}_r{row:04d}_c{col:04d}"
        for row, col in zip(rows, cols, strict=False)
    ]
    position = pd.Series(
        np.arange(len(coarse.grid), dtype=np.int32),
        index=coarse.grid["cell_id"].astype(str),
    )
    mapped = position.reindex(coarse_ids)
    if mapped.isna().any():
        raise ValueError(f"Could not map {int(mapped.isna().sum())} fine cells to the coarse grid")
    return mapped.to_numpy(dtype=np.int32)


def replay_vol3(
    fine: TargetDataset,
    coarse: TargetDataset,
    events: pd.DataFrame,
    transition: np.ndarray,
    crime_pos: dict[str, int],
    area_pos: dict[str, int],
    cell_area_index: np.ndarray,
    fine_to_coarse: np.ndarray,
    coarse_predictions: dict[str, np.ndarray],
    test_dates: pd.DatetimeIndex,
    args: argparse.Namespace,
    run_label: str = "Vol3",
) -> tuple[pd.Index, list[dict], np.ndarray, pd.DataFrame, pd.DataFrame]:
    grid = fine.grid.copy()
    cells = pd.Index(grid["cell_id"].astype(str), name="cell_id")
    cell_pos = {cell: index for index, cell in enumerate(cells)}
    areas = pd.Index(sorted(grid["area_id"].astype(str).unique()), name="area_id")
    events_by_area = {
        area: events[events["area_id"].astype(str) == area].copy() for area in areas
    }
    grid_by_area = {
        area: grid[grid["area_id"].astype(str) == area].copy() for area in areas
    }
    states: dict[str, dict] = {
        area: {"params": None, "last_refit": None, "refits": 0} for area in areas
    }

    frame_offsets = np.arange(0, len(test_dates), args.frame_days, dtype=np.int64)
    risk_matrix = np.zeros((len(frame_offsets), len(cells)), dtype=np.float32)
    records: list[dict] = []
    parameter_rows: list[dict] = []
    coarse_rows: list[dict] = []

    for frame_index, prediction_offset in enumerate(frame_offsets):
        cutoff = test_dates[int(prediction_offset)]
        theta_by_area = np.zeros(len(areas), dtype=np.float32)
        omega_by_area = np.zeros(len(areas), dtype=np.float32)
        frame_refits = 0
        fit_events = 0
        for area in areas:
            state = states[str(area)]
            due = (
                state["params"] is None
                or state["last_refit"] is None
                or (cutoff - state["last_refit"]).days >= args.refit_days
            )
            if due:
                state["params"] = fit_area_etas_params(
                    area_grid=grid_by_area[str(area)],
                    area_events=events_by_area[str(area)],
                    cutoff=cutoff,
                    previous=state["params"],
                    args=args,
                )
                state["last_refit"] = cutoff
                state["refits"] += 1
                frame_refits += 1
            params: ETASParameters = state["params"]
            area_index = area_pos[str(area)]
            theta_by_area[area_index] = float(params.theta)
            omega_by_area[area_index] = float(params.omega)
            fit_events += int(params.fit_events)
            parameter_rows.append(
                {
                    "frame_index": frame_index,
                    "forecast_date": cutoff.date().isoformat(),
                    "area_id": str(area),
                    "theta": float(params.theta),
                    "omega": float(params.omega),
                    "fit_method": params.fit_method,
                    "fit_events": int(params.fit_events),
                    "neg_log_likelihood": params.neg_log_likelihood,
                    "optimizer_success": params.optimizer_success,
                    "refits_for_area": int(state["refits"]),
                }
            )

        area_state = AreaETASState(theta_by_area=theta_by_area, omega_by_area=omega_by_area)
        point_risk, point_risk_by_crime = forecast_risk_for_cutoff(
            dataset=fine,
            events=events,
            transition=transition,
            cell_pos=cell_pos,
            crime_pos=crime_pos,
            cell_area_index=cell_area_index,
            area_state=area_state,
            cutoff=cutoff,
            args=args,
        )
        coarse_mean = coarse_predictions["mean"][prediction_offset, :, 0, :]
        coarse_variance = coarse_predictions["variance"][prediction_offset, :, 0, :]
        coarse_point_risk = np.zeros_like(coarse_mean, dtype=np.float32)
        np.add.at(coarse_point_risk, fine_to_coarse, point_risk_by_crime)
        fused_coarse_mean, zinb_weight, fusion_diagnostics = (
            uncertainty_weighted_count_fusion(
                point_process_expected_count=coarse_point_risk,
                zinb_mean=coarse_mean,
                zinb_variance=coarse_variance,
                minimum_variance=args.minimum_fusion_variance,
            )
        )
        calibrated_by_crime, diagnostics = allocate_coarse_counts_to_fine_cells(
            point_process_risk=point_risk_by_crime,
            coarse_expected_count=fused_coarse_mean,
            fine_to_coarse=fine_to_coarse,
            epsilon=args.allocation_epsilon,
        )
        risk = calibrated_by_crime.sum(axis=1).astype(np.float32)
        risk_matrix[frame_index] = risk
        display_time = cutoff + pd.Timedelta(hours=args.horizon_hours)
        records.append(
            {
                "frame_index": frame_index,
                "forecast_date": cutoff,
                "display_time": display_time,
                "actual_events_in_window": observed_window_counts(events, cutoff, display_time),
                "predicted_events": float(risk.sum()),
                "point_process_events_before_calibration": float(point_risk.sum()),
                "zinb_expected_events": float(coarse_mean.sum()),
                "mean_zinb_fusion_weight": fusion_diagnostics.mean_zinb_weight,
                "mean_pi": float(coarse_predictions["pi"][prediction_offset].mean()),
                "maximum_coarse_count_error": diagnostics.maximum_coarse_count_error,
                "allocation_fallback_groups": diagnostics.fallback_groups,
                "area_refits_this_frame": frame_refits,
                "fit_events_total": fit_events,
                "theta_median": float(np.median(theta_by_area)),
                "omega_median": float(np.median(omega_by_area)),
            }
        )

        for coarse_index, coarse_cell in enumerate(coarse.grid["cell_id"].astype(str)):
            for crime, crime_index in crime_pos.items():
                coarse_rows.append(
                    {
                        "frame_index": frame_index,
                        "forecast_date": cutoff.date().isoformat(),
                        "coarse_cell_id": coarse_cell,
                        "target_crime": crime,
                        "point_process_mean": float(
                            coarse_point_risk[coarse_index, crime_index]
                        ),
                        "predicted_mean": float(coarse_mean[coarse_index, crime_index]),
                        "predicted_variance": float(
                            coarse_variance[coarse_index, crime_index]
                        ),
                        "zinb_fusion_weight": float(zinb_weight[coarse_index, crime_index]),
                        "fused_mean": float(fused_coarse_mean[coarse_index, crime_index]),
                        "pi": float(
                            coarse_predictions["pi"][prediction_offset, coarse_index, 0, crime_index]
                        ),
                        "p": float(
                            coarse_predictions["p"][prediction_offset, coarse_index, 0, crime_index]
                        ),
                        "r": float(
                            coarse_predictions["r"][prediction_offset, coarse_index, 0, crime_index]
                        ),
                    }
                )
        if (frame_index + 1) % 10 == 0 or frame_index + 1 == len(frame_offsets):
            print(f"{run_label} forecasted {frame_index + 1}/{len(frame_offsets)} frames")

    return (
        cells,
        records,
        risk_matrix,
        pd.DataFrame(parameter_rows),
        pd.DataFrame(coarse_rows),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-start", default="2010-01-01")
    parser.add_argument("--train-end", default="2020-01-01")
    parser.add_argument("--data-end-exclusive", default="2024-12-01")
    parser.add_argument("--validation-days", type=int, default=30)
    parser.add_argument("--cell-size-m", type=int, default=3000)
    parser.add_argument("--input-days", type=int, default=28)
    parser.add_argument("--output-days", type=int, default=1)
    parser.add_argument("--hidden-dim", type=int, default=24)
    parser.add_argument("--graph-layers", type=int, default=1)
    parser.add_argument("--temporal-hidden-steps", type=int, default=14)
    parser.add_argument("--zinb-dropout", type=float, default=0.10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--zinb-epochs", type=int, default=100)
    parser.add_argument("--zinb-learning-rate", type=float, default=1e-3)
    parser.add_argument("--zinb-weight-decay", type=float, default=1e-5)
    parser.add_argument("--zinb-grad-clip", type=float, default=5.0)
    parser.add_argument("--zinb-patience", type=int, default=12)
    parser.add_argument("--zinb-min-delta", type=float, default=1e-5)
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
        raise ValueError("Vol3 currently requires the paper-aligned 3000 m coarse grid")
    if args.output_days != 1 or args.horizon_hours != 24:
        raise ValueError("Vol3 currently supports one-day / 24-hour forecasts")
    args.history_days = int(args.history_days)
    result_dir = ROOT / "our_experiment" / "vol3" / "results"

    fine = load_target_dataset(overwrite=args.overwrite_derived_data, cell_size_m=150)
    coarse = build_coarse_dataset(fine, args.cell_size_m, args.overwrite_derived_data)
    daily = make_daily_tensor(coarse, args.data_end_exclusive)
    train_indices, validation_indices, test_indices, validation_start_index = (
        chronological_indices(
            dates=daily.dates,
            input_steps=args.input_days,
            output_steps=args.output_days,
            train_start=pd.Timestamp(args.train_start),
            train_end=pd.Timestamp(args.train_end),
            validation_days=args.validation_days,
        )
    )
    features, feature_mean, feature_std = standardize_features(
        daily.counts,
        training_day_stop=validation_start_index,
    )
    train_loader = make_loader(features, daily.counts, train_indices, args, shuffle=True)
    validation_loader = make_loader(
        features, daily.counts, validation_indices, args, shuffle=False
    )
    test_loader = make_loader(features, daily.counts, test_indices, args, shuffle=False)
    edge_index, edge_weight, graph_summary = build_graph(coarse.grid)
    config = MultiscaleSTNPPGATZINBConfig(
        n_coarse_nodes=len(daily.cells),
        n_crimes=len(daily.crimes),
        input_steps=args.input_days,
        output_steps=args.output_days,
        hidden_dim=args.hidden_dim,
        graph_layers=args.graph_layers,
        temporal_hidden_steps=args.temporal_hidden_steps,
        dropout=args.zinb_dropout,
        allocation_epsilon=args.allocation_epsilon,
        minimum_fusion_variance=args.minimum_fusion_variance,
    )
    model_state_path = result_dir / "model" / "model_state.pt"
    checkpoint = None
    if args.reuse_model_state:
        if not model_state_path.exists():
            raise FileNotFoundError(f"Cannot reuse missing model state: {model_state_path}")
        checkpoint = torch.load(model_state_path, map_location="cpu", weights_only=False)
        device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        count_model = MultiscaleSTNPPGATZINB(config, edge_index, edge_weight).to(device)
        count_model.load_state_dict(checkpoint["zinb_state_dict"])
        zinb_train_summary = checkpoint["zinb_training_summary"]
        zinb_history = pd.read_csv(result_dir / "metrics" / "zinb_training_history.csv")
        print(f"Reused ZINB and GAT states from {model_state_path}")
    else:
        count_model, zinb_history, zinb_train_summary = train_count_calibrator(
            config,
            edge_index,
            edge_weight,
            train_loader,
            validation_loader,
            args,
        )
    coarse_predictions = predict_count_distributions(count_model, test_loader)
    test_dates = daily.dates[test_indices]
    zinb_summary, zinb_daily, lower, upper = evaluate_predictions(
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
    )

    gif_path = result_dir / "animations" / f"{MODEL_ID}_24h_forecast_heatmap.gif"
    if not args.skip_gif:
        gif_path = save_forecast_animation(
            out_path=gif_path,
            dataset=fine,
            records=records,
            risk_matrix=risk_matrix,
            args=args,
            title="LAPD STNPP-GAT-ZINB vol3",
            note=(
                "Heatmap: uncertainty-fused 24h point-process mean. "
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
        "train_samples": len(train_indices),
        "validation_samples": len(validation_indices),
        "test_samples": len(test_indices),
        "gif_frames": len(records),
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
                "Vol3 adds paper-aligned DGCN/MTCN ZINB count distributions on 3 km cells. "
                "Inverse-variance fusion combines their means with the adaptive GAT/ETAS "
                "count surface before allocation over 150 m cells."
            ),
            "coarse_dataset_id": coarse.metadata["dataset_id"],
            "coarse_cell_size_m": args.cell_size_m,
            "fine_cell_size_m": fine.metadata["spatial"]["cell_size_m"],
            "split": split_summary,
            "count_model_config": asdict(config),
            "graph_summary": graph_summary,
            "zinb_training_summary": zinb_train_summary,
            "gat_training_summary": gat_train_summary,
            "transition_summary": transition_summary,
            "uncertainty_scope": (
                "ZINB uncertainty is calibrated at 3 km x crime x day resolution; the GIF uses "
                "the predictive mean and does not claim 150 m prediction intervals."
            ),
        },
    )

    metric_dir = result_dir / "metrics"
    table_dir = result_dir / "tables"
    model_dir = result_dir / "model"
    metric_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    zinb_summary.to_csv(metric_dir / "zinb_summary_metrics.csv", index=False)
    zinb_daily.to_csv(metric_dir / "zinb_daily_metrics.csv", index=False)
    zinb_history.to_csv(metric_dir / "zinb_training_history.csv", index=False)
    save_metric_plots(metric_dir, zinb_summary, zinb_daily, zinb_history)
    save_prediction_table(
        table_dir / "coarse_zinb_daily_predictions.parquet",
        coarse_predictions,
        lower,
        upper,
        test_dates,
        daily.cells,
        daily.crimes,
    )
    area_params.to_csv(table_dir / "area_etas_parameters.csv", index=False)
    coarse_frame_predictions.to_parquet(
        table_dir / "coarse_zinb_frame_predictions.parquet", index=False
    )
    pd.DataFrame(
        transition,
        columns=[f"source_{index}" for index in range(transition.shape[1])],
    ).to_csv(table_dir / "learned_mark_transition.csv", index_label="target_mark")
    torch.save(
        {
            "model_id": MODEL_ID,
            "zinb_state_dict": count_model.state_dict(),
            "zinb_config": asdict(config),
            "gat_state_dict": gat_model.state_dict(),
            "gat_config": asdict(gat_config),
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "coarse_cells": daily.cells.tolist(),
            "crimes": daily.crimes.tolist(),
            "split": split_summary,
            "zinb_training_summary": zinb_train_summary,
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
                "zinb_training": zinb_train_summary,
                "gat_training": gat_train_summary,
                "transition": transition_summary,
            },
            handle,
            sort_keys=False,
            allow_unicode=True,
        )
    print(f"Wrote {result_dir}")


if __name__ == "__main__":
    main()
