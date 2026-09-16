"""Run vol2: STNPP-GAT with ETAS-calibrated area-wise triggering."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import yaml

from our_experiment.common.lapd_target_common import (
    TARGET_CRIME_ORDER,
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
from our_model.vol1.stnpp_gat import STNPPGATConfig
from our_model.vol2.etas_enhanced_stnpp_gat import AreaETASState, marked_trigger_component
from ref_models.models.etas.adaptive_etas import ETASParameters, fit_initial_or_refit


MODEL_ID = "O2_etas_enhanced_stnpp_gat"


def _empty_params(cells: pd.Index, theta: float, omega: float, history_days: int) -> ETASParameters:
    return ETASParameters(
        mu=pd.Series(1e-3 / history_days, index=cells, dtype=float),
        theta=theta,
        omega=omega,
        fit_method="fallback_empty_area",
        fit_events=0,
    )


def fit_area_etas_params(
    area_grid: pd.DataFrame,
    area_events: pd.DataFrame,
    cutoff: pd.Timestamp,
    previous: ETASParameters | None,
    args: argparse.Namespace,
) -> ETASParameters:
    history_start = cutoff - pd.Timedelta(days=args.history_days)
    history = area_events[
        (area_events["occurred_at"] >= history_start) & (area_events["occurred_at"] < cutoff)
    ].copy()
    cells = pd.Index(area_grid["cell_id"].astype(str), name="cell_id")
    if history.empty:
        return _empty_params(
            cells=cells,
            theta=args.initial_theta if previous is None else previous.theta,
            omega=args.initial_omega if previous is None else previous.omega,
            history_days=args.history_days,
        )
    return fit_initial_or_refit(
        area_grid,
        history,
        history_start=history_start,
        cutoff=cutoff,
        history_days=args.history_days,
        init_theta=args.initial_theta if previous is None else previous.theta,
        init_omega=args.initial_omega if previous is None else previous.omega,
        theta_min=args.theta_min,
        alpha=args.background_alpha,
    )


def forecast_risk_for_cutoff(
    dataset: TargetDataset,
    events: pd.DataFrame,
    transition: np.ndarray,
    cell_pos: dict[str, int],
    crime_pos: dict[str, int],
    cell_area_index: np.ndarray,
    area_state: AreaETASState,
    cutoff: pd.Timestamp,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray]:
    n_cells = len(dataset.grid)
    n_crimes = len(crime_pos)
    horizon_days = args.horizon_hours / 24.0
    history_start = cutoff - pd.Timedelta(days=args.history_days)
    history = events[
        (events["occurred_at"] >= history_start) & (events["occurred_at"] < cutoff)
    ].copy()

    background = np.full((n_cells, n_crimes), args.background_alpha / args.history_days, dtype=np.float32)
    if not history.empty:
        grouped = history.groupby(["cell_id", "crime_idx"], observed=True).size().rename("count").reset_index()
        for row in grouped.itertuples(index=False):
            pos = cell_pos.get(str(row.cell_id))
            if pos is not None:
                background[pos, int(row.crime_idx)] += float(row.count) / args.history_days
    background *= horizon_days

    trigger_start = cutoff - pd.Timedelta(days=args.lookback_days)
    trigger_events = events[
        (events["occurred_at"] >= trigger_start) & (events["occurred_at"] < cutoff)
    ].copy()
    z = np.zeros((n_cells, n_crimes), dtype=np.float32)
    if not trigger_events.empty:
        ages = (cutoff - trigger_events["occurred_at"]).dt.total_seconds().to_numpy(dtype=float) / 86400.0
        for cell_id, crime_idx, area_idx, age in zip(
            trigger_events["cell_id"].astype(str),
            trigger_events["crime_idx"].astype(int),
            trigger_events["area_idx"].astype(int),
            ages,
            strict=False,
        ):
            pos = cell_pos.get(cell_id)
            if pos is not None:
                omega = float(area_state.omega_by_area[int(area_idx)])
                z[pos, int(crime_idx)] += float(np.exp(-omega * age))

    trigger = marked_trigger_component(
        transition=transition,
        z_by_cell_crime=z,
        cell_area_index=cell_area_index,
        area_state=area_state,
        horizon_days=horizon_days,
    )
    risk_by_crime = background + trigger
    return risk_by_crime.sum(axis=1).astype(np.float32), risk_by_crime


def replay_etas_enhanced_stnpp_gat(
    dataset: TargetDataset,
    events: pd.DataFrame,
    transition: np.ndarray,
    crime_pos: dict[str, int],
    area_pos: dict[str, int],
    cell_area_index: np.ndarray,
    args: argparse.Namespace,
) -> tuple[pd.Index, list[dict], np.ndarray, pd.DataFrame]:
    grid = dataset.grid.copy()
    cells = pd.Index(grid["cell_id"].astype(str), name="cell_id")
    cell_pos = {cell: i for i, cell in enumerate(cells)}
    areas = pd.Index(sorted(grid["area_id"].astype(str).unique()), name="area_id")
    events_by_area = {area: events[events["area_id"].astype(str) == area].copy() for area in areas}
    grid_by_area = {area: grid[grid["area_id"].astype(str) == area].copy() for area in areas}
    states: dict[str, dict] = {area: {"params": None, "last_refit": None, "refits": 0} for area in areas}

    cutoffs = pd.date_range(pd.Timestamp(args.start), pd.Timestamp(args.end), freq=f"{args.frame_days}D", inclusive="left")
    risk_matrix = np.zeros((len(cutoffs), len(cells)), dtype=np.float32)
    records = []
    param_rows = []

    for frame_index, cutoff in enumerate(cutoffs):
        frame_refits = 0
        theta_by_area = np.zeros(len(areas), dtype=np.float32)
        omega_by_area = np.zeros(len(areas), dtype=np.float32)
        fit_events = 0
        for area in areas:
            state = states[area]
            due = (
                state["params"] is None
                or state["last_refit"] is None
                or (cutoff - state["last_refit"]).days >= args.refit_days
            )
            if due:
                state["params"] = fit_area_etas_params(
                    area_grid=grid_by_area[area],
                    area_events=events_by_area[area],
                    cutoff=cutoff,
                    previous=state["params"],
                    args=args,
                )
                state["last_refit"] = cutoff
                state["refits"] += 1
                frame_refits += 1
            params: ETASParameters = state["params"]
            area_idx = area_pos[str(area)]
            theta_by_area[area_idx] = float(params.theta)
            omega_by_area[area_idx] = float(params.omega)
            fit_events += int(params.fit_events)
            param_rows.append(
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

        actual_end = cutoff + pd.Timedelta(hours=args.horizon_hours)
        area_state = AreaETASState(theta_by_area=theta_by_area, omega_by_area=omega_by_area)
        risk, _ = forecast_risk_for_cutoff(
            dataset=dataset,
            events=events,
            transition=transition,
            cell_pos=cell_pos,
            crime_pos=crime_pos,
            cell_area_index=cell_area_index,
            area_state=area_state,
            cutoff=cutoff,
            args=args,
        )
        risk_matrix[frame_index] = risk
        records.append(
            {
                "frame_index": frame_index,
                "forecast_date": cutoff,
                "display_time": actual_end,
                "actual_events_in_window": observed_window_counts(events, cutoff, actual_end),
                "predicted_events": float(risk.sum()),
                "area_refits_this_frame": frame_refits,
                "area_refits_total": int(sum(s["refits"] for s in states.values())),
                "fit_events_total": fit_events,
                "theta_median": float(np.median(theta_by_area)),
                "omega_median": float(np.median(omega_by_area)),
            }
        )
        if (frame_index + 1) % 10 == 0 or frame_index + 1 == len(cutoffs):
            print(f"Vol2 forecasted {frame_index + 1}/{len(cutoffs)} frames")
    return cells, records, risk_matrix, pd.DataFrame(param_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2025-01-01")
    parser.add_argument("--train-start", default="2010-01-01")
    parser.add_argument("--train-end", default="2020-01-01")
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
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--attention-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--learning-rate", type=float, default=1.0)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--log-every", type=int, default=150)
    parser.add_argument("--device")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--fade-days", type=int, default=28)
    parser.add_argument("--pop-days", type=float, default=3.0)
    parser.add_argument("--smooth-sigma", type=float, default=2.1)
    parser.add_argument("--vmax-quantile", type=float, default=0.985)
    parser.add_argument("--heatmap-gamma", type=float, default=0.42)
    parser.add_argument("--duration-ms", type=int, default=105)
    parser.add_argument("--dpi", type=int, default=82)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite-derived-data", action="store_true")
    parser.add_argument("--skip-gif", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_dir = ROOT / "our_experiment" / "vol2" / "results"
    dataset = load_target_dataset(overwrite=args.overwrite_derived_data, cell_size_m=150)
    crimes, areas, crime_pos, area_pos, cell_area_index = build_mark_indices(dataset)
    events = event_arrays(dataset, crime_pos, area_pos)
    mark_features = make_mark_features(crimes, areas, dataset)
    config = STNPPGATConfig(
        n_crimes=len(crimes),
        n_areas=len(areas),
        hidden_dim=args.hidden_dim,
        attention_heads=args.attention_heads,
        dropout=args.dropout,
    )
    empirical, source_weight, transition_summary = empirical_transition_matrix(
        events=events,
        n_marks=config.n_marks,
        omega=args.initial_omega,
        lookback_days=args.lookback_days,
        train_start=pd.Timestamp(args.train_start),
        train_end=pd.Timestamp(args.train_end),
        smoothing=args.transition_smoothing,
    )
    model, train_summary, transition = train_mark_gat(
        mark_features=mark_features,
        empirical=empirical,
        source_weight=source_weight,
        config=config,
        args=args,
    )
    cells, records, risk_matrix, param_table = replay_etas_enhanced_stnpp_gat(
        dataset=dataset,
        events=events,
        transition=transition,
        crime_pos=crime_pos,
        area_pos=area_pos,
        cell_area_index=cell_area_index,
        args=args,
    )
    gif_path = result_dir / "animations" / f"{MODEL_ID}_24h_forecast_heatmap.gif"
    if not args.skip_gif:
        gif_path = save_forecast_animation(
            out_path=gif_path,
            dataset=dataset,
            records=records,
            risk_matrix=risk_matrix,
            args=args,
            title="LAPD ETAS-enhanced STNPP-GAT",
            note=(
                "Red heatmap: predicted 24h vol2 risk. "
                "Colored rings: observed target-crime events."
            ),
        )
    write_forecast_tables(
        result_dir=result_dir,
        model_id=MODEL_ID,
        dataset=dataset,
        records=records,
        cells=cells,
        risk_matrix=risk_matrix,
        args=args,
        gif_path=gif_path,
        extra_config={
            "model_development": (
                "Vol2 keeps vol1 STNPP-GAT mark interactions and adopts ETAS-style "
                "area-wise rolling theta/omega calibration for the self-excitation term."
            ),
            "train_start": args.train_start,
            "train_end": args.train_end,
            "attention_heads": args.attention_heads,
            "hidden_dim": args.hidden_dim,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "refit_days": args.refit_days,
            "theta_min": args.theta_min,
            "initial_theta": args.initial_theta,
            "initial_omega": args.initial_omega,
            "transition_summary": transition_summary,
            "train_summary": train_summary,
        },
    )
    model_dir = result_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_id": MODEL_ID,
            "state_dict": model.state_dict(),
            "config": config,
            "train_summary": train_summary,
            "transition_summary": transition_summary,
        },
        model_dir / "model_state.pt",
    )
    tables_dir = result_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    param_table.to_csv(tables_dir / "area_etas_parameters.csv", index=False)
    pd.DataFrame(transition, columns=[f"source_{i}" for i in range(transition.shape[1])]).to_csv(
        tables_dir / "learned_mark_transition.csv",
        index_label="target_mark",
    )
    with (tables_dir / "training_summary.yml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            {
                "train_summary": train_summary,
                "transition_summary": transition_summary,
                "crimes": list(crimes),
                "areas": list(areas),
            },
            f,
            sort_keys=False,
            allow_unicode=True,
        )
    print(f"Wrote {result_dir}")


if __name__ == "__main__":
    main()
