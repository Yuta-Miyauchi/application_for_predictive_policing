"""Run a Mohler-style ETAS forecast animation on LAPD target crimes.

This experiment moves the ETAS setup closer to Mohler et al.'s Los Angeles
field experiment: target crimes, 150m cells, 365-day background history, 24h
forecast horizon, and division-wise parameter estimation. The GIF still shows
all LAPD divisions together so regional forecasts can be inspected as one map.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import yaml

from our_experiment.common.lapd_target_common import (
    TARGET_CRIME_ORDER,
    TargetDataset,
    load_target_dataset,
    observed_window_counts,
    save_forecast_animation,
    write_forecast_tables,
)
from ref_models.models.etas.adaptive_etas import (
    ETASParameters,
    fit_initial_or_refit,
    integrated_risk_components,
    rebuild_trigger_state,
)


MODEL_ID = "M7_mohler_style_lapd_etas"


def _empty_params(cells: pd.Index, theta: float, omega: float, history_days: int) -> ETASParameters:
    mu = pd.Series(1e-3 / history_days, index=cells, dtype=float)
    return ETASParameters(
        mu=mu,
        theta=theta,
        omega=omega,
        fit_method="fallback_empty_area",
        fit_events=0,
    )


def fit_area_params(
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
    init_theta = args.initial_theta if previous is None else previous.theta
    init_omega = args.initial_omega if previous is None else previous.omega
    return fit_initial_or_refit(
        area_grid,
        history,
        history_start=history_start,
        cutoff=cutoff,
        history_days=args.history_days,
        init_theta=init_theta,
        init_omega=init_omega,
        theta_min=args.theta_min,
        alpha=args.background_alpha,
    )


def replay_mohler_etas(
    dataset: TargetDataset,
    args: argparse.Namespace,
) -> tuple[pd.Index, list[dict], np.ndarray, pd.DataFrame]:
    events = dataset.events.sort_values("occurred_at").reset_index(drop=True)
    grid = dataset.grid.copy()
    cells = pd.Index(grid["cell_id"].astype(str), name="cell_id")
    cell_pos = {cell: i for i, cell in enumerate(cells)}
    area_ids = pd.Index(sorted(grid["area_id"].astype(str).unique()), name="area_id")

    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)
    cutoffs = pd.date_range(start, end, freq=f"{args.frame_days}D", inclusive="left")
    if len(cutoffs) == 0:
        raise ValueError("No forecast frames. Check --start, --end, and --frame-days.")

    events_by_area = {
        area_id: events[events["area_id"].astype(str) == area_id].copy() for area_id in area_ids
    }
    grid_by_area = {
        area_id: grid[grid["area_id"].astype(str) == area_id].copy() for area_id in area_ids
    }
    states: dict[str, dict] = {
        area_id: {"params": None, "last_refit": None, "refits": 0} for area_id in area_ids
    }
    risk_matrix = np.zeros((len(cutoffs), len(cells)), dtype=np.float32)
    records: list[dict] = []
    param_rows: list[dict] = []
    horizon_days = args.horizon_hours / 24.0

    for frame_index, cutoff in enumerate(cutoffs):
        actual_end = cutoff + pd.Timedelta(hours=args.horizon_hours)
        frame_risk = np.zeros(len(cells), dtype=np.float32)
        theta_values = []
        omega_values = []
        fit_events = 0
        frame_refits = 0

        for area_id in area_ids:
            state = states[area_id]
            due = (
                state["params"] is None
                or state["last_refit"] is None
                or (cutoff - state["last_refit"]).days >= args.refit_days
            )
            area_grid = grid_by_area[area_id]
            area_events = events_by_area[area_id]
            if due:
                state["params"] = fit_area_params(
                    area_grid=area_grid,
                    area_events=area_events,
                    cutoff=cutoff,
                    previous=state["params"],
                    args=args,
                )
                state["last_refit"] = cutoff
                state["refits"] += 1
                frame_refits += 1
            params: ETASParameters = state["params"]
            history_start = cutoff - pd.Timedelta(days=args.history_days)
            history = area_events[
                (area_events["occurred_at"] >= history_start) & (area_events["occurred_at"] < cutoff)
            ].copy()
            z = rebuild_trigger_state(area_grid, history, cutoff, params.omega)
            risk, background, trigger = integrated_risk_components(params, z, horizon_days)
            idx = [cell_pos[cell] for cell in risk.index if cell in cell_pos]
            values = risk.reindex([cells[i] for i in idx], fill_value=0.0).to_numpy(dtype=np.float32)
            frame_risk[idx] = values
            theta_values.append(params.theta)
            omega_values.append(params.omega)
            fit_events += int(params.fit_events)
            param_rows.append(
                {
                    "frame_index": frame_index,
                    "forecast_date": cutoff.date().isoformat(),
                    "area_id": area_id,
                    "area_name": str(area_grid["area_name"].iloc[0]) if not area_grid.empty else area_id,
                    "theta": float(params.theta),
                    "omega": float(params.omega),
                    "fit_method": params.fit_method,
                    "fit_events": int(params.fit_events),
                    "neg_log_likelihood": params.neg_log_likelihood,
                    "optimizer_success": params.optimizer_success,
                    "background_expected": float(background.sum()),
                    "trigger_expected": float(trigger.sum()),
                    "risk_expected": float(risk.sum()),
                    "refits_for_area": int(state["refits"]),
                }
            )

        risk_matrix[frame_index] = frame_risk
        actual_count = observed_window_counts(events, cutoff, actual_end)
        records.append(
            {
                "frame_index": frame_index,
                "forecast_date": cutoff,
                "display_time": actual_end,
                "actual_events_in_window": actual_count,
                "predicted_events": float(frame_risk.sum()),
                "area_refits_this_frame": frame_refits,
                "area_refits_total": int(sum(s["refits"] for s in states.values())),
                "fit_events_total": int(fit_events),
                "theta_median": float(np.median(theta_values)),
                "omega_median": float(np.median(omega_values)),
            }
        )
        if (frame_index + 1) % 10 == 0 or frame_index + 1 == len(cutoffs):
            print(
                f"ETAS replayed {frame_index + 1}/{len(cutoffs)} frames; "
                f"refits this frame={frame_refits}"
            )
    return cells, records, risk_matrix, pd.DataFrame(param_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="lapd_legacy_2020_2024_mohler_etas_target_150m")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2025-01-01")
    parser.add_argument("--history-days", type=int, default=365)
    parser.add_argument("--horizon-hours", type=int, default=24)
    parser.add_argument("--frame-days", type=int, default=14)
    parser.add_argument("--refit-days", type=int, default=28)
    parser.add_argument("--initial-theta", type=float, default=0.35)
    parser.add_argument("--initial-omega", type=float, default=1.0 / 14.0)
    parser.add_argument("--theta-min", type=float, default=0.01)
    parser.add_argument("--background-alpha", type=float, default=1e-3)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--fade-days", type=int, default=28)
    parser.add_argument("--pop-days", type=float, default=3.0)
    parser.add_argument("--smooth-sigma", type=float, default=2.1)
    parser.add_argument("--vmax-quantile", type=float, default=0.985)
    parser.add_argument("--heatmap-gamma", type=float, default=0.42)
    parser.add_argument("--duration-ms", type=int, default=105)
    parser.add_argument("--dpi", type=int, default=82)
    parser.add_argument("--overwrite-derived-data", action="store_true")
    parser.add_argument("--skip-gif", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_dir = ROOT / "ref_models" / "experiments" / "etas" / "results"
    dataset = load_target_dataset(overwrite=args.overwrite_derived_data, cell_size_m=150)
    cells, records, risk_matrix, param_table = replay_mohler_etas(dataset, args)
    gif_path = result_dir / "animations" / f"{MODEL_ID}_24h_forecast_heatmap.gif"
    if not args.skip_gif:
        gif_path = save_forecast_animation(
            out_path=gif_path,
            dataset=dataset,
            records=records,
            risk_matrix=risk_matrix,
            args=args,
            title="LAPD Mohler-style ETAS",
            note=(
                "Red heatmap: predicted 24h target-crime risk. "
                "Colored rings: observed burglary, car theft, and theft-from-vehicle events."
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
            "paper_alignment": (
                "Target crimes, 150m cells, 365-day history, 24h horizon, and "
                "division-wise ETAS parameter estimation. Forecast frames are sampled "
                "every frame_days for a compact GIF."
            ),
            "target_crime_order": TARGET_CRIME_ORDER,
            "refit_days": args.refit_days,
            "initial_theta": args.initial_theta,
            "initial_omega": args.initial_omega,
            "theta_min": args.theta_min,
            "background_alpha": args.background_alpha,
        },
    )
    tables_dir = result_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    param_table.to_csv(tables_dir / "etas_area_parameters.csv", index=False)
    with (tables_dir / "dataset_metadata.yml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(dataset.metadata, f, sort_keys=False, allow_unicode=True)
    print(f"Wrote {result_dir}")


if __name__ == "__main__":
    main()
