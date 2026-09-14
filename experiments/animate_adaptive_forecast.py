"""Create a GIF animation for an adaptive ETAS forecast run."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from PIL import Image
from scipy.ndimage import gaussian_filter

from experiments.common import load_dataset
from models.adaptive_etas import (
    ETASParameters,
    decay_state_to,
    fit_initial_or_refit,
    integrated_risk_components,
    rebuild_trigger_state,
    smoothed_mu,
    update_state_with_events,
)


@dataclass
class AnimationState:
    params: ETASParameters
    z: pd.Series
    last_time: pd.Timestamp
    state_updates: int = 0
    refits: int = 0


def build_initial_state(
    model_id: str,
    grid: gpd.GeoDataFrame,
    initial_history: pd.DataFrame,
    history_start: pd.Timestamp,
    start_ts: pd.Timestamp,
    history_days: int,
    fixed_theta: float,
    fixed_omega: float,
    theta_floor: float,
) -> AnimationState:
    if model_id == "M2_state_online_etas_fixed_public":
        params = ETASParameters(
            mu=smoothed_mu(grid, initial_history, history_days),
            theta=fixed_theta,
            omega=fixed_omega,
            fit_method="fixed_public_scaffold",
            fit_events=int(len(initial_history)),
        )
    elif model_id == "M3_adaptive_etas_weekly_mle":
        params = fit_initial_or_refit(
            grid,
            initial_history,
            history_start=history_start,
            cutoff=start_ts,
            history_days=history_days,
        )
    elif model_id == "M4_adaptive_etas_weekly_theta_floor":
        params = fit_initial_or_refit(
            grid,
            initial_history,
            history_start=history_start,
            cutoff=start_ts,
            history_days=history_days,
            init_theta=max(theta_floor, fixed_theta),
            init_omega=fixed_omega,
            theta_min=theta_floor,
        )
    else:
        raise ValueError(f"Unsupported animation model: {model_id}")
    return AnimationState(
        params=params,
        z=rebuild_trigger_state(grid, initial_history, start_ts, params.omega),
        last_time=start_ts,
    )


def replay_forecasts(
    dataset_id: str,
    model_id: str,
    start: str,
    end: str,
    history_days: int,
    horizon_hours: int,
    refit_days: int,
    fixed_theta: float,
    fixed_omega: float,
    theta_floor: float,
) -> tuple[gpd.GeoDataFrame, pd.DataFrame, dict, list[dict]]:
    events, grid, metadata, _ = load_dataset(dataset_id)
    events = events.sort_values("occurred_at").copy()
    events["cell_id"] = events["cell_id"].astype(str)
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    history_start = start_ts - pd.Timedelta(days=history_days)
    initial_history = events[
        (events["occurred_at"] >= history_start) & (events["occurred_at"] < start_ts)
    ].copy()
    state = build_initial_state(
        model_id,
        grid,
        initial_history,
        history_start,
        start_ts,
        history_days,
        fixed_theta,
        fixed_omega,
        theta_floor,
    )
    horizon = pd.Timedelta(hours=horizon_hours)
    horizon_days = horizon_hours / 24.0
    records: list[dict] = []
    cutoffs = pd.date_range(start_ts, end_ts, freq="1D", inclusive="left")
    for step, cutoff in enumerate(cutoffs):
        state.z, state.last_time = decay_state_to(
            state.z, state.last_time, cutoff, state.params.omega
        )
        risk, background, trigger = integrated_risk_components(
            state.params, state.z, horizon_days
        )
        actual_end = cutoff + horizon
        actual = events[
            (events["occurred_at"] >= cutoff) & (events["occurred_at"] < actual_end)
        ].copy()
        records.append(
            {
                "cutoff": cutoff,
                "display_time": actual_end,
                "risk": risk,
                "background": background,
                "trigger": trigger,
                "actual": actual,
                "theta": state.params.theta,
                "omega": state.params.omega,
                "refits": state.refits,
                "state_updates": state.state_updates,
            }
        )
        state.z, state.last_time, added = update_state_with_events(
            state.z, state.last_time, actual, state.params.omega
        )
        state.state_updates += added
        if model_id.startswith("M3") or model_id.startswith("M4"):
            if (step + 1) % refit_days == 0:
                refit_cutoff = actual_end
                refit_start = refit_cutoff - pd.Timedelta(days=history_days)
                refit_history = events[
                    (events["occurred_at"] >= refit_start)
                    & (events["occurred_at"] < refit_cutoff)
                ].copy()
                state.params = fit_initial_or_refit(
                    grid,
                    refit_history,
                    history_start=refit_start,
                    cutoff=refit_cutoff,
                    history_days=history_days,
                    init_theta=state.params.theta,
                    init_omega=state.params.omega,
                    theta_min=theta_floor if model_id.startswith("M4") else 0.0,
                )
                state.z = rebuild_trigger_state(
                    grid, refit_history, refit_cutoff, state.params.omega
                )
                state.last_time = refit_cutoff
                state.refits += 1
    return grid, events, metadata, records


def to_projected_grid(grid: gpd.GeoDataFrame, metadata: dict) -> gpd.GeoDataFrame:
    projected_crs = metadata.get("spatial", {}).get("projected_crs")
    out = grid.copy()
    if projected_crs:
        out = out.to_crs(projected_crs)
    if "area_m2" not in out.columns:
        out["area_m2"] = out.geometry.area
    out["cell_id"] = out["cell_id"].astype(str)
    return out


def render_frame(
    plot_grid: gpd.GeoDataFrame,
    record: dict,
    event_trail: pd.DataFrame,
    model_id: str,
    dataset_id: str,
    fade_days: int,
    pop_days: float,
    vmax: float,
    dpi: int,
    smooth_sigma: float,
) -> Image.Image:
    risk = record["risk"]
    display_time = record["display_time"]
    trail = event_trail[
        (event_trail["occurred_at"] <= display_time)
        & (event_trail["occurred_at"] >= display_time - pd.Timedelta(days=fade_days))
    ].copy()

    heatmap, extent = risk_heatmap(plot_grid, risk, smooth_sigma)
    heatmap = np.ma.masked_invalid(np.clip(heatmap, 0, vmax))

    fig, ax = plt.subplots(figsize=(9.2, 7.4), dpi=dpi, constrained_layout=True)
    im = ax.imshow(
        heatmap,
        extent=extent,
        origin="lower",
        cmap="YlOrRd",
        vmin=0,
        vmax=vmax,
        interpolation="bilinear",
        alpha=0.95,
    )
    division_boundary(plot_grid).plot(ax=ax, color="#555555", linewidth=0.8, alpha=0.7)
    if not trail.empty:
        points = gpd.GeoDataFrame(
            trail,
            geometry=gpd.points_from_xy(trail["lon"], trail["lat"]),
            crs="EPSG:4326",
        ).to_crs(plot_grid.crs)
        ages = (
            (display_time - points["occurred_at"]).dt.total_seconds() / 86400.0
        ).to_numpy(dtype=float)
        fade = np.clip(1.0 - ages / fade_days, 0.0, 1.0)
        blast = np.exp(-ages / max(pop_days, 1e-6))
        ring_size = 120 + 1500 * (1.0 - blast) * fade
        ring_alpha = np.clip(0.65 * blast + 0.20 * fade, 0.0, 0.75)
        core_size = 28 + 430 * blast
        core_alpha = np.clip(0.92 * fade, 0.0, 0.92)
        ax.scatter(
            points.geometry.x,
            points.geometry.y,
            s=ring_size,
            facecolors="none",
            edgecolors=[(0.95, 0.0, 0.0, float(a)) for a in ring_alpha],
            linewidths=2.0,
            zorder=4,
        )
        ax.scatter(
            points.geometry.x,
            points.geometry.y,
            s=core_size,
            c=[(0.95, 0.0, 0.0, float(a)) for a in core_alpha],
            edgecolors="white",
            linewidths=0.4,
            zorder=5,
        )
    ax.set_axis_off()
    ax.set_title(
        f"{dataset_id}\n{model_id} forecast {record['cutoff'].date()} | "
        f"theta={record['theta']:.3f}, omega={record['omega']:.3f}, refits={record['refits']}",
        fontsize=10,
    )
    ax.text(
        0.015,
        0.025,
        f"smoothed heatmap: predicted risk for next 24h\n"
        f"red bursts: observed events, fading over {fade_days} days",
        transform=ax.transAxes,
        va="bottom",
        ha="left",
        fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.85},
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.01)
    cbar.set_label("predicted expected events", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    fig.canvas.draw()
    image = Image.fromarray(np.asarray(fig.canvas.buffer_rgba()))
    plt.close(fig)
    return image.convert("P", palette=Image.Palette.ADAPTIVE)


def risk_heatmap(
    plot_grid: gpd.GeoDataFrame,
    risk: pd.Series,
    smooth_sigma: float,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    rows = plot_grid["row"].astype(int)
    cols = plot_grid["col"].astype(int)
    row_min, row_max = int(rows.min()), int(rows.max())
    col_min, col_max = int(cols.min()), int(cols.max())
    values = np.full((row_max - row_min + 1, col_max - col_min + 1), np.nan, dtype=float)
    mask = np.zeros_like(values, dtype=float)
    risk_lookup = risk.astype(float).to_dict()
    for item in plot_grid[["cell_id", "row", "col"]].itertuples(index=False):
        r = int(item.row) - row_min
        c = int(item.col) - col_min
        values[r, c] = float(risk_lookup.get(str(item.cell_id), 0.0))
        mask[r, c] = 1.0
    filled = np.nan_to_num(values, nan=0.0)
    if smooth_sigma > 0:
        smooth_values = gaussian_filter(filled * mask, sigma=smooth_sigma, mode="constant", cval=0.0)
        smooth_mask = gaussian_filter(mask, sigma=smooth_sigma, mode="constant", cval=0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            heatmap = smooth_values / smooth_mask
    else:
        heatmap = filled
    heatmap[mask == 0] = np.nan
    minx, miny, maxx, maxy = plot_grid.total_bounds
    return heatmap, (float(minx), float(maxx), float(miny), float(maxy))


def division_boundary(plot_grid: gpd.GeoDataFrame) -> gpd.GeoSeries:
    return gpd.GeoSeries([plot_grid.geometry.union_all().boundary], crs=plot_grid.crs)


def save_animation(
    out_path: Path,
    grid: gpd.GeoDataFrame,
    events: pd.DataFrame,
    metadata: dict,
    records: list[dict],
    dataset_id: str,
    model_id: str,
    top_k: int,
    fade_days: int,
    pop_days: float,
    frame_step: int,
    duration_ms: int,
    dpi: int,
    smooth_sigma: float,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plot_grid = to_projected_grid(grid, metadata)
    sampled_records = records[::frame_step]
    risk_values = np.concatenate([r["risk"].to_numpy(dtype=float) for r in sampled_records])
    vmax = float(np.quantile(risk_values, 0.995))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = float(np.nanmax(risk_values)) if len(risk_values) else 1.0
    vmax = max(vmax, 1e-9)

    frames: list[Image.Image] = []
    for i, record in enumerate(sampled_records, start=1):
        frames.append(
            render_frame(
                plot_grid=plot_grid,
                record=record,
                event_trail=events,
                model_id=model_id,
                dataset_id=dataset_id,
                fade_days=fade_days,
                pop_days=pop_days,
                vmax=vmax,
                dpi=dpi,
                smooth_sigma=smooth_sigma,
            )
        )
        if i % 25 == 0 or i == len(sampled_records):
            print(f"Rendered {i}/{len(sampled_records)} frames")
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )


def export_replay_tables(
    result_dir: Path,
    dataset_id: str,
    model_id: str,
    records: list[dict],
    events: pd.DataFrame,
    args: argparse.Namespace,
    gif_path: Path,
) -> None:
    """Save the non-metric tables needed to reproduce the GIF."""

    tables_dir = result_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "dataset_id": dataset_id,
        "model_id": model_id,
        "start": args.start,
        "end": args.end,
        "history_days": args.history_days,
        "horizon_hours": args.horizon_hours,
        "top_k": args.top_k,
        "refit_days": args.refit_days,
        "fixed_theta": args.fixed_theta,
        "fixed_omega": args.fixed_omega,
        "theta_floor": args.theta_floor,
        "fade_days": args.fade_days,
        "pop_days": args.pop_days,
        "smooth_sigma": args.smooth_sigma,
        "frame_step": args.frame_step,
        "duration_ms": args.duration_ms,
        "dpi": args.dpi,
        "gif": str(gif_path.relative_to(result_dir)),
    }
    with (tables_dir / "animation_config.yml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)

    frame_rows = []
    forecast_rows = []
    top_rows = []
    for record in records:
        cutoff = pd.Timestamp(record["cutoff"])
        actual = record["actual"]
        frame_rows.append(
            {
                "forecast_date": cutoff.date().isoformat(),
                "display_time": pd.Timestamp(record["display_time"]).isoformat(),
                "theta": float(record["theta"]),
                "omega": float(record["omega"]),
                "refits": int(record["refits"]),
                "state_updates": int(record["state_updates"]),
                "actual_events_in_window": int(len(actual)),
            }
        )
        risk = record["risk"].rename("risk").astype(float)
        background = record["background"].rename("background").astype(float)
        trigger = record["trigger"].rename("trigger").astype(float)
        joined = pd.concat([risk, background, trigger], axis=1).reset_index()
        joined = joined.rename(columns={joined.columns[0]: "cell_id"})
        joined.insert(0, "forecast_date", cutoff.date().isoformat())
        forecast_rows.append(joined)

        top = joined.sort_values("risk", ascending=False).head(int(args.top_k)).copy()
        top["rank"] = np.arange(1, len(top) + 1)
        top_rows.append(top[["forecast_date", "rank", "cell_id", "risk", "background", "trigger"]])

    pd.DataFrame(frame_rows).to_csv(tables_dir / "forecast_frames.csv", index=False)
    pd.concat(forecast_rows, ignore_index=True).to_parquet(
        tables_dir / "forecast_cell_risk.parquet", index=False
    )
    pd.concat(top_rows, ignore_index=True).to_csv(tables_dir / "forecast_top_cells.csv", index=False)

    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)
    observed = events[(events["occurred_at"] >= start) & (events["occurred_at"] < end)].copy()
    keep_cols = [
        c
        for c in [
            "event_id",
            "occurred_at",
            "crime_type",
            "area_id",
            "area_name",
            "lat",
            "lon",
            "x",
            "y",
            "cell_id",
            "source_dataset",
        ]
        if c in observed.columns
    ]
    observed[keep_cols].to_parquet(tables_dir / "observed_events.parquet", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--model-id",
        default="M4_adaptive_etas_weekly_theta_floor",
        choices=[
            "M2_state_online_etas_fixed_public",
            "M3_adaptive_etas_weekly_mle",
            "M4_adaptive_etas_weekly_theta_floor",
        ],
    )
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--history-days", type=int, default=365)
    parser.add_argument("--horizon-hours", type=int, default=24)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--refit-days", type=int, default=7)
    parser.add_argument("--fixed-theta", type=float, default=0.35)
    parser.add_argument("--fixed-omega", type=float, default=1.0 / 14.0)
    parser.add_argument("--theta-floor", type=float, default=0.05)
    parser.add_argument("--fade-days", type=int, default=14)
    parser.add_argument("--pop-days", type=float, default=1.5)
    parser.add_argument("--smooth-sigma", type=float, default=2.2)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--duration-ms", type=int, default=120)
    parser.add_argument("--dpi", type=int, default=105)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    grid, events, metadata, records = replay_forecasts(
        dataset_id=args.dataset_id,
        model_id=args.model_id,
        start=args.start,
        end=args.end,
        history_days=args.history_days,
        horizon_hours=args.horizon_hours,
        refit_days=args.refit_days,
        fixed_theta=args.fixed_theta,
        fixed_omega=args.fixed_omega,
        theta_floor=args.theta_floor,
    )
    out_path = (
        ROOT
        / "results"
        / args.run_id
        / "animations"
        / f"{args.model_id}_forecast_heatmap.gif"
    )
    result_dir = ROOT / "results" / args.run_id
    save_animation(
        out_path=out_path,
        grid=grid,
        events=events,
        metadata=metadata,
        records=records,
        dataset_id=args.dataset_id,
        model_id=args.model_id,
        top_k=args.top_k,
        fade_days=args.fade_days,
        pop_days=args.pop_days,
        frame_step=args.frame_step,
        duration_ms=args.duration_ms,
        dpi=args.dpi,
        smooth_sigma=args.smooth_sigma,
    )
    export_replay_tables(
        result_dir=result_dir,
        dataset_id=args.dataset_id,
        model_id=args.model_id,
        records=records,
        events=events,
        args=args,
        gif_path=out_path,
    )
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
