"""Create ETAS v2 GIFs with marked events and combined LAPD display."""

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
from matplotlib.colors import LinearSegmentedColormap, PowerNorm, to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from PIL import Image

from experiments.animate_adaptive_forecast import risk_heatmap, to_projected_grid
from experiments.common import load_dataset
from models.marked_adaptive_etas import (
    MarkedETASParameters,
    crime_type_index,
    decay_marked_state_to,
    ensure_crime_type,
    fit_marked_initial_or_refit,
    integrated_marked_risk_components,
    rebuild_marked_trigger_state,
    update_marked_state_with_events,
)


MODEL_ID = "M5_marked_adaptive_etas_weekly_theta_floor"


@dataclass(frozen=True)
class ComponentSpec:
    component_id: str
    label: str
    dataset_id: str
    cmap_name: str
    color: str


@dataclass
class MarkedAnimationState:
    params: MarkedETASParameters
    z: pd.DataFrame
    last_time: pd.Timestamp
    state_updates: int = 0
    refits: int = 0


@dataclass
class ComponentRun:
    spec: ComponentSpec
    grid: gpd.GeoDataFrame
    plot_grid: gpd.GeoDataFrame
    events: pd.DataFrame
    metadata: dict
    records: list[dict]
    crime_types: pd.Index
    vmax: float


PRESETS = {
    "lapd_three_divisions": [
        ComponentSpec(
            component_id="foothill",
            label="Foothill",
            dataset_id="lapd_foothill_multicrime_burglary_vehicle_stolen_2011_2013_grid150m_h24h",
            cmap_name="etas_v2_foothill_red",
            color="#cb181d",
        ),
        ComponentSpec(
            component_id="n_hollywood",
            label="N Hollywood",
            dataset_id="lapd_n_hollywood_multicrime_burglary_vehicle_stolen_2011_2013_grid150m_h24h",
            cmap_name="etas_v2_n_hollywood_blue",
            color="#2171b5",
        ),
        ComponentSpec(
            component_id="southwest",
            label="Southwest",
            dataset_id="lapd_southwest_multicrime_burglary_vehicle_stolen_2011_2013_grid150m_h24h",
            cmap_name="etas_v2_southwest_yellow",
            color="#d8a000",
        ),
    ],
    "chicago_district011": [
        ComponentSpec(
            component_id="district011",
            label="Chicago District 011",
            dataset_id="chicago_district011_burglary_2011_2013_grid150m_h24h",
            cmap_name="etas_v2_chicago_green",
            color="#238b45",
        )
    ],
}


CRIME_COLORS = {
    "BURGLARY": "#7b3294",
    "VEHICLE - STOLEN": "#008837",
}
FALLBACK_CRIME_COLORS = ["#4d4d4d", "#762a83", "#1b7837", "#543005", "#01665e"]


def cmap_for(name: str) -> LinearSegmentedColormap:
    palettes = {
        "etas_v2_foothill_red": ["#ffffff", "#fee5d9", "#fb6a4a", "#a50f15"],
        "etas_v2_n_hollywood_blue": ["#ffffff", "#deebf7", "#6baed6", "#08519c"],
        "etas_v2_southwest_yellow": ["#ffffff", "#fff7bc", "#fec44f", "#b8860b"],
        "etas_v2_chicago_green": ["#ffffff", "#e5f5e0", "#74c476", "#006d2c"],
    }
    return LinearSegmentedColormap.from_list(name, palettes[name])


def color_for_crime(crime_type: str, known: dict[str, str]) -> str:
    if crime_type not in known:
        known[crime_type] = FALLBACK_CRIME_COLORS[len(known) % len(FALLBACK_CRIME_COLORS)]
    return known[crime_type]


def build_initial_state(
    grid: gpd.GeoDataFrame,
    initial_history: pd.DataFrame,
    history_start: pd.Timestamp,
    start_ts: pd.Timestamp,
    history_days: int,
    fixed_theta: float,
    fixed_omega: float,
    theta_floor: float,
    crime_types: pd.Index,
) -> MarkedAnimationState:
    params = fit_marked_initial_or_refit(
        grid,
        initial_history,
        history_start=history_start,
        cutoff=start_ts,
        history_days=history_days,
        crime_types=crime_types,
        init_theta=max(theta_floor, fixed_theta),
        init_omega=fixed_omega,
        theta_min=theta_floor,
    )
    return MarkedAnimationState(
        params=params,
        z=rebuild_marked_trigger_state(
            grid, initial_history, start_ts, params.omega, crime_types
        ),
        last_time=start_ts,
    )


def replay_marked_forecasts(
    dataset_id: str,
    start: str,
    end: str,
    history_days: int,
    horizon_hours: int,
    refit_days: int,
    fixed_theta: float,
    fixed_omega: float,
    theta_floor: float,
) -> tuple[gpd.GeoDataFrame, pd.DataFrame, dict, list[dict], pd.Index]:
    events, grid, metadata, _ = load_dataset(dataset_id)
    events = ensure_crime_type(events.sort_values("occurred_at").copy())
    events["cell_id"] = events["cell_id"].astype(str)
    crime_types = crime_type_index(events)
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    history_start = start_ts - pd.Timedelta(days=history_days)
    initial_history = events[
        (events["occurred_at"] >= history_start) & (events["occurred_at"] < start_ts)
    ].copy()
    state = build_initial_state(
        grid,
        initial_history,
        history_start,
        start_ts,
        history_days,
        fixed_theta,
        fixed_omega,
        theta_floor,
        crime_types,
    )
    horizon = pd.Timedelta(hours=horizon_hours)
    horizon_days = horizon_hours / 24.0
    records: list[dict] = []
    cutoffs = pd.date_range(start_ts, end_ts, freq="1D", inclusive="left")
    for step, cutoff in enumerate(cutoffs):
        state.z, state.last_time = decay_marked_state_to(
            state.z, state.last_time, cutoff, state.params.omega
        )
        (
            risk,
            background,
            trigger,
            risk_by_type,
            background_by_type,
            trigger_by_type,
        ) = integrated_marked_risk_components(state.params, state.z, horizon_days)
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
                "risk_by_type": risk_by_type,
                "background_by_type": background_by_type,
                "trigger_by_type": trigger_by_type,
                "actual": actual,
                "theta": state.params.theta,
                "omega": state.params.omega,
                "refits": state.refits,
                "state_updates": state.state_updates,
                "transition": state.params.transition.copy(),
            }
        )
        state.z, state.last_time, added = update_marked_state_with_events(
            state.z, state.last_time, actual, state.params.omega
        )
        state.state_updates += added
        if (step + 1) % refit_days == 0:
            refit_cutoff = actual_end
            refit_start = refit_cutoff - pd.Timedelta(days=history_days)
            refit_history = events[
                (events["occurred_at"] >= refit_start)
                & (events["occurred_at"] < refit_cutoff)
            ].copy()
            state.params = fit_marked_initial_or_refit(
                grid,
                refit_history,
                history_start=refit_start,
                cutoff=refit_cutoff,
                history_days=history_days,
                crime_types=crime_types,
                init_theta=state.params.theta,
                init_omega=state.params.omega,
                theta_min=theta_floor,
            )
            state.z = rebuild_marked_trigger_state(
                grid, refit_history, refit_cutoff, state.params.omega, crime_types
            )
            state.last_time = refit_cutoff
            state.refits += 1
    return grid, events, metadata, records, crime_types


def run_component(spec: ComponentSpec, args: argparse.Namespace) -> ComponentRun:
    grid, events, metadata, records, crime_types = replay_marked_forecasts(
        dataset_id=spec.dataset_id,
        start=args.start,
        end=args.end,
        history_days=args.history_days,
        horizon_hours=args.horizon_hours,
        refit_days=args.refit_days,
        fixed_theta=args.fixed_theta,
        fixed_omega=args.fixed_omega,
        theta_floor=args.theta_floor,
    )
    sampled = records[:: args.frame_step]
    risk_values = np.concatenate([r["risk"].to_numpy(dtype=float) for r in sampled])
    vmax = float(np.quantile(risk_values, args.vmax_quantile))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = float(np.nanmax(risk_values)) if len(risk_values) else 1.0
    return ComponentRun(
        spec=spec,
        grid=grid,
        plot_grid=to_projected_grid(grid, metadata),
        events=events,
        metadata=metadata,
        records=records,
        crime_types=crime_types,
        vmax=max(vmax, 1e-9),
    )


def component_boundary(plot_grid: gpd.GeoDataFrame) -> gpd.GeoSeries:
    return gpd.GeoSeries([plot_grid.geometry.union_all().boundary], crs=plot_grid.crs)


def render_combined_frame(
    components: list[ComponentRun],
    frame_index: int,
    args: argparse.Namespace,
    crime_colors: dict[str, str],
) -> Image.Image:
    display_time = components[0].records[frame_index]["display_time"]
    forecast_date = components[0].records[frame_index]["cutoff"].date()
    bounds = np.array([component.plot_grid.total_bounds for component in components])
    minx, miny = bounds[:, 0].min(), bounds[:, 1].min()
    maxx, maxy = bounds[:, 2].max(), bounds[:, 3].max()
    pad = max(maxx - minx, maxy - miny) * 0.035

    fig, ax = plt.subplots(figsize=(9.6, 8.8), dpi=args.dpi, constrained_layout=True)
    for component in components:
        record = component.records[frame_index]
        heatmap, extent = risk_heatmap(
            component.plot_grid, record["risk"], args.smooth_sigma
        )
        heatmap = np.ma.masked_invalid(np.clip(heatmap, 0, component.vmax))
        ax.imshow(
            heatmap,
            extent=extent,
            origin="lower",
            cmap=cmap_for(component.spec.cmap_name),
            norm=PowerNorm(gamma=args.heatmap_gamma, vmin=0, vmax=component.vmax),
            interpolation="bilinear",
            alpha=0.92,
            zorder=1,
        )
        component_boundary(component.plot_grid).plot(
            ax=ax, color=component.spec.color, linewidth=1.2, alpha=0.95, zorder=3
        )

    trail_frames = []
    for component in components:
        trail = component.events[
            (component.events["occurred_at"] <= display_time)
            & (
                component.events["occurred_at"]
                >= display_time - pd.Timedelta(days=args.fade_days)
            )
        ].copy()
        if trail.empty:
            continue
        trail["component_id"] = component.spec.component_id
        trail_frames.append(trail)
    if trail_frames:
        trail = pd.concat(trail_frames, ignore_index=True)
        points = gpd.GeoDataFrame(
            trail,
            geometry=gpd.points_from_xy(trail["lon"], trail["lat"]),
            crs="EPSG:4326",
        ).to_crs(components[0].plot_grid.crs)
        ages = (
            (pd.Timestamp(display_time) - points["occurred_at"]).dt.total_seconds()
            / 86400.0
        ).to_numpy(dtype=float)
        fade = np.clip(1.0 - ages / args.fade_days, 0.0, 1.0)
        blast = np.exp(-ages / max(args.pop_days, 1e-6))
        ring_size = 100 + 1300 * (1.0 - blast) * fade
        ring_alpha = np.clip(0.55 * blast + 0.18 * fade, 0.0, 0.68)
        core_size = 24 + 360 * blast
        core_alpha = np.clip(0.90 * fade, 0.0, 0.90)
        for crime_type in sorted(points["crime_type"].astype(str).unique()):
            mask = points["crime_type"].astype(str) == crime_type
            color = color_for_crime(crime_type, crime_colors)
            rgb = to_rgb(color)
            rgba_ring = [(*rgb, float(a)) for a in ring_alpha[mask.to_numpy()]]
            rgba_core = [(*rgb, float(a)) for a in core_alpha[mask.to_numpy()]]
            subset = points[mask]
            ax.scatter(
                subset.geometry.x,
                subset.geometry.y,
                s=ring_size[mask.to_numpy()],
                facecolors="none",
                edgecolors=rgba_ring,
                linewidths=1.9,
                zorder=5,
            )
            ax.scatter(
                subset.geometry.x,
                subset.geometry.y,
                s=core_size[mask.to_numpy()],
                c=rgba_core,
                edgecolors="white",
                linewidths=0.35,
                zorder=6,
            )

    ax.set_xlim(minx - pad, maxx + pad)
    ax.set_ylim(miny - pad, maxy + pad)
    ax.set_aspect("equal")
    ax.set_axis_off()
    title = f"{args.title} | {forecast_date.isoformat()} forecast"
    ax.set_title(title, fontsize=11)
    subtitle = (
        "Separate ETAS fit per area; heatmap scale is local to each area. "
        f"Events fade over {args.fade_days} days."
    )
    ax.text(
        0.012,
        0.018,
        subtitle,
        transform=ax.transAxes,
        va="bottom",
        ha="left",
        fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.82},
    )
    area_handles = [
        Patch(facecolor=component.spec.color, edgecolor=component.spec.color, alpha=0.55, label=component.spec.label)
        for component in components
    ]
    crime_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="white",
            markerfacecolor=color_for_crime(str(crime), crime_colors),
            markeredgecolor="white",
            markersize=7,
            label=str(crime),
            linewidth=0,
        )
        for crime in sorted(
            {str(c) for component in components for c in component.crime_types.to_list()}
        )
    ]
    legend = ax.legend(
        handles=area_handles + crime_handles,
        loc="upper right",
        fontsize=7,
        frameon=True,
        framealpha=0.82,
        borderpad=0.5,
    )
    legend.get_frame().set_edgecolor("#cccccc")
    fig.canvas.draw()
    image = Image.fromarray(np.asarray(fig.canvas.buffer_rgba()))
    plt.close(fig)
    return image.convert("P", palette=Image.Palette.ADAPTIVE)


def save_animation(
    result_dir: Path,
    components: list[ComponentRun],
    args: argparse.Namespace,
) -> Path:
    out_path = result_dir / "animations" / f"{MODEL_ID}_forecast_heatmap.gif"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_frames = len(components[0].records[:: args.frame_step])
    frame_indices = list(range(0, len(components[0].records), args.frame_step))
    crime_colors = dict(CRIME_COLORS)
    frames = []
    for i, frame_index in enumerate(frame_indices, start=1):
        frames.append(render_combined_frame(components, frame_index, args, crime_colors))
        if i % 25 == 0 or i == n_frames:
            print(f"Rendered {i}/{n_frames} frames")
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=args.duration_ms,
        loop=0,
        optimize=True,
    )
    return out_path


def export_tables(
    result_dir: Path,
    components: list[ComponentRun],
    args: argparse.Namespace,
    gif_path: Path,
) -> None:
    tables_dir = result_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "preset": args.preset,
        "model_id": MODEL_ID,
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
        "vmax_quantile": args.vmax_quantile,
        "heatmap_gamma": args.heatmap_gamma,
        "gif": str(gif_path.relative_to(result_dir)),
        "components": [
            {
                "component_id": component.spec.component_id,
                "label": component.spec.label,
                "dataset_id": component.spec.dataset_id,
                "heatmap_color": component.spec.color,
                "heatmap_vmax_local": component.vmax,
                "crime_types": component.crime_types.to_list(),
            }
            for component in components
        ],
    }
    with (tables_dir / "animation_config.yml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)

    component_rows = []
    frame_rows = []
    cell_rows = []
    crime_cell_rows = []
    top_rows = []
    observed_frames = []
    for component in components:
        component_rows.append(
            {
                "component_id": component.spec.component_id,
                "label": component.spec.label,
                "dataset_id": component.spec.dataset_id,
                "heatmap_color": component.spec.color,
                "heatmap_vmax_local": component.vmax,
                "crime_types": "|".join(component.crime_types.astype(str)),
            }
        )
        for record in component.records:
            cutoff = pd.Timestamp(record["cutoff"])
            actual = record["actual"]
            frame_rows.append(
                {
                    "component_id": component.spec.component_id,
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
            joined.insert(0, "component_id", component.spec.component_id)
            cell_rows.append(joined)

            top = joined.sort_values("risk", ascending=False).head(int(args.top_k)).copy()
            top["rank"] = np.arange(1, len(top) + 1)
            top_rows.append(
                top[
                    [
                        "component_id",
                        "forecast_date",
                        "rank",
                        "cell_id",
                        "risk",
                        "background",
                        "trigger",
                    ]
                ]
            )

            crime_parts = []
            for value_name, table in [
                ("risk", record["risk_by_type"]),
                ("background", record["background_by_type"]),
                ("trigger", record["trigger_by_type"]),
            ]:
                long = table.rename_axis("cell_id").reset_index().melt(
                    id_vars="cell_id", var_name="crime_type", value_name=value_name
                )
                crime_parts.append(long)
            keys = ["cell_id", "crime_type"]
            crime_joined = crime_parts[0].merge(crime_parts[1], on=keys).merge(
                crime_parts[2], on=keys
            )
            crime_joined.insert(0, "forecast_date", cutoff.date().isoformat())
            crime_joined.insert(0, "component_id", component.spec.component_id)
            crime_cell_rows.append(crime_joined)

        start = pd.Timestamp(args.start)
        end = pd.Timestamp(args.end)
        observed = component.events[
            (component.events["occurred_at"] >= start) & (component.events["occurred_at"] < end)
        ].copy()
        observed["component_id"] = component.spec.component_id
        observed_frames.append(observed)

    pd.DataFrame(component_rows).to_csv(tables_dir / "component_runs.csv", index=False)
    pd.DataFrame(frame_rows).to_csv(tables_dir / "forecast_frames.csv", index=False)
    pd.concat(cell_rows, ignore_index=True).to_parquet(
        tables_dir / "forecast_cell_risk.parquet", index=False
    )
    pd.concat(crime_cell_rows, ignore_index=True).to_parquet(
        tables_dir / "forecast_cell_crime_risk.parquet", index=False
    )
    pd.concat(top_rows, ignore_index=True).to_csv(tables_dir / "forecast_top_cells.csv", index=False)

    keep_cols = [
        "component_id",
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
        "source_single_crime_dataset",
    ]
    observed_all = pd.concat(observed_frames, ignore_index=True)
    keep_cols = [c for c in keep_cols if c in observed_all.columns]
    observed_all[keep_cols].to_parquet(tables_dir / "observed_events.parquet", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=sorted(PRESETS), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--title", default="ETAS v2")
    parser.add_argument("--start", default="2012-05-16")
    parser.add_argument("--end", default="2013-01-10")
    parser.add_argument("--history-days", type=int, default=365)
    parser.add_argument("--horizon-hours", type=int, default=24)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--refit-days", type=int, default=7)
    parser.add_argument("--fixed-theta", type=float, default=0.35)
    parser.add_argument("--fixed-omega", type=float, default=1.0 / 14.0)
    parser.add_argument("--theta-floor", type=float, default=0.05)
    parser.add_argument("--fade-days", type=int, default=14)
    parser.add_argument("--pop-days", type=float, default=1.5)
    parser.add_argument("--smooth-sigma", type=float, default=2.4)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--duration-ms", type=int, default=90)
    parser.add_argument("--dpi", type=int, default=90)
    parser.add_argument("--vmax-quantile", type=float, default=0.985)
    parser.add_argument("--heatmap-gamma", type=float, default=0.55)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_dir = ROOT / "results" / "ETAS_v2" / args.run_id
    components = []
    for spec in PRESETS[args.preset]:
        print(f"Replaying {spec.label}: {spec.dataset_id}")
        components.append(run_component(spec, args))
    gif_path = save_animation(result_dir, components, args)
    export_tables(result_dir, components, args, gif_path)
    print(f"Wrote {gif_path}")


if __name__ == "__main__":
    main()
