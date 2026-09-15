"""Run a pooled weekly ETAS experiment on full legacy LAPD data."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
from matplotlib.lines import Line2D
from PIL import Image

from experiments.animate_adaptive_forecast import risk_heatmap


DATASET_ID = "lapd_legacy_2010_2024_all_crimes_grid300m_h168h"
MODEL_ID = "M6_pooled_lapd_weekly_online_etas_fixed_theta"
DATASET_DIR = ROOT / "datas" / "lapd_full" / DATASET_ID

GROUP_COLORS = {
    "BURGLARY": "#7b3294",
    "FRAUD": "#a6611a",
    "OTHER": "#4d4d4d",
    "ROBBERY": "#018571",
    "SEX_OFFENSE": "#c51b7d",
    "THEFT": "#1f78b4",
    "VANDALISM": "#dfc27d",
    "VEHICLE": "#4daf4a",
    "VIOLENT": "#542788",
    "WEAPON": "#80cdc1",
}

RISK_CMAP = LinearSegmentedColormap.from_list(
    "predictive_risk_red",
    ["#fff7bc", "#fec44f", "#fb6a4a", "#de2d26", "#a50f15", "#4d0013"],
)


def load_full_dataset() -> tuple[pd.DataFrame, gpd.GeoDataFrame, dict]:
    metadata_path = DATASET_DIR / "metadata.yml"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Missing {metadata_path}. Run datas/lapd_full/prepare_lapd_legacy_full.py first."
        )
    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = yaml.safe_load(f)
    events = pd.read_parquet(DATASET_DIR / "processed" / "events.parquet")
    events["occurred_at"] = pd.to_datetime(events["occurred_at"])
    events["cell_id"] = events["cell_id"].astype(str)
    events["crime_group"] = events["crime_group"].fillna("OTHER").astype(str)
    grid = gpd.read_file(DATASET_DIR / "processed" / "grid.geojson")
    grid["cell_id"] = grid["cell_id"].astype(str)
    projected_crs = metadata.get("spatial", {}).get("projected_crs", "EPSG:3310")
    grid = grid.to_crs(projected_crs)
    if "area_m2" not in grid.columns:
        grid["area_m2"] = grid.geometry.area
    return events.sort_values("occurred_at").reset_index(drop=True), grid, metadata


def build_weekly_records(
    events: pd.DataFrame,
    grid: gpd.GeoDataFrame,
    args: argparse.Namespace,
) -> tuple[list[dict], pd.Index, pd.Index, np.ndarray]:
    cells = pd.Index(grid["cell_id"].astype(str), name="cell_id")
    groups = pd.Index(sorted(events["crime_group"].dropna().astype(str).unique()), name="crime_group")
    cell_pos = {cell: i for i, cell in enumerate(cells)}
    group_pos = {group: i for i, group in enumerate(groups)}

    event_cells = events["cell_id"].map(cell_pos).to_numpy(dtype=int)
    event_groups = events["crime_group"].map(group_pos).to_numpy(dtype=int)
    times = events["occurred_at"].to_numpy(dtype="datetime64[ns]")
    n = len(events)
    n_cells = len(cells)
    n_groups = len(groups)

    start = pd.Timestamp(args.start) if args.start else events["occurred_at"].min().normalize()
    end = pd.Timestamp(args.end) if args.end else (events["occurred_at"].max().normalize() + pd.Timedelta(days=7))
    cutoffs = pd.date_range(start, end, freq=f"{args.frame_days}D", inclusive="left")
    horizon_days = args.horizon_hours / 24.0
    history_days = float(args.history_days)
    theta = float(args.theta)
    omega = float(args.omega)
    alpha = float(args.background_alpha)
    trigger_integral = 1.0 - float(np.exp(-omega * horizon_days))

    z = np.zeros(n_cells, dtype=np.float64)
    z_group = np.zeros(n_groups, dtype=np.float64)
    history_counts = np.zeros(n_cells, dtype=np.float64)
    history_group_counts = np.zeros(n_groups, dtype=np.float64)
    update_ptr = 0
    hist_start_ptr = 0
    hist_end_ptr = 0
    state_time = np.datetime64(start.to_datetime64())
    records: list[dict] = []
    risk_matrix = np.zeros((len(cutoffs), n_cells), dtype=np.float32)

    for frame_index, cutoff in enumerate(cutoffs):
        cutoff64 = np.datetime64(cutoff.to_datetime64())
        history_start64 = np.datetime64((cutoff - pd.Timedelta(days=history_days)).to_datetime64())

        while hist_end_ptr < n and times[hist_end_ptr] < cutoff64:
            history_counts[event_cells[hist_end_ptr]] += 1.0
            history_group_counts[event_groups[hist_end_ptr]] += 1.0
            hist_end_ptr += 1
        while hist_start_ptr < n and times[hist_start_ptr] < history_start64:
            history_counts[event_cells[hist_start_ptr]] -= 1.0
            history_group_counts[event_groups[hist_start_ptr]] -= 1.0
            hist_start_ptr += 1

        while update_ptr < n and times[update_ptr] < cutoff64:
            event_time = times[update_ptr]
            delta_days = (event_time - state_time) / np.timedelta64(1, "D")
            if delta_days > 0:
                decay = float(np.exp(-omega * float(delta_days)))
                z *= decay
                z_group *= decay
                state_time = event_time
            z[event_cells[update_ptr]] += 1.0
            z_group[event_groups[update_ptr]] += 1.0
            update_ptr += 1

        delta_days = (cutoff64 - state_time) / np.timedelta64(1, "D")
        if delta_days > 0:
            decay = float(np.exp(-omega * float(delta_days)))
            z *= decay
            z_group *= decay
            state_time = cutoff64

        background = ((history_counts + alpha) / max(history_days, 1.0)) * horizon_days
        trigger = theta * z * trigger_integral
        risk = background + trigger
        risk_matrix[frame_index, :] = risk.astype(np.float32)

        group_background = ((history_group_counts + alpha) / max(history_days, 1.0)) * horizon_days
        group_trigger = theta * z_group * trigger_integral
        actual_end = cutoff + pd.Timedelta(hours=args.horizon_hours)
        actual_start_idx = int(np.searchsorted(times, cutoff64, side="left"))
        actual_end_idx = int(np.searchsorted(times, np.datetime64(actual_end.to_datetime64()), side="left"))
        records.append(
            {
                "frame_index": frame_index,
                "forecast_date": cutoff,
                "display_time": actual_end,
                "history_events": int(hist_end_ptr - hist_start_ptr),
                "state_events": int(update_ptr),
                "actual_start_idx": actual_start_idx,
                "actual_end_idx": actual_end_idx,
                "actual_events_in_window": int(actual_end_idx - actual_start_idx),
                "group_background": group_background.astype(np.float32),
                "group_trigger": group_trigger.astype(np.float32),
                "group_risk": (group_background + group_trigger).astype(np.float32),
            }
        )
        if (frame_index + 1) % 25 == 0 or frame_index + 1 == len(cutoffs):
            print(f"Replayed {frame_index + 1}/{len(cutoffs)} weekly forecasts")
    return records, cells, groups, risk_matrix


def save_tables(
    result_dir: Path,
    records: list[dict],
    cells: pd.Index,
    groups: pd.Index,
    risk_matrix: np.ndarray,
    events: pd.DataFrame,
    args: argparse.Namespace,
    gif_path: Path,
) -> None:
    tables_dir = result_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "dataset_id": DATASET_ID,
        "model_id": MODEL_ID,
        "start": records[0]["forecast_date"].date().isoformat(),
        "end": records[-1]["display_time"].date().isoformat(),
        "history_days": args.history_days,
        "horizon_hours": args.horizon_hours,
        "frame_days": args.frame_days,
        "theta": args.theta,
        "omega": args.omega,
        "background_alpha": args.background_alpha,
        "top_k": args.top_k,
        "fade_weeks": args.fade_weeks,
        "smooth_sigma": args.smooth_sigma,
        "vmax_quantile": args.vmax_quantile,
        "heatmap_gamma": args.heatmap_gamma,
        "duration_ms": args.duration_ms,
        "dpi": args.dpi,
        "gif": str(gif_path.relative_to(result_dir)),
    }
    with (tables_dir / "animation_config.yml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)

    pd.DataFrame(
        [
            {
                "frame_index": r["frame_index"],
                "forecast_date": r["forecast_date"].date().isoformat(),
                "display_time": r["display_time"].isoformat(),
                "history_events": r["history_events"],
                "state_events": r["state_events"],
                "actual_events_in_window": r["actual_events_in_window"],
            }
            for r in records
        ]
    ).to_csv(tables_dir / "forecast_frames.csv", index=False)

    top_rows = []
    for r in records:
        risk = risk_matrix[r["frame_index"]]
        top_idx = np.argsort(risk)[-args.top_k :][::-1]
        for rank, idx in enumerate(top_idx, start=1):
            top_rows.append(
                {
                    "frame_index": r["frame_index"],
                    "forecast_date": r["forecast_date"].date().isoformat(),
                    "rank": rank,
                    "cell_id": cells[idx],
                    "risk": float(risk[idx]),
                }
            )
    pd.DataFrame(top_rows).to_csv(tables_dir / "forecast_top_cells.csv", index=False)

    group_rows = []
    for r in records:
        for i, group in enumerate(groups):
            group_rows.append(
                {
                    "frame_index": r["frame_index"],
                    "forecast_date": r["forecast_date"].date().isoformat(),
                    "crime_group": group,
                    "risk": float(r["group_risk"][i]),
                    "background": float(r["group_background"][i]),
                    "trigger": float(r["group_trigger"][i]),
                }
            )
    pd.DataFrame(group_rows).to_csv(tables_dir / "forecast_group_risk.csv", index=False)

    writer = None
    try:
        for r in records:
            frame_idx = r["frame_index"]
            df = pd.DataFrame(
                {
                    "frame_index": frame_idx,
                    "forecast_date": r["forecast_date"].date().isoformat(),
                    "cell_id": cells.to_numpy(),
                    "risk": risk_matrix[frame_idx],
                }
            )
            table = pa.Table.from_pandas(df, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(tables_dir / "forecast_cell_risk.parquet", table.schema)
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()

    keep_cols = [
        "event_id",
        "occurred_at",
        "crime_code",
        "crime_type",
        "crime_group",
        "area_id",
        "area_name",
        "lat",
        "lon",
        "x",
        "y",
        "cell_id",
        "source_legacy_dataset",
    ]
    keep_cols = [c for c in keep_cols if c in events.columns]
    start = records[0]["forecast_date"]
    end = records[-1]["display_time"]
    observed = events[(events["occurred_at"] >= start) & (events["occurred_at"] < end)]
    observed[keep_cols].to_parquet(tables_dir / "observed_events.parquet", index=False)


def render_frame(
    grid: gpd.GeoDataFrame,
    events: pd.DataFrame,
    times: np.ndarray,
    record: dict,
    risk: np.ndarray,
    vmax: float,
    args: argparse.Namespace,
) -> Image.Image:
    risk_series = pd.Series(risk, index=grid["cell_id"].astype(str), dtype=float)
    heatmap, extent = risk_heatmap(grid, risk_series, args.smooth_sigma)
    heatmap = np.ma.masked_invalid(np.clip(heatmap, 0, vmax))

    fig, ax = plt.subplots(figsize=(9.4, 9.6), dpi=args.dpi, constrained_layout=True)
    ax.imshow(
        heatmap,
        extent=extent,
        origin="lower",
        cmap=RISK_CMAP,
        norm=PowerNorm(gamma=args.heatmap_gamma, vmin=0, vmax=vmax),
        interpolation="bilinear",
        alpha=0.98,
        zorder=1,
    )
    boundary = gpd.GeoSeries([grid.geometry.union_all().boundary], crs=grid.crs)
    boundary.plot(ax=ax, color="#4f4f4f", linewidth=0.45, alpha=0.8, zorder=3)

    display_time = pd.Timestamp(record["display_time"])
    trail_start = display_time - pd.Timedelta(days=7 * args.fade_weeks)
    start_idx = int(np.searchsorted(times, np.datetime64(trail_start.to_datetime64()), side="left"))
    end_idx = int(np.searchsorted(times, np.datetime64(display_time.to_datetime64()), side="left"))
    trail = events.iloc[start_idx:end_idx].copy()
    if not trail.empty:
        centroids = grid[["cell_id", "geometry"]].copy()
        centroids["cx"] = centroids.geometry.centroid.x
        centroids["cy"] = centroids.geometry.centroid.y
        grouped = (
            trail.groupby(["cell_id", "crime_group"])
            .agg(
                count=("event_id", "size"),
                latest=("occurred_at", "max"),
            )
            .reset_index()
            .merge(centroids[["cell_id", "cx", "cy"]], on="cell_id", how="left")
            .dropna(subset=["cx", "cy"])
        )
        ages = (display_time - grouped["latest"]).dt.total_seconds() / 86400.0
        fade = np.clip(1.0 - ages / max(7 * args.fade_weeks, 1), 0.0, 1.0)
        pop = np.exp(-ages / max(args.pop_days, 1e-6))
        sizes = 10 + 85 * np.sqrt(grouped["count"].to_numpy(dtype=float)) * (0.35 + pop) * fade
        for crime_group, subset in grouped.groupby("crime_group"):
            idx = subset.index.to_numpy()
            color = GROUP_COLORS.get(str(crime_group), "#4d4d4d")
            ax.scatter(
                subset["cx"],
                subset["cy"],
                s=sizes[idx],
                facecolors="none",
                edgecolors=color,
                linewidths=0.85,
                alpha=0.35,
                zorder=4,
            )

    minx, miny, maxx, maxy = grid.total_bounds
    pad = max(maxx - minx, maxy - miny) * 0.025
    ax.set_xlim(minx - pad, maxx + pad)
    ax.set_ylim(miny - pad, maxy + pad)
    ax.set_axis_off()
    ax.set_aspect("equal")
    ax.set_title(
        f"LAPD pooled ETAS | week of {record['forecast_date'].date().isoformat()}",
        fontsize=11,
    )
    ax.text(
        0.01,
        0.018,
        "Pooled 21-area ETAS. Red heatmap: next-week risk. Rings: observed crime groups.",
        transform=ax.transAxes,
        va="bottom",
        ha="left",
        fontsize=7.5,
        bbox={"facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.82},
    )
    handles = [
        Line2D([0], [0], marker="o", color=color, markerfacecolor="none", label=group, linewidth=0)
        for group, color in GROUP_COLORS.items()
    ]
    legend = ax.legend(handles=handles, loc="upper right", fontsize=6.5, framealpha=0.84)
    legend.get_frame().set_edgecolor("#cccccc")
    fig.canvas.draw()
    image = Image.fromarray(np.asarray(fig.canvas.buffer_rgba()))
    plt.close(fig)
    return image.convert("P", palette=Image.Palette.ADAPTIVE)


def save_animation(
    result_dir: Path,
    grid: gpd.GeoDataFrame,
    events: pd.DataFrame,
    records: list[dict],
    risk_matrix: np.ndarray,
    args: argparse.Namespace,
) -> Path:
    out_path = result_dir / "animations" / f"{MODEL_ID}_weekly_forecast_heatmap.gif"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    risk_values = risk_matrix.ravel()
    vmax = float(np.quantile(risk_values[np.isfinite(risk_values)], args.vmax_quantile))
    vmax = max(vmax, 1e-9)
    times = events["occurred_at"].to_numpy(dtype="datetime64[ns]")
    frames = []
    for i, record in enumerate(records, start=1):
        frames.append(
            render_frame(
                grid=grid,
                events=events,
                times=times,
                record=record,
                risk=risk_matrix[record["frame_index"]],
                vmax=vmax,
                args=args,
            )
        )
        if i % 25 == 0 or i == len(records):
            print(f"Rendered {i}/{len(records)} frames")
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=args.duration_ms,
        loop=0,
        optimize=True,
    )
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="lapd_legacy_2010_2024_pooled_etas_weekly")
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--history-days", type=int, default=365)
    parser.add_argument("--horizon-hours", type=int, default=168)
    parser.add_argument("--frame-days", type=int, default=7)
    parser.add_argument("--theta", type=float, default=0.35)
    parser.add_argument("--omega", type=float, default=1.0 / 14.0)
    parser.add_argument("--background-alpha", type=float, default=1e-3)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--fade-weeks", type=int, default=4)
    parser.add_argument("--pop-days", type=float, default=2.5)
    parser.add_argument("--smooth-sigma", type=float, default=1.8)
    parser.add_argument("--vmax-quantile", type=float, default=0.985)
    parser.add_argument("--heatmap-gamma", type=float, default=0.4)
    parser.add_argument("--duration-ms", type=int, default=85)
    parser.add_argument("--dpi", type=int, default=85)
    parser.add_argument("--skip-gif", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_dir = ROOT / "results" / "ETAS_full" / args.run_id
    events, grid, _ = load_full_dataset()
    records, cells, groups, risk_matrix = build_weekly_records(events, grid, args)
    if args.skip_gif:
        gif_path = result_dir / "animations" / f"{MODEL_ID}_weekly_forecast_heatmap.gif"
    else:
        gif_path = save_animation(result_dir, grid, events, records, risk_matrix, args)
    save_tables(result_dir, records, cells, groups, risk_matrix, events, args, gif_path)
    print(f"Wrote {result_dir}")


if __name__ == "__main__":
    main()
