"""Shared LAPD target-crime data and animation helpers."""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
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
from scipy.ndimage import gaussian_filter
from shapely.geometry import box
from shapely.prepared import prep


SOURCE_DATASET_ID = "lapd_legacy_2010_2024_all_crimes_grid300m_h168h"
SOURCE_DIR = ROOT / "datas" / "lapd_full" / SOURCE_DATASET_ID
TARGET_DATASET_ID = "lapd_legacy_2010_2024_mohler_target_crimes_grid150m_h24h"
TARGET_DIR = SOURCE_DIR / "derived" / TARGET_DATASET_ID

TARGET_CRIME_ORDER = ["BURGLARY", "CAR_THEFT", "THEFT_FROM_VEHICLE"]
TARGET_CRIME_COLORS = {
    "BURGLARY": "#2b8cbe",
    "CAR_THEFT": "#31a354",
    "THEFT_FROM_VEHICLE": "#756bb1",
}

RISK_CMAP = LinearSegmentedColormap.from_list(
    "predictive_risk_red_strong",
    ["#fff7bc", "#fec44f", "#fb6a4a", "#de2d26", "#a50f15", "#4d0013"],
)


@dataclass(frozen=True)
class TargetDataset:
    events: pd.DataFrame
    grid: gpd.GeoDataFrame
    boundaries: gpd.GeoDataFrame
    metadata: dict


def classify_target_crime(crime_type: pd.Series) -> pd.Series:
    """Map LAPD crime descriptions to Mohler et al. Los Angeles target crimes."""

    text = crime_type.fillna("").astype(str).str.upper()
    out = pd.Series(pd.NA, index=crime_type.index, dtype="object")
    burglary = text.isin(["BURGLARY", "BURGLARY, ATTEMPTED"])
    car_theft = (
        text.str.contains("VEHICLE - STOLEN", regex=False)
        | text.str.contains("VEHICLE, STOLEN", regex=False)
        | text.eq("VEHICLE - ATTEMPT STOLEN")
    )
    theft_from_vehicle = (
        text.str.contains("BURGLARY FROM VEHICLE", regex=False)
        | text.str.contains("THEFT FROM MOTOR VEHICLE", regex=False)
    )
    out.loc[burglary] = "BURGLARY"
    out.loc[car_theft] = "CAR_THEFT"
    out.loc[theft_from_vehicle] = "THEFT_FROM_VEHICLE"
    return out


def _source_metadata() -> dict:
    metadata_path = SOURCE_DIR / "metadata.yml"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Missing {metadata_path}. Run datas/lapd_full/prepare_lapd_legacy_full.py first."
        )
    with metadata_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _cell_id(row: int, col: int) -> str:
    return f"r{row:04d}_c{col:04d}"


def _build_grid150(boundaries: gpd.GeoDataFrame, cell_size_m: int) -> tuple[gpd.GeoDataFrame, float, float]:
    minx, miny, maxx, maxy = boundaries.total_bounds
    origin_x = math.floor(minx / cell_size_m) * cell_size_m
    origin_y = math.floor(miny / cell_size_m) * cell_size_m
    col_min = int(math.floor((minx - origin_x) / cell_size_m))
    col_max = int(math.ceil((maxx - origin_x) / cell_size_m))
    row_min = int(math.floor((miny - origin_y) / cell_size_m))
    row_max = int(math.ceil((maxy - origin_y) / cell_size_m))

    union = boundaries.geometry.union_all()
    prepared_union = prep(union)
    records: list[dict] = []
    for row in range(row_min, row_max):
        y0 = origin_y + row * cell_size_m
        y1 = y0 + cell_size_m
        for col in range(col_min, col_max):
            x0 = origin_x + col * cell_size_m
            x1 = x0 + cell_size_m
            geom = box(x0, y0, x1, y1)
            if not prepared_union.intersects(geom):
                continue
            records.append(
                {
                    "cell_id": _cell_id(row, col),
                    "row": row,
                    "col": col,
                    "area_m2": float(cell_size_m * cell_size_m),
                    "geometry": geom,
                }
            )
    grid = gpd.GeoDataFrame(records, geometry="geometry", crs=boundaries.crs)
    centers = gpd.GeoDataFrame(
        {"grid_index": grid.index},
        geometry=grid.geometry.centroid,
        crs=grid.crs,
    )
    joined = gpd.sjoin(
        centers,
        boundaries[["area_id", "area_name", "geometry"]],
        how="left",
        predicate="within",
    ).drop_duplicates("grid_index")
    grid["area_id"] = joined.set_index("grid_index").reindex(grid.index)["area_id"].astype("object")
    grid["area_name"] = joined.set_index("grid_index").reindex(grid.index)["area_name"].astype("object")
    missing = grid["area_id"].isna()
    if missing.any():
        for idx, geom in grid.loc[missing, "geometry"].items():
            intersections = boundaries.geometry.intersection(geom).area
            best = boundaries.loc[intersections.idxmax()]
            grid.loc[idx, "area_id"] = str(best["area_id"]).zfill(2)
            grid.loc[idx, "area_name"] = str(best["area_name"])
    grid["area_id"] = grid["area_id"].astype(str).str.zfill(2)
    return grid.sort_values(["row", "col"]).reset_index(drop=True), origin_x, origin_y


def _prepare_target_dataset(cell_size_m: int = 150) -> TargetDataset:
    metadata = _source_metadata()
    projected_crs = metadata.get("spatial", {}).get("projected_crs", "EPSG:3310")
    raw_events = pd.read_parquet(
        SOURCE_DIR / "processed" / "events.parquet",
        columns=[
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
            "source_legacy_dataset",
        ],
    )
    raw_events["occurred_at"] = pd.to_datetime(raw_events["occurred_at"])
    raw_events["target_crime"] = classify_target_crime(raw_events["crime_type"])
    events = raw_events.dropna(subset=["target_crime", "x", "y"]).copy()
    events["target_crime"] = pd.Categorical(
        events["target_crime"].astype(str),
        categories=TARGET_CRIME_ORDER,
        ordered=True,
    )
    events["area_id"] = events["area_id"].astype(str).str.zfill(2)

    boundaries = gpd.read_file(SOURCE_DIR / "processed" / "division_boundaries.geojson")
    boundaries["area_id"] = boundaries["area_id"].astype(str).str.zfill(2)
    boundaries = boundaries.to_crs(projected_crs)
    grid, origin_x, origin_y = _build_grid150(boundaries, cell_size_m)

    events["col"] = np.floor((events["x"].to_numpy(dtype=float) - origin_x) / cell_size_m).astype(int)
    events["row"] = np.floor((events["y"].to_numpy(dtype=float) - origin_y) / cell_size_m).astype(int)
    events["cell_id"] = [
        _cell_id(int(row), int(col)) for row, col in zip(events["row"], events["col"], strict=False)
    ]
    valid_cells = set(grid["cell_id"].astype(str))
    events = events[events["cell_id"].isin(valid_cells)].copy()
    events = events.rename(columns={"area_id": "source_area_id", "area_name": "source_area_name"})
    cell_area = grid[["cell_id", "area_id", "area_name"]].copy()
    events = events.merge(cell_area, on="cell_id", how="left")
    events["area_id"] = events["area_id"].astype(str).str.zfill(2)
    events = events.sort_values("occurred_at").reset_index(drop=True)

    target_metadata = {
        "dataset_id": TARGET_DATASET_ID,
        "source_dataset_id": SOURCE_DATASET_ID,
        "paper_alignment": {
            "reference": "Mohler et al. JASA predictive policing field experiment",
            "target_crimes": TARGET_CRIME_ORDER,
            "cell_size_m": cell_size_m,
            "forecast_horizon_hours": 24,
            "history_days": 365,
            "note": (
                "Derived locally from full LAPD legacy public data. The original source "
                "data and this derived dataset remain git-ignored under datas/."
            ),
        },
        "spatial": {
            "projected_crs": projected_crs,
            "cell_size_m": cell_size_m,
            "origin_x": float(origin_x),
            "origin_y": float(origin_y),
        },
        "counts": {
            "events": int(len(events)),
            "cells": int(len(grid)),
            "areas": int(grid["area_id"].nunique()),
            "target_crimes": int(events["target_crime"].nunique()),
        },
    }
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    events.to_parquet(TARGET_DIR / "events.parquet", index=False)
    grid.to_parquet(TARGET_DIR / "grid.parquet", index=False)
    boundaries.to_parquet(TARGET_DIR / "division_boundaries.parquet", index=False)
    with (TARGET_DIR / "metadata.yml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(target_metadata, f, sort_keys=False, allow_unicode=True)
    return TargetDataset(events=events, grid=grid, boundaries=boundaries, metadata=target_metadata)


def load_target_dataset(overwrite: bool = False, cell_size_m: int = 150) -> TargetDataset:
    """Load or locally derive the Mohler target-crime 150m LAPD dataset."""

    metadata_path = TARGET_DIR / "metadata.yml"
    events_path = TARGET_DIR / "events.parquet"
    grid_path = TARGET_DIR / "grid.parquet"
    boundaries_path = TARGET_DIR / "division_boundaries.parquet"
    if overwrite or not (metadata_path.exists() and events_path.exists() and grid_path.exists()):
        return _prepare_target_dataset(cell_size_m=cell_size_m)
    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = yaml.safe_load(f)
    events = pd.read_parquet(events_path)
    events["occurred_at"] = pd.to_datetime(events["occurred_at"])
    events["cell_id"] = events["cell_id"].astype(str)
    events["area_id"] = events["area_id"].astype(str).str.zfill(2)
    events["target_crime"] = pd.Categorical(
        events["target_crime"].astype(str),
        categories=TARGET_CRIME_ORDER,
        ordered=True,
    )
    grid = gpd.read_parquet(grid_path)
    grid["cell_id"] = grid["cell_id"].astype(str)
    grid["area_id"] = grid["area_id"].astype(str).str.zfill(2)
    boundaries = gpd.read_parquet(boundaries_path)
    boundaries["area_id"] = boundaries["area_id"].astype(str).str.zfill(2)
    return TargetDataset(events=events, grid=grid, boundaries=boundaries, metadata=metadata)


def observed_window_counts(
    events: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> int:
    return int(((events["occurred_at"] >= start) & (events["occurred_at"] < end)).sum())


def write_forecast_tables(
    result_dir: Path,
    model_id: str,
    dataset: TargetDataset,
    records: list[dict],
    cells: pd.Index,
    risk_matrix: np.ndarray,
    args: SimpleNamespace,
    gif_path: Path,
    extra_config: dict | None = None,
) -> None:
    tables_dir = result_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "dataset_id": TARGET_DATASET_ID,
        "source_dataset_id": SOURCE_DATASET_ID,
        "model_id": model_id,
        "forecast_start": records[0]["forecast_date"].date().isoformat(),
        "forecast_end": records[-1]["display_time"].date().isoformat(),
        "target_crimes": TARGET_CRIME_ORDER,
        "cell_size_m": dataset.metadata["spatial"]["cell_size_m"],
        "horizon_hours": getattr(args, "horizon_hours", None),
        "history_days": getattr(args, "history_days", None),
        "frame_days": getattr(args, "frame_days", None),
        "fade_days": getattr(args, "fade_days", None),
        "smooth_sigma": getattr(args, "smooth_sigma", None),
        "vmax_quantile": getattr(args, "vmax_quantile", None),
        "heatmap_gamma": getattr(args, "heatmap_gamma", None),
        "duration_ms": getattr(args, "duration_ms", None),
        "dpi": getattr(args, "dpi", None),
        "gif": str(gif_path.relative_to(result_dir)),
    }
    if extra_config:
        config.update(extra_config)
    with (tables_dir / "animation_config.yml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)

    pd.DataFrame(
        [
            {
                key: value
                for key, value in r.items()
                if key not in {"risk", "risk_by_crime"}
            }
            for r in records
        ]
    ).to_csv(tables_dir / "forecast_frames.csv", index=False)

    writer = None
    try:
        for r in records:
            frame_idx = int(r["frame_index"])
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

    top_k = int(getattr(args, "top_k", 20))
    top_rows = []
    for r in records:
        risk = risk_matrix[int(r["frame_index"])]
        top_idx = np.argsort(risk)[-top_k:][::-1]
        for rank, idx in enumerate(top_idx, start=1):
            top_rows.append(
                {
                    "frame_index": int(r["frame_index"]),
                    "forecast_date": r["forecast_date"].date().isoformat(),
                    "rank": rank,
                    "cell_id": cells[idx],
                    "risk": float(risk[idx]),
                }
            )
    pd.DataFrame(top_rows).to_csv(tables_dir / "forecast_top_cells.csv", index=False)

    start = records[0]["forecast_date"]
    end = records[-1]["display_time"]
    observed = dataset.events[
        (dataset.events["occurred_at"] >= start) & (dataset.events["occurred_at"] < end)
    ].copy()
    observed.to_parquet(tables_dir / "observed_events.parquet", index=False)


def _event_trail_for_frame(
    grid: gpd.GeoDataFrame,
    events: pd.DataFrame,
    display_time: pd.Timestamp,
    fade_days: int,
) -> pd.DataFrame:
    trail = events[
        (events["occurred_at"] < display_time)
        & (events["occurred_at"] >= display_time - pd.Timedelta(days=fade_days))
    ].copy()
    if trail.empty:
        return trail
    centroids = grid[["cell_id", "geometry"]].copy()
    centroids["cx"] = centroids.geometry.centroid.x
    centroids["cy"] = centroids.geometry.centroid.y
    grouped = (
        trail.groupby(["cell_id", "target_crime"], observed=True)
        .agg(count=("event_id", "size"), latest=("occurred_at", "max"))
        .reset_index()
        .merge(centroids[["cell_id", "cx", "cy"]], on="cell_id", how="left")
        .dropna(subset=["cx", "cy"])
    )
    return grouped


def render_forecast_frame(
    grid: gpd.GeoDataFrame,
    boundaries: gpd.GeoDataFrame,
    events: pd.DataFrame,
    record: dict,
    risk: np.ndarray,
    vmax: float,
    args: SimpleNamespace,
    title: str,
    note: str,
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
    boundaries.boundary.plot(ax=ax, color="#505050", linewidth=0.33, alpha=0.78, zorder=3)

    display_time = pd.Timestamp(record["display_time"])
    trail = _event_trail_for_frame(grid, events, display_time, args.fade_days)
    if not trail.empty:
        ages = (display_time - trail["latest"]).dt.total_seconds() / 86400.0
        fade = np.clip(1.0 - ages / max(args.fade_days, 1), 0.0, 1.0)
        pop = np.exp(-ages / max(args.pop_days, 1e-6))
        ring_size = 18 + 155 * np.sqrt(trail["count"].to_numpy(dtype=float)) * (0.35 + pop) * fade
        for target_crime, subset in trail.groupby("target_crime", observed=True):
            idx = subset.index.to_numpy()
            color = TARGET_CRIME_COLORS.get(str(target_crime), "#4d4d4d")
            ax.scatter(
                subset["cx"],
                subset["cy"],
                s=ring_size[idx],
                facecolors="none",
                edgecolors=color,
                linewidths=0.95,
                alpha=0.48,
                zorder=4,
            )

    minx, miny, maxx, maxy = grid.total_bounds
    pad = max(maxx - minx, maxy - miny) * 0.025
    ax.set_xlim(minx - pad, maxx + pad)
    ax.set_ylim(miny - pad, maxy + pad)
    ax.set_axis_off()
    ax.set_aspect("equal")
    ax.set_title(
        f"{title} | forecast {record['forecast_date'].date().isoformat()}",
        fontsize=11,
    )
    ax.text(
        0.01,
        0.018,
        note,
        transform=ax.transAxes,
        va="bottom",
        ha="left",
        fontsize=7.4,
        bbox={"facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.84},
    )
    handles = [
        Line2D([0], [0], marker="o", color=color, markerfacecolor="none", label=crime, linewidth=0)
        for crime, color in TARGET_CRIME_COLORS.items()
    ]
    legend = ax.legend(handles=handles, loc="upper right", fontsize=6.6, framealpha=0.84)
    legend.get_frame().set_edgecolor("#cccccc")
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
        row = int(item.row) - row_min
        col = int(item.col) - col_min
        values[row, col] = float(risk_lookup.get(str(item.cell_id), 0.0))
        mask[row, col] = 1.0
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


def save_forecast_animation(
    out_path: Path,
    dataset: TargetDataset,
    records: list[dict],
    risk_matrix: np.ndarray,
    args: SimpleNamespace,
    title: str,
    note: str,
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    finite = risk_matrix[np.isfinite(risk_matrix)]
    vmax = float(np.quantile(finite, args.vmax_quantile)) if len(finite) else 1.0
    vmax = max(vmax, 1e-9)
    frames = []
    for i, record in enumerate(records, start=1):
        frames.append(
            render_forecast_frame(
                grid=dataset.grid,
                boundaries=dataset.boundaries,
                events=dataset.events,
                record=record,
                risk=risk_matrix[int(record["frame_index"])],
                vmax=vmax,
                args=args,
                title=title,
                note=note,
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
