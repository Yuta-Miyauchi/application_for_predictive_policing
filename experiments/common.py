"""Shared experiment utilities."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]


def load_dataset(dataset_id: str) -> tuple[pd.DataFrame, gpd.GeoDataFrame, dict, Path]:
    """Load a prepared dataset by ID."""

    candidates = list((ROOT / "datas").glob(f"*/{dataset_id}"))
    if not candidates:
        raise FileNotFoundError(f"Dataset ID not found under datas/: {dataset_id}")
    dataset_dir = candidates[0]
    metadata_path = dataset_dir / "metadata.yml"
    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = yaml.safe_load(f)
    events = pd.read_parquet(dataset_dir / "processed" / "events.parquet")
    events["occurred_at"] = pd.to_datetime(events["occurred_at"])
    grid = gpd.read_file(dataset_dir / "processed" / "grid.geojson")
    grid["cell_id"] = grid["cell_id"].astype(str)
    return events, grid, metadata, dataset_dir


def evaluate_day(
    risk: pd.Series,
    grid: gpd.GeoDataFrame,
    actual_events: pd.DataFrame,
    top_k: int,
) -> dict:
    """Evaluate top-k cell predictions against events in one forecast window."""

    ranked = risk.sort_values(ascending=False)
    selected = ranked.head(top_k).index.astype(str)
    selected_set = set(selected)
    actual_n = len(actual_events)
    hit_events = int(actual_events["cell_id"].astype(str).isin(selected_set).sum())
    selected_grid = grid[grid["cell_id"].astype(str).isin(selected_set)]
    selected_area = (
        float(selected_grid["area_m2"].sum()) if "area_m2" in grid else float(len(selected))
    )
    total_area = float(grid["area_m2"].sum()) if "area_m2" in grid else float(len(grid))
    hit_rate = hit_events / actual_n if actual_n else None
    area_share = selected_area / total_area if total_area else None
    pai = hit_rate / area_share if actual_n and area_share else None
    actual_cells = set(actual_events["cell_id"].astype(str))
    precision_at_k = len(actual_cells & selected_set) / len(selected_set) if selected_set else None
    recall_at_k = hit_events / actual_n if actual_n else None
    return {
        "actual_events": actual_n,
        "hit_events": hit_events,
        "selected_cells": len(selected_set),
        "selected_area_m2": selected_area,
        "total_area_m2": total_area,
        "area_share": area_share,
        "hit_rate": hit_rate,
        "pai": pai,
        "precision_at_k": precision_at_k,
        "recall_at_k": recall_at_k,
    }
