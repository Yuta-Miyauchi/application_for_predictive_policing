"""Prepare Chicago public-crime dataset variants.

The first external-reproduction variant is intentionally small: one police
district, one crime type, and the same broad time period used for the first
LAPD experiments.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests
import yaml
from shapely.geometry import box


ROOT = Path(__file__).resolve().parents[2]
CHICAGO_ROOT = ROOT / "datas" / "chicago"
CRIME_ENDPOINT = "https://data.cityofchicago.org/resource/ijzp-q8t2.json"
BOUNDARY_ENDPOINT = "https://data.cityofchicago.org/resource/9vmg-9p8p.geojson"
SOURCE_FIELDS = [
    "id",
    "case_number",
    "date",
    "block",
    "iucr",
    "primary_type",
    "description",
    "location_description",
    "arrest",
    "domestic",
    "beat",
    "district",
    "ward",
    "community_area",
    "year",
    "latitude",
    "longitude",
    "updated_on",
]
CRS_WGS84 = "EPSG:4326"
CRS_METERS = "EPSG:26971"


@dataclass(frozen=True)
class ChicagoVariant:
    dataset_id: str
    district: str
    crime_type: str
    start: str
    end: str
    grid_m: int = 150
    horizon_hours: int = 24


VARIANTS = [
    ChicagoVariant(
        "chicago_district011_burglary_2011_2013_grid150m_h24h",
        "011",
        "BURGLARY",
        "2011-01-01",
        "2013-01-11",
    )
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_json(url: str, params: dict, timeout: int = 60) -> object:
    response = requests.get(url, params=params, timeout=timeout)
    response.raise_for_status()
    return response.json()


def fetch_boundary(district: str) -> gpd.GeoDataFrame:
    district_num = str(int(district))
    params = {"$where": f"dist_num='{district_num}'", "$limit": 5000}
    data = fetch_json(BOUNDARY_ENDPOINT, params)
    features = data.get("features", []) if isinstance(data, dict) else []
    if not features:
        raise RuntimeError(f"No Chicago boundary features found for district {district}")
    return gpd.GeoDataFrame.from_features(features, crs=CRS_WGS84).to_crs(CRS_METERS)


def fetch_events(variant: ChicagoVariant) -> pd.DataFrame:
    rows: list[dict] = []
    limit = 50000
    offset = 0
    where = (
        f"date between '{variant.start}T00:00:00' and '{variant.end}T00:00:00' "
        f"and district='{variant.district}' and primary_type='{variant.crime_type}'"
    )
    while True:
        params = {
            "$select": ",".join(SOURCE_FIELDS),
            "$where": where,
            "$limit": limit,
            "$offset": offset,
            "$order": "date,id",
        }
        batch = fetch_json(CRIME_ENDPOINT, params)
        if not isinstance(batch, list):
            raise RuntimeError(f"Unexpected Socrata response: {batch!r}")
        if not batch:
            break
        rows.extend(batch)
        if len(batch) < limit:
            break
        offset += limit
    return pd.DataFrame(rows)


def clean_events(raw: pd.DataFrame, variant: ChicagoVariant) -> tuple[pd.DataFrame, dict]:
    original_count = len(raw)
    df = raw.copy().drop_duplicates(subset=["id"])
    duplicate_count = original_count - len(df)
    df["occurred_at"] = pd.to_datetime(df["date"], errors="coerce")
    before_coord = len(df[df["occurred_at"].notna()])
    df = df[df["occurred_at"].notna()].copy()
    df["lat"] = pd.to_numeric(df.get("latitude"), errors="coerce")
    df["lon"] = pd.to_numeric(df.get("longitude"), errors="coerce")
    valid_coord = (
        df["lat"].notna()
        & df["lon"].notna()
        & df["lat"].between(41.5, 42.1)
        & df["lon"].between(-88.0, -87.3)
    )
    df = df[valid_coord].copy()
    df["event_id"] = df["id"].astype(str)
    df["crime_type"] = variant.crime_type
    df["area_id"] = variant.district
    df["area_name"] = f"Chicago District {variant.district}"
    df["source_dataset"] = variant.dataset_id
    summary = {
        "raw_rows": int(original_count),
        "duplicate_id_rows": int(duplicate_count),
        "invalid_datetime_rows": int(original_count - duplicate_count - before_coord),
        "invalid_coordinate_rows": int(before_coord - len(df)),
        "after_basic_cleaning_rows": int(len(df)),
    }
    return df, summary


def make_grid(boundary: gpd.GeoDataFrame, grid_m: int) -> gpd.GeoDataFrame:
    polygon = boundary.geometry.union_all()
    minx, miny, maxx, maxy = polygon.bounds
    minx = math.floor(minx / grid_m) * grid_m
    miny = math.floor(miny / grid_m) * grid_m
    maxx = math.ceil(maxx / grid_m) * grid_m
    maxy = math.ceil(maxy / grid_m) * grid_m
    cells = []
    y = miny
    row = 0
    while y < maxy:
        x = minx
        col = 0
        while x < maxx:
            geom = box(x, y, x + grid_m, y + grid_m)
            if geom.intersects(polygon):
                clipped = geom.intersection(polygon)
                if not clipped.is_empty:
                    cells.append(
                        {
                            "cell_id": f"c{row:04d}_{col:04d}",
                            "row": row,
                            "col": col,
                            "area_m2": float(clipped.area),
                            "geometry": clipped,
                        }
                    )
            x += grid_m
            col += 1
        y += grid_m
        row += 1
    return gpd.GeoDataFrame(cells, crs=CRS_METERS)


def assign_cells(
    clean: pd.DataFrame, boundary: gpd.GeoDataFrame, grid: gpd.GeoDataFrame
) -> tuple[pd.DataFrame, dict]:
    points = gpd.GeoDataFrame(
        clean,
        geometry=gpd.points_from_xy(clean["lon"], clean["lat"]),
        crs=CRS_WGS84,
    ).to_crs(CRS_METERS)
    points["x"] = points.geometry.x
    points["y"] = points.geometry.y
    before_clip = len(points)
    points = gpd.sjoin(points, boundary[["geometry"]], predicate="within", how="inner")
    points = points.drop(columns=[c for c in ["index_right"] if c in points.columns])
    after_boundary = len(points)
    joined = gpd.sjoin(points, grid[["cell_id", "geometry"]], predicate="within", how="left")
    joined = joined.drop(columns=[c for c in ["index_right"] if c in joined.columns])
    joined = joined[joined["cell_id"].notna()].copy()
    output_cols = [
        "event_id",
        "occurred_at",
        "crime_type",
        "area_id",
        "area_name",
        "case_number",
        "block",
        "iucr",
        "primary_type",
        "description",
        "location_description",
        "beat",
        "district",
        "ward",
        "community_area",
        "year",
        "lat",
        "lon",
        "x",
        "y",
        "cell_id",
        "source_dataset",
    ]
    summary = {
        "outside_district_rows": int(before_clip - after_boundary),
        "unassigned_cell_rows": int(after_boundary - len(joined)),
        "processed_rows": int(len(joined)),
    }
    return pd.DataFrame(joined[output_cols]), summary


def write_metadata(
    dataset_dir: Path,
    variant: ChicagoVariant,
    raw_path: Path,
    grid_path: Path,
    events_path: Path,
    cleaning_summary: dict,
    boundary: gpd.GeoDataFrame,
    grid: gpd.GeoDataFrame,
) -> None:
    metadata = {
        "dataset_id": variant.dataset_id,
        "source": {
            "city": "Chicago",
            "provider": "Chicago Police Department via City of Chicago Data Portal",
            "crime_endpoint": CRIME_ENDPOINT,
            "boundary_endpoint": BOUNDARY_ENDPOINT,
            "retrieved_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        },
        "filters": {
            "district": variant.district,
            "crime_type": variant.crime_type,
            "start": variant.start,
            "end": variant.end,
        },
        "spatial": {
            "source_crs": CRS_WGS84,
            "projected_crs": CRS_METERS,
            "grid_m": variant.grid_m,
            "grid_cells": int(len(grid)),
            "district_area_m2": float(boundary.geometry.union_all().area),
            "grid_area_m2": float(grid["area_m2"].sum()),
        },
        "forecast": {"horizon_hours": variant.horizon_hours},
        "files": {
            "raw": str(raw_path.relative_to(dataset_dir)),
            "processed_events": str(events_path.relative_to(dataset_dir)),
            "processed_grid": str(grid_path.relative_to(dataset_dir)),
            "raw_sha256": sha256_file(raw_path),
            "processed_events_sha256": sha256_file(events_path),
            "processed_grid_sha256": sha256_file(grid_path),
        },
        "cleaning_summary": cleaning_summary,
        "limitations": [
            "Chicago public crime locations are approximate and block-level; do not infer exact addresses.",
            "This dataset uses public records and is intended for technical reproduction only.",
        ],
    }
    with (dataset_dir / "metadata.yml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(metadata, f, sort_keys=False, allow_unicode=True)


def prepare_variant(variant: ChicagoVariant, overwrite: bool = False) -> Path:
    dataset_dir = CHICAGO_ROOT / variant.dataset_id
    raw_dir = dataset_dir / "raw"
    processed_dir = dataset_dir / "processed"
    if dataset_dir.exists() and not overwrite:
        raise FileExistsError(f"{dataset_dir} already exists; pass --overwrite to rebuild")
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    raw = fetch_events(variant)
    if raw.empty:
        raise RuntimeError(f"No rows returned for {variant.dataset_id}")
    raw_path = raw_dir / "events_raw.parquet"
    raw.to_parquet(raw_path, index=False)
    boundary = fetch_boundary(variant.district)
    boundary.to_crs(CRS_WGS84).to_file(processed_dir / "district_boundary.geojson", driver="GeoJSON")
    clean, clean_summary = clean_events(raw, variant)
    grid = make_grid(boundary, variant.grid_m)
    events, assign_summary = assign_cells(clean, boundary, grid)
    cleaning_summary = {**clean_summary, **assign_summary}
    events_path = processed_dir / "events.parquet"
    grid_path = processed_dir / "grid.geojson"
    events.to_parquet(events_path, index=False)
    grid.to_crs(CRS_WGS84).to_file(grid_path, driver="GeoJSON")
    write_metadata(
        dataset_dir,
        variant,
        raw_path,
        grid_path,
        events_path,
        cleaning_summary,
        boundary,
        grid,
    )
    return dataset_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        action="append",
        choices=[v.dataset_id for v in VARIANTS],
        help="Dataset variant to prepare. Repeatable. Defaults to all variants.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--list", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list:
        print(json.dumps([v.__dict__ for v in VARIANTS], indent=2))
        return
    selected = set(args.variant or [v.dataset_id for v in VARIANTS])
    for variant in VARIANTS:
        if variant.dataset_id not in selected:
            continue
        print(f"Preparing {variant.dataset_id}...")
        path = prepare_variant(variant, overwrite=args.overwrite)
        print(f"  wrote {path}")


if __name__ == "__main__":
    main()
