"""Prepare LAPD public-crime dataset variants.

This script downloads filtered rows from the LA City Socrata API, clips them
to official LAPD division boundaries, builds a metric grid, and writes a
model-ready dataset variant under datas/lapd/<dataset_id>/.
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
LAPD_ROOT = ROOT / "datas" / "lapd"
CRIME_ENDPOINT = "https://data.lacity.org/resource/63jg-8b9z.json"
BOUNDARY_ENDPOINT = (
    "https://maps.lacity.org/arcgis/rest/services/Mapping/Boundaries/MapServer/11/query"
)
SOURCE_FIELDS = [
    "dr_no",
    "date_rptd",
    "date_occ",
    "time_occ",
    "area",
    "area_name",
    "rpt_dist_no",
    "crm_cd",
    "crm_cd_desc",
    "premis_cd",
    "premis_desc",
    "location",
    "lat",
    "lon",
]
CRS_WGS84 = "EPSG:4326"
CRS_METERS = "EPSG:3310"


@dataclass(frozen=True)
class LapdVariant:
    dataset_id: str
    area_code: str
    area_name: str
    crime_code: str
    crime_name: str
    start: str
    end: str
    grid_m: int = 150
    horizon_hours: int = 24


VARIANTS = [
    LapdVariant(
        "lapd_foothill_burglary_2011_2013_grid150m_h24h",
        "16",
        "Foothill",
        "310",
        "BURGLARY",
        "2011-01-01",
        "2013-01-11",
    ),
    LapdVariant(
        "lapd_n_hollywood_burglary_2011_2013_grid150m_h24h",
        "15",
        "N Hollywood",
        "310",
        "BURGLARY",
        "2011-01-01",
        "2013-01-11",
    ),
    LapdVariant(
        "lapd_southwest_burglary_2011_2013_grid150m_h24h",
        "03",
        "Southwest",
        "310",
        "BURGLARY",
        "2011-01-01",
        "2013-01-11",
    ),
    LapdVariant(
        "lapd_foothill_vehicle_stolen_2011_2013_grid150m_h24h",
        "16",
        "Foothill",
        "510",
        "VEHICLE - STOLEN",
        "2011-01-01",
        "2013-01-11",
    ),
    LapdVariant(
        "lapd_n_hollywood_vehicle_stolen_2011_2013_grid150m_h24h",
        "15",
        "N Hollywood",
        "510",
        "VEHICLE - STOLEN",
        "2011-01-01",
        "2013-01-11",
    ),
    LapdVariant(
        "lapd_southwest_vehicle_stolen_2011_2013_grid150m_h24h",
        "03",
        "Southwest",
        "510",
        "VEHICLE - STOLEN",
        "2011-01-01",
        "2013-01-11",
    ),
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


def fetch_lapd_boundary(area_name: str, area_code: str) -> gpd.GeoDataFrame:
    params = {
        "where": f"PREC={int(area_code)}",
        "outFields": "*",
        "f": "geojson",
        "outSR": "4326",
        "returnGeometry": "true",
    }
    data = fetch_json(BOUNDARY_ENDPOINT, params)
    features = data.get("features", []) if isinstance(data, dict) else []
    if not features:
        raise RuntimeError(f"No LAPD boundary features found for {area_name}")
    gdf = gpd.GeoDataFrame.from_features(features, crs=CRS_WGS84)
    gdf["APREC"] = gdf["APREC"].astype(str)
    dissolved = gdf.dissolve(by="APREC", as_index=False)
    match = dissolved[dissolved["APREC"].str.casefold() == area_name.casefold()]
    if match.empty:
        match = dissolved
    return match.to_crs(CRS_METERS)


def fetch_lapd_events(variant: LapdVariant) -> pd.DataFrame:
    rows: list[dict] = []
    limit = 50000
    offset = 0
    where = (
        f"date_occ between '{variant.start}T00:00:00' and "
        f"'{variant.end}T00:00:00' and area='{variant.area_code}' "
        f"and crm_cd='{variant.crime_code}'"
    )
    while True:
        params = {
            "$select": ",".join(SOURCE_FIELDS),
            "$where": where,
            "$limit": limit,
            "$offset": offset,
            "$order": "date_occ,time_occ,dr_no",
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


def clean_lapd_events(raw: pd.DataFrame, variant: LapdVariant) -> tuple[pd.DataFrame, dict]:
    df = raw.copy()
    original_count = len(df)
    df = df.drop_duplicates(subset=["dr_no"])
    duplicate_count = original_count - len(df)

    df["date_occ"] = pd.to_datetime(df["date_occ"], errors="coerce")
    df["date_rptd"] = pd.to_datetime(df["date_rptd"], errors="coerce")
    df["time_occ_clean"] = df["time_occ"].fillna("").astype(str).str.extract(r"(\d+)")[0]
    df["time_occ_clean"] = df["time_occ_clean"].fillna("").str.zfill(4)
    df["hour"] = pd.to_numeric(df["time_occ_clean"].str[:2], errors="coerce")
    df["minute"] = pd.to_numeric(df["time_occ_clean"].str[2:4], errors="coerce")
    valid_time = df["hour"].between(0, 23) & df["minute"].between(0, 59)
    df = df[df["date_occ"].notna() & valid_time].copy()
    df["occurred_at"] = df["date_occ"].dt.normalize() + pd.to_timedelta(
        df["hour"], unit="h"
    ) + pd.to_timedelta(df["minute"], unit="m")

    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    valid_coord = (
        df["lat"].notna()
        & df["lon"].notna()
        & (df["lat"] != 0)
        & (df["lon"] != 0)
        & df["lat"].between(33.5, 34.5)
        & df["lon"].between(-119.0, -117.5)
    )
    before_coord = len(df)
    df = df[valid_coord].copy()

    df["event_id"] = df["dr_no"].astype(str)
    df["crime_type"] = variant.crime_name
    df["area_id"] = variant.area_code
    df["source_dataset"] = variant.dataset_id
    summary = {
        "raw_rows": int(original_count),
        "duplicate_dr_no_rows": int(duplicate_count),
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
        "date_rptd",
        "crime_type",
        "area_id",
        "area_name",
        "crm_cd",
        "crm_cd_desc",
        "rpt_dist_no",
        "premis_cd",
        "premis_desc",
        "location",
        "lat",
        "lon",
        "x",
        "y",
        "cell_id",
        "source_dataset",
    ]
    summary = {
        "outside_division_rows": int(before_clip - after_boundary),
        "unassigned_cell_rows": int(after_boundary - len(joined)),
        "processed_rows": int(len(joined)),
    }
    return pd.DataFrame(joined[output_cols]), summary


def write_metadata(
    dataset_dir: Path,
    variant: LapdVariant,
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
            "city": "Los Angeles",
            "provider": "Los Angeles Police Department via LA City Open Data",
            "crime_endpoint": CRIME_ENDPOINT,
            "boundary_endpoint": BOUNDARY_ENDPOINT,
            "retrieved_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        },
        "filters": {
            "area_code": variant.area_code,
            "area_name": variant.area_name,
            "crime_code": variant.crime_code,
            "crime_name": variant.crime_name,
            "start": variant.start,
            "end": variant.end,
        },
        "spatial": {
            "source_crs": CRS_WGS84,
            "projected_crs": CRS_METERS,
            "grid_m": variant.grid_m,
            "grid_cells": int(len(grid)),
            "division_area_m2": float(boundary.geometry.union_all().area),
            "grid_area_m2": float(grid["area_m2"].sum()),
        },
        "forecast": {
            "horizon_hours": variant.horizon_hours,
        },
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
            "Public historical records may differ from operational records available at prediction time.",
            "Occurrence datetime is based on DATE OCC and TIME OCC; reporting delay is not modeled here.",
            "This dataset is prepared for technical reproduction only, not operational deployment.",
        ],
    }
    with (dataset_dir / "metadata.yml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(metadata, f, sort_keys=False, allow_unicode=True)


def prepare_variant(variant: LapdVariant, overwrite: bool = False) -> Path:
    dataset_dir = LAPD_ROOT / variant.dataset_id
    raw_dir = dataset_dir / "raw"
    processed_dir = dataset_dir / "processed"
    if dataset_dir.exists() and not overwrite:
        raise FileExistsError(f"{dataset_dir} already exists; pass --overwrite to rebuild")
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    raw = fetch_lapd_events(variant)
    if raw.empty:
        raise RuntimeError(f"No rows returned for {variant.dataset_id}")
    raw_path = raw_dir / "events_raw.parquet"
    raw.to_parquet(raw_path, index=False)

    boundary = fetch_lapd_boundary(variant.area_name, variant.area_code)
    boundary_path = processed_dir / "division_boundary.geojson"
    boundary.to_crs(CRS_WGS84).to_file(boundary_path, driver="GeoJSON")

    clean, clean_summary = clean_lapd_events(raw, variant)
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
    parser.add_argument("--list", action="store_true", help="List available variants.")
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
