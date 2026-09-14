# datas

Prepared datasets live here. Each dataset variant should be kept separate and should be usable independently by experiments.

## Intended Structure

```text
datas/
  lapd/
    README.md
    <dataset_id>/
      metadata.yml
      raw/
      interim/
      processed/
  chicago/
    README.md
    <dataset_id>/
      metadata.yml
      raw/
      interim/
      processed/
  philadelphia/
    README.md
    <dataset_id>/
      metadata.yml
      raw/
      interim/
      processed/
```

## Dataset ID Convention

Use explicit IDs that describe city, area, crime type, period, grid, and time horizon.

Examples:

- `lapd_foothill_burglary_2011_2013_grid150m_h24h`
- `lapd_north_hollywood_burglary_2011_2013_grid150m_h24h`
- `lapd_southwest_vehicle_stolen_2011_2013_grid150m_h24h`
- `chicago_districtXX_burglary_YYYY_YYYY_grid150m_h24h`
- `philadelphia_citywide_burglary_YYYY_YYYY_grid500ft_shift`

## Storage Policy

- Keep source-specific raw data under that dataset's own `raw/` directory.
- Keep cleaning outputs under `interim/`.
- Keep model-ready tables, grids, and event-cell assignments under `processed/`.
- Store source URL, download timestamp, hash, filtering rules, coordinate reference system, grid size, and exclusion counts in `metadata.yml`.
- Do not silently overwrite dataset variants. Create a new dataset ID when filters, periods, geography, grid size, or target crime type change.
