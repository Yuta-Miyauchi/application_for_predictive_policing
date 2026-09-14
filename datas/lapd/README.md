# LAPD Dataset Family

This folder is reserved for prepared variants of Los Angeles public crime data.

## Planned Variants

Initial LAPD variants should be prepared separately rather than merged:

- `lapd_foothill_burglary_2011_2013_grid150m_h24h`
- `lapd_north_hollywood_burglary_2011_2013_grid150m_h24h`
- `lapd_southwest_burglary_2011_2013_grid150m_h24h`
- `lapd_foothill_vehicle_stolen_2011_2013_grid150m_h24h`
- `lapd_north_hollywood_vehicle_stolen_2011_2013_grid150m_h24h`
- `lapd_southwest_vehicle_stolen_2011_2013_grid150m_h24h`

Later variants can change grid size, period, forecast horizon, or crime type.

## Preparation Notes

- Use occurrence datetime from `DATE OCC` and `TIME OCC`.
- Keep `Date Rptd` for later sensitivity analysis about reporting delay.
- Exclude missing coordinates, `(0, 0)` coordinates, out-of-bound points, and unresolved crime-type rows.
- Record all exclusions in each dataset's metadata.
- Prepare division-specific datasets first, then citywide or multi-division variants only when needed.
