# Chicago Dataset Family

This folder is reserved for prepared variants of Chicago public crime data.

Chicago is planned as an external reproduction dataset after the LAPD pipeline is stable.

## Planned Use

- Test whether baseline and ETAS/SEPP-like models behave similarly outside Los Angeles.
- Start with one police district, one crime type, and a limited multi-year period.
- Expand only after data cleaning, grid assignment, and rolling evaluation are stable.

## Preparation Notes

- Keep Chicago variants separate from LAPD variants.
- Record whether locations are block-level approximations.
- Use chronological splits only.
- Store source URL, retrieval timestamp, filters, CRS, and exclusion counts in each dataset's metadata.
