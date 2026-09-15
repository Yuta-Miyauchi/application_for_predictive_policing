# models

Algorithm implementations live here.

Models should be reusable across datasets. A model should accept standardized processed data from `datas/` and should not assume LAPD, Chicago, or Philadelphia-specific column names internally.

## Implemented Model Family

- `adaptive_etas`
- `marked_adaptive_etas`

The current retained ETAS experiment is a pooled 21-area LAPD run with fixed public-scaffold ETAS parameters and weekly online state updates.

## Planned Model Families

- adaptive SEPP variants
- spatial-kernel ETAS
- report-delay-aware adaptive ETAS
- observation-feedback simulation models

## Feature-Based Models

- later HunchLab-like gradient boosting model

## Implementation Policy

- Keep model code independent from dataset download and cleaning code.
- Put model-specific parameters in experiment configs rather than hard-coding them into the implementation.
- Return comparable risk scores or expected counts per grid cell and forecast window.
- Save model metadata needed for interpretation and reproduction.

## Adaptive ETAS Notes

`models/adaptive_etas.py` provides utilities for event-by-event state updates and periodic rolling refits. The current adaptive experiment is a research scaffold:

- same-cell triggering only
- test-then-update evaluation
- rolling 365-day background history
- constrained theta/omega likelihood with branching-ratio background shrinkage
- optional theta floor sensitivity run

`models/marked_adaptive_etas.py` extends the same scaffold by preserving `crime_type` as a mark:

- same-cell triggering with per-crime trigger states
- crime-type-specific background rates
- a lightweight source-to-target crime-type transition matrix
- total cell risk for heatmaps plus per-crime risk tables for inspection

`experiments/run_lapd_full_etas.py` currently implements the large pooled replay directly for memory-efficient weekly table writing. It keeps `crime_group` risk totals for inspection while using total cell risk for the GIF heatmap.
