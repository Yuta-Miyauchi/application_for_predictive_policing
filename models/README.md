# models

Algorithm implementations live here.

Models should be reusable across datasets. A model should accept standardized processed data from `datas/` and should not assume LAPD, Chicago, or Philadelphia-specific column names internally.

## Implemented Model Family

- `adaptive_etas`

The current experiments still include baseline-like comparison conditions such as `M0_historical_count`, but those conditions are implemented inside the adaptive experiment runner so the evaluation order stays consistent.

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
- weekly 365-day refit
- constrained theta/omega likelihood with branching-ratio background shrinkage
- optional theta floor sensitivity run
