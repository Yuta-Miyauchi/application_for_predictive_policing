# models

Algorithm implementations live here.

Models should be reusable across datasets. A model should accept standardized processed data from `datas/` and should not assume LAPD, Chicago, or Philadelphia-specific column names internally.

## Implemented Model Family

- `adaptive_etas`
- `marked_adaptive_etas`
- `temporal_attention_transformer`

The current retained ETAS experiment is a pooled 21-area LAPD run with fixed public-scaffold ETAS parameters and weekly online state updates.
The current Transformer v1 experiment is a compact PyTorch encoder over weekly cell histories, with learned area embeddings and a Poisson output head.

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

## Transformer v1 Notes

`models/temporal_attention_transformer.py` defines `WeeklyCellTransformer`, a first PyTorch scaffold for moving toward STNPP-style marked point process models:

- weekly cell-count sequence tokens
- learned temporal positional embeddings
- LAPD area embeddings
- normalized spatial centroid features
- seasonal forecast-week features
- nonnegative next-week expected-count output trained with weighted Poisson loss

This v1 is intentionally small enough to run on CPU. It does not yet model event-level continuous time, spatial network edges, or full crime-type marks inside the transformer.
