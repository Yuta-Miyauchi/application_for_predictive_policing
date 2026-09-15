# results

Experiment outputs live here.

Each run should write to a unique result directory. The current retained result is a GIF animation; metric CSVs and static plots were removed while evaluation design is deferred.
Prediction and replay tables are retained so that a GIF can be reproduced or inspected without treating those tables as evaluation metrics.

## Intended Structure

```text
results/
  ETAS_full/
    <run_id>/
      animations/
        M6_pooled_lapd_weekly_online_etas_fixed_theta_weekly_forecast_heatmap.gif
      tables/
        animation_config.yml
        forecast_frames.csv
        forecast_cell_risk.parquet
        forecast_top_cells.csv
        forecast_group_risk.csv
        observed_events.parquet
```

Large generated artifacts should stay out of Git unless they are intentionally selected for publication.
