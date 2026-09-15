"""Run a Transformer v1-style weekly LAPD forecast experiment.

The current v1 is a lightweight temporal-attention model that uses the full
LAPD legacy dataset but trains a compact Poisson head over fixed attention
sequence summaries. It is meant as a bridge toward a full neural transformer.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import yaml
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
from matplotlib.lines import Line2D
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

from experiments.animate_adaptive_forecast import risk_heatmap
from models.temporal_attention_transformer import (
    TransformerV1Config,
    WeeklyCellTransformer,
    weighted_poisson_loss,
)


DATASET_ID = "lapd_legacy_2010_2024_all_crimes_grid300m_h168h"
MODEL_ID = "T1_temporal_attention_transformer_v1"
DATASET_DIR = ROOT / "datas" / "lapd_full" / DATASET_ID

GROUP_COLORS = {
    "BURGLARY": "#7b3294",
    "FRAUD": "#a6611a",
    "OTHER": "#4d4d4d",
    "ROBBERY": "#018571",
    "SEX_OFFENSE": "#c51b7d",
    "THEFT": "#1f78b4",
    "VANDALISM": "#dfc27d",
    "VEHICLE": "#4daf4a",
    "VIOLENT": "#542788",
    "WEAPON": "#80cdc1",
}

RISK_CMAP = LinearSegmentedColormap.from_list(
    "predictive_risk_red",
    ["#fff7bc", "#fec44f", "#fb6a4a", "#de2d26", "#a50f15", "#4d0013"],
)


def load_full_dataset() -> tuple[pd.DataFrame, gpd.GeoDataFrame, dict]:
    metadata_path = DATASET_DIR / "metadata.yml"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Missing {metadata_path}. Run datas/lapd_full/prepare_lapd_legacy_full.py first."
        )
    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = yaml.safe_load(f)
    events = pd.read_parquet(DATASET_DIR / "processed" / "events.parquet")
    events["occurred_at"] = pd.to_datetime(events["occurred_at"])
    events["cell_id"] = events["cell_id"].astype(str)
    events["crime_group"] = events["crime_group"].fillna("OTHER").astype(str)
    events["area_id"] = events["area_id"].astype(str).str.zfill(2)
    grid = gpd.read_file(DATASET_DIR / "processed" / "grid.geojson")
    grid["cell_id"] = grid["cell_id"].astype(str)
    projected_crs = metadata.get("spatial", {}).get("projected_crs", "EPSG:3310")
    grid = grid.to_crs(projected_crs)
    if "area_m2" not in grid.columns:
        grid["area_m2"] = grid.geometry.area
    return events.sort_values("occurred_at").reset_index(drop=True), grid, metadata


def make_weekly_matrices(
    events: pd.DataFrame,
    grid: gpd.GeoDataFrame,
    frame_days: int,
) -> tuple[pd.DatetimeIndex, pd.Index, pd.Index, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cells = pd.Index(grid["cell_id"].astype(str), name="cell_id")
    groups = pd.Index(sorted(events["crime_group"].dropna().astype(str).unique()), name="crime_group")
    cell_pos = {cell: i for i, cell in enumerate(cells)}
    group_pos = {group: i for i, group in enumerate(groups)}

    start = events["occurred_at"].min().normalize()
    end = events["occurred_at"].max().normalize() + pd.Timedelta(days=frame_days)
    week_starts = pd.date_range(start, end, freq=f"{frame_days}D", inclusive="left")
    week_idx = np.searchsorted(
        week_starts.to_numpy(dtype="datetime64[ns]"),
        events["occurred_at"].to_numpy(dtype="datetime64[ns]"),
        side="right",
    ) - 1
    valid = (week_idx >= 0) & (week_idx < len(week_starts))
    event_cells = events.loc[valid, "cell_id"].map(cell_pos).to_numpy(dtype=np.int64)
    event_groups = events.loc[valid, "crime_group"].map(group_pos).to_numpy(dtype=np.int64)
    week_idx = week_idx[valid].astype(np.int64)

    counts = np.zeros((len(week_starts), len(cells)), dtype=np.float32)
    group_counts = np.zeros((len(week_starts), len(groups)), dtype=np.float32)
    np.add.at(counts, (week_idx, event_cells), 1.0)
    np.add.at(group_counts, (week_idx, event_groups), 1.0)

    cell_area = (
        events.groupby(["cell_id", "area_id"], observed=True)
        .size()
        .rename("n")
        .reset_index()
        .sort_values(["cell_id", "n"], ascending=[True, False])
        .drop_duplicates("cell_id")
        .set_index("cell_id")["area_id"]
    )
    area_ids = pd.Index(sorted(events["area_id"].dropna().astype(str).str.zfill(2).unique()))
    area_pos = {area: i for i, area in enumerate(area_ids)}
    fallback_area = area_ids[0]
    area_index_by_cell = np.array(
        [area_pos.get(cell_area.get(cell, fallback_area), 0) for cell in cells],
        dtype=np.int64,
    )
    area_counts = np.zeros((len(week_starts), len(area_ids)), dtype=np.float32)
    for area_idx in range(len(area_ids)):
        area_counts[:, area_idx] = counts[:, area_index_by_cell == area_idx].sum(axis=1)
    area_cell_counts = np.bincount(area_index_by_cell, minlength=len(area_ids)).astype(np.float32)
    area_cell_counts = np.maximum(area_cell_counts, 1.0)
    return week_starts, cells, groups, counts, group_counts, area_counts, area_index_by_cell


def trailing_mean(values: np.ndarray, window: int) -> np.ndarray:
    cumsum = np.vstack([np.zeros((1, values.shape[1]), dtype=np.float32), np.cumsum(values, axis=0)])
    out = np.zeros_like(values, dtype=np.float32)
    for t in range(values.shape[0]):
        start = max(0, t - window)
        width = max(t - start, 1)
        out[t] = (cumsum[t] - cumsum[start]) / width
    return out


def normalized_centroids(grid: gpd.GeoDataFrame) -> tuple[np.ndarray, np.ndarray]:
    centroids = grid.geometry.centroid
    x = centroids.x.to_numpy(dtype=np.float32)
    y = centroids.y.to_numpy(dtype=np.float32)
    x_norm = (x - x.mean()) / max(float(x.std()), 1e-6)
    y_norm = (y - y.mean()) / max(float(y.std()), 1e-6)
    return x_norm.astype(np.float32), y_norm.astype(np.float32)


def build_history_tensor(
    counts: np.ndarray,
    week_indices: np.ndarray,
    cell_indices: np.ndarray,
    context_weeks: int,
) -> np.ndarray:
    offsets = np.arange(context_weeks, 0, -1, dtype=np.int64)
    return counts[week_indices[:, None] - offsets[None, :], cell_indices[:, None]].astype(np.float32)


def build_week_features(week_indices: np.ndarray) -> np.ndarray:
    phase = 2.0 * np.pi * (week_indices.astype(np.float32) % 52.1775) / 52.1775
    return np.column_stack([np.sin(phase), np.cos(phase)]).astype(np.float32)


def sample_training_indices(
    counts: np.ndarray,
    context_weeks: int,
    train_end_idx: int,
    max_train_rows: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    target = counts[context_weeks:train_end_idx]
    positive = np.argwhere(target > 0)
    n_total = target.size
    n_positive_total = len(positive)
    n_negative_total = n_total - n_positive_total
    n_pos = min(n_positive_total, max_train_rows // 2)
    n_neg = max_train_rows - n_pos
    if n_pos < n_positive_total:
        positive = positive[rng.choice(n_positive_total, size=n_pos, replace=False)]
    else:
        n_pos = len(positive)

    neg_weeks = []
    neg_cells = []
    while len(neg_weeks) < n_neg:
        batch = min((n_neg - len(neg_weeks)) * 2, max_train_rows)
        weeks = rng.integers(0, target.shape[0], size=batch)
        cells = rng.integers(0, target.shape[1], size=batch)
        keep = target[weeks, cells] == 0
        neg_weeks.extend(weeks[keep].tolist())
        neg_cells.extend(cells[keep].tolist())
    neg_weeks = np.asarray(neg_weeks[:n_neg], dtype=np.int64)
    neg_cells = np.asarray(neg_cells[:n_neg], dtype=np.int64)

    pos_weeks = positive[:, 0].astype(np.int64)
    pos_cells = positive[:, 1].astype(np.int64)
    week_idx = np.concatenate([pos_weeks, neg_weeks]) + context_weeks
    cell_idx = np.concatenate([pos_cells, neg_cells])
    weights = np.concatenate(
        [
            np.full(n_pos, n_positive_total / max(n_pos, 1), dtype=np.float32),
            np.full(n_neg, n_negative_total / max(n_neg, 1), dtype=np.float32),
        ]
    )
    order = rng.permutation(len(week_idx))
    return week_idx[order], cell_idx[order], weights[order]


def train_transformer_v1(
    counts: np.ndarray,
    area_counts: np.ndarray,
    area_index_by_cell: np.ndarray,
    x_norm: np.ndarray,
    y_norm: np.ndarray,
    train_end_idx: int,
    args: argparse.Namespace,
) -> tuple[WeeklyCellTransformer, dict]:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    config = TransformerV1Config(
        context_weeks=args.context_weeks,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
    )
    model = WeeklyCellTransformer(n_areas=area_counts.shape[1], config=config).to(device)
    week_idx, cell_idx, weights = sample_training_indices(
        counts=counts,
        context_weeks=args.context_weeks,
        train_end_idx=train_end_idx,
        max_train_rows=args.max_train_rows,
        seed=args.seed,
    )
    histories = build_history_tensor(counts, week_idx, cell_idx, args.context_weeks)
    area_idx = area_index_by_cell[cell_idx].astype(np.int64)
    static = np.column_stack([x_norm[cell_idx], y_norm[cell_idx]]).astype(np.float32)
    week_features = build_week_features(week_idx)
    y_train = counts[week_idx, cell_idx].astype(np.float32)
    weights = (weights / max(float(weights.mean()), 1e-6)).astype(np.float32)

    dataset = TensorDataset(
        torch.from_numpy(histories),
        torch.from_numpy(area_idx),
        torch.from_numpy(static),
        torch.from_numpy(week_features),
        torch.from_numpy(y_train),
        torch.from_numpy(weights),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    epoch_losses = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        seen = 0
        for history_b, area_b, static_b, week_b, y_b, w_b in loader:
            history_b = history_b.to(device)
            area_b = area_b.to(device)
            static_b = static_b.to(device)
            week_b = week_b.to(device)
            y_b = y_b.to(device)
            w_b = w_b.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(history_b, area_b, static_b, week_b)
            loss = weighted_poisson_loss(pred, y_b, w_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            running += float(loss.detach().cpu()) * len(y_b)
            seen += len(y_b)
        epoch_loss = running / max(seen, 1)
        epoch_losses.append(epoch_loss)
        print(f"Epoch {epoch}/{args.epochs}: weighted_poisson_loss={epoch_loss:.5f}")

    summary = {
        "train_rows": int(len(y_train)),
        "train_positive_rows": int((y_train > 0).sum()),
        "train_target_mean_unweighted": float(y_train.mean()),
        "train_target_mean_weighted": float(np.average(y_train, weights=weights)),
        "epoch_losses": [float(v) for v in epoch_losses],
        "device": str(device),
        "torch_version": str(torch.__version__),
        "config": {
            "context_weeks": config.context_weeks,
            "d_model": config.d_model,
            "n_heads": config.n_heads,
            "n_layers": config.n_layers,
            "dim_feedforward": config.dim_feedforward,
            "dropout": config.dropout,
        },
    }
    return model, summary


@torch.no_grad()
def predict_transformer_week(
    model: WeeklyCellTransformer,
    counts: np.ndarray,
    week_idx: int,
    area_index_by_cell: np.ndarray,
    x_norm: np.ndarray,
    y_norm: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    device = next(model.parameters()).device
    model.eval()
    n_cells = counts.shape[1]
    out = np.zeros(n_cells, dtype=np.float32)
    all_cells = np.arange(n_cells, dtype=np.int64)
    week_features_all = build_week_features(np.full(n_cells, week_idx, dtype=np.int64))
    for start in range(0, n_cells, args.inference_batch_size):
        end = min(start + args.inference_batch_size, n_cells)
        cells = all_cells[start:end]
        weeks = np.full(len(cells), week_idx, dtype=np.int64)
        histories = build_history_tensor(counts, weeks, cells, args.context_weeks)
        static = np.column_stack([x_norm[cells], y_norm[cells]]).astype(np.float32)
        pred = model(
            torch.from_numpy(histories).to(device),
            torch.from_numpy(area_index_by_cell[cells].astype(np.int64)).to(device),
            torch.from_numpy(static).to(device),
            torch.from_numpy(week_features_all[start:end]).to(device),
        )
        out[start:end] = pred.detach().cpu().numpy().astype(np.float32)
    return out


def forecast_weekly(
    model: WeeklyCellTransformer,
    counts: np.ndarray,
    group_counts: np.ndarray,
    area_index_by_cell: np.ndarray,
    x_norm: np.ndarray,
    y_norm: np.ndarray,
    week_starts: pd.DatetimeIndex,
    cells: pd.Index,
    groups: pd.Index,
    forecast_start_idx: int,
    forecast_end_idx: int,
    args: argparse.Namespace,
) -> tuple[list[dict], np.ndarray]:
    group_context = trailing_mean(group_counts, args.group_window_weeks)
    records = []
    risk_matrix = np.zeros((forecast_end_idx - forecast_start_idx, len(cells)), dtype=np.float32)
    for frame_index, week_idx in enumerate(range(forecast_start_idx, forecast_end_idx)):
        risk = predict_transformer_week(
            model=model,
            counts=counts,
            week_idx=week_idx,
            area_index_by_cell=area_index_by_cell,
            x_norm=x_norm,
            y_norm=y_norm,
            args=args,
        )
        risk_matrix[frame_index] = risk
        group_base = group_context[week_idx].astype(np.float32)
        if float(group_base.sum()) <= 0:
            group_share = np.full(len(groups), 1.0 / len(groups), dtype=np.float32)
        else:
            group_share = group_base / group_base.sum()
        group_risk = group_share * float(risk.sum())
        records.append(
            {
                "frame_index": frame_index,
                "week_index": week_idx,
                "forecast_date": week_starts[week_idx],
                "display_time": week_starts[week_idx] + pd.Timedelta(days=args.frame_days),
                "actual_events_in_window": int(counts[week_idx].sum()),
                "predicted_events": float(risk.sum()),
                "group_risk": group_risk,
            }
        )
        if (frame_index + 1) % 25 == 0 or week_idx + 1 == forecast_end_idx:
            print(f"Forecasted {frame_index + 1}/{forecast_end_idx - forecast_start_idx} weeks")
    return records, risk_matrix


def save_tables(
    result_dir: Path,
    records: list[dict],
    cells: pd.Index,
    groups: pd.Index,
    risk_matrix: np.ndarray,
    events: pd.DataFrame,
    train_summary: dict,
    model: WeeklyCellTransformer,
    args: argparse.Namespace,
    gif_path: Path,
) -> None:
    tables_dir = result_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "dataset_id": DATASET_ID,
        "model_id": MODEL_ID,
        "model_note": "Transformer v1-style fixed temporal attention encoder with Poisson output head",
        "train_end": args.train_end,
        "forecast_start": records[0]["forecast_date"].date().isoformat(),
        "forecast_end": records[-1]["display_time"].date().isoformat(),
        "context_weeks": args.context_weeks,
        "frame_days": args.frame_days,
        "max_train_rows": args.max_train_rows,
        "d_model": args.d_model,
        "n_heads": args.n_heads,
        "n_layers": args.n_layers,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "top_k": args.top_k,
        "fade_weeks": args.fade_weeks,
        "smooth_sigma": args.smooth_sigma,
        "vmax_quantile": args.vmax_quantile,
        "heatmap_gamma": args.heatmap_gamma,
        "duration_ms": args.duration_ms,
        "dpi": args.dpi,
        "gif": str(gif_path.relative_to(result_dir)),
        "train_summary": train_summary,
    }
    with (tables_dir / "animation_config.yml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)
    model_dir = result_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_id": MODEL_ID,
            "state_dict": model.state_dict(),
            "train_summary": train_summary,
            "config": config,
        },
        model_dir / "model_state.pt",
    )
    pd.DataFrame(
        [
            {
                "frame_index": r["frame_index"],
                "forecast_date": r["forecast_date"].date().isoformat(),
                "display_time": r["display_time"].isoformat(),
                "actual_events_in_window": r["actual_events_in_window"],
                "predicted_events": r["predicted_events"],
            }
            for r in records
        ]
    ).to_csv(tables_dir / "forecast_frames.csv", index=False)
    with (tables_dir / "training_summary.yml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(train_summary, f, sort_keys=False, allow_unicode=True)

    top_rows = []
    for r in records:
        risk = risk_matrix[r["frame_index"]]
        top_idx = np.argsort(risk)[-args.top_k :][::-1]
        for rank, idx in enumerate(top_idx, start=1):
            top_rows.append(
                {
                    "frame_index": r["frame_index"],
                    "forecast_date": r["forecast_date"].date().isoformat(),
                    "rank": rank,
                    "cell_id": cells[idx],
                    "risk": float(risk[idx]),
                }
            )
    pd.DataFrame(top_rows).to_csv(tables_dir / "forecast_top_cells.csv", index=False)

    group_rows = []
    for r in records:
        for i, group in enumerate(groups):
            group_rows.append(
                {
                    "frame_index": r["frame_index"],
                    "forecast_date": r["forecast_date"].date().isoformat(),
                    "crime_group": group,
                    "risk": float(r["group_risk"][i]),
                }
            )
    pd.DataFrame(group_rows).to_csv(tables_dir / "forecast_group_risk.csv", index=False)

    writer = None
    try:
        for r in records:
            frame_idx = r["frame_index"]
            df = pd.DataFrame(
                {
                    "frame_index": frame_idx,
                    "forecast_date": r["forecast_date"].date().isoformat(),
                    "cell_id": cells.to_numpy(),
                    "risk": risk_matrix[frame_idx],
                }
            )
            table = pa.Table.from_pandas(df, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(tables_dir / "forecast_cell_risk.parquet", table.schema)
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()

    keep_cols = [
        "event_id",
        "occurred_at",
        "crime_code",
        "crime_type",
        "crime_group",
        "area_id",
        "area_name",
        "lat",
        "lon",
        "x",
        "y",
        "cell_id",
        "source_legacy_dataset",
    ]
    keep_cols = [c for c in keep_cols if c in events.columns]
    start = records[0]["forecast_date"]
    end = records[-1]["display_time"]
    observed = events[(events["occurred_at"] >= start) & (events["occurred_at"] < end)]
    observed[keep_cols].to_parquet(tables_dir / "observed_events.parquet", index=False)


def render_frame(
    grid: gpd.GeoDataFrame,
    events: pd.DataFrame,
    times: np.ndarray,
    record: dict,
    risk: np.ndarray,
    vmax: float,
    args: argparse.Namespace,
) -> Image.Image:
    risk_series = pd.Series(risk, index=grid["cell_id"].astype(str), dtype=float)
    heatmap, extent = risk_heatmap(grid, risk_series, args.smooth_sigma)
    heatmap = np.ma.masked_invalid(np.clip(heatmap, 0, vmax))

    fig, ax = plt.subplots(figsize=(9.4, 9.6), dpi=args.dpi, constrained_layout=True)
    ax.imshow(
        heatmap,
        extent=extent,
        origin="lower",
        cmap=RISK_CMAP,
        norm=PowerNorm(gamma=args.heatmap_gamma, vmin=0, vmax=vmax),
        interpolation="bilinear",
        alpha=0.98,
        zorder=1,
    )
    boundary = gpd.GeoSeries([grid.geometry.union_all().boundary], crs=grid.crs)
    boundary.plot(ax=ax, color="#4f4f4f", linewidth=0.45, alpha=0.8, zorder=3)

    display_time = pd.Timestamp(record["display_time"])
    trail_start = display_time - pd.Timedelta(days=7 * args.fade_weeks)
    start_idx = int(np.searchsorted(times, np.datetime64(trail_start.to_datetime64()), side="left"))
    end_idx = int(np.searchsorted(times, np.datetime64(display_time.to_datetime64()), side="left"))
    trail = events.iloc[start_idx:end_idx].copy()
    if not trail.empty:
        centroids = grid[["cell_id", "geometry"]].copy()
        centroids["cx"] = centroids.geometry.centroid.x
        centroids["cy"] = centroids.geometry.centroid.y
        grouped = (
            trail.groupby(["cell_id", "crime_group"], observed=True)
            .agg(count=("event_id", "size"), latest=("occurred_at", "max"))
            .reset_index()
            .merge(centroids[["cell_id", "cx", "cy"]], on="cell_id", how="left")
            .dropna(subset=["cx", "cy"])
        )
        ages = (display_time - grouped["latest"]).dt.total_seconds() / 86400.0
        fade = np.clip(1.0 - ages / max(7 * args.fade_weeks, 1), 0.0, 1.0)
        pop = np.exp(-ages / max(args.pop_days, 1e-6))
        sizes = 10 + 85 * np.sqrt(grouped["count"].to_numpy(dtype=float)) * (0.35 + pop) * fade
        for crime_group, subset in grouped.groupby("crime_group", observed=True):
            idx = subset.index.to_numpy()
            color = GROUP_COLORS.get(str(crime_group), "#4d4d4d")
            ax.scatter(
                subset["cx"],
                subset["cy"],
                s=sizes[idx],
                facecolors="none",
                edgecolors=color,
                linewidths=0.85,
                alpha=0.35,
                zorder=4,
            )

    minx, miny, maxx, maxy = grid.total_bounds
    pad = max(maxx - minx, maxy - miny) * 0.025
    ax.set_xlim(minx - pad, maxx + pad)
    ax.set_ylim(miny - pad, maxy + pad)
    ax.set_axis_off()
    ax.set_aspect("equal")
    ax.set_title(
        f"LAPD Transformer v1 | week of {record['forecast_date'].date().isoformat()}",
        fontsize=11,
    )
    ax.text(
        0.01,
        0.018,
        "Temporal-attention Transformer v1. Red heatmap: next-week risk. Rings: observed crime groups.",
        transform=ax.transAxes,
        va="bottom",
        ha="left",
        fontsize=7.5,
        bbox={"facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.82},
    )
    handles = [
        Line2D([0], [0], marker="o", color=color, markerfacecolor="none", label=group, linewidth=0)
        for group, color in GROUP_COLORS.items()
    ]
    legend = ax.legend(handles=handles, loc="upper right", fontsize=6.5, framealpha=0.84)
    legend.get_frame().set_edgecolor("#cccccc")
    fig.canvas.draw()
    image = Image.fromarray(np.asarray(fig.canvas.buffer_rgba()))
    plt.close(fig)
    return image.convert("P", palette=Image.Palette.ADAPTIVE)


def save_animation(
    result_dir: Path,
    grid: gpd.GeoDataFrame,
    events: pd.DataFrame,
    records: list[dict],
    risk_matrix: np.ndarray,
    args: argparse.Namespace,
) -> Path:
    out_path = result_dir / "animations" / f"{MODEL_ID}_weekly_forecast_heatmap.gif"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    risk_values = risk_matrix.ravel()
    vmax = float(np.quantile(risk_values[np.isfinite(risk_values)], args.vmax_quantile))
    vmax = max(vmax, 1e-9)
    times = events["occurred_at"].to_numpy(dtype="datetime64[ns]")
    frames = []
    for i, record in enumerate(records, start=1):
        frames.append(
            render_frame(
                grid=grid,
                events=events,
                times=times,
                record=record,
                risk=risk_matrix[record["frame_index"]],
                vmax=vmax,
                args=args,
            )
        )
        if i % 25 == 0 or i == len(records):
            print(f"Rendered {i}/{len(records)} frames")
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=args.duration_ms,
        loop=0,
        optimize=True,
    )
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="lapd_legacy_2020_2024_transformer_v1_weekly")
    parser.add_argument("--train-end", default="2020-01-03")
    parser.add_argument("--forecast-start")
    parser.add_argument("--forecast-end")
    parser.add_argument("--context-weeks", type=int, default=52)
    parser.add_argument("--frame-days", type=int, default=7)
    parser.add_argument("--group-window-weeks", type=int, default=13)
    parser.add_argument("--max-train-rows", type=int, default=250_000)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=1)
    parser.add_argument("--dim-feedforward", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--inference-batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--device")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--fade-weeks", type=int, default=4)
    parser.add_argument("--pop-days", type=float, default=2.5)
    parser.add_argument("--smooth-sigma", type=float, default=1.8)
    parser.add_argument("--vmax-quantile", type=float, default=0.985)
    parser.add_argument("--heatmap-gamma", type=float, default=0.4)
    parser.add_argument("--duration-ms", type=int, default=95)
    parser.add_argument("--dpi", type=int, default=85)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-gif", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_dir = ROOT / "results" / "Transformer_v1" / args.run_id
    events, grid, _ = load_full_dataset()
    week_starts, cells, groups, counts, group_counts, area_counts, area_index_by_cell = make_weekly_matrices(
        events=events,
        grid=grid,
        frame_days=args.frame_days,
    )
    x_norm, y_norm = normalized_centroids(grid)
    train_end_idx = int(np.searchsorted(week_starts, pd.Timestamp(args.train_end), side="left"))
    forecast_start = pd.Timestamp(args.forecast_start) if args.forecast_start else pd.Timestamp(args.train_end)
    forecast_end = (
        pd.Timestamp(args.forecast_end)
        if args.forecast_end
        else events["occurred_at"].max().normalize() + pd.Timedelta(days=args.frame_days)
    )
    forecast_start_idx = int(np.searchsorted(week_starts, forecast_start, side="left"))
    forecast_end_idx = int(np.searchsorted(week_starts, forecast_end, side="left"))
    if train_end_idx <= args.context_weeks:
        raise ValueError("train_end leaves too little history for context_weeks")
    if forecast_start_idx < args.context_weeks:
        raise ValueError("forecast_start leaves too little history for context_weeks")

    model, train_summary = train_transformer_v1(
        counts=counts,
        area_counts=area_counts,
        area_index_by_cell=area_index_by_cell,
        x_norm=x_norm,
        y_norm=y_norm,
        train_end_idx=train_end_idx,
        args=args,
    )
    records, risk_matrix = forecast_weekly(
        model=model,
        counts=counts,
        group_counts=group_counts,
        area_index_by_cell=area_index_by_cell,
        x_norm=x_norm,
        y_norm=y_norm,
        week_starts=week_starts,
        cells=cells,
        groups=groups,
        forecast_start_idx=forecast_start_idx,
        forecast_end_idx=forecast_end_idx,
        args=args,
    )
    if args.skip_gif:
        gif_path = result_dir / "animations" / f"{MODEL_ID}_weekly_forecast_heatmap.gif"
    else:
        gif_path = save_animation(result_dir, grid, events, records, risk_matrix, args)
    save_tables(
        result_dir=result_dir,
        records=records,
        cells=cells,
        groups=groups,
        risk_matrix=risk_matrix,
        events=events,
        train_summary=train_summary,
        model=model,
        args=args,
        gif_path=gif_path,
    )
    print(f"Wrote {result_dir}")


if __name__ == "__main__":
    main()
