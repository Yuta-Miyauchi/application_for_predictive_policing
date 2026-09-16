"""Run a compact STNPP-GAT-style LAPD forecast animation.

The experiment replaces the earlier weekly Transformer baseline with a marked
point-process model closer to the STNPP paper's core mechanism: a graph
attention network learns crime-area mark interactions, and the forecast uses a
marked self-exciting intensity on 150m cells.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import yaml

from our_experiment.common.lapd_target_common import (
    TARGET_CRIME_ORDER,
    TargetDataset,
    load_target_dataset,
    observed_window_counts,
    save_forecast_animation,
    write_forecast_tables,
)
from our_model.vol1.stnpp_gat import MarkGraphAttention, STNPPGATConfig, transition_kl_loss


MODEL_ID = "T2_stnpp_gat_marked_point_process"


def build_mark_indices(
    dataset: TargetDataset,
) -> tuple[pd.Index, pd.Index, dict[str, int], dict[str, int], np.ndarray]:
    crimes = pd.Index(TARGET_CRIME_ORDER, name="target_crime")
    areas = pd.Index(sorted(dataset.grid["area_id"].astype(str).unique()), name="area_id")
    crime_pos = {crime: i for i, crime in enumerate(crimes)}
    area_pos = {area: i for i, area in enumerate(areas)}
    cell_area_index = dataset.grid["area_id"].astype(str).map(area_pos).to_numpy(dtype=np.int64)
    return crimes, areas, crime_pos, area_pos, cell_area_index


def make_mark_features(
    crimes: pd.Index,
    areas: pd.Index,
    dataset: TargetDataset,
) -> np.ndarray:
    area_centroids = (
        dataset.boundaries.assign(cx=dataset.boundaries.geometry.centroid.x)
        .assign(cy=dataset.boundaries.geometry.centroid.y)
        .set_index("area_id")[["cx", "cy"]]
        .reindex(areas)
    )
    xy = area_centroids.to_numpy(dtype=np.float32)
    xy = (xy - xy.mean(axis=0, keepdims=True)) / np.maximum(xy.std(axis=0, keepdims=True), 1e-6)
    rows = []
    for area_idx in range(len(areas)):
        for crime_idx in range(len(crimes)):
            crime_onehot = np.zeros(len(crimes), dtype=np.float32)
            crime_onehot[crime_idx] = 1.0
            area_onehot = np.zeros(len(areas), dtype=np.float32)
            area_onehot[area_idx] = 1.0
            rows.append(np.concatenate([crime_onehot, area_onehot, xy[area_idx]], dtype=np.float32))
    return np.vstack(rows).astype(np.float32)


def event_arrays(
    dataset: TargetDataset,
    crime_pos: dict[str, int],
    area_pos: dict[str, int],
) -> pd.DataFrame:
    events = dataset.events.copy()
    events["crime_idx"] = events["target_crime"].astype(str).map(crime_pos).astype(int)
    events["area_idx"] = events["area_id"].astype(str).map(area_pos).astype(int)
    events["mark_idx"] = events["area_idx"] * len(crime_pos) + events["crime_idx"]
    return events.sort_values("occurred_at").reset_index(drop=True)


def empirical_transition_matrix(
    events: pd.DataFrame,
    n_marks: int,
    omega: float,
    lookback_days: float,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    smoothing: float,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Estimate weighted same-cell source->target mark transitions."""

    train = events[
        (events["occurred_at"] >= train_start) & (events["occurred_at"] < train_end)
    ].copy()
    mat = np.full((n_marks, n_marks), smoothing, dtype=np.float64)
    source_weight = np.full(n_marks, smoothing * n_marks, dtype=np.float64)
    pair_count = 0
    for _, group in train.groupby("cell_id", sort=False):
        group = group.sort_values("occurred_at")
        times = group["occurred_at"].to_numpy(dtype="datetime64[ns]")
        marks = group["mark_idx"].to_numpy(dtype=np.int64)
        left = 0
        for right in range(len(group)):
            while left < right:
                age_left = (times[right] - times[left]) / np.timedelta64(1, "D")
                if float(age_left) <= lookback_days:
                    break
                left += 1
            if right <= left:
                continue
            ages = (times[right] - times[left:right]) / np.timedelta64(1, "D")
            mask = ages > 0
            if not mask.any():
                continue
            src = marks[left:right][mask]
            weights = np.exp(-omega * ages[mask].astype(float))
            target = int(marks[right])
            np.add.at(mat[target], src, weights)
            np.add.at(source_weight, src, weights)
            pair_count += int(mask.sum())
    col_sums = np.maximum(mat.sum(axis=0, keepdims=True), 1e-12)
    empirical = mat / col_sums
    summary = {
        "train_events": int(len(train)),
        "weighted_pair_count": int(pair_count),
        "lookback_days": float(lookback_days),
        "transition_smoothing": float(smoothing),
    }
    return empirical.astype(np.float32), source_weight.astype(np.float32), summary


def train_mark_gat(
    mark_features: np.ndarray,
    empirical: np.ndarray,
    source_weight: np.ndarray,
    config: STNPPGATConfig,
    args: argparse.Namespace,
) -> tuple[MarkGraphAttention, dict, np.ndarray]:
    torch.manual_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = MarkGraphAttention(config).to(device)
    features_t = torch.from_numpy(mark_features).to(device)
    empirical_t = torch.from_numpy(empirical).to(device)
    weight_t = torch.from_numpy(source_weight).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
    losses = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        pred = model(features_t)
        loss = transition_kl_loss(pred, empirical_t, weight_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            print(f"STNPP-GAT epoch {epoch}/{args.epochs}: transition_kl={losses[-1]:.6f}")
    model.eval()
    with torch.no_grad():
        transition = model(features_t).detach().cpu().numpy().astype(np.float32)
    summary = {
        "device": str(device),
        "torch_version": str(torch.__version__),
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "attention_heads": config.attention_heads,
        "hidden_dim": config.hidden_dim,
        "dropout": config.dropout,
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
    }
    return model, summary, transition


def forecast_risk_for_cutoff(
    dataset: TargetDataset,
    events: pd.DataFrame,
    transition: np.ndarray,
    cell_pos: dict[str, int],
    crime_pos: dict[str, int],
    cell_area_index: np.ndarray,
    cutoff: pd.Timestamp,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray]:
    n_cells = len(dataset.grid)
    n_crimes = len(crime_pos)
    horizon_days = args.horizon_hours / 24.0
    history_start = cutoff - pd.Timedelta(days=args.history_days)
    history = events[
        (events["occurred_at"] >= history_start) & (events["occurred_at"] < cutoff)
    ].copy()
    background = np.full((n_cells, n_crimes), args.background_alpha / args.history_days, dtype=np.float32)
    if not history.empty:
        grouped = history.groupby(["cell_id", "crime_idx"], observed=True).size().rename("count").reset_index()
        for row in grouped.itertuples(index=False):
            pos = cell_pos.get(str(row.cell_id))
            if pos is not None:
                background[pos, int(row.crime_idx)] += float(row.count) / args.history_days
    background *= horizon_days

    trigger_start = cutoff - pd.Timedelta(days=args.lookback_days)
    trigger_events = events[
        (events["occurred_at"] >= trigger_start) & (events["occurred_at"] < cutoff)
    ].copy()
    z = np.zeros((n_cells, n_crimes), dtype=np.float32)
    if not trigger_events.empty:
        ages = (cutoff - trigger_events["occurred_at"]).dt.total_seconds().to_numpy(dtype=float) / 86400.0
        weights = np.exp(-args.omega * ages).astype(np.float32)
        for cell_id, crime_idx, weight in zip(
            trigger_events["cell_id"].astype(str),
            trigger_events["crime_idx"].astype(int),
            weights,
            strict=False,
        ):
            pos = cell_pos.get(cell_id)
            if pos is not None:
                z[pos, int(crime_idx)] += float(weight)

    trigger_integral = 1.0 - float(np.exp(-args.omega * horizon_days))
    risk_by_crime = np.array(background, copy=True)
    for cell_idx in range(n_cells):
        area_idx = int(cell_area_index[cell_idx])
        target_marks = area_idx * n_crimes + np.arange(n_crimes)
        source_marks = target_marks
        local_transition = transition[np.ix_(target_marks, source_marks)]
        risk_by_crime[cell_idx] += (
            args.theta * trigger_integral * (local_transition @ z[cell_idx])
        ).astype(np.float32)
    return risk_by_crime.sum(axis=1).astype(np.float32), risk_by_crime


def replay_stnpp_gat(
    dataset: TargetDataset,
    events: pd.DataFrame,
    transition: np.ndarray,
    crime_pos: dict[str, int],
    cell_area_index: np.ndarray,
    args: argparse.Namespace,
) -> tuple[pd.Index, list[dict], np.ndarray]:
    cells = pd.Index(dataset.grid["cell_id"].astype(str), name="cell_id")
    cell_pos = {cell: i for i, cell in enumerate(cells)}
    cutoffs = pd.date_range(pd.Timestamp(args.start), pd.Timestamp(args.end), freq=f"{args.frame_days}D", inclusive="left")
    risk_matrix = np.zeros((len(cutoffs), len(cells)), dtype=np.float32)
    records = []
    for frame_index, cutoff in enumerate(cutoffs):
        actual_end = cutoff + pd.Timedelta(hours=args.horizon_hours)
        risk, _ = forecast_risk_for_cutoff(
            dataset=dataset,
            events=events,
            transition=transition,
            cell_pos=cell_pos,
            crime_pos=crime_pos,
            cell_area_index=cell_area_index,
            cutoff=cutoff,
            args=args,
        )
        risk_matrix[frame_index] = risk
        records.append(
            {
                "frame_index": frame_index,
                "forecast_date": cutoff,
                "display_time": actual_end,
                "actual_events_in_window": observed_window_counts(events, cutoff, actual_end),
                "predicted_events": float(risk.sum()),
                "theta": float(args.theta),
                "omega": float(args.omega),
            }
        )
        if (frame_index + 1) % 10 == 0 or frame_index + 1 == len(cutoffs):
            print(f"STNPP-GAT forecasted {frame_index + 1}/{len(cutoffs)} frames")
    return cells, records, risk_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="lapd_legacy_2020_2024_stnpp_gat_target_150m")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2025-01-01")
    parser.add_argument("--train-start", default="2010-01-01")
    parser.add_argument("--train-end", default="2020-01-01")
    parser.add_argument("--history-days", type=int, default=365)
    parser.add_argument("--horizon-hours", type=int, default=24)
    parser.add_argument("--frame-days", type=int, default=14)
    parser.add_argument("--lookback-days", type=float, default=30.0)
    parser.add_argument("--omega", type=float, default=1.0 / 14.0)
    parser.add_argument("--theta", type=float, default=0.35)
    parser.add_argument("--background-alpha", type=float, default=1e-3)
    parser.add_argument("--transition-smoothing", type=float, default=0.25)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--attention-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--learning-rate", type=float, default=1.0)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--log-every", type=int, default=150)
    parser.add_argument("--device")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--fade-days", type=int, default=28)
    parser.add_argument("--pop-days", type=float, default=3.0)
    parser.add_argument("--smooth-sigma", type=float, default=2.1)
    parser.add_argument("--vmax-quantile", type=float, default=0.985)
    parser.add_argument("--heatmap-gamma", type=float, default=0.42)
    parser.add_argument("--duration-ms", type=int, default=105)
    parser.add_argument("--dpi", type=int, default=82)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite-derived-data", action="store_true")
    parser.add_argument("--skip-gif", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_dir = ROOT / "our_experiment" / "vol1" / "results"
    dataset = load_target_dataset(overwrite=args.overwrite_derived_data, cell_size_m=150)
    crimes, areas, crime_pos, area_pos, cell_area_index = build_mark_indices(dataset)
    events = event_arrays(dataset, crime_pos, area_pos)
    mark_features = make_mark_features(crimes, areas, dataset)
    config = STNPPGATConfig(
        n_crimes=len(crimes),
        n_areas=len(areas),
        hidden_dim=args.hidden_dim,
        attention_heads=args.attention_heads,
        dropout=args.dropout,
    )
    empirical, source_weight, transition_summary = empirical_transition_matrix(
        events=events,
        n_marks=config.n_marks,
        omega=args.omega,
        lookback_days=args.lookback_days,
        train_start=pd.Timestamp(args.train_start),
        train_end=pd.Timestamp(args.train_end),
        smoothing=args.transition_smoothing,
    )
    model, train_summary, transition = train_mark_gat(
        mark_features=mark_features,
        empirical=empirical,
        source_weight=source_weight,
        config=config,
        args=args,
    )
    cells, records, risk_matrix = replay_stnpp_gat(
        dataset=dataset,
        events=events,
        transition=transition,
        crime_pos=crime_pos,
        cell_area_index=cell_area_index,
        args=args,
    )
    gif_path = result_dir / "animations" / f"{MODEL_ID}_24h_forecast_heatmap.gif"
    if not args.skip_gif:
        gif_path = save_forecast_animation(
            out_path=gif_path,
            dataset=dataset,
            records=records,
            risk_matrix=risk_matrix,
            args=args,
            title="LAPD STNPP-GAT",
            note=(
                "Red heatmap: predicted 24h marked point-process risk. "
                "Colored rings: observed target-crime events."
            ),
        )
    write_forecast_tables(
        result_dir=result_dir,
        model_id=MODEL_ID,
        dataset=dataset,
        records=records,
        cells=cells,
        risk_matrix=risk_matrix,
        args=args,
        gif_path=gif_path,
        extra_config={
            "paper_alignment": (
                "Uses STNPP-style GAT mark interaction learning with R attention heads. "
                "Street-network distances are not available here, so triggering is restricted "
                "to the same 150m cell for this reproducible LAPD public-data experiment."
            ),
            "train_start": args.train_start,
            "train_end": args.train_end,
            "attention_heads": args.attention_heads,
            "hidden_dim": args.hidden_dim,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "batch_size_note": (
                "The paper uses event-sequence SGD batches of M=3. This compact version "
                "optimizes the full learned mark-transition matrix directly because the "
                "mark graph has only 63 nodes."
            ),
            "transition_summary": transition_summary,
            "train_summary": train_summary,
        },
    )
    model_dir = result_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_id": MODEL_ID,
            "state_dict": model.state_dict(),
            "config": config,
            "train_summary": train_summary,
            "transition_summary": transition_summary,
        },
        model_dir / "model_state.pt",
    )
    tables_dir = result_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(transition, columns=[f"source_{i}" for i in range(transition.shape[1])]).to_csv(
        tables_dir / "learned_mark_transition.csv",
        index_label="target_mark",
    )
    with (tables_dir / "training_summary.yml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            {
                "train_summary": train_summary,
                "transition_summary": transition_summary,
                "crimes": list(crimes),
                "areas": list(areas),
            },
            f,
            sort_keys=False,
            allow_unicode=True,
        )
    print(f"Wrote {result_dir}")


if __name__ == "__main__":
    main()
