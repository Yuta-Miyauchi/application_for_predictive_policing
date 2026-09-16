from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parents[1] / ".matplotlib-cache"),
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import gammaln
from scipy.stats import linregress, pearsonr, spearmanr
from shapely import wkb
from sklearn.metrics import (
    average_precision_score,
    log_loss,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = (
    ROOT
    / "datas/lapd_full/lapd_legacy_2010_2024_all_crimes_grid300m_h168h"
    / "derived/lapd_legacy_2010_2024_mohler_target_crimes_grid150m_h24h"
)
METRIC_DIRNAME = "metrics"

Q_VALUES = tuple(float(x) for x in np.round(np.arange(0.01, 0.101, 0.01), 2))
MAIN_Q_VALUES = (0.01, 0.05, 0.10)
K_VALUES = (20, 50, 100, 500)
CALIBRATION_BINS = 10
EPS = 1e-12


@dataclass(frozen=True)
class ModelResult:
    name: str
    label: str
    result_dir: Path


MODELS = (
    ModelResult(
        name="etas_ref",
        label="Reference ETAS",
        result_dir=ROOT / "ref_models/experiments/etas/results",
    ),
    ModelResult(
        name="our_vol1",
        label="Our vol1 STNPP-GAT",
        result_dir=ROOT / "our_experiment/vol1/results",
    ),
    ModelResult(
        name="our_vol2",
        label="Our vol2 ETAS-enhanced STNPP-GAT",
        result_dir=ROOT / "our_experiment/vol2/results",
    ),
)


def model_metric_dir(model: ModelResult) -> Path:
    return model.result_dir / METRIC_DIRNAME


def model_plot_dir(model: ModelResult) -> Path:
    return model_metric_dir(model) / "plots"


def safe_div(num: float, den: float) -> float:
    if den == 0 or not np.isfinite(den):
        return np.nan
    return float(num / den)


def gini(values: np.ndarray) -> float:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    if np.min(x) < 0:
        x = x - np.min(x)
    total = np.sum(x)
    if total <= 0:
        return 0.0
    x = np.sort(x)
    n = x.size
    index = np.arange(1, n + 1, dtype=np.float64)
    return float((2.0 * np.sum(index * x) / (n * total)) - ((n + 1.0) / n))


def poisson_deviance_mean(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=np.float64)
    mu = np.clip(np.asarray(y_pred, dtype=np.float64), EPS, None)
    term = mu.copy()
    positive = y > 0
    term[positive] = y[positive] * np.log(y[positive] / mu[positive]) - (
        y[positive] - mu[positive]
    )
    return float(2.0 * np.mean(term))


def poisson_log_likelihood(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=np.float64)
    mu = np.clip(np.asarray(y_pred, dtype=np.float64), EPS, None)
    return float(np.sum(y * np.log(mu) - mu - gammaln(y + 1.0)))


def morans_i(residual: np.ndarray, src: np.ndarray, dst: np.ndarray) -> float:
    x = np.asarray(residual, dtype=np.float64)
    if x.size < 2:
        return np.nan
    centered = x - np.mean(x)
    denom = np.sum(centered * centered)
    if denom <= 0:
        return np.nan
    numerator = np.sum(centered[src] * centered[dst])
    return float((x.size / src.size) * (numerator / denom))


def select_by_area(order: np.ndarray, area_m2: np.ndarray, target_area: float) -> np.ndarray:
    sorted_area = area_m2[order]
    cut = int(np.searchsorted(np.cumsum(sorted_area), target_area, side="left") + 1)
    cut = max(1, min(cut, order.size))
    return order[:cut]


def confusion_metrics(tp: int, fp: int, tn: int, fn: int) -> dict[str, float]:
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
    fpr = safe_div(fp, fp + tn)
    fnr = safe_div(fn, fn + tp)
    f1 = safe_div(2.0 * precision * recall, precision + recall)
    accuracy = safe_div(tp + tn, tp + fp + tn + fn)
    balanced_accuracy = np.nanmean([recall, specificity])
    mcc_den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = safe_div(tp * tn - fp * fn, mcc_den)
    return {
        "accuracy": accuracy,
        "recall_tpr": recall,
        "specificity_tnr": specificity,
        "precision_ppv": precision,
        "fpr": fpr,
        "fnr": fnr,
        "f1": f1,
        "balanced_accuracy": balanced_accuracy,
        "mcc": mcc,
    }


def load_grid() -> pd.DataFrame:
    grid = pd.read_parquet(DATASET_DIR / "grid.parquet").copy()
    centroids = [wkb.loads(bytes(g)).centroid for g in grid["geometry"]]
    grid["centroid_x"] = [pt.x for pt in centroids]
    grid["centroid_y"] = [pt.y for pt in centroids]
    grid = grid.drop(columns=["geometry"])
    grid = grid.sort_values("cell_id").reset_index(drop=True)
    return grid


def adjacency_arrays(grid: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    index_by_rc = {
        (int(row), int(col)): idx
        for idx, row, col in grid[["row", "col"]].itertuples(index=True, name=None)
    }
    src: list[int] = []
    dst: list[int] = []
    for idx, row, col in grid[["row", "col"]].itertuples(index=True, name=None):
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            j = index_by_rc.get((int(row) + dr, int(col) + dc))
            if j is not None:
                src.append(idx)
                dst.append(j)
    return np.asarray(src, dtype=np.int32), np.asarray(dst, dtype=np.int32)


def load_frames() -> pd.DataFrame:
    frames = pd.read_csv(MODELS[0].result_dir / "tables/forecast_frames.csv")
    frames["forecast_date"] = pd.to_datetime(frames["forecast_date"])
    frames["display_time"] = pd.to_datetime(frames["display_time"])
    return frames


def build_actual_matrix(frames: pd.DataFrame, grid: pd.DataFrame) -> np.ndarray:
    events = pd.read_parquet(MODELS[0].result_dir / "tables/observed_events.parquet")
    events["occurred_at"] = pd.to_datetime(events["occurred_at"])
    cell_index = pd.Series(np.arange(len(grid), dtype=np.int32), index=grid["cell_id"])
    actual = np.zeros((len(frames), len(grid)), dtype=np.int16)

    for frame in frames.itertuples(index=False):
        window = events[
            (events["occurred_at"] >= frame.forecast_date)
            & (events["occurred_at"] < frame.display_time)
        ].copy()
        window["frame_index"] = int(frame.frame_index)
        counts = window["cell_id"].value_counts()
        idx = cell_index.reindex(counts.index).dropna().astype(np.int32).to_numpy()
        actual[int(frame.frame_index), idx] = counts.reindex(cell_index.index[idx]).to_numpy(
            dtype=np.int16
        )
    return actual


def load_risk_matrix(model: ModelResult, grid: pd.DataFrame, n_frames: int) -> np.ndarray:
    risk_long = pd.read_parquet(model.result_dir / "tables/forecast_cell_risk.parquet")
    cell_index = pd.Series(np.arange(len(grid), dtype=np.int32), index=grid["cell_id"])
    risk_long["cell_index"] = cell_index.reindex(risk_long["cell_id"]).to_numpy()
    risk_long = risk_long.dropna(subset=["cell_index"])
    risk_long["cell_index"] = risk_long["cell_index"].astype(np.int32)
    risk = np.zeros((n_frames, len(grid)), dtype=np.float32)
    risk[
        risk_long["frame_index"].to_numpy(dtype=np.int32),
        risk_long["cell_index"].to_numpy(dtype=np.int32),
    ] = risk_long["risk"].to_numpy(dtype=np.float32)
    return risk


def training_naive_prediction(grid: pd.DataFrame) -> np.ndarray:
    events = pd.read_parquet(DATASET_DIR / "events.parquet")
    events["occurred_at"] = pd.to_datetime(events["occurred_at"])
    start = pd.Timestamp("2010-01-01")
    end = pd.Timestamp("2020-01-01")
    train = events[(events["occurred_at"] >= start) & (events["occurred_at"] < end)]
    train_days = (end - start).total_seconds() / 86400.0
    counts = train["cell_id"].value_counts()
    baseline = grid["cell_id"].map(counts).fillna(0.0).to_numpy(dtype=np.float64)
    return baseline / train_days


def summarize_frame_metrics(
    df: pd.DataFrame, group_cols: list[str], metric_cols: Iterable[str], category: str
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, group in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_data = dict(zip(group_cols, keys))
        for metric in metric_cols:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            rows.append(
                {
                    **key_data,
                    "category": category,
                    "metric": metric,
                    "mean": values.mean() if not values.empty else np.nan,
                    "std": values.std(ddof=1) if len(values) > 1 else np.nan,
                    "n": int(values.size),
                }
            )
    return pd.DataFrame(rows)


def add_global_summary(
    rows: list[dict[str, object]],
    model: ModelResult,
    category: str,
    metric: str,
    value: float,
    q: float | None = None,
    k: int | None = None,
    note: str | None = None,
) -> None:
    rows.append(
        {
            "model": model.name,
            "model_label": model.label,
            "category": category,
            "metric": metric,
            "q": q,
            "k": k,
            "mean": value,
            "std": np.nan,
            "n": 1,
            "note": note,
        }
    )


def evaluate_model(
    model: ModelResult,
    grid: pd.DataFrame,
    frames: pd.DataFrame,
    actual: np.ndarray,
    naive_pred: np.ndarray,
    adjacency_src: np.ndarray,
    adjacency_dst: np.ndarray,
) -> dict[str, pd.DataFrame]:
    area = grid["area_m2"].to_numpy(dtype=np.float64)
    total_area = float(area.sum())
    x = grid["centroid_x"].to_numpy(dtype=np.float64)
    y = grid["centroid_y"].to_numpy(dtype=np.float64)
    area_ids = grid["area_id"].astype(str).to_numpy()
    area_names = grid["area_name"].astype(str).to_numpy()
    unique_areas = pd.DataFrame({"area_id": area_ids, "area_name": area_names}).drop_duplicates()
    risk = load_risk_matrix(model, grid, len(frames))
    naive = np.broadcast_to(naive_pred.reshape(1, -1), actual.shape)
    naive_frame_mae = np.mean(np.abs(actual.astype(np.float64) - naive), axis=1)
    global_naive_mae = float(np.mean(np.abs(actual.astype(np.float64) - naive)))

    hotspot_rows: list[dict[str, object]] = []
    class_rows: list[dict[str, object]] = []
    k_rows: list[dict[str, object]] = []
    count_rows: list[dict[str, object]] = []
    rri_rows: list[dict[str, object]] = []
    area_rows: list[dict[str, object]] = []
    selected_by_q: dict[float, list[np.ndarray]] = {q: [] for q in Q_VALUES}
    placement_counts: dict[float, np.ndarray] = {
        q: np.zeros(len(grid), dtype=np.int16) for q in MAIN_Q_VALUES
    }

    for frame in frames.itertuples(index=False):
        t = int(frame.frame_index)
        forecast_date = frame.forecast_date.date().isoformat()
        actual_t = actual[t].astype(np.float64)
        risk_t = np.clip(risk[t].astype(np.float64), 0.0, None)
        binary_t = actual_t > 0
        n_events = float(actual_t.sum())
        n_positive_cells = int(binary_t.sum())
        risk_order = np.argsort(-risk_t, kind="mergesort")
        actual_density_order = np.argsort(-(actual_t / area), kind="mergesort")

        if risk_t.sum() > 0:
            pred_cx = float(np.sum(risk_t * x) / np.sum(risk_t))
            pred_cy = float(np.sum(risk_t * y) / np.sum(risk_t))
        else:
            pred_cx = pred_cy = np.nan
        if n_events > 0:
            obs_cx = float(np.sum(actual_t * x) / n_events)
            obs_cy = float(np.sum(actual_t * y) / n_events)
            centroid_distance = math.hypot(pred_cx - obs_cx, pred_cy - obs_cy)
        else:
            centroid_distance = np.nan

        residual = actual_t - risk_t
        p = 1.0 - np.exp(-risk_t)
        p = np.clip(p, EPS, 1.0 - EPS)
        has_two_classes = binary_t.any() and (~binary_t).any()

        try:
            pearson = float(pearsonr(actual_t, risk_t).statistic)
        except Exception:
            pearson = np.nan
        try:
            spearman = float(spearmanr(actual_t, risk_t).statistic)
        except Exception:
            spearman = np.nan

        count_rows.append(
            {
                "model": model.name,
                "model_label": model.label,
                "frame_index": t,
                "forecast_date": forecast_date,
                "mae": float(np.mean(np.abs(residual))),
                "rmse": float(np.sqrt(np.mean(residual * residual))),
                "poisson_deviance": poisson_deviance_mean(actual_t, risk_t),
                "rmsle": float(np.sqrt(np.mean((np.log1p(actual_t) - np.log1p(risk_t)) ** 2))),
                "mase": safe_div(float(np.mean(np.abs(residual))), float(naive_frame_mae[t])),
                "r2": (
                    1.0
                    - safe_div(
                        float(np.sum(residual * residual)),
                        float(np.sum((actual_t - np.mean(actual_t)) ** 2)),
                    )
                ),
                "pearson": pearson,
                "spearman": spearman,
                "residual_morans_i": morans_i(residual, adjacency_src, adjacency_dst),
                "brier_score": float(np.mean((p - binary_t.astype(np.float64)) ** 2)),
                "log_loss": float(log_loss(binary_t, p, labels=[False, True])),
                "probability_oe_ratio": safe_div(float(binary_t.sum()), float(p.sum())),
                "count_oe_ratio": safe_div(float(actual_t.sum()), float(risk_t.sum())),
                "roc_auc": float(roc_auc_score(binary_t, risk_t)) if has_two_classes else np.nan,
                "pr_auc_average_precision": (
                    float(average_precision_score(binary_t, risk_t))
                    if has_two_classes
                    else np.nan
                ),
                "poisson_log_likelihood": poisson_log_likelihood(actual_t, risk_t),
                "poisson_log_likelihood_per_event": safe_div(
                    poisson_log_likelihood(actual_t, risk_t), n_events
                ),
                "centroid_distance_m": centroid_distance,
                "predicted_events": float(risk_t.sum()),
                "actual_events": n_events,
            }
        )

        for k in K_VALUES:
            selected = risk_order[: min(k, len(risk_order))]
            k_rows.append(
                {
                    "model": model.name,
                    "model_label": model.label,
                    "frame_index": t,
                    "forecast_date": forecast_date,
                    "k": k,
                    "precision_at_k": safe_div(float(np.sum(binary_t[selected])), float(k)),
                    "hit_count_at_k": float(np.sum(actual_t[selected])),
                }
            )

        for q in Q_VALUES:
            target_area = q * total_area
            selected = select_by_area(risk_order, area, target_area)
            selected_by_q[q].append(selected)
            selected_area = float(area[selected].sum())
            q_eff = selected_area / total_area
            hit_count = float(actual_t[selected].sum())
            hit_rate = safe_div(hit_count, n_events)
            pai = safe_div(hit_rate, q_eff)
            opt_selected = select_by_area(actual_density_order, area, selected_area)
            opt_area = float(area[opt_selected].sum())
            opt_hit_count = float(actual_t[opt_selected].sum())

            selected_mask = np.zeros(len(grid), dtype=bool)
            opt_mask = np.zeros(len(grid), dtype=bool)
            selected_mask[selected] = True
            opt_mask[opt_selected] = True
            intersection_area = float(area[selected_mask & opt_mask].sum())
            union_area = float(area[selected_mask | opt_mask].sum())

            hotspot_rows.append(
                {
                    "model": model.name,
                    "model_label": model.label,
                    "frame_index": t,
                    "forecast_date": forecast_date,
                    "q": q,
                    "q_eff": q_eff,
                    "selected_cells": int(selected.size),
                    "hit_count": hit_count,
                    "hit_rate": hit_rate,
                    "miss_rate": 1.0 - hit_rate if np.isfinite(hit_rate) else np.nan,
                    "area_coverage": q_eff,
                    "hotspot_density_per_km2": safe_div(hit_count, selected_area / 1_000_000.0),
                    "pai": pai,
                    "gain": hit_rate,
                    "lift": pai,
                    "pei_star": safe_div(hit_count, opt_hit_count),
                    "oracle_hit_count": opt_hit_count,
                    "iou_jaccard": safe_div(intersection_area, union_area),
                    "dice": safe_div(2.0 * intersection_area, selected_area + opt_area),
                    "centroid_distance_m": centroid_distance,
                }
            )

            if q in MAIN_Q_VALUES:
                placement_counts[q][selected] += 1

                tp = int(np.sum(binary_t[selected]))
                fp = int(selected.size - tp)
                fn = int(n_positive_cells - tp)
                tn = int(len(grid) - selected.size - fn)
                class_rows.append(
                    {
                        "model": model.name,
                        "model_label": model.label,
                        "frame_index": t,
                        "forecast_date": forecast_date,
                        "q": q,
                        "tp": tp,
                        "fp": fp,
                        "tn": tn,
                        "fn": fn,
                        **confusion_metrics(tp, fp, tn, fn),
                    }
                )

                selected_df = pd.DataFrame(
                    {
                        "area_id": area_ids,
                        "area_name": area_names,
                        "selected": selected_mask,
                        "area_m2": area,
                        "actual": actual_t,
                        "binary": binary_t,
                    }
                )
                area_group = selected_df.groupby(["area_id", "area_name"], as_index=False).agg(
                    selected_area_m2=("area_m2", lambda s: float(s[selected_df.loc[s.index, "selected"]].sum())),
                    total_area_m2=("area_m2", "sum"),
                    actual_events=("actual", "sum"),
                    positive_cells=("binary", "sum"),
                    selected_cells=("selected", "sum"),
                )
                allocation_share = area_group["selected_area_m2"] / max(selected_area, EPS)
                crime_share = area_group["actual_events"] / max(n_events, EPS)
                ratio = allocation_share / crime_share.replace(0, np.nan)

                subgroup_fpr: list[float] = []
                subgroup_fnr: list[float] = []
                for area_id in unique_areas["area_id"]:
                    m = area_ids == area_id
                    area_selected = selected_mask[m]
                    area_binary = binary_t[m]
                    a_tp = int(np.sum(area_selected & area_binary))
                    a_fp = int(np.sum(area_selected & ~area_binary))
                    a_fn = int(np.sum(~area_selected & area_binary))
                    a_tn = int(np.sum(~area_selected & ~area_binary))
                    subgroup_fpr.append(safe_div(a_fp, a_fp + a_tn))
                    subgroup_fnr.append(safe_div(a_fn, a_fn + a_tp))
                fpr_values = np.asarray(subgroup_fpr, dtype=np.float64)
                fnr_values = np.asarray(subgroup_fnr, dtype=np.float64)

                area_rows.append(
                    {
                        "model": model.name,
                        "model_label": model.label,
                        "frame_index": t,
                        "forecast_date": forecast_date,
                        "q": q,
                        "allocation_share_gap": float(allocation_share.max() - allocation_share.min()),
                        "allocation_to_crime_ratio_gap": float(
                            np.nanmax(ratio.to_numpy(dtype=np.float64))
                            - np.nanmin(ratio.to_numpy(dtype=np.float64))
                        )
                        if np.isfinite(ratio).any()
                        else np.nan,
                        "subgroup_fpr_gap": float(np.nanmax(fpr_values) - np.nanmin(fpr_values)),
                        "subgroup_fnr_gap": float(np.nanmax(fnr_values) - np.nanmin(fnr_values)),
                        "allocation_gini_frame": gini(selected_mask.astype(np.float64)),
                    }
                )

    for q, selections in selected_by_q.items():
        for t in range(len(selections) - 1):
            selected = selections[t]
            n1 = float(actual[t, selected].sum())
            n2 = float(actual[t + 1, selected].sum())
            n_all1 = float(actual[t].sum())
            n_all2 = float(actual[t + 1].sum())
            rri_rows.append(
                {
                    "model": model.name,
                    "model_label": model.label,
                    "frame_index": t,
                    "forecast_date": frames.loc[t, "forecast_date"].date().isoformat(),
                    "q": q,
                    "rri": safe_div(safe_div(n2, n1), safe_div(n_all2, n_all1)),
                }
            )

    area_calibration_rows: list[dict[str, object]] = []
    for area_id, area_name in unique_areas.itertuples(index=False):
        m = area_ids == area_id
        y_area = (actual[:, m] > 0).astype(np.float64).ravel()
        p_area = (1.0 - np.exp(-risk[:, m])).ravel()
        p_area = np.clip(p_area, EPS, 1.0 - EPS)
        area_calibration_rows.append(
            {
                "model": model.name,
                "model_label": model.label,
                "area_id": area_id,
                "area_name": area_name,
                "brier_score": float(np.mean((p_area - y_area) ** 2)),
                "probability_oe_ratio": safe_div(float(y_area.sum()), float(p_area.sum())),
                "mean_probability": float(np.mean(p_area)),
                "observed_rate": float(np.mean(y_area)),
            }
        )

    placement_rows: list[dict[str, object]] = []
    for q, counts in placement_counts.items():
        placement_rows.append(
            {
                "model": model.name,
                "model_label": model.label,
                "q": q,
                "allocation_gini": gini(counts.astype(np.float64)),
                "selected_cell_frame_share": safe_div(float(counts.sum()), float(counts.size * len(frames))),
            }
        )

    calibration_rows = calibration_table(model, risk, actual)
    cumulative_rows = cumulative_intensity_table(model, frames, risk, actual)

    return {
        "hotspot": pd.DataFrame(hotspot_rows),
        "classification": pd.DataFrame(class_rows),
        "precision_at_k": pd.DataFrame(k_rows),
        "count_probability": pd.DataFrame(count_rows),
        "rri": pd.DataFrame(rri_rows),
        "area_proxy": pd.DataFrame(area_rows),
        "area_calibration": pd.DataFrame(area_calibration_rows),
        "placement": pd.DataFrame(placement_rows),
        "calibration": pd.DataFrame(calibration_rows),
        "cumulative": pd.DataFrame(cumulative_rows),
        "global": global_metrics(model, risk, actual, naive_pred, global_naive_mae),
    }


def calibration_table(
    model: ModelResult, risk: np.ndarray, actual: np.ndarray
) -> list[dict[str, object]]:
    p = np.clip(1.0 - np.exp(-risk.astype(np.float64).ravel()), EPS, 1.0 - EPS)
    y = (actual.ravel() > 0).astype(np.float64)
    quantiles = np.unique(np.quantile(p, np.linspace(0.0, 1.0, CALIBRATION_BINS + 1)))
    if quantiles.size <= 2:
        quantiles = np.linspace(p.min(), p.max(), CALIBRATION_BINS + 1)
    bins = np.digitize(p, quantiles[1:-1], right=True)
    rows: list[dict[str, object]] = []
    for b in range(int(bins.max()) + 1):
        mask = bins == b
        if not np.any(mask):
            continue
        rows.append(
            {
                "model": model.name,
                "model_label": model.label,
                "bin": b,
                "n": int(mask.sum()),
                "mean_probability": float(np.mean(p[mask])),
                "observed_rate": float(np.mean(y[mask])),
                "abs_calibration_error": float(abs(np.mean(y[mask]) - np.mean(p[mask]))),
            }
        )
    return rows


def cumulative_intensity_table(
    model: ModelResult, frames: pd.DataFrame, risk: np.ndarray, actual: np.ndarray
) -> list[dict[str, object]]:
    pred = risk.sum(axis=1)
    obs = actual.sum(axis=1)
    cum_pred = np.cumsum(pred)
    cum_obs = np.cumsum(obs)
    return [
        {
            "model": model.name,
            "model_label": model.label,
            "frame_index": int(frame.frame_index),
            "forecast_date": frame.forecast_date.date().isoformat(),
            "predicted_events": float(pred[int(frame.frame_index)]),
            "actual_events": float(obs[int(frame.frame_index)]),
            "cumulative_predicted_events": float(cum_pred[int(frame.frame_index)]),
            "cumulative_actual_events": float(cum_obs[int(frame.frame_index)]),
            "cumulative_error": float(cum_pred[int(frame.frame_index)] - cum_obs[int(frame.frame_index)]),
        }
        for frame in frames.itertuples(index=False)
    ]


def global_metrics(
    model: ModelResult, risk: np.ndarray, actual: np.ndarray, naive_pred: np.ndarray, naive_mae: float
) -> pd.DataFrame:
    y = actual.astype(np.float64).ravel()
    mu = np.clip(risk.astype(np.float64).ravel(), 0.0, None)
    binary = y > 0
    p = np.clip(1.0 - np.exp(-mu), EPS, 1.0 - EPS)
    rows: list[dict[str, object]] = []
    for metric, value, category in (
        ("global_mae", np.mean(np.abs(y - mu)), "count"),
        ("global_rmse", np.sqrt(np.mean((y - mu) ** 2)), "count"),
        ("global_poisson_deviance", poisson_deviance_mean(y, mu), "count"),
        ("global_rmsle", np.sqrt(np.mean((np.log1p(y) - np.log1p(mu)) ** 2)), "count"),
        ("global_mase", safe_div(float(np.mean(np.abs(y - mu))), naive_mae), "count"),
        ("global_brier_score", np.mean((p - binary.astype(np.float64)) ** 2), "probability"),
        ("global_log_loss", log_loss(binary, p, labels=[False, True]), "probability"),
        ("global_probability_oe_ratio", safe_div(float(binary.sum()), float(p.sum())), "probability"),
        ("global_count_oe_ratio", safe_div(float(y.sum()), float(mu.sum())), "count"),
        ("global_poisson_log_likelihood", poisson_log_likelihood(y, mu), "point_process_approx"),
        (
            "global_poisson_log_likelihood_per_event",
            safe_div(poisson_log_likelihood(y, mu), float(y.sum())),
            "point_process_approx",
        ),
    ):
        rows.append(
            {
                "model": model.name,
                "model_label": model.label,
                "category": category,
                "metric": metric,
                "mean": float(value),
                "std": np.nan,
                "n": 1,
                "note": "Computed over all frame-cell pairs.",
            }
        )
    return pd.DataFrame(rows)


def metric_applicability() -> pd.DataFrame:
    rows = [
        ("Hit Count", "computed", "hotspot_frame_metrics.csv", "Computed for q=1%..10%."),
        ("Hit Rate", "computed", "hotspot_frame_metrics.csv", "Equivalent to Gain/Recall@q."),
        ("Miss Rate", "computed", "hotspot_frame_metrics.csv", "1 - Hit Rate."),
        ("Area Coverage", "computed", "hotspot_frame_metrics.csv", "Uses actual selected cell area."),
        ("Hotspot Density", "computed", "hotspot_frame_metrics.csv", "Reported per km2."),
        ("PAI", "computed", "hotspot_frame_metrics.csv", "Hit Rate / Area Coverage."),
        ("RRI", "computed", "rri_frame_metrics.csv", "Uses consecutive forecast frames and the same selected region."),
        ("PEI*", "computed", "hotspot_frame_metrics.csv", "Oracle region uses realized cell crime density."),
        ("Gain@q / Recall@q", "computed", "hotspot_frame_metrics.csv", "Same value as Hit Rate."),
        ("Lift@q", "computed", "hotspot_frame_metrics.csv", "Same value as PAI under area-based q."),
        ("Precision@k", "computed", "precision_at_k_frame_metrics.csv", f"k={K_VALUES}."),
        ("Hit-rate curve", "computed", "plots/hit_rate_curve.png", "q=1%..10%."),
        ("PAI curve", "computed", "plots/pai_curve.png", "q=1%..10%."),
        ("Area Under Hitrate Curve", "computed", "summary_metrics.csv", "Trapezoid over q=1%..10%."),
        ("IoU / Jaccard", "computed", "hotspot_frame_metrics.csv", "Predicted region vs same-area oracle region."),
        ("Dice", "computed", "hotspot_frame_metrics.csv", "Predicted region vs same-area oracle region."),
        ("Centroid Distance", "computed", "count_probability_frame_metrics.csv", "Risk-weighted vs realized count-weighted centroid."),
        ("Accuracy", "computed", "classification_frame_metrics.csv", "Top-q region is predicted positive."),
        ("Recall / TPR", "computed", "classification_frame_metrics.csv", "Top-q region is predicted positive."),
        ("Specificity / TNR", "computed", "classification_frame_metrics.csv", "Top-q region is predicted positive."),
        ("Precision / PPV", "computed", "classification_frame_metrics.csv", "Top-q region is predicted positive."),
        ("FPR", "computed", "classification_frame_metrics.csv", "Top-q region is predicted positive."),
        ("FNR", "computed", "classification_frame_metrics.csv", "Top-q region is predicted positive."),
        ("F1", "computed", "classification_frame_metrics.csv", "Top-q region is predicted positive."),
        ("Balanced Accuracy", "computed", "classification_frame_metrics.csv", "Top-q region is predicted positive."),
        ("MCC", "computed", "classification_frame_metrics.csv", "Top-q region is predicted positive."),
        ("ROC-AUC", "computed", "count_probability_frame_metrics.csv", "Uses cell risk as score."),
        ("PR-AUC / Average Precision", "computed", "count_probability_frame_metrics.csv", "Uses cell risk as score."),
        ("MAE", "computed", "count_probability_frame_metrics.csv", "Cell-frame count error."),
        ("RMSE", "computed", "count_probability_frame_metrics.csv", "Cell-frame count error."),
        ("Poisson Deviance", "computed", "count_probability_frame_metrics.csv", "Cell-frame count error."),
        ("RMSLE", "computed", "count_probability_frame_metrics.csv", "Cell-frame count error."),
        ("MASE", "computed", "count_probability_frame_metrics.csv", "Naive baseline is 2010-2019 cell daily mean."),
        ("R2", "computed", "count_probability_frame_metrics.csv", "Cell-frame count error."),
        ("Pearson", "computed", "count_probability_frame_metrics.csv", "Cell-frame count/risk correlation."),
        ("Spearman", "computed", "count_probability_frame_metrics.csv", "Cell-frame count/risk rank correlation."),
        ("Residual Moran's I", "computed_no_pvalue", "count_probability_frame_metrics.csv", "Rook adjacency; permutation p-value not computed."),
        ("Log Loss", "computed", "count_probability_frame_metrics.csv", "Probability p=1-exp(-expected count)."),
        ("Brier Score", "computed", "count_probability_frame_metrics.csv", "Probability p=1-exp(-expected count)."),
        ("Calibration Curve", "computed", "calibration_bins.csv", "Equal-frequency bins."),
        ("ECE", "computed", "summary_metrics.csv", "Weighted mean absolute calibration error."),
        ("O/E ratio", "computed", "count_probability_frame_metrics.csv", "Probability and count versions are reported."),
        ("Test LogLikelihood", "approximated", "summary_metrics.csv", "Binned Poisson test log-likelihood, not continuous point-process LL."),
        ("AIC", "not_computed", "", "Requires training log-likelihood and comparable parameter count; not saved for all models."),
        ("BIC", "not_computed", "", "Requires training log-likelihood and comparable parameter count; not saved for all models."),
        ("N(t) and Lambda(t)", "computed", "cumulative_intensity.csv", "Computed over forecast frames."),
        ("Cumulative intensity error", "computed", "summary_metrics.csv", "RMSE and max absolute cumulative error over forecast frames."),
        ("Time-rescaling KS", "not_computed", "", "Requires continuous integrated intensity between events; current outputs are 14-day-spaced 24h snapshots."),
        ("Next-event Time MAE", "not_computed", "", "Current models do not output a next-event time prediction."),
        ("Next-event Location MAE", "not_computed", "", "Current models do not output a next-event location prediction."),
        ("Type Macro-F1", "not_computed", "", "Current saved forecasts are total target-crime risk, not per-crime-type predicted labels."),
        ("Allocation Share Gap", "computed_area_proxy", "area_proxy_frame_metrics.csv", "Uses LAPD area as subgroup, not demographic attributes."),
        ("Allocation-to-Crime Ratio Gap", "computed_area_proxy", "area_proxy_frame_metrics.csv", "Uses LAPD area as subgroup, not demographic attributes."),
        ("Subgroup FPR/FNR Gap", "computed_area_proxy", "area_proxy_frame_metrics.csv", "Uses LAPD area as subgroup, not demographic attributes."),
        ("Subgroup Calibration", "computed_area_proxy", "area_calibration.csv", "Uses LAPD area as subgroup, not demographic attributes."),
        ("Allocation Gini", "computed", "placement_metrics.csv", "Cell-level selected-frame counts."),
        ("Bias Amplification Slope", "computed_area_proxy", "summary_metrics.csv", "Linear slope of area allocation-to-crime ratio gap over frame index."),
    ]
    return pd.DataFrame(rows, columns=["metric", "status", "output", "note"])


def build_summary(results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    summary_parts: list[pd.DataFrame] = []
    summary_parts.append(
        summarize_frame_metrics(
            results["hotspot"],
            ["model", "model_label", "q"],
            [
                "hit_count",
                "hit_rate",
                "miss_rate",
                "area_coverage",
                "hotspot_density_per_km2",
                "pai",
                "pei_star",
                "iou_jaccard",
                "dice",
                "centroid_distance_m",
            ],
            "hotspot_ranking",
        )
    )
    summary_parts.append(
        summarize_frame_metrics(
            results["classification"],
            ["model", "model_label", "q"],
            [
                "accuracy",
                "recall_tpr",
                "specificity_tnr",
                "precision_ppv",
                "fpr",
                "fnr",
                "f1",
                "balanced_accuracy",
                "mcc",
            ],
            "binary_classification",
        )
    )
    summary_parts.append(
        summarize_frame_metrics(
            results["precision_at_k"],
            ["model", "model_label", "k"],
            ["precision_at_k", "hit_count_at_k"],
            "ranking_at_k",
        )
    )
    summary_parts.append(
        summarize_frame_metrics(
            results["count_probability"],
            ["model", "model_label"],
            [
                "mae",
                "rmse",
                "poisson_deviance",
                "rmsle",
                "mase",
                "r2",
                "pearson",
                "spearman",
                "residual_morans_i",
                "brier_score",
                "log_loss",
                "probability_oe_ratio",
                "count_oe_ratio",
                "roc_auc",
                "pr_auc_average_precision",
                "poisson_log_likelihood_per_event",
                "centroid_distance_m",
            ],
            "count_probability_point_process",
        )
    )
    summary_parts.append(
        summarize_frame_metrics(
            results["rri"], ["model", "model_label", "q"], ["rri"], "hotspot_persistence"
        )
    )
    summary_parts.append(
        summarize_frame_metrics(
            results["area_proxy"],
            ["model", "model_label", "q"],
            [
                "allocation_share_gap",
                "allocation_to_crime_ratio_gap",
                "subgroup_fpr_gap",
                "subgroup_fnr_gap",
                "allocation_gini_frame",
            ],
            "area_proxy_fairness",
        )
    )
    summary_parts.append(results["global"])

    summary = pd.concat(summary_parts, ignore_index=True)

    extra_rows: list[dict[str, object]] = []
    present_models = {
        str(model_name) for model_name in results["hotspot"]["model"].dropna().unique()
    }
    models = [model for model in MODELS if model.name in present_models]

    for model in models:
        hotspot = results["hotspot"][results["hotspot"]["model"] == model.name]
        for metric_name, source_col in (
            ("area_under_hitrate_curve_q01_q10", "hit_rate"),
            ("area_under_pai_curve_q01_q10", "pai"),
        ):
            curve = hotspot.groupby("q", as_index=False)[source_col].mean().sort_values("q")
            value = float(np.trapezoid(curve[source_col], curve["q"]))
            add_global_summary(extra_rows, model, "hotspot_ranking", metric_name, value)

        calibration = results["calibration"][results["calibration"]["model"] == model.name]
        ece = float(
            np.average(
                calibration["abs_calibration_error"],
                weights=calibration["n"],
            )
        )
        add_global_summary(extra_rows, model, "probability", "expected_calibration_error", ece)

        cumulative = results["cumulative"][results["cumulative"]["model"] == model.name]
        cum_error = cumulative["cumulative_error"].to_numpy(dtype=np.float64)
        add_global_summary(
            extra_rows,
            model,
            "point_process_approx",
            "cumulative_intensity_rmse",
            float(np.sqrt(np.mean(cum_error * cum_error))),
        )
        add_global_summary(
            extra_rows,
            model,
            "point_process_approx",
            "cumulative_intensity_max_abs_error",
            float(np.max(np.abs(cum_error))),
        )

        area_proxy = results["area_proxy"][
            (results["area_proxy"]["model"] == model.name)
            & (results["area_proxy"]["q"] == 0.05)
        ].sort_values("frame_index")
        if len(area_proxy) > 1:
            slope = linregress(
                area_proxy["frame_index"].to_numpy(dtype=np.float64),
                area_proxy["allocation_to_crime_ratio_gap"].to_numpy(dtype=np.float64),
            ).slope
            add_global_summary(
                extra_rows,
                model,
                "area_proxy_fairness",
                "bias_amplification_slope_q05",
                float(slope),
                q=0.05,
                note="Slope over frame index; LAPD area proxy, not demographic subgroup.",
            )

    summary = pd.concat([summary, pd.DataFrame(extra_rows)], ignore_index=True)
    for col in ("q", "k", "note"):
        if col not in summary.columns:
            summary[col] = np.nan
    return summary


def write_model_outputs(
    model: ModelResult, results: dict[str, pd.DataFrame], summary: pd.DataFrame
) -> None:
    output_dir = model_metric_dir(model)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "hotspot_frame_metrics.csv": results["hotspot"],
        "classification_frame_metrics.csv": results["classification"],
        "precision_at_k_frame_metrics.csv": results["precision_at_k"],
        "count_probability_frame_metrics.csv": results["count_probability"],
        "rri_frame_metrics.csv": results["rri"],
        "area_proxy_frame_metrics.csv": results["area_proxy"],
        "area_calibration.csv": results["area_calibration"],
        "placement_metrics.csv": results["placement"],
        "calibration_bins.csv": results["calibration"],
        "cumulative_intensity.csv": results["cumulative"],
        "summary_metrics.csv": summary,
        "metric_applicability.csv": metric_applicability(),
    }
    for name, df in outputs.items():
        df.to_csv(output_dir / name, index=False)


def plot_grouped_bars(
    ax: plt.Axes,
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    hue_col: str = "model_label",
    title: str = "",
    ylabel: str = "",
) -> None:
    x_values = list(data[x_col].drop_duplicates())
    hue_values = list(data[hue_col].drop_duplicates())
    width = 0.8 / max(len(hue_values), 1)
    x = np.arange(len(x_values))
    for i, hue in enumerate(hue_values):
        values = [
            data[(data[x_col] == xv) & (data[hue_col] == hue)][y_col].mean()
            for xv in x_values
        ]
        ax.bar(x + (i - (len(hue_values) - 1) / 2) * width, values, width, label=hue)
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in x_values])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)


def create_model_plots(
    model: ModelResult, results: dict[str, pd.DataFrame], summary: pd.DataFrame
) -> None:
    plot_dir = model_plot_dir(model)
    plot_dir.mkdir(parents=True, exist_ok=True)

    hotspot = results["hotspot"]
    main_hotspot = hotspot[hotspot["q"].isin(MAIN_Q_VALUES)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    for ax, metric, title in zip(
        axes,
        ["hit_rate", "pai", "pei_star"],
        ["Hit Rate", "PAI", "PEI*"],
    ):
        plot_grouped_bars(
            ax,
            main_hotspot.groupby(["model_label", "q"], as_index=False)[metric].mean(),
            "q",
            metric,
            title=title,
            ylabel=metric,
        )
    axes[0].legend(fontsize=8)
    fig.savefig(plot_dir / "hotspot_main_metrics.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    for label, group in hotspot.groupby("model_label"):
        curve = group.groupby("q", as_index=False)["hit_rate"].mean()
        ax.plot(curve["q"], curve["hit_rate"], marker="o", label=label)
    ax.set_title("Hit-rate Curve")
    ax.set_xlabel("Area coverage q")
    ax.set_ylabel("Hit Rate / Gain")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.savefig(plot_dir / "hit_rate_curve.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    for label, group in hotspot.groupby("model_label"):
        curve = group.groupby("q", as_index=False)["pai"].mean()
        ax.plot(curve["q"], curve["pai"], marker="o", label=label)
    ax.set_title("PAI Curve")
    ax.set_xlabel("Area coverage q")
    ax.set_ylabel("PAI")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.savefig(plot_dir / "pai_curve.png", dpi=170)
    plt.close(fig)

    classification = results["classification"]
    q05 = classification[np.isclose(classification["q"], 0.05)]
    class_long = q05.melt(
        id_vars=["model_label"],
        value_vars=["precision_ppv", "recall_tpr", "specificity_tnr", "f1", "mcc"],
        var_name="metric",
        value_name="value",
    )
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    plot_grouped_bars(
        ax,
        class_long.groupby(["model_label", "metric"], as_index=False)["value"].mean(),
        "metric",
        "value",
        title="Binary Classification Metrics at q=5%",
        ylabel="Mean over frames",
    )
    ax.tick_params(axis="x", rotation=25)
    ax.legend(fontsize=8)
    fig.savefig(plot_dir / "classification_q05.png", dpi=170)
    plt.close(fig)

    count = results["count_probability"]
    metric_sets = [
        ("Count Errors", ["mae", "rmse", "poisson_deviance", "rmsle"]),
        ("Rank/Correlation", ["roc_auc", "pr_auc_average_precision", "spearman", "pearson"]),
        ("Probability", ["brier_score", "log_loss", "probability_oe_ratio", "count_oe_ratio"]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5), constrained_layout=True)
    for ax, (title, cols) in zip(axes, metric_sets):
        long = count.melt(id_vars=["model_label"], value_vars=cols, var_name="metric", value_name="value")
        plot_grouped_bars(
            ax,
            long.groupby(["model_label", "metric"], as_index=False)["value"].mean(),
            "metric",
            "value",
            title=title,
            ylabel="Mean over frames",
        )
        ax.tick_params(axis="x", rotation=25)
    axes[0].legend(fontsize=8)
    fig.savefig(plot_dir / "count_probability_metrics.png", dpi=170)
    plt.close(fig)

    calibration = results["calibration"]
    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
    ax.plot([0, calibration["mean_probability"].max()], [0, calibration["mean_probability"].max()], "--", color="gray", label="perfect")
    for label, group in calibration.groupby("model_label"):
        ax.plot(group["mean_probability"], group["observed_rate"], marker="o", label=label)
    ax.set_title("Calibration Curve")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed cell crime rate")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.savefig(plot_dir / "calibration_curve.png", dpi=170)
    plt.close(fig)

    cumulative = results["cumulative"]
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    actual_curve = cumulative.sort_values("frame_index")
    ax.plot(
        pd.to_datetime(actual_curve["forecast_date"]),
        actual_curve["cumulative_actual_events"],
        color="black",
        linewidth=2,
        label="Observed N(t)",
    )
    ax.plot(
        pd.to_datetime(actual_curve["forecast_date"]),
        actual_curve["cumulative_predicted_events"],
        label=f"{model.label} Lambda(t)",
    )
    ax.set_title("Cumulative Observed Events and Predicted Intensity")
    ax.set_ylabel("Cumulative events")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.savefig(plot_dir / "cumulative_intensity.png", dpi=170)
    plt.close(fig)

    spatial = main_hotspot[main_hotspot["q"] == 0.05].melt(
        id_vars=["model_label"],
        value_vars=["iou_jaccard", "dice", "centroid_distance_m"],
        var_name="metric",
        value_name="value",
    )
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    plot_grouped_bars(
        ax,
        spatial.groupby(["model_label", "metric"], as_index=False)["value"].mean(),
        "metric",
        "value",
        title="Spatial Match Metrics at q=5%",
        ylabel="Mean over frames",
    )
    ax.tick_params(axis="x", rotation=25)
    ax.legend(fontsize=8)
    fig.savefig(plot_dir / "spatial_match_q05.png", dpi=170)
    plt.close(fig)

    area_proxy = results["area_proxy"]
    area_q05 = area_proxy[np.isclose(area_proxy["q"], 0.05)]
    area_long = area_q05.melt(
        id_vars=["model_label"],
        value_vars=[
            "allocation_share_gap",
            "allocation_to_crime_ratio_gap",
            "subgroup_fpr_gap",
            "subgroup_fnr_gap",
            "allocation_gini_frame",
        ],
        var_name="metric",
        value_name="value",
    )
    fig, ax = plt.subplots(figsize=(13, 5), constrained_layout=True)
    plot_grouped_bars(
        ax,
        area_long.groupby(["model_label", "metric"], as_index=False)["value"].mean(),
        "metric",
        "value",
        title="Area Proxy Fairness/Burden Metrics at q=5%",
        ylabel="Mean over frames",
    )
    ax.tick_params(axis="x", rotation=25)
    ax.legend(fontsize=8)
    fig.savefig(plot_dir / "area_proxy_metrics_q05.png", dpi=170)
    plt.close(fig)

    area_cal = results["area_calibration"].sort_values(["area_id", "model_label"])
    fig, ax = plt.subplots(figsize=(13, 5), constrained_layout=True)
    plot_grouped_bars(
        ax,
        area_cal,
        "area_id",
        "brier_score",
        title="Area-level Brier Score",
        ylabel="Brier Score",
    )
    ax.tick_params(axis="x", rotation=45)
    ax.legend(fontsize=8)
    fig.savefig(plot_dir / "area_calibration_brier.png", dpi=170)
    plt.close(fig)


def main() -> None:
    grid = load_grid()
    frames = load_frames()
    actual = build_actual_matrix(frames, grid)
    naive_pred = training_naive_prediction(grid)
    adjacency_src, adjacency_dst = adjacency_arrays(grid)

    for model in MODELS:
        print(f"Evaluating {model.label}")
        result = evaluate_model(
            model,
            grid,
            frames,
            actual,
            naive_pred,
            adjacency_src,
            adjacency_dst,
        )
        summary = build_summary(result)
        write_model_outputs(model, result, summary)
        create_model_plots(model, result, summary)
        print(f"Wrote metrics to {model_metric_dir(model)}")
        print(f"Wrote plots to {model_plot_dir(model)}")


if __name__ == "__main__":
    main()
