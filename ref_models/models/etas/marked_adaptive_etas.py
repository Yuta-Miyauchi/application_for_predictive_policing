"""Marked adaptive grid ETAS utilities.

This extends the project scaffold ETAS model with crime-type marks. The
spatial triggering remains same-cell and lightweight, while the trigger state
and replay tables retain crime_type-specific information.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ref_models.models.etas.adaptive_etas import (
    ETASParameters,
    cell_index,
    decay_state_to,
    fit_initial_or_refit,
)


@dataclass
class MarkedETASParameters:
    """Marked ETAS parameters at one forecast cutoff."""

    mu: pd.DataFrame
    transition: pd.DataFrame
    theta: float
    omega: float
    fit_method: str
    fit_events: int
    base_params: ETASParameters


def crime_type_index(events: pd.DataFrame) -> pd.Index:
    if "crime_type" not in events.columns or events.empty:
        return pd.Index(["ALL"], name="crime_type")
    values = sorted(events["crime_type"].dropna().astype(str).unique())
    return pd.Index(values or ["ALL"], name="crime_type")


def ensure_crime_type(events: pd.DataFrame) -> pd.DataFrame:
    out = events.copy()
    if "crime_type" not in out.columns:
        out["crime_type"] = "ALL"
    out["crime_type"] = out["crime_type"].fillna("ALL").astype(str)
    return out


def smoothed_mu_by_type(
    grid: pd.DataFrame,
    history_events: pd.DataFrame,
    history_days: int,
    crime_types: pd.Index,
    theta: float,
    alpha: float = 1e-3,
) -> pd.DataFrame:
    """Estimate per-day background intensity for every cell and crime type."""

    cells = cell_index(grid)
    mu = pd.DataFrame(alpha / history_days, index=cells, columns=crime_types, dtype=float)
    if history_events.empty:
        return mu
    history = ensure_crime_type(history_events)
    counts = (
        history.groupby([history["cell_id"].astype(str), history["crime_type"]])
        .size()
        .rename("count")
        .reset_index()
    )
    pivot = counts.pivot(index="cell_id", columns="crime_type", values="count")
    pivot = (
        pivot.reindex(index=cells, columns=crime_types, fill_value=0.0)
        .fillna(0.0)
        .astype(float)
    )
    return ((1.0 - theta) * pivot + alpha) / history_days


def fit_mark_transition(
    history_events: pd.DataFrame,
    crime_types: pd.Index,
    omega: float,
    lookback_days: float = 60.0,
    alpha: float = 0.25,
) -> pd.DataFrame:
    """Estimate source-to-target crime-type trigger proportions.

    Columns are source crime types and rows are target crime types. Each column
    sums to one, so a source event distributes its expected triggered offspring
    across target types.
    """

    mat = pd.DataFrame(alpha, index=crime_types, columns=crime_types, dtype=float)
    if history_events.empty:
        return mat.div(mat.sum(axis=0), axis=1)

    history = ensure_crime_type(history_events).sort_values("occurred_at")
    for _, group in history.groupby(history["cell_id"].astype(str), sort=False):
        group = group.sort_values("occurred_at").reset_index(drop=True)
        times = group["occurred_at"].to_numpy(dtype="datetime64[ns]")
        types = group["crime_type"].astype(str).to_numpy()
        for i in range(1, len(group)):
            age_days = (times[i] - times[:i]) / np.timedelta64(1, "D")
            mask = (age_days > 0) & (age_days <= lookback_days)
            if not mask.any():
                continue
            target = types[i]
            sources = types[:i][mask]
            weights = np.exp(-omega * age_days[mask].astype(float))
            for source, weight in zip(sources, weights, strict=False):
                if target in mat.index and source in mat.columns:
                    mat.loc[target, source] += float(weight)

    col_sums = mat.sum(axis=0).replace(0.0, 1.0)
    return mat.div(col_sums, axis=1)


def fit_marked_initial_or_refit(
    grid: pd.DataFrame,
    history_events: pd.DataFrame,
    history_start: pd.Timestamp,
    cutoff: pd.Timestamp,
    history_days: int,
    crime_types: pd.Index,
    init_theta: float = 0.25,
    init_omega: float = 1.0 / 14.0,
    theta_min: float = 0.0,
    alpha: float = 1e-3,
) -> MarkedETASParameters:
    history = ensure_crime_type(history_events)
    base_params = fit_initial_or_refit(
        grid,
        history,
        history_start=history_start,
        cutoff=cutoff,
        history_days=history_days,
        init_theta=init_theta,
        init_omega=init_omega,
        theta_min=theta_min,
        alpha=alpha,
    )
    history = history[
        (history["occurred_at"] >= history_start) & (history["occurred_at"] < cutoff)
    ].copy()
    mu = smoothed_mu_by_type(
        grid,
        history,
        history_days=history_days,
        crime_types=crime_types,
        theta=base_params.theta,
        alpha=alpha,
    )
    transition = fit_mark_transition(history, crime_types, omega=base_params.omega)
    return MarkedETASParameters(
        mu=mu,
        transition=transition,
        theta=base_params.theta,
        omega=base_params.omega,
        fit_method=f"marked_{base_params.fit_method}",
        fit_events=base_params.fit_events,
        base_params=base_params,
    )


def rebuild_marked_trigger_state(
    grid: pd.DataFrame,
    history_events: pd.DataFrame,
    cutoff: pd.Timestamp,
    omega: float,
    crime_types: pd.Index,
) -> pd.DataFrame:
    cells = cell_index(grid)
    z = pd.DataFrame(0.0, index=cells, columns=crime_types, dtype=float)
    if history_events.empty:
        return z
    history = ensure_crime_type(history_events)
    ages = (cutoff - history["occurred_at"]).dt.total_seconds() / 86400.0
    valid = ages >= 0
    if not valid.any():
        return z
    weights = np.exp(-omega * ages[valid].to_numpy(dtype=float))
    weighted = pd.DataFrame(
        {
            "cell_id": history.loc[valid, "cell_id"].astype(str).to_numpy(),
            "crime_type": history.loc[valid, "crime_type"].astype(str).to_numpy(),
            "weight": weights,
        }
    )
    grouped = weighted.groupby(["cell_id", "crime_type"])["weight"].sum().reset_index()
    pivot = grouped.pivot(index="cell_id", columns="crime_type", values="weight")
    pivot = (
        pivot.reindex(index=cells, columns=crime_types, fill_value=0.0)
        .fillna(0.0)
        .astype(float)
    )
    z.loc[:, :] = pivot.to_numpy(dtype=float)
    return z


def decay_marked_state_to(
    z: pd.DataFrame,
    last_time: pd.Timestamp,
    new_time: pd.Timestamp,
    omega: float,
) -> tuple[pd.DataFrame, pd.Timestamp]:
    decayed, out_time = decay_state_to(z, last_time, new_time, omega)
    return decayed, out_time


def update_marked_state_with_events(
    z: pd.DataFrame,
    last_time: pd.Timestamp,
    new_events: pd.DataFrame,
    omega: float,
) -> tuple[pd.DataFrame, pd.Timestamp, int]:
    if new_events.empty:
        return z, last_time, 0
    current_z = z.copy()
    current_time = last_time
    updates = 0
    events = ensure_crime_type(new_events).sort_values("occurred_at")
    for event in events.itertuples(index=False):
        event_time = pd.Timestamp(event.occurred_at)
        current_z, current_time = decay_marked_state_to(
            current_z, current_time, event_time, omega
        )
        cell = str(event.cell_id)
        crime_type = str(event.crime_type)
        if cell in current_z.index and crime_type in current_z.columns:
            current_z.loc[cell, crime_type] += 1.0
        updates += 1
    return current_z, current_time, updates


def integrated_marked_risk_components(
    params: MarkedETASParameters,
    z: pd.DataFrame,
    horizon_days: float,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return total and crime-type-specific risk components."""

    z = z.reindex(index=params.mu.index, columns=params.mu.columns, fill_value=0.0)
    background_by_type = params.mu * horizon_days
    trigger_kernel_integral = 1.0 - float(np.exp(-params.omega * horizon_days))
    transition = params.transition.reindex(
        index=params.mu.columns, columns=params.mu.columns, fill_value=0.0
    )
    trigger_values = z.to_numpy(dtype=float) @ transition.T.to_numpy(dtype=float)
    trigger_by_type = pd.DataFrame(
        params.theta * trigger_kernel_integral * trigger_values,
        index=params.mu.index,
        columns=params.mu.columns,
    )
    risk_by_type = background_by_type + trigger_by_type
    background = background_by_type.sum(axis=1).rename("background")
    trigger = trigger_by_type.sum(axis=1).rename("trigger")
    risk = risk_by_type.sum(axis=1).rename("risk")
    return risk, background, trigger, risk_by_type, background_by_type, trigger_by_type
