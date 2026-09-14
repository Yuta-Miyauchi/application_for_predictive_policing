"""Adaptive grid ETAS utilities.

The implementation here is a compact research scaffold for an Adaptive
PredPol-like ETAS experiment. It uses same-cell triggering and supports
test-then-update backtests with event-by-event state updates and periodic
rolling parameter refits.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize


@dataclass
class ETASParameters:
    """Grid ETAS parameters at one forecast cutoff."""

    mu: pd.Series
    theta: float
    omega: float
    fit_method: str
    fit_events: int
    neg_log_likelihood: float | None = None
    optimizer_success: bool | None = None


def cell_index(grid: pd.DataFrame) -> pd.Index:
    return pd.Index(grid["cell_id"].astype(str), name="cell_id")


def smoothed_mu(
    grid: pd.DataFrame,
    history_events: pd.DataFrame,
    history_days: int,
    alpha: float = 1e-3,
) -> pd.Series:
    """Estimate per-day background intensity for every grid cell."""

    cells = cell_index(grid)
    mu = pd.Series(alpha / history_days, index=cells, dtype=float)
    if history_events.empty:
        return mu
    counts = history_events.groupby(history_events["cell_id"].astype(str)).size()
    aligned = counts.reindex(cells, fill_value=0).astype(float)
    return (aligned + alpha) / history_days


def rebuild_trigger_state(
    grid: pd.DataFrame,
    history_events: pd.DataFrame,
    cutoff: pd.Timestamp,
    omega: float,
) -> pd.Series:
    """Rebuild z_n(cutoff) from an event buffer."""

    cells = cell_index(grid)
    z = pd.Series(0.0, index=cells, dtype=float)
    if history_events.empty:
        return z
    ages = (cutoff - history_events["occurred_at"]).dt.total_seconds() / 86400.0
    valid = ages >= 0
    if not valid.any():
        return z
    weights = np.exp(-omega * ages[valid].to_numpy(dtype=float))
    grouped = pd.Series(
        weights,
        index=history_events.loc[valid, "cell_id"].astype(str).to_numpy(),
    ).groupby(level=0).sum()
    z.loc[grouped.index] = grouped.reindex(cells, fill_value=0.0)
    return z


def decay_state_to(
    z: pd.Series,
    last_time: pd.Timestamp,
    new_time: pd.Timestamp,
    omega: float,
) -> tuple[pd.Series, pd.Timestamp]:
    """Decay state from last_time to new_time."""

    if new_time <= last_time:
        return z, last_time
    delta_days = (new_time - last_time).total_seconds() / 86400.0
    return z * float(np.exp(-omega * delta_days)), new_time


def update_state_with_events(
    z: pd.Series,
    last_time: pd.Timestamp,
    new_events: pd.DataFrame,
    omega: float,
) -> tuple[pd.Series, pd.Timestamp, int]:
    """Process events one by one in chronological order."""

    if new_events.empty:
        return z, last_time, 0
    updates = 0
    current_z = z.copy()
    current_time = last_time
    for event in new_events.sort_values("occurred_at").itertuples(index=False):
        event_time = pd.Timestamp(event.occurred_at)
        current_z, current_time = decay_state_to(current_z, current_time, event_time, omega)
        cell = str(event.cell_id)
        if cell in current_z.index:
            current_z.loc[cell] += 1.0
        updates += 1
    return current_z, current_time, updates


def integrated_risk_components(
    params: ETASParameters,
    z: pd.Series,
    horizon_days: float,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return total risk, background contribution, and trigger contribution."""

    z = z.reindex(params.mu.index, fill_value=0.0)
    background = params.mu * horizon_days
    trigger = params.theta * z * (1.0 - float(np.exp(-params.omega * horizon_days)))
    total = background + trigger
    return total, background, trigger


def fit_initial_or_refit(
    grid: pd.DataFrame,
    history_events: pd.DataFrame,
    history_start: pd.Timestamp,
    cutoff: pd.Timestamp,
    history_days: int,
    init_theta: float = 0.25,
    init_omega: float = 1.0 / 14.0,
    theta_min: float = 0.0,
    alpha: float = 1e-3,
) -> ETASParameters:
    """Fit theta and omega by constrained likelihood.

    This is intentionally modest. It uses a branching-ratio approximation:
    if theta is the expected triggered share, the background rate is shrunk
    toward (1 - theta) times the observed cell rate. This prevents the
    background term from absorbing all events and leaving theta at zero.
    """

    history = history_events.copy()
    history = history[
        (history["occurred_at"] >= history_start) & (history["occurred_at"] < cutoff)
    ].sort_values("occurred_at")
    mu = smoothed_mu(grid, history, history_days, alpha=alpha)
    if len(history) < 20:
        return ETASParameters(
            mu=mu,
            theta=float(np.clip(init_theta, theta_min, 0.999)),
            omega=float(np.clip(init_omega, 1.0 / 90.0, 2.0)),
            fit_method="fallback_small_history",
            fit_events=int(len(history)),
        )

    cells = list(mu.index)
    cell_pos = {cell: i for i, cell in enumerate(cells)}
    event_cells = history["cell_id"].astype(str).map(cell_pos).to_numpy()
    valid = ~pd.isna(event_cells)
    history = history.loc[valid].copy()
    event_cells = event_cells[valid].astype(int)
    times = (
        (history["occurred_at"] - history_start).dt.total_seconds() / 86400.0
    ).to_numpy(dtype=float)
    duration = max((cutoff - history_start).total_seconds() / 86400.0, 1e-6)
    counts_arr = (
        history.groupby(history["cell_id"].astype(str))
        .size()
        .reindex(cells, fill_value=0)
        .to_numpy(dtype=float)
    )

    def mu_for_theta(theta: float) -> np.ndarray:
        return ((1.0 - theta) * counts_arr + alpha) / duration

    def neg_log_likelihood(x: np.ndarray) -> float:
        theta = float(x[0])
        omega = float(x[1])
        if theta < theta_min or theta >= 1 or omega <= 0:
            return 1e30
        mu_arr = mu_for_theta(theta)
        z = np.zeros(len(cells), dtype=float)
        last_t = 0.0
        ll = 0.0
        for pos, t in zip(event_cells, times, strict=False):
            if t > last_t:
                z *= np.exp(-omega * (t - last_t))
                last_t = t
            lam = mu_arr[pos] + theta * omega * z[pos]
            if lam <= 0 or not np.isfinite(lam):
                return 1e30
            ll += np.log(lam)
            z[pos] += 1.0
        background_integral = float(mu_arr.sum() * duration)
        trigger_integral = float(theta * np.sum(1.0 - np.exp(-omega * (duration - times))))
        return -(ll - background_integral - trigger_integral)

    result = minimize(
        neg_log_likelihood,
        x0=np.array(
            [
                np.clip(init_theta, max(theta_min, 1e-6), 0.95),
                np.clip(init_omega, 1.0 / 90.0, 2.0),
            ]
        ),
        method="L-BFGS-B",
        bounds=[(theta_min, 0.999), (1.0 / 90.0, 2.0)],
        options={"maxiter": 80, "ftol": 1e-7},
    )
    if result.success and np.isfinite(result.fun):
        theta = float(result.x[0])
        omega = float(result.x[1])
        nll = float(result.fun)
        success = True
    else:
        theta = float(np.clip(init_theta, theta_min, 0.999))
        omega = float(np.clip(init_omega, 1.0 / 90.0, 2.0))
        nll = None
        success = False
    mu = pd.Series(mu_for_theta(theta), index=pd.Index(cells, name="cell_id"), dtype=float)
    return ETASParameters(
        mu=mu,
        theta=theta,
        omega=omega,
        fit_method=(
            "constrained_mle_branching_mu"
            if theta_min == 0
            else f"constrained_mle_branching_mu_theta_min_{theta_min:g}"
        ),
        fit_events=int(len(history)),
        neg_log_likelihood=nll,
        optimizer_success=success,
    )
