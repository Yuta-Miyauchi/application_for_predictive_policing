"""ETAS-enhanced STNPP-GAT utilities for vol2.

Vol2 keeps the vol1 GAT mark-transition learner and adds an ETAS-inspired
adaptive trigger calibration. Each LAPD area can have its own theta and omega,
estimated from a rolling history by the ETAS reference implementation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from our_model.vol1.stnpp_gat import MarkGraphAttention, STNPPGATConfig, transition_kl_loss


@dataclass(frozen=True)
class AreaETASState:
    """Area-wise ETAS parameters used to calibrate marked triggering."""

    theta_by_area: np.ndarray
    omega_by_area: np.ndarray


def marked_trigger_component(
    transition: np.ndarray,
    z_by_cell_crime: np.ndarray,
    cell_area_index: np.ndarray,
    area_state: AreaETASState,
    horizon_days: float,
) -> np.ndarray:
    """Compute a cell x crime trigger matrix using GAT transitions and ETAS parameters."""

    n_cells, n_crimes = z_by_cell_crime.shape
    out = np.zeros((n_cells, n_crimes), dtype=np.float32)
    crime_offsets = np.arange(n_crimes)
    for cell_idx in range(n_cells):
        area_idx = int(cell_area_index[cell_idx])
        marks = area_idx * n_crimes + crime_offsets
        local_transition = transition[np.ix_(marks, marks)]
        omega = float(area_state.omega_by_area[area_idx])
        theta = float(area_state.theta_by_area[area_idx])
        trigger_integral = 1.0 - float(np.exp(-omega * horizon_days))
        out[cell_idx] = (theta * trigger_integral * (local_transition @ z_by_cell_crime[cell_idx])).astype(
            np.float32
        )
    return out
