"""Multiscale uncertainty calibration for STNPP-GAT vol3.

Vol3 keeps the fine-grid GAT and adaptive ETAS intensity from vol2, but treats
that intensity as a within-region allocation surface. A DGCN/MTCN network
learns a ZINB distribution for daily crime counts on a coarser graph. Its mean
calibrates the total count assigned to each coarse cell and crime type, while
the vol2 surface determines where that count is placed on the 150 m grid.

This division of responsibilities avoids fitting a ZINB model to the extremely
sparse 150 m tensor and avoids smoothing away the event-scale hotspots that the
point-process component was designed to represent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from ref_models.models.stmgnn_zinb import (
    STMGNNZINB,
    STMGNNZINBConfig,
    ZINBOutput,
)


@dataclass(frozen=True)
class MultiscaleSTNPPGATZINBConfig:
    n_coarse_nodes: int
    n_crimes: int
    input_steps: int = 28
    output_steps: int = 1
    hidden_dim: int = 24
    graph_layers: int = 1
    temporal_hidden_steps: int = 14
    dropout: float = 0.10
    allocation_epsilon: float = 1e-8
    minimum_fusion_variance: float = 1e-4

    def count_model_config(self) -> STMGNNZINBConfig:
        return STMGNNZINBConfig(
            n_nodes=self.n_coarse_nodes,
            n_crimes=self.n_crimes,
            input_steps=self.input_steps,
            output_steps=self.output_steps,
            hidden_dim=self.hidden_dim,
            graph_layers=self.graph_layers,
            temporal_hidden_steps=self.temporal_hidden_steps,
            dropout=self.dropout,
        )


@dataclass(frozen=True)
class AllocationDiagnostics:
    point_process_total: float
    calibrated_total: float
    coarse_target_total: float
    maximum_coarse_count_error: float
    fallback_groups: int


@dataclass(frozen=True)
class CountFusionDiagnostics:
    mean_zinb_weight: float
    minimum_zinb_weight: float
    maximum_zinb_weight: float
    point_process_total: float
    zinb_mean_total: float
    fused_total: float


class MultiscaleSTNPPGATZINB(nn.Module):
    """ZINB coarse-count branch used to calibrate vol2 point-process risk."""

    def __init__(
        self,
        config: MultiscaleSTNPPGATZINBConfig,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
    ):
        super().__init__()
        self.config = config
        self.count_model = STMGNNZINB(
            config.count_model_config(),
            edge_index=edge_index,
            edge_weight=edge_weight,
        )

    def forward(self, count_history: torch.Tensor) -> ZINBOutput:
        return self.count_model(count_history)


def uncertainty_weighted_count_fusion(
    point_process_expected_count: np.ndarray,
    zinb_mean: np.ndarray,
    zinb_variance: np.ndarray,
    minimum_variance: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray, CountFusionDiagnostics]:
    """Fuse two count means using their Poisson and ZINB predictive variances.

    The point-process count is treated as Poisson, so its variance equals its
    expected count. Inverse-variance fusion then gives the ZINB estimate weight
    ``var_point / (var_point + var_zinb)``. The returned weight is therefore
    smaller where the ZINB branch reports broad uncertainty.
    """

    point_mean = np.clip(np.asarray(point_process_expected_count, dtype=np.float64), 0.0, None)
    distribution_mean = np.clip(np.asarray(zinb_mean, dtype=np.float64), 0.0, None)
    distribution_variance = np.clip(
        np.asarray(zinb_variance, dtype=np.float64), minimum_variance, None
    )
    if point_mean.shape != distribution_mean.shape or point_mean.shape != distribution_variance.shape:
        raise ValueError("Point-process mean, ZINB mean, and ZINB variance must share a shape")
    point_variance = np.clip(point_mean, minimum_variance, None)
    zinb_weight = point_variance / (point_variance + distribution_variance)
    fused = (1.0 - zinb_weight) * point_mean + zinb_weight * distribution_mean
    diagnostics = CountFusionDiagnostics(
        mean_zinb_weight=float(zinb_weight.mean()),
        minimum_zinb_weight=float(zinb_weight.min()),
        maximum_zinb_weight=float(zinb_weight.max()),
        point_process_total=float(point_mean.sum()),
        zinb_mean_total=float(distribution_mean.sum()),
        fused_total=float(fused.sum()),
    )
    return fused.astype(np.float32), zinb_weight.astype(np.float32), diagnostics


def allocate_coarse_counts_to_fine_cells(
    point_process_risk: np.ndarray,
    coarse_expected_count: np.ndarray,
    fine_to_coarse: np.ndarray,
    epsilon: float = 1e-8,
) -> tuple[np.ndarray, AllocationDiagnostics]:
    """Preserve vol2 rankings while matching each coarse crime-count mean.

    A uniform fallback is used only when a coarse cell has a positive target
    count but the point-process component gives all of its fine cells zero risk.
    """

    fine_risk = np.asarray(point_process_risk, dtype=np.float64)
    target = np.clip(np.asarray(coarse_expected_count, dtype=np.float64), 0.0, None)
    mapping = np.asarray(fine_to_coarse, dtype=np.int64)
    if fine_risk.ndim != 2 or target.ndim != 2:
        raise ValueError("Risk and coarse expected counts must both be two-dimensional")
    if fine_risk.shape[0] != mapping.size:
        raise ValueError("fine_to_coarse must contain one entry for each fine cell")
    if fine_risk.shape[1] != target.shape[1]:
        raise ValueError("Fine and coarse arrays must use the same crime channels")
    if mapping.size and (mapping.min() < 0 or mapping.max() >= target.shape[0]):
        raise ValueError("fine_to_coarse contains an out-of-range coarse index")

    n_coarse, n_crimes = target.shape
    coarse_point_risk = np.zeros((n_coarse, n_crimes), dtype=np.float64)
    np.add.at(coarse_point_risk, mapping, fine_risk)
    fine_count_by_coarse = np.bincount(mapping, minlength=n_coarse).astype(np.float64)

    calibrated = np.zeros_like(fine_risk)
    fallback_groups = 0
    for crime_index in range(n_crimes):
        denominator = coarse_point_risk[:, crime_index]
        regular = denominator > epsilon
        scale = np.zeros(n_coarse, dtype=np.float64)
        scale[regular] = target[regular, crime_index] / denominator[regular]
        calibrated[:, crime_index] = fine_risk[:, crime_index] * scale[mapping]

        fallback = (~regular) & (target[:, crime_index] > epsilon) & (fine_count_by_coarse > 0)
        if fallback.any():
            fallback_groups += int(fallback.sum())
            per_cell = np.zeros(n_coarse, dtype=np.float64)
            per_cell[fallback] = target[fallback, crime_index] / fine_count_by_coarse[fallback]
            fine_fallback = fallback[mapping]
            calibrated[fine_fallback, crime_index] = per_cell[mapping[fine_fallback]]

    allocated_coarse = np.zeros_like(target)
    np.add.at(allocated_coarse, mapping, calibrated)
    represented = fine_count_by_coarse > 0
    max_error = (
        float(np.max(np.abs(allocated_coarse[represented] - target[represented])))
        if represented.any()
        else 0.0
    )
    diagnostics = AllocationDiagnostics(
        point_process_total=float(fine_risk.sum()),
        calibrated_total=float(calibrated.sum()),
        coarse_target_total=float(target[represented].sum()),
        maximum_coarse_count_error=max_error,
        fallback_groups=fallback_groups,
    )
    return calibrated.astype(np.float32), diagnostics
