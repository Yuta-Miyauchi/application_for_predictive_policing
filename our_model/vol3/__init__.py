"""Vol3: uncertainty-calibrated multiscale STNPP-GAT."""

from .stnpp_gat_zinb import (
    AllocationDiagnostics,
    CountFusionDiagnostics,
    MultiscaleSTNPPGATZINB,
    MultiscaleSTNPPGATZINBConfig,
    allocate_coarse_counts_to_fine_cells,
    uncertainty_weighted_count_fusion,
)

__all__ = [
    "AllocationDiagnostics",
    "CountFusionDiagnostics",
    "MultiscaleSTNPPGATZINB",
    "MultiscaleSTNPPGATZINBConfig",
    "allocate_coarse_counts_to_fine_cells",
    "uncertainty_weighted_count_fusion",
]
