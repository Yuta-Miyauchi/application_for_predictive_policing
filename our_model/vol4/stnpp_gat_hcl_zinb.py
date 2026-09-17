"""HCL-regularized ZINB count branch for STNPP-GAT vol4.

Vol4 keeps vol3's division of labor: event-level GAT/ETAS produces the sharp
150 m allocation surface, while a 3 km daily model predicts crime-specific
count distributions. The coarse branch now applies HCL's crime-type,
cardinal-neighbor, and representation-level Hawkes correlations before
parameterizing a zero-inflated negative binomial distribution.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from ref_models.models.hcl import (
    HCLConfig,
    HawkesContrastiveHypergraph,
    hcl_correlation_losses,
)
from ref_models.models.stmgnn_zinb import ZINBOutput, zinb_nll


@dataclass(frozen=True)
class HCLZINBConfig:
    n_nodes: int
    n_crimes: int
    input_steps: int = 28
    hidden_dim: int = 16
    hypergraph_layers: int = 1
    dropout: float = 0.20
    hawkes_scope: int = 3
    hawkes_delta: float = 0.01
    min_probability: float = 1e-5
    min_dispersion: float = 1e-4
    max_dispersion: float = 1e4

    def encoder_config(self) -> HCLConfig:
        return HCLConfig(
            n_nodes=self.n_nodes,
            n_crimes=self.n_crimes,
            input_steps=self.input_steps,
            hidden_dim=self.hidden_dim,
            hypergraph_layers=self.hypergraph_layers,
            dropout=self.dropout,
            hawkes_scope=self.hawkes_scope,
            hawkes_delta=self.hawkes_delta,
        )


@dataclass(frozen=True)
class HCLZINBForward:
    distribution: ZINBOutput
    primitive: torch.Tensor
    enhanced: torch.Tensor


@dataclass(frozen=True)
class HCLZINBLoss:
    total: torch.Tensor
    nll: torch.Tensor
    type_contrast: torch.Tensor
    neighbor_contrast: torch.Tensor


class HCLZINBCountModel(nn.Module):
    """Compact HCL encoder with a distributional ZINB task head."""

    def __init__(
        self,
        config: HCLZINBConfig,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
    ):
        super().__init__()
        self.config = config
        self.encoder = HawkesContrastiveHypergraph(
            config.encoder_config(),
            edge_index=edge_index,
            edge_weight=edge_weight,
        )
        for parameter in self.encoder.output_head.parameters():
            parameter.requires_grad_(False)
        self.parameter_head = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 3),
        )

    def forward(self, features: torch.Tensor) -> HCLZINBForward:
        primitive = self.encoder.primitive_encode(features)
        enhanced = self.encoder.hawkes_enhance(primitive)
        attention = torch.softmax(self.encoder.temporal_score(enhanced), dim=2)
        history_context = torch.sum(attention * enhanced, dim=2)
        last_context = enhanced[:, :, -1]
        raw = self.parameter_head(
            torch.cat([last_context, history_context], dim=-1)
        ).unsqueeze(2)

        eps = self.config.min_probability
        pi = torch.sigmoid(raw[..., 0]).clamp(eps, 1.0 - eps)
        p = torch.sigmoid(raw[..., 1]).clamp(eps, 1.0 - eps)
        r = F.softplus(raw[..., 2]).clamp(
            self.config.min_dispersion,
            self.config.max_dispersion,
        )
        nb_mean = r * p / (1.0 - p)
        nb_variance = r * p / (1.0 - p).square()
        mean = (1.0 - pi) * nb_mean
        variance = (1.0 - pi) * nb_variance + pi * (1.0 - pi) * nb_mean.square()
        distribution = ZINBOutput(
            pi=pi,
            p=p,
            r=r,
            mean=mean,
            variance=variance,
        )
        return HCLZINBForward(
            distribution=distribution,
            primitive=primitive,
            enhanced=enhanced,
        )


def hcl_zinb_loss(
    output: HCLZINBForward,
    target: torch.Tensor,
    raw_history: torch.Tensor,
    neighbor_center: torch.Tensor,
    neighbor_node: torch.Tensor,
    category_coefficient: torch.Tensor,
    lambda_type: float,
    lambda_neighbor: float,
    contrast_steps: int = 1,
    reduction: str = "mean",
) -> HCLZINBLoss:
    """Combine direct ZINB NLL with HCL's two spatial alignment losses."""

    nll = zinb_nll(target, output.distribution, reduction=reduction)
    type_contrast, neighbor_contrast = hcl_correlation_losses(
        enhanced=output.enhanced,
        raw_history=raw_history,
        neighbor_center=neighbor_center,
        neighbor_node=neighbor_node,
        category_coefficient=category_coefficient,
        contrast_steps=contrast_steps,
    )
    total = nll + lambda_type * type_contrast + lambda_neighbor * neighbor_contrast
    return HCLZINBLoss(
        total=total,
        nll=nll,
        type_contrast=type_contrast,
        neighbor_contrast=neighbor_contrast,
    )
