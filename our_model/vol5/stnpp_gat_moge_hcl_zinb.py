"""ST-MoGE-enhanced HCL-ZINB count branch for STNPP-GAT vol5.

Vol5 keeps vol4's multiscale boundary. The 3 km branch uses the HCL encoder as
a universal expert and adds category-specific spatial-temporal graph experts,
attentive spatial gates, regional ZINB predictors, and hierarchical adaptive
loss re-weighting. The 150 m GAT/ETAS branch remains outside this module.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from ref_models.models.hcl import HCLConfig, HawkesContrastiveHypergraph
from ref_models.models.hcl import hcl_correlation_losses
from ref_models.models.st_moge.st_moge import (
    AttentiveSpatialGate,
    STMOGEConfig,
    SpatialTemporalGraphExpert,
)
from ref_models.models.stmgnn_zinb import ZINBOutput, zinb_log_prob


@dataclass(frozen=True)
class MoGEHCLZINBConfig:
    n_nodes: int
    n_crimes: int
    input_steps: int = 28
    hidden_dim: int = 16
    hypergraph_layers: int = 1
    hcl_dropout: float = 0.20
    hawkes_scope: int = 3
    hawkes_delta: float = 0.01
    node_embedding_dim: int = 16
    specific_st_blocks: int = 1
    specific_spatial_layers: int = 1
    specific_temporal_layers: int = 2
    temporal_kernel_size: int = 3
    attention_heads: int = 4
    n_clusters: int = 4
    specific_dropout: float = 0.10
    min_probability: float = 1e-5
    min_dispersion: float = 1e-4
    max_dispersion: float = 1e4

    def hcl_config(self) -> HCLConfig:
        return HCLConfig(
            n_nodes=self.n_nodes,
            n_crimes=self.n_crimes,
            input_steps=self.input_steps,
            hidden_dim=self.hidden_dim,
            hypergraph_layers=self.hypergraph_layers,
            dropout=self.hcl_dropout,
            hawkes_scope=self.hawkes_scope,
            hawkes_delta=self.hawkes_delta,
        )

    def specific_expert_config(self) -> STMOGEConfig:
        return STMOGEConfig(
            n_nodes=self.n_nodes,
            n_crimes=self.n_crimes,
            input_steps=self.input_steps,
            hidden_dim=self.hidden_dim,
            node_embedding_dim=self.node_embedding_dim,
            st_blocks=self.specific_st_blocks,
            spatial_layers=self.specific_spatial_layers,
            temporal_layers=self.specific_temporal_layers,
            temporal_kernel_size=self.temporal_kernel_size,
            attention_heads=self.attention_heads,
            n_clusters=self.n_clusters,
            dropout=self.specific_dropout,
        )


@dataclass(frozen=True)
class MoGEHCLZINBForward:
    distribution: ZINBOutput
    primitive: torch.Tensor
    enhanced: torch.Tensor
    specific: torch.Tensor
    universal: torch.Tensor
    gate: torch.Tensor


@dataclass(frozen=True)
class MoGEHCLZINBLoss:
    total: torch.Tensor
    weighted_nll: torch.Tensor
    unweighted_nll: torch.Tensor
    type_contrast: torch.Tensor
    neighbor_contrast: torch.Tensor
    cluster_nll: torch.Tensor


class MoGEHCLZINBCountModel(nn.Module):
    """HCL universal expert fused with category-specific graph experts."""

    def __init__(
        self,
        config: MoGEHCLZINBConfig,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
    ):
        super().__init__()
        if config.hidden_dim % config.attention_heads:
            raise ValueError("hidden_dim must be divisible by attention_heads")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, n_edges]")
        if edge_index.shape[1] != edge_weight.numel():
            raise ValueError("edge_index and edge_weight must describe the same edges")

        prior = torch.zeros(config.n_nodes, config.n_nodes, dtype=torch.float32)
        source, target = edge_index.long()
        prior[target, source] = edge_weight.float()

        self.config = config
        self.universal_encoder = HawkesContrastiveHypergraph(
            config.hcl_config(),
            edge_index=edge_index,
            edge_weight=edge_weight,
        )
        for parameter in self.universal_encoder.output_head.parameters():
            parameter.requires_grad_(False)
        self.universal_projection = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(config.hidden_dim),
        )

        expert_config = config.specific_expert_config()
        self.specific_experts = nn.ModuleList(
            SpatialTemporalGraphExpert(
                expert_config,
                input_channels=1,
                prior_adjacency=prior,
            )
            for _ in range(config.n_crimes)
        )
        self.gates = nn.ModuleList(
            AttentiveSpatialGate(
                config.hidden_dim,
                config.attention_heads,
                config.specific_dropout,
            )
            for _ in range(config.n_crimes)
        )
        self.predictors = nn.ModuleList(
            nn.ModuleList(
                nn.Sequential(
                    nn.Linear(config.hidden_dim, config.hidden_dim),
                    nn.GELU(),
                    nn.Dropout(config.specific_dropout),
                    nn.Linear(config.hidden_dim, 3),
                )
                for _ in range(config.n_clusters)
            )
            for _ in range(config.n_crimes)
        )
        self.register_buffer(
            "cluster_assignments",
            torch.arange(config.n_nodes).view(1, -1).repeat(config.n_crimes, 1)
            % config.n_clusters,
        )

    def set_cluster_assignments(self, assignments: torch.Tensor) -> None:
        expected = (self.config.n_crimes, self.config.n_nodes)
        if tuple(assignments.shape) != expected:
            raise ValueError(f"Expected cluster assignments {expected}")
        if assignments.min() < 0 or assignments.max() >= self.config.n_clusters:
            raise ValueError("Cluster assignment is outside the configured range")
        self.cluster_assignments.copy_(assignments.to(self.cluster_assignments.device))

    def category_node_embeddings(self) -> torch.Tensor:
        return torch.stack(
            [expert.node_embedding_first for expert in self.specific_experts],
            dim=0,
        )

    def _distribution(self, raw: torch.Tensor) -> ZINBOutput:
        eps = self.config.min_probability
        pi = torch.sigmoid(raw[..., 0]).clamp(eps, 1.0 - eps).unsqueeze(2)
        p = torch.sigmoid(raw[..., 1]).clamp(eps, 1.0 - eps).unsqueeze(2)
        r = F.softplus(raw[..., 2]).clamp(
            self.config.min_dispersion,
            self.config.max_dispersion,
        ).unsqueeze(2)
        nb_mean = r * p / (1.0 - p)
        nb_variance = r * p / (1.0 - p).square()
        mean = (1.0 - pi) * nb_mean
        variance = (1.0 - pi) * nb_variance + pi * (1.0 - pi) * nb_mean.square()
        return ZINBOutput(
            pi=pi,
            p=p,
            r=r,
            mean=mean,
            variance=variance,
        )

    def forward(self, features: torch.Tensor) -> MoGEHCLZINBForward:
        expected = (
            self.config.n_nodes,
            self.config.input_steps,
            self.config.n_crimes,
        )
        if tuple(features.shape[1:]) != expected:
            raise ValueError(f"Expected [batch, {expected}], got {tuple(features.shape)}")

        primitive = self.universal_encoder.primitive_encode(features)
        enhanced = self.universal_encoder.hawkes_enhance(primitive)
        attention = torch.softmax(
            self.universal_encoder.temporal_score(enhanced),
            dim=2,
        )
        history_context = torch.sum(attention * enhanced, dim=2)
        last_context = enhanced[:, :, -1]
        universal = self.universal_projection(
            torch.cat([last_context, history_context], dim=-1)
        )

        specific = torch.stack(
            [
                expert(features[..., crime : crime + 1])
                for crime, expert in enumerate(self.specific_experts)
            ],
            dim=1,
        )
        fused_parts = []
        gate_parts = []
        for crime, gate_layer in enumerate(self.gates):
            fused, gate = gate_layer(specific[:, crime], universal[:, :, crime])
            fused_parts.append(fused)
            gate_parts.append(gate.mean(dim=-1))
        fused_all = torch.stack(fused_parts, dim=1)
        gate_all = torch.stack(gate_parts, dim=-1)

        raw_parts = []
        for crime in range(self.config.n_crimes):
            candidates = torch.stack(
                [
                    predictor(fused_all[:, crime])
                    for predictor in self.predictors[crime]
                ],
                dim=2,
            )
            selected = torch.gather(
                candidates,
                dim=2,
                index=self.cluster_assignments[crime]
                .view(1, -1, 1, 1)
                .expand(features.shape[0], -1, 1, 3),
            ).squeeze(2)
            raw_parts.append(selected)
        raw = torch.stack(raw_parts, dim=2)
        return MoGEHCLZINBForward(
            distribution=self._distribution(raw),
            primitive=primitive,
            enhanced=enhanced,
            specific=specific,
            universal=universal.permute(0, 2, 1, 3),
            gate=gate_all,
        )


def regional_zinb_nll(
    output: MoGEHCLZINBForward,
    target: torch.Tensor,
    cluster_assignments: torch.Tensor,
    category_weights: torch.Tensor,
    cluster_weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return HALR-weighted, global, and category/cluster ZINB NLL values."""

    element = -zinb_log_prob(
        target,
        output.distribution.pi,
        output.distribution.p,
        output.distribution.r,
    ).squeeze(2)
    cluster_losses = torch.zeros(
        element.shape[-1],
        cluster_weights.shape[1],
        dtype=element.dtype,
        device=element.device,
    )
    for crime in range(element.shape[-1]):
        for cluster in range(cluster_weights.shape[1]):
            mask = cluster_assignments[crime] == cluster
            if torch.any(mask):
                cluster_losses[crime, cluster] = element[:, mask, crime].mean()
    category_losses = torch.sum(cluster_weights * cluster_losses, dim=1)
    weighted = torch.sum(category_weights * category_losses)
    return weighted, element.mean(), cluster_losses


def moge_hcl_zinb_loss(
    output: MoGEHCLZINBForward,
    target: torch.Tensor,
    raw_history: torch.Tensor,
    neighbor_center: torch.Tensor,
    neighbor_node: torch.Tensor,
    category_coefficient: torch.Tensor,
    cluster_assignments: torch.Tensor,
    category_weights: torch.Tensor,
    cluster_weights: torch.Tensor,
    lambda_type: float,
    lambda_neighbor: float,
    contrast_steps: int = 1,
) -> MoGEHCLZINBLoss:
    """Combine regional HALR ZINB likelihood with vol4's HCL losses."""

    weighted_nll, unweighted_nll, cluster_nll = regional_zinb_nll(
        output,
        target,
        cluster_assignments,
        category_weights,
        cluster_weights,
    )
    type_contrast, neighbor_contrast = hcl_correlation_losses(
        enhanced=output.enhanced,
        raw_history=raw_history,
        neighbor_center=neighbor_center,
        neighbor_node=neighbor_node,
        category_coefficient=category_coefficient,
        contrast_steps=contrast_steps,
    )
    total = (
        weighted_nll
        + lambda_type * type_contrast
        + lambda_neighbor * neighbor_contrast
    )
    return MoGEHCLZINBLoss(
        total=total,
        weighted_nll=weighted_nll,
        unweighted_nll=unweighted_nll,
        type_contrast=type_contrast,
        neighbor_contrast=neighbor_contrast,
        cluster_nll=cluster_nll,
    )
