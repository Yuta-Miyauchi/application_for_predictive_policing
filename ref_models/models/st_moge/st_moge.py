"""Paper-constrained ST-MoGE for multi-type crime quantity prediction.

The implementation follows Wu et al.'s MGE, attentive spatial gate, CECL, and
HALR equations. The paper does not publish source code and leaves a few
operational details unspecified; those choices are exposed by the experiment
and recorded with its outputs.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class STMOGEConfig:
    n_nodes: int
    n_crimes: int
    input_steps: int = 7
    hidden_dim: int = 32
    node_embedding_dim: int = 16
    st_blocks: int = 3
    spatial_layers: int = 2
    temporal_layers: int = 3
    temporal_kernel_size: int = 3
    attention_heads: int = 4
    n_clusters: int = 4
    dropout: float = 0.10


@dataclass(frozen=True)
class STMOGEForward:
    prediction: torch.Tensor
    specific: torch.Tensor
    universal: torch.Tensor
    gate: torch.Tensor
    auxiliary_first: torch.Tensor | None
    auxiliary_second: torch.Tensor | None


class DilatedCausalTemporalLayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ):
        super().__init__()
        self.left_padding = dilation * (kernel_size - 1)
        self.convolution = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=kernel_size,
            dilation=dilation,
        )
        self.normalization = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        batch, nodes, steps, hidden = values.shape
        sequence = values.permute(0, 1, 3, 2).reshape(batch * nodes, hidden, steps)
        convolved = self.convolution(F.pad(sequence, (self.left_padding, 0)))
        convolved = self.dropout(F.relu(self.normalization(convolved)))
        convolved = convolved.reshape(batch, nodes, hidden, steps).permute(0, 1, 3, 2)
        return values + convolved


class AdaptiveGraphLayer(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.adaptive_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.prior_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(hidden_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        values: torch.Tensor,
        adaptive_adjacency: torch.Tensor,
        prior_adjacency: torch.Tensor,
    ) -> torch.Tensor:
        adaptive_context = torch.einsum(
            "ij,bjtd->bitd", adaptive_adjacency, values
        )
        prior_context = torch.einsum("ij,bjtd->bitd", prior_adjacency, values)
        update = F.relu(
            self.adaptive_projection(adaptive_context)
            + self.prior_projection(prior_context)
            + self.bias
        )
        return values + self.dropout(update)


class SpatialTemporalBlock(nn.Module):
    def __init__(self, config: STMOGEConfig):
        super().__init__()
        self.temporal = nn.ModuleList(
            DilatedCausalTemporalLayer(
                hidden_dim=config.hidden_dim,
                kernel_size=config.temporal_kernel_size,
                dilation=2**layer,
                dropout=config.dropout,
            )
            for layer in range(config.temporal_layers)
        )
        self.spatial = nn.ModuleList(
            AdaptiveGraphLayer(config.hidden_dim, config.dropout)
            for _ in range(config.spatial_layers)
        )
        self.skip_projections = nn.ModuleList(
            nn.Linear(config.hidden_dim, config.hidden_dim)
            for _ in range(config.temporal_layers)
        )

    def forward(
        self,
        values: torch.Tensor,
        adaptive_adjacency: torch.Tensor,
        prior_adjacency: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = values
        skip_total = torch.zeros_like(hidden)
        for index, temporal_layer in enumerate(self.temporal):
            hidden = temporal_layer(hidden)
            skip_total = skip_total + self.skip_projections[index](hidden)
            if index < len(self.spatial):
                hidden = self.spatial[index](
                    hidden,
                    adaptive_adjacency,
                    prior_adjacency,
                )
        return hidden, skip_total


class SpatialTemporalGraphExpert(nn.Module):
    def __init__(
        self,
        config: STMOGEConfig,
        input_channels: int,
        prior_adjacency: torch.Tensor,
    ):
        super().__init__()
        self.input_projection = nn.Linear(input_channels, config.hidden_dim)
        self.node_embedding_first = nn.Parameter(
            torch.empty(config.n_nodes, config.node_embedding_dim)
        )
        self.node_embedding_second = nn.Parameter(
            torch.empty(config.n_nodes, config.node_embedding_dim)
        )
        nn.init.xavier_uniform_(self.node_embedding_first)
        nn.init.xavier_uniform_(self.node_embedding_second)
        self.blocks = nn.ModuleList(
            SpatialTemporalBlock(config) for _ in range(config.st_blocks)
        )
        self.output_normalization = nn.LayerNorm(config.hidden_dim)
        self.register_buffer("prior_adjacency", prior_adjacency.float())

    def adaptive_adjacency(self) -> torch.Tensor:
        scores = F.relu(self.node_embedding_first @ self.node_embedding_second.T)
        return torch.softmax(scores, dim=-1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden = self.input_projection(features)
        adaptive = self.adaptive_adjacency()
        skip_total = torch.zeros_like(hidden)
        for block in self.blocks:
            hidden, skip = block(hidden, adaptive, self.prior_adjacency)
            skip_total = skip_total + skip
        return self.output_normalization(F.relu(skip_total[:, :, -1]))


class AttentiveSpatialGate(nn.Module):
    def __init__(self, hidden_dim: int, attention_heads: int, dropout: float):
        super().__init__()
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim,
            attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.specific_gate = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.universal_gate = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.gate_bias = nn.Parameter(torch.zeros(hidden_dim))
        self.normalization = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        specific: torch.Tensor,
        universal: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        selected, _ = self.cross_attention(
            query=specific,
            key=universal,
            value=universal,
            need_weights=False,
        )
        selected = self.normalization(universal + selected)
        gate = torch.sigmoid(
            self.specific_gate(specific)
            + self.universal_gate(selected)
            + self.gate_bias
        )
        fused = gate * specific + (1.0 - gate) * selected
        return fused, gate


class SpatialTemporalMixtureOfGraphExperts(nn.Module):
    def __init__(
        self,
        config: STMOGEConfig,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
    ):
        super().__init__()
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, n_edges]")
        if edge_index.shape[1] != edge_weight.numel():
            raise ValueError("edge_index and edge_weight must describe the same edges")
        if config.hidden_dim % config.attention_heads:
            raise ValueError("hidden_dim must be divisible by attention_heads")

        prior = torch.zeros(config.n_nodes, config.n_nodes, dtype=torch.float32)
        source, target = edge_index.long()
        prior[target, source] = edge_weight.float()
        self.config = config
        self.specific_experts = nn.ModuleList(
            SpatialTemporalGraphExpert(config, input_channels=1, prior_adjacency=prior)
            for _ in range(config.n_crimes)
        )
        self.universal_expert = SpatialTemporalGraphExpert(
            config,
            input_channels=config.n_crimes,
            prior_adjacency=prior,
        )
        self.gates = nn.ModuleList(
            AttentiveSpatialGate(
                config.hidden_dim,
                config.attention_heads,
                config.dropout,
            )
            for _ in range(config.n_crimes)
        )
        self.predictors = nn.ModuleList(
            nn.ModuleList(
                nn.Sequential(
                    nn.Linear(config.hidden_dim, config.hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(config.dropout),
                    nn.Linear(config.hidden_dim, 1),
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
            [expert.node_embedding_first for expert in self.specific_experts], dim=0
        )

    def _auxiliary_views(
        self,
        features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        masked = []
        for crime in range(self.config.n_crimes):
            category = torch.zeros_like(features)
            category[..., crime] = features[..., crime]
            masked.append(category)
        stacked = torch.cat(masked, dim=0)
        corrupted = self.universal_expert(stacked)
        first = F.dropout(corrupted, p=self.config.dropout, training=True)
        second = F.dropout(corrupted, p=self.config.dropout, training=True)
        batch = features.shape[0]
        first = first.reshape(self.config.n_crimes, batch, self.config.n_nodes, -1)
        second = second.reshape(self.config.n_crimes, batch, self.config.n_nodes, -1)
        return first.permute(1, 0, 2, 3), second.permute(1, 0, 2, 3)

    def forward(
        self,
        features: torch.Tensor,
        compute_contrastive: bool = True,
    ) -> STMOGEForward:
        expected = (
            self.config.n_nodes,
            self.config.input_steps,
            self.config.n_crimes,
        )
        if tuple(features.shape[1:]) != expected:
            raise ValueError(f"Expected [batch, {expected}], got {tuple(features.shape)}")

        specific = torch.stack(
            [
                expert(features[..., crime : crime + 1])
                for crime, expert in enumerate(self.specific_experts)
            ],
            dim=1,
        )
        universal = self.universal_expert(features)
        fused_parts = []
        gate_parts = []
        for crime, gate_layer in enumerate(self.gates):
            fused, gate = gate_layer(specific[:, crime], universal)
            fused_parts.append(fused)
            gate_parts.append(gate.mean(dim=-1))
        fused_all = torch.stack(fused_parts, dim=1)
        gate_all = torch.stack(gate_parts, dim=-1)

        prediction_parts = []
        for crime in range(self.config.n_crimes):
            candidates = torch.stack(
                [
                    predictor(fused_all[:, crime]).squeeze(-1)
                    for predictor in self.predictors[crime]
                ],
                dim=-1,
            )
            selected = torch.gather(
                candidates,
                dim=2,
                index=self.cluster_assignments[crime]
                .view(1, -1, 1)
                .expand(features.shape[0], -1, 1),
            ).squeeze(-1)
            prediction_parts.append(F.softplus(selected))
        prediction = torch.stack(prediction_parts, dim=-1)

        auxiliary_first = None
        auxiliary_second = None
        if compute_contrastive:
            auxiliary_first, auxiliary_second = self._auxiliary_views(features)
        return STMOGEForward(
            prediction=prediction,
            specific=specific,
            universal=universal,
            gate=gate_all,
            auxiliary_first=auxiliary_first,
            auxiliary_second=auxiliary_second,
        )


def cecl_loss(
    output: STMOGEForward,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stable InfoNCE completion of the paper's Equations (1)-(2)."""

    if output.auxiliary_first is None or output.auxiliary_second is None:
        raise ValueError("CECL requires auxiliary representations")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    first = F.normalize(output.auxiliary_first, dim=-1)
    second = F.normalize(output.auxiliary_second, dim=-1)
    specific = F.normalize(output.specific, dim=-1)
    universal = F.normalize(output.universal, dim=-1)
    specific_losses = []
    universal_losses = []
    for crime in range(first.shape[1]):
        positive = torch.sum(first[:, crime] * second[:, crime], dim=-1).mean()
        specific_negative = torch.stack(
            [
                torch.sum(first[:, crime] * specific[:, other], dim=-1).mean()
                for other in range(specific.shape[1])
            ]
        )
        universal_negative = torch.stack(
            [
                torch.sum(first[:, other] * universal, dim=-1).mean()
                for other in range(first.shape[1])
            ]
        )
        specific_logits = torch.cat([positive.view(1), specific_negative]) / temperature
        universal_logits = torch.cat([positive.view(1), universal_negative]) / temperature
        target = torch.zeros(1, dtype=torch.long, device=first.device)
        specific_losses.append(
            F.cross_entropy(specific_logits.view(1, -1), target)
        )
        universal_losses.append(
            F.cross_entropy(universal_logits.view(1, -1), target)
        )
    return torch.stack(specific_losses).mean(), torch.stack(universal_losses).mean()


def cluster_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    cluster_assignments: torch.Tensor,
    n_clusters: int,
) -> torch.Tensor:
    losses = torch.zeros(
        prediction.shape[-1],
        n_clusters,
        dtype=prediction.dtype,
        device=prediction.device,
    )
    squared = (prediction - target).square()
    for crime in range(prediction.shape[-1]):
        for cluster in range(n_clusters):
            mask = cluster_assignments[crime] == cluster
            if torch.any(mask):
                losses[crime, cluster] = squared[:, mask, crime].mean()
    return losses


def hierarchical_weighted_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    cluster_assignments: torch.Tensor,
    category_weights: torch.Tensor,
    cluster_weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    losses = cluster_mse(
        prediction,
        target,
        cluster_assignments,
        cluster_weights.shape[1],
    )
    weighted = category_weights * torch.sum(cluster_weights * losses, dim=1)
    return weighted.sum(), losses
