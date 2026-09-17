"""Paper-constrained HCL implementation for daily crime quantity prediction.

Liang et al. (AAAI 2024) define HCL as a plug-in around a spatial-temporal
hypergraph backbone. The paper specifies its Hawkes representation update and
two correlation losses, but does not publish the HCL code or all backbone
details. This module implements those HCL-specific equations directly and uses
a compact structural hypergraph encoder for reproducible CPU experiments.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class HCLConfig:
    n_nodes: int
    n_crimes: int
    input_steps: int = 30
    hidden_dim: int = 16
    hypergraph_layers: int = 1
    dropout: float = 0.20
    hawkes_scope: int = 3
    hawkes_delta: float = 0.01


@dataclass(frozen=True)
class HCLForward:
    prediction: torch.Tensor
    primitive: torch.Tensor
    enhanced: torch.Tensor


@dataclass(frozen=True)
class HCLLoss:
    total: torch.Tensor
    task: torch.Tensor
    type_contrast: torch.Tensor
    neighbor_contrast: torch.Tensor


class StructuralHypergraphLayer(nn.Module):
    """Message passing over crime-type and cardinal-neighborhood hyperedges."""

    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.self_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.type_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.neighbor_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(hidden_dim))
        self.normalization = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        values: torch.Tensor,
        edge_source: torch.Tensor,
        edge_target: torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> torch.Tensor:
        type_context = values.mean(dim=3, keepdim=True).expand_as(values)
        messages = values[:, edge_source] * edge_weight.view(1, -1, 1, 1, 1)
        neighbor_context = torch.zeros_like(values)
        neighbor_context.index_add_(1, edge_target, messages)
        update = (
            self.self_projection(values)
            + self.type_projection(type_context)
            + self.neighbor_projection(neighbor_context)
            + self.bias
        )
        return self.normalization(values + self.dropout(F.gelu(update)))


class HawkesContrastiveHypergraph(nn.Module):
    """HCL quantity model with an explicit representation-level Hawkes update."""

    def __init__(
        self,
        config: HCLConfig,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
    ):
        super().__init__()
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, n_edges]")
        if edge_index.shape[1] != edge_weight.numel():
            raise ValueError("edge_index and edge_weight must have matching edges")
        if config.hawkes_scope < 1 or config.hawkes_scope >= config.input_steps:
            raise ValueError("hawkes_scope must be in [1, input_steps)")

        self.config = config
        self.value_projection = nn.Linear(1, config.hidden_dim)
        self.node_embedding = nn.Embedding(config.n_nodes, config.hidden_dim)
        self.crime_embedding = nn.Embedding(config.n_crimes, config.hidden_dim)
        self.time_embedding = nn.Embedding(config.input_steps, config.hidden_dim)
        self.input_normalization = nn.LayerNorm(config.hidden_dim)
        self.layers = nn.ModuleList(
            StructuralHypergraphLayer(config.hidden_dim, config.dropout)
            for _ in range(config.hypergraph_layers)
        )
        self.temporal_score = nn.Linear(config.hidden_dim, 1)
        self.output_head = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 1),
        )
        self.register_buffer("edge_source", edge_index[0].long())
        self.register_buffer("edge_target", edge_index[1].long())
        self.register_buffer("edge_weight", edge_weight.float())

    def primitive_encode(self, features: torch.Tensor) -> torch.Tensor:
        expected = (
            self.config.n_nodes,
            self.config.input_steps,
            self.config.n_crimes,
        )
        if tuple(features.shape[1:]) != expected:
            raise ValueError(
                f"Expected [batch, {expected[0]}, {expected[1]}, {expected[2]}], "
                f"got {tuple(features.shape)}"
            )
        nodes = self.node_embedding.weight.view(1, self.config.n_nodes, 1, 1, -1)
        crimes = self.crime_embedding.weight.view(1, 1, 1, self.config.n_crimes, -1)
        times = self.time_embedding.weight.view(1, 1, self.config.input_steps, 1, -1)
        hidden = self.value_projection(features.unsqueeze(-1)) + nodes + crimes + times
        hidden = self.input_normalization(hidden)
        for layer in self.layers:
            hidden = layer(hidden, self.edge_source, self.edge_target, self.edge_weight)
        return hidden

    def hawkes_enhance(self, primitive: torch.Tensor) -> torch.Tensor:
        enhanced = primitive.clone()
        denominator = float(self.config.hawkes_scope + 1)
        for lag in range(1, self.config.hawkes_scope + 1):
            weight = math.exp(-float(lag + 1) / denominator)
            enhanced[:, :, lag:] = enhanced[:, :, lag:] + (
                self.config.hawkes_delta * weight * primitive[:, :, :-lag]
            )
        return enhanced

    def forward(self, features: torch.Tensor) -> HCLForward:
        primitive = self.primitive_encode(features)
        enhanced = self.hawkes_enhance(primitive)
        attention = torch.softmax(self.temporal_score(enhanced), dim=2)
        history_context = torch.sum(attention * enhanced, dim=2)
        last_context = enhanced[:, :, -1]
        raw_prediction = self.output_head(
            torch.cat([last_context, history_context], dim=-1)
        ).squeeze(-1)
        prediction = F.softplus(raw_prediction)
        return HCLForward(
            prediction=prediction,
            primitive=primitive,
            enhanced=enhanced,
        )


def _normalized_squared_distance(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left = F.normalize(left, p=2, dim=-1, eps=1e-8)
    right = F.normalize(right, p=2, dim=-1, eps=1e-8)
    return torch.sum((left - right).square(), dim=-1)


def _type_alignment_loss(
    enhanced: torch.Tensor,
    raw_history: torch.Tensor,
    contrast_steps: int,
) -> torch.Tensor:
    hidden = enhanced[:, :, -contrast_steps:]
    present = raw_history[:, :, -contrast_steps:] > 0
    absent = ~present
    present_count = present.sum(dim=3)
    absent_count = absent.sum(dim=3)
    present_pool = torch.sum(hidden * present.unsqueeze(-1), dim=3) / present_count.clamp_min(
        1
    ).unsqueeze(-1)
    absent_pool = torch.sum(hidden * absent.unsqueeze(-1), dim=3) / absent_count.clamp_min(
        1
    ).unsqueeze(-1)
    valid = (present_count > 0) & (absent_count > 0)
    if not torch.any(valid):
        return hidden.sum() * 0.0
    return _normalized_squared_distance(present_pool, absent_pool)[valid].mean()


def _neighbor_alignment_loss(
    enhanced: torch.Tensor,
    raw_history: torch.Tensor,
    neighbor_center: torch.Tensor,
    neighbor_node: torch.Tensor,
    category_coefficient: torch.Tensor,
    contrast_steps: int,
) -> torch.Tensor:
    hidden = enhanced[:, :, -contrast_steps:]
    raw = raw_history[:, :, -contrast_steps:]
    neighbor_hidden = hidden[:, neighbor_node]
    neighbor_absent = raw[:, neighbor_node] <= 0

    pooled = torch.zeros_like(hidden)
    pooled.index_add_(
        1,
        neighbor_center,
        neighbor_hidden * neighbor_absent.unsqueeze(-1),
    )
    counts = torch.zeros_like(raw)
    counts.index_add_(1, neighbor_center, neighbor_absent.to(raw.dtype))
    pooled = pooled / counts.clamp_min(1).unsqueeze(-1)

    anchor_present = raw > 0
    valid = anchor_present & (counts > 0)
    if not torch.any(valid):
        return hidden.sum() * 0.0
    distance = _normalized_squared_distance(hidden, pooled)
    weighted = distance * category_coefficient.view(1, 1, 1, -1)
    return weighted[valid].mean()


def hcl_quantity_loss(
    output: HCLForward,
    target: torch.Tensor,
    raw_history: torch.Tensor,
    neighbor_center: torch.Tensor,
    neighbor_node: torch.Tensor,
    category_coefficient: torch.Tensor,
    lambda_type: float,
    lambda_neighbor: float,
    contrast_steps: int = 1,
) -> HCLLoss:
    """Equation (9) with ST-HSL-style MSE quantity loss."""

    type_contrast, neighbor_contrast = hcl_correlation_losses(
        enhanced=output.enhanced,
        raw_history=raw_history,
        neighbor_center=neighbor_center,
        neighbor_node=neighbor_node,
        category_coefficient=category_coefficient,
        contrast_steps=contrast_steps,
    )
    task = F.mse_loss(output.prediction, target)
    total = task + lambda_type * type_contrast + lambda_neighbor * neighbor_contrast
    return HCLLoss(
        total=total,
        task=task,
        type_contrast=type_contrast,
        neighbor_contrast=neighbor_contrast,
    )


def hcl_correlation_losses(
    enhanced: torch.Tensor,
    raw_history: torch.Tensor,
    neighbor_center: torch.Tensor,
    neighbor_node: torch.Tensor,
    category_coefficient: torch.Tensor,
    contrast_steps: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the type and neighbor losses from HCL Equations (10)-(11)."""

    if contrast_steps < 1 or contrast_steps > raw_history.shape[2]:
        raise ValueError("contrast_steps must fit inside the input history")
    type_contrast = _type_alignment_loss(
        enhanced, raw_history, contrast_steps
    )
    neighbor_contrast = _neighbor_alignment_loss(
        enhanced,
        raw_history,
        neighbor_center,
        neighbor_node,
        category_coefficient,
        contrast_steps,
    )
    return type_contrast, neighbor_contrast
