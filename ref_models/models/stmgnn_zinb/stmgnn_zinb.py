"""Paper-constrained implementation of STMGNN-ZINB.

The model follows the architecture described by Wang et al. (2024): a
diffusion graph convolution branch captures spatial dependence, a gated
multivariate-temporal branch mixes all crime types and historical time steps,
and the two branches are fused with Hadamard products to parameterize a
zero-inflated negative binomial distribution.

The paper does not publish layer widths or implementation code. Those choices
therefore live in :class:`STMGNNZINBConfig` instead of being presented as
paper-specified settings.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class STMGNNZINBConfig:
    n_nodes: int
    n_crimes: int
    input_steps: int = 28
    output_steps: int = 1
    hidden_dim: int = 32
    graph_layers: int = 2
    temporal_hidden_steps: int = 14
    dropout: float = 0.10
    min_probability: float = 1e-5
    min_dispersion: float = 1e-4
    max_dispersion: float = 1e4


@dataclass(frozen=True)
class ZINBOutput:
    pi: torch.Tensor
    p: torch.Tensor
    r: torch.Tensor
    mean: torch.Tensor
    variance: torch.Tensor


class DiffusionGraphLayer(nn.Module):
    """D^-1 A H W plus a learned self-state transform."""

    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.neighbor = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.self_state = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        source: torch.Tensor,
        target: torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> torch.Tensor:
        messages = x[:, source] * edge_weight.view(1, -1, 1)
        diffused = torch.zeros_like(x)
        diffused.index_add_(1, target, messages)
        update = self.neighbor(diffused) + self.self_state(x)
        return self.dropout(F.relu(update))


class SpatialParameterBranch(nn.Module):
    """Encode the full history and diffuse it over the spatial graph."""

    def __init__(self, config: STMGNNZINBConfig):
        super().__init__()
        input_dim = config.input_steps * config.n_crimes
        output_dim = config.output_steps * config.n_crimes * 3
        self.input_projection = nn.Linear(input_dim, config.hidden_dim)
        self.layers = nn.ModuleList(
            [
                DiffusionGraphLayer(config.hidden_dim, config.dropout)
                for _ in range(config.graph_layers)
            ]
        )
        self.output_projection = nn.Linear(config.hidden_dim, output_dim)
        self.output_steps = config.output_steps
        self.n_crimes = config.n_crimes

    def forward(
        self,
        x: torch.Tensor,
        source: torch.Tensor,
        target: torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> torch.Tensor:
        batch, nodes, _, _ = x.shape
        hidden = F.relu(self.input_projection(x.reshape(batch, nodes, -1)))
        for layer in self.layers:
            hidden = layer(hidden, source, target, edge_weight)
        raw = self.output_projection(hidden)
        return raw.view(batch, nodes, self.output_steps, self.n_crimes, 3)


class GatedMultivariateTemporalLayer(nn.Module):
    """Shared gated projection over flattened time and crime dimensions."""

    def __init__(self, in_features: int, out_features: int, dropout: float):
        super().__init__()
        self.candidate = nn.Linear(in_features, out_features)
        self.gate = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = torch.tanh(self.candidate(x)) * torch.sigmoid(self.gate(x))
        return self.dropout(hidden)


class MultivariateTemporalParameterBranch(nn.Module):
    """MTCN branch that jointly mixes historical steps and crime types."""

    def __init__(self, config: STMGNNZINBConfig):
        super().__init__()
        input_dim = config.input_steps * config.n_crimes
        hidden_dim = config.temporal_hidden_steps * config.n_crimes
        output_dim = config.output_steps * config.n_crimes
        self.shared = GatedMultivariateTemporalLayer(input_dim, hidden_dim, config.dropout)
        self.parameter_heads = nn.ModuleList(
            [
                GatedMultivariateTemporalLayer(hidden_dim, output_dim, config.dropout)
                for _ in range(3)
            ]
        )
        self.output_steps = config.output_steps
        self.n_crimes = config.n_crimes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, nodes, _, _ = x.shape
        hidden = self.shared(x.reshape(batch, nodes, -1))
        params = [
            head(hidden).view(batch, nodes, self.output_steps, self.n_crimes)
            for head in self.parameter_heads
        ]
        return torch.stack(params, dim=-1)


class STMGNNZINB(nn.Module):
    """Spatial-temporal multivariate graph network with a ZINB output."""

    def __init__(
        self,
        config: STMGNNZINBConfig,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
    ):
        super().__init__()
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, n_edges]")
        if edge_index.shape[1] != edge_weight.numel():
            raise ValueError("edge_index and edge_weight must describe the same edges")
        self.config = config
        self.spatial = SpatialParameterBranch(config)
        self.temporal = MultivariateTemporalParameterBranch(config)
        self.register_buffer("edge_source", edge_index[0].long())
        self.register_buffer("edge_target", edge_index[1].long())
        self.register_buffer("edge_weight", edge_weight.float())

    def forward(self, x: torch.Tensor) -> ZINBOutput:
        expected_shape = (self.config.n_nodes, self.config.input_steps, self.config.n_crimes)
        if tuple(x.shape[1:]) != expected_shape:
            raise ValueError(
                f"Expected input [batch, {expected_shape[0]}, {expected_shape[1]}, "
                f"{expected_shape[2]}], got {tuple(x.shape)}"
            )
        spatial = self.spatial(
            x,
            self.edge_source,
            self.edge_target,
            self.edge_weight,
        )
        temporal = self.temporal(x)

        eps = self.config.min_probability
        pi = (torch.sigmoid(spatial[..., 0]) * torch.sigmoid(temporal[..., 0])).clamp(
            eps, 1.0 - eps
        )
        p = (torch.sigmoid(spatial[..., 1]) * torch.sigmoid(temporal[..., 1])).clamp(
            eps, 1.0 - eps
        )
        r = (
            F.softplus(spatial[..., 2]) * F.softplus(temporal[..., 2])
        ).clamp(self.config.min_dispersion, self.config.max_dispersion)

        nb_mean = r * p / (1.0 - p)
        nb_variance = r * p / (1.0 - p).square()
        mean = (1.0 - pi) * nb_mean
        variance = (1.0 - pi) * nb_variance + pi * (1.0 - pi) * nb_mean.square()
        return ZINBOutput(pi=pi, p=p, r=r, mean=mean, variance=variance)


def zinb_log_prob(
    target: torch.Tensor,
    pi: torch.Tensor,
    p: torch.Tensor,
    r: torch.Tensor,
) -> torch.Tensor:
    """Element-wise log probability under the paper's ZINB parameterization."""

    target = target.to(dtype=pi.dtype)
    log_pi = torch.log(pi)
    log_one_minus_pi = torch.log1p(-pi)
    log_p = torch.log(p)
    log_one_minus_p = torch.log1p(-p)

    nb_log_prob = (
        torch.lgamma(target + r)
        - torch.lgamma(r)
        - torch.lgamma(target + 1.0)
        + target * log_p
        + r * log_one_minus_p
    )
    zero_log_prob = torch.logaddexp(log_pi, log_one_minus_pi + r * log_one_minus_p)
    return torch.where(target == 0, zero_log_prob, log_one_minus_pi + nb_log_prob)


def zinb_nll(target: torch.Tensor, output: ZINBOutput, reduction: str = "mean") -> torch.Tensor:
    """Negative log likelihood used as Equation (3) in the paper."""

    loss = -zinb_log_prob(target, output.pi, output.p, output.r)
    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    if reduction == "mean":
        return loss.mean()
    raise ValueError(f"Unsupported reduction: {reduction}")
