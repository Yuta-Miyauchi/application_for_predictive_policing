"""PyTorch Transformer v1 model for weekly cell-level crime forecasting."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class TransformerV1Config:
    context_weeks: int = 52
    d_model: int = 32
    n_heads: int = 4
    n_layers: int = 1
    dim_feedforward: int = 96
    dropout: float = 0.1


class WeeklyCellTransformer(nn.Module):
    """A compact Transformer encoder over per-cell weekly count sequences.

    Inputs are deliberately simple for v1:
    - one token per historical week containing log1p(count)
    - learned positional embeddings
    - learned area embeddings
    - normalized cell centroid coordinates
    - forecast-week seasonal sin/cos features

    The output is a nonnegative expected event count for the next week in the
    cell. This is not the full STNPP architecture yet; it is a practical first
    Transformer-backed version of the current weekly GIF experiment.
    """

    def __init__(self, n_areas: int, config: TransformerV1Config):
        super().__init__()
        self.config = config
        self.count_projection = nn.Linear(1, config.d_model)
        self.position_embedding = nn.Parameter(torch.zeros(1, config.context_weeks, config.d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layers)
        self.area_embedding = nn.Embedding(n_areas, config.d_model)
        self.static_projection = nn.Linear(4, config.d_model)
        self.output = nn.Sequential(
            nn.LayerNorm(config.d_model * 3),
            nn.Linear(config.d_model * 3, config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, 1),
        )
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)

    def forward(
        self,
        history_counts: torch.Tensor,
        area_index: torch.Tensor,
        static_features: torch.Tensor,
        week_features: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.log1p(history_counts).unsqueeze(-1)
        tokens = self.count_projection(x) + self.position_embedding[:, : x.shape[1], :]
        encoded = self.encoder(tokens)
        sequence_state = encoded[:, -1, :]
        area_state = self.area_embedding(area_index)
        static_state = self.static_projection(torch.cat([static_features, week_features], dim=1))
        raw = self.output(torch.cat([sequence_state, area_state, static_state], dim=1)).squeeze(1)
        return F.softplus(raw) + 1e-6


def weighted_poisson_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    loss = prediction - target * torch.log(prediction)
    if sample_weight is not None:
        weight = sample_weight / sample_weight.mean().clamp_min(1e-6)
        loss = loss * weight
    return loss.mean()
