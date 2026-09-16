"""GAT-based marked point-process utilities inspired by STNPP.

The paper model learns mark-to-mark excitation coefficients with a graph
attention network. This module keeps that central idea in a compact form that
can be combined with the LAPD 150m grid experiment: marks are crime type x LAPD
division, columns of the learned transition matrix sum to one, and predictions
use a marked self-exciting intensity.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class STNPPGATConfig:
    n_crimes: int
    n_areas: int
    hidden_dim: int = 64
    attention_heads: int = 8
    dropout: float = 0.05

    @property
    def n_marks(self) -> int:
        return self.n_crimes * self.n_areas


class MarkGraphAttention(nn.Module):
    """Multi-head graph attention over crime-area marks."""

    def __init__(self, config: STNPPGATConfig):
        super().__init__()
        self.config = config
        in_dim = config.n_crimes + config.n_areas + 2
        self.projections = nn.ModuleList(
            [nn.Linear(in_dim, config.hidden_dim, bias=False) for _ in range(config.attention_heads)]
        )
        self.attn_left = nn.Parameter(torch.empty(config.attention_heads, config.hidden_dim))
        self.attn_right = nn.Parameter(torch.empty(config.attention_heads, config.hidden_dim))
        self.dropout = nn.Dropout(config.dropout)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for proj in self.projections:
            nn.init.xavier_uniform_(proj.weight)
        nn.init.xavier_uniform_(self.attn_left)
        nn.init.xavier_uniform_(self.attn_right)

    def forward(self, mark_features: torch.Tensor) -> torch.Tensor:
        """Return P[target_mark, source_mark] with each source column summing to one."""

        head_probs = []
        for head, proj in enumerate(self.projections):
            h = self.dropout(F.elu(proj(mark_features)))
            left = h @ self.attn_left[head]
            right = h @ self.attn_right[head]
            logits = F.leaky_relu(left[:, None] + right[None, :], negative_slope=0.2)
            head_probs.append(torch.softmax(logits, dim=0))
        return torch.stack(head_probs, dim=0).mean(dim=0)


def transition_kl_loss(
    predicted: torch.Tensor,
    empirical: torch.Tensor,
    source_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Column-wise KL loss between empirical and predicted transition matrices."""

    eps = 1e-8
    empirical = empirical.clamp_min(eps)
    predicted = predicted.clamp_min(eps)
    kl_by_source = (empirical * (empirical.log() - predicted.log())).sum(dim=0)
    if source_weight is None:
        return kl_by_source.mean()
    weight = source_weight / source_weight.mean().clamp_min(eps)
    return (kl_by_source * weight).mean()
