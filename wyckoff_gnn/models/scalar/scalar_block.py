"""Scalar (0e) message-passing blocks for WyckoffCoGN0e.

Mirrors the coGN ProcessingBlock equation set but operates on the Wyckoff
quotient graph where nodes are symmetry-distinct orbits (not atoms).

Equations per block
-------------------
    edge_input = [e, h[target], h[src]]       # (E, 3*hidden)
    messages   = edge_mlp(edge_input)          # (E, hidden)
    agg        = scatter_sum(messages, target)  # (N, hidden)
    h_new      = h + node_mlp(agg)             # residual

Edge features are NOT updated between blocks (reuse initial edge state).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class GaussBasisExpansion(nn.Module):
    """Gaussian radial basis functions.

    Matches kgcnn ``GaussBasisExpansion.from_bounds(n, low, high, variance=1.0)``.
    Centers are at ``linspace(low, high, n+1)[1:]`` with sigma derived from spacing.
    """

    def __init__(self, mu: torch.Tensor, sigma: torch.Tensor):
        super().__init__()
        self.register_buffer("mu", mu.unsqueeze(0))
        self.register_buffer("sigma", sigma.unsqueeze(0))

    @classmethod
    def from_bounds(cls, n: int, low: float, high: float, variance: float = 1.0):
        mus = np.linspace(low, high, num=n + 1)
        var = np.diff(mus)
        mus = mus[1:]
        return cls(
            torch.tensor(mus, dtype=torch.float32),
            torch.tensor(np.sqrt(var * variance), dtype=torch.float32),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(-(x.unsqueeze(-1) - self.mu).pow(2) / (2 * self.sigma.pow(2)))


class ScalarProcessingBlock(nn.Module):
    """One scalar message-passing block.

    edge_mlp: [3*hidden -> hidden] -> hidden, 5 layers with SiLU
    node_mlp: [hidden -> hidden], 1 layer with SiLU
    Residual node update, no edge update.
    """

    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        edge_layers = []
        for i in range(5):
            in_dim = hidden_dim * 3 if i == 0 else hidden_dim
            edge_layers.append(nn.Linear(in_dim, hidden_dim, bias=True))
            edge_layers.append(nn.SiLU())
        self.edge_mlp = nn.Sequential(*edge_layers)
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.SiLU(),
        )

    def forward(self, e, h, edge_index):
        target, src = edge_index[0], edge_index[1]
        edge_input = torch.cat([e, h[target], h[src]], dim=-1)
        messages = self.edge_mlp(edge_input)
        agg = torch.zeros(h.size(0), e.size(1), device=h.device, dtype=h.dtype)
        agg.index_add_(0, target, messages)
        h = h + self.node_mlp(agg)
        return h
