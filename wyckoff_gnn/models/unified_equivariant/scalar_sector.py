"""Embedding layers for the unified quotient-equivariant model.

Provides InputEmbedding (atom embed + node linear) and EdgeEmbedding
(RBF expansion + edge linear) as separate modules so that the model
state_dict has clean, unified key paths:

    input_embedding.atom_embed.*
    input_embedding.node_input.*
    edge_embedding.edge_rbf.*
    edge_embedding.edge_input.*
"""

from __future__ import annotations

import torch
import torch.nn as nn

from wyckoff_gnn.models.scalar.scalar_block import GaussBasisExpansion
from wyckoff_gnn.models.scalar.node_embedding import AtomEmbedding


class InputEmbedding(nn.Module):
    """Node embedding: AtomEmbedding(z) -> Linear -> scalar features."""

    def __init__(self, scalar_mul: int = 128, num_rbf: int = 32):
        super().__init__()
        self.atom_embed = AtomEmbedding(scalar_mul)
        node_input_dim = scalar_mul + 4 + 14
        self.node_input = nn.Linear(node_input_dim, scalar_mul, bias=True)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.node_input(self.atom_embed(z))


class EdgeEmbedding(nn.Module):
    """Edge embedding: GaussRBF(d) -> Linear -> scalar edge features."""

    def __init__(self, num_rbf: int = 32, rbf_max: float = 8.0,
                 scalar_mul: int = 128):
        super().__init__()
        self.edge_rbf = GaussBasisExpansion.from_bounds(
            num_rbf, 0.0, rbf_max, variance=1.0
        )
        self.edge_input = nn.Linear(num_rbf, scalar_mul, bias=True)

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        return self.edge_input(self.edge_rbf(distances))
