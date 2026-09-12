"""EdgeState-v1: Scalar invariant edge state for WyckoffGNN.

OPTIONAL EXPERIMENTAL BRANCH.
Not used unless use_edge_state=True in config.
Default mp20_lite/mp20_full configs set use_edge_state=False.

The edge state is a per-sub-edge scalar vector that:
1. Is initialized from invariant edge features (RBF, weight, etc.)
2. Conditions the dynamic TP weight MLP (concatenated with edge_rbf)
3. Gets updated after each MP layer from invariant message summaries
4. Participates in readout via edge pooling

All edge state values are SE(3)-invariant scalars.
"""

import torch
import torch.nn as nn
from torch_scatter import scatter


class EdgeStateInit(nn.Module):
    """Initialize scalar edge state from invariant edge features."""

    def __init__(self, input_dim: int, edge_state_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, edge_state_dim * 2),
            nn.SiLU(),
            nn.Linear(edge_state_dim * 2, edge_state_dim),
        )

    def forward(self, edge_features: torch.Tensor) -> torch.Tensor:
        return self.mlp(edge_features)


class EdgeStateUpdate(nn.Module):
    """Update edge state from invariant message summary + current state."""

    def __init__(self, edge_state_dim: int, msg_inv_dim: int, num_rbf: int):
        super().__init__()
        input_dim = edge_state_dim + msg_inv_dim + num_rbf
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, edge_state_dim * 2),
            nn.SiLU(),
            nn.Linear(edge_state_dim * 2, edge_state_dim),
        )
        self.norm = nn.LayerNorm(edge_state_dim)

    def forward(
        self,
        edge_state: torch.Tensor,
        msg_invariant: torch.Tensor,
        edge_rbf: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([edge_state, msg_invariant, edge_rbf], dim=-1)
        delta = self.mlp(x)
        return self.norm(edge_state + delta)


class InvariantMessageSummary(nn.Module):
    """Extract SE(3)-invariant summary from equivariant messages.

    Extracts:
    - l=0 scalar components directly
    - Per-copy norm for l>0 components
    This ensures strict invariance: rotation does not change the output.
    """

    def __init__(self, irreps):
        super().__init__()
        from e3nn import o3
        self.irreps = o3.Irreps(irreps) if isinstance(irreps, str) else irreps
        self._output_dim = self._compute_output_dim()

    def _compute_output_dim(self) -> int:
        dim = 0
        for mul, ir in self.irreps:
            if ir.l == 0:
                dim += mul
            else:
                dim += mul
        return dim

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, msg: torch.Tensor) -> torch.Tensor:
        parts = []
        offset = 0
        for mul, ir in self.irreps:
            block_dim = mul * ir.dim
            block = msg[:, offset:offset + block_dim]
            if ir.l == 0:
                parts.append(block)
            else:
                block_reshaped = block.reshape(block.size(0), mul, ir.dim)
                norms = block_reshaped.norm(dim=-1)
                parts.append(norms)
            offset += block_dim
        return torch.cat(parts, dim=-1)


class EdgePool(nn.Module):
    """Pool edge states to graph-level invariant representation."""

    def __init__(self, edge_state_dim: int, pool_mode: str = "normalized"):
        super().__init__()
        self.pool_mode = pool_mode
        self.edge_state_dim = edge_state_dim

    def forward(
        self,
        edge_state: torch.Tensor,
        edge_weight: torch.Tensor,
        batch_edge: torch.Tensor,
        num_graphs: int,
    ) -> torch.Tensor:
        if self.pool_mode == "sum":
            weighted = edge_state * edge_weight.unsqueeze(-1)
            return scatter(weighted, batch_edge, dim=0, dim_size=num_graphs, reduce="sum")
        elif self.pool_mode == "mean":
            weighted = edge_state * edge_weight.unsqueeze(-1)
            pooled = scatter(weighted, batch_edge, dim=0, dim_size=num_graphs, reduce="sum")
            counts = scatter(edge_weight, batch_edge, dim=0, dim_size=num_graphs, reduce="sum")
            return pooled / counts.unsqueeze(-1).clamp(min=1e-8)
        else:
            weighted = edge_state * edge_weight.unsqueeze(-1)
            pooled = scatter(weighted, batch_edge, dim=0, dim_size=num_graphs, reduce="sum")
            norm = scatter(edge_weight.pow(2), batch_edge, dim=0, dim_size=num_graphs, reduce="sum").sqrt()
            return pooled / norm.unsqueeze(-1).clamp(min=1e-8)
