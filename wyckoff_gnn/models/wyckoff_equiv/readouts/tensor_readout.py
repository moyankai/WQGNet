"""Tensor property head for arbitrary O(3) irrep outputs.

Handles graph-level prediction of scalars, vectors, and rank-2 tensors.

**Scope (Phase 4)**: this head is a *linear* (per-node) equivariant projection
+ weighted-sum pooling + a second linear projection to the target irrep. It is
SO(3)- (and O(3)-) equivariant by construction, and its output transforms as
declared by ``target_irreps`` under a rotation of the *input* node features.

**Non-goals / current limitations** (documented explicitly so downstream
callers do not assume more than the head delivers):

* No graph-level crystal point-group projection is applied. The
  ``apply_global_symmetry`` constructor argument is retained as an interface
  slot but setting it to ``True`` will raise ``NotImplementedError`` — a
  silent no-op would be misleading given the constructor name. Enforcement of
  crystal point-group covariance must be added in Phase 4.1.
* Because the mapping ``h → out`` factors through two ``o3.Linear`` operators
  with a scalar-multiplied sum in the middle, the head is *fully linear* in
  the equivariant features. It cannot represent nonlinear tensor functions.
* Predicted vectors/tensors are unconstrained beyond irrep type. In
  particular, positivity, symmetry classes stronger than the target irrep
  decomposition (e.g. Cauchy-symmetric elasticity, Voigt structure), and
  point-group projections are the caller's responsibility.
"""

from __future__ import annotations

from typing import Optional, Union

import torch
import torch.nn as nn
from e3nn import o3
from torch_scatter import scatter


__all__ = ["TensorPropertyHead"]


class TensorPropertyHead(nn.Module):
    """Predict arbitrary O(3) tensor properties from per-node equivariant features.

    Args:
        irreps_node_in: e3nn Irreps of node features (e.g., "64x0e+32x1o+16x2e").
        target_irreps: e3nn Irreps of the output tensor (e.g., "1x1o" for polar vector,
            "1x0e+1x2e" for symmetric rank-2 tensor).
        pool: "sum" | "mean" — how to aggregate per-node features to graph level.
        hidden_multiplier: intermediate irreps width factor (multiplies each irrep's mul).
        apply_global_symmetry: **Not implemented**. Passing ``True`` raises
            ``NotImplementedError``; ``False`` (default) is a no-op that
            explicitly declares no graph-level point-group projection is
            applied. See module docstring.
    """

    def __init__(
        self,
        irreps_node_in: Union[str, o3.Irreps],
        target_irreps: Union[str, o3.Irreps],
        pool: str = "sum",
        hidden_multiplier: int = 2,
        apply_global_symmetry: bool = False,
    ):
        super().__init__()
        self.irreps_in = o3.Irreps(irreps_node_in) if isinstance(irreps_node_in, str) else irreps_node_in
        self.target_irreps = o3.Irreps(target_irreps) if isinstance(target_irreps, str) else target_irreps
        self.pool = pool
        if apply_global_symmetry:
            raise NotImplementedError(
                "TensorPropertyHead.apply_global_symmetry=True is reserved for "
                "Phase 4.1 (graph-level crystal point-group projection) and is "
                "not yet implemented. Pass apply_global_symmetry=False and "
                "handle point-group covariance externally."
            )
        self.apply_global_symmetry = False

        # Build intermediate irreps: expand each target irrep by hidden_multiplier.
        # If target = "1x0e+1x2e", hidden = "2x0e+2x2e" (for multiplier=2).
        hidden_parts = []
        for mul, ir in self.target_irreps:
            hidden_parts.append((mul * hidden_multiplier, ir))
        self.irreps_hidden = o3.Irreps(hidden_parts)

        # Two-stage equivariant projection:
        #   Linear1: irreps_in -> irreps_hidden (per-node)
        #   Linear2: irreps_hidden -> target_irreps (per-node, after pooling)
        self.linear_in = o3.Linear(self.irreps_in, self.irreps_hidden)
        self.linear_out = o3.Linear(self.irreps_hidden, self.target_irreps)

    def forward(
        self,
        h: torch.Tensor,
        batch: torch.Tensor,
        node_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict target tensor from per-node features.

        Args:
            h: (K, irreps_in.dim) node features.
            batch: (K,) graph index per node.
            node_weights: optional (K,) per-node weight (e.g., Wyckoff multiplicity).

        Returns:
            If pool != "none": (num_graphs, target_irreps.dim).
            If pool == "none": (K, target_irreps.dim) — per-node output.
        """
        # Per-node equivariant transform
        h_hidden = self.linear_in(h)  # (K, irreps_hidden.dim)

        if self.pool == "none":
            return self.linear_out(h_hidden)

        # Weighted equivariant pooling to graph level
        num_graphs = int(batch.max().item()) + 1 if h.shape[0] > 0 else 0
        if node_weights is None:
            w = torch.ones(h.shape[0], device=h.device, dtype=h.dtype)
        else:
            w = node_weights.to(dtype=h.dtype)

        h_weighted = h_hidden * w.unsqueeze(-1)
        if self.pool == "sum":
            h_graph = scatter(h_weighted, batch, dim=0, dim_size=num_graphs, reduce="sum")
        elif self.pool == "mean":
            h_graph = scatter(h_weighted, batch, dim=0, dim_size=num_graphs, reduce="sum")
            norm = scatter(w, batch, dim=0, dim_size=num_graphs, reduce="sum").clamp(min=1e-6)
            h_graph = h_graph / norm.unsqueeze(-1)
        else:
            raise ValueError(f"Unknown pool: {self.pool}")

        # Final equivariant projection to target
        out = self.linear_out(h_graph)  # (num_graphs, target_irreps.dim)
        return out
