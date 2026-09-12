"""Invariant pooling utilities for edge features.

Provides SE(3)-invariant scalar reductions of edge quantities:
- 0e channels of edge_state
- L2 norm of non-scalar irrep blocks in message features
- (l,l → 0e) self-contraction (dot-product) for any l ≥ 1
- RBF histogram pooling
- Edge count / weight-sum statistics

All outputs are graph-level scalars of shape (num_graphs, feature_dim).
Empty edges → zero output (safe fallback).

Reduction semantics:
- ``sum``: sum_e w_e * f_e  (weighted sum)
- ``mean``: (sum_e w_e * f_e) / max(sum_e w_e, eps)  — TRUE weighted mean.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch
import torch.nn as nn
from e3nn import o3
from torch_scatter import scatter


__all__ = [
    "extract_scalar_channels",
    "extract_irrep_norms",
    "irrep_pair_invariant",
    "rbf_histogram_pool",
    "edge_count_pool",
    "InvariantPoolingBank",
    "weighted_scatter_reduce",
]


def weighted_scatter_reduce(
    values: torch.Tensor,          # (E, d) already multiplied by edge_weight if desired
    weights: torch.Tensor,         # (E,) per-edge weight (positive)
    batch_edge: torch.Tensor,      # (E,) graph index
    num_graphs: int,
    reduce: str,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Reduce (E, d) → (num_graphs, d) with sum or weighted-mean semantics.

    - ``sum``: scatter_sum(values, batch_edge). ``values`` is expected to
      already include any edge-weighting the caller wants.
    - ``mean``: (sum values) / max(sum weights, eps). This is a TRUE weighted
      mean; the previous behavior of ``scatter_mean`` gave an unweighted mean
      even though ``values`` had already been multiplied by weight — a bug.
    """
    if values.numel() == 0:
        return values.new_zeros(num_graphs, values.shape[-1] if values.dim() > 1 else 1)
    if reduce == "sum":
        return scatter(values, batch_edge, dim=0, dim_size=num_graphs, reduce="sum")
    if reduce == "mean":
        num = scatter(values, batch_edge, dim=0, dim_size=num_graphs, reduce="sum")
        den = scatter(weights, batch_edge, dim=0, dim_size=num_graphs, reduce="sum")
        den = den.clamp(min=eps).unsqueeze(-1)
        return num / den
    raise ValueError(f"Unknown reduce mode: {reduce!r} (use 'sum' or 'mean')")


def extract_scalar_channels(
    edge_state: torch.Tensor,
    edge_state_dim: int,
    n_scalars: Optional[int] = None,
) -> torch.Tensor:
    """Take the first n_scalars channels (interpret as 0e) of edge_state."""
    if edge_state.numel() == 0:
        return edge_state.new_zeros(0, n_scalars or edge_state_dim)
    n = n_scalars if n_scalars is not None else edge_state_dim
    return edge_state[:, :n]


def extract_irrep_norms(
    features: torch.Tensor,
    irreps: o3.Irreps,
) -> torch.Tensor:
    """Compute per-copy L2 norm of each l>0 irrep block. Norm is SE(3)-invariant."""
    if features.numel() == 0:
        n_norms = sum(mul for mul, ir in irreps if ir.l > 0)
        return features.new_zeros(0, n_norms)

    outputs = []
    offset = 0
    for mul, ir in irreps:
        dim_l = ir.dim
        if ir.l == 0:
            offset += mul * dim_l
            continue
        block = features[:, offset:offset + mul * dim_l].reshape(-1, mul, dim_l)
        norms = block.norm(dim=-1)
        outputs.append(norms)
        offset += mul * dim_l

    if not outputs:
        return features.new_zeros(features.shape[0], 0)
    return torch.cat(outputs, dim=-1)


def irrep_pair_invariant(
    features: torch.Tensor,
    irreps: o3.Irreps,
    target_l: int = 1,
) -> torch.Tensor:
    """Compute (target_l × target_l → 0e) self-contraction per copy.

    For any l ≥ 1, contracting an SH-basis vector with itself gives an
    SE(3)-invariant scalar (the squared norm along that l block). This
    generalizes the "1o · 1o" dot-product to arbitrary l.

    Args:
        features: (E, irreps.dim) equivariant features.
        irreps: layout.
        target_l: which l to contract (≥ 1).

    Returns:
        (E, num_copies_of_l) scalar invariants.
    """
    if features.numel() == 0:
        n = sum(mul for mul, ir in irreps if ir.l == target_l)
        return features.new_zeros(0, n)

    outputs = []
    offset = 0
    for mul, ir in irreps:
        dim_l = ir.dim
        if ir.l != target_l:
            offset += mul * dim_l
            continue
        block = features[:, offset:offset + mul * dim_l].reshape(-1, mul, dim_l)
        inv = (block * block).sum(dim=-1)
        outputs.append(inv)
        offset += mul * dim_l

    if not outputs:
        return features.new_zeros(features.shape[0], 0)
    return torch.cat(outputs, dim=-1)


def rbf_histogram_pool(
    edge_rbf: torch.Tensor,
    batch_edge: torch.Tensor,
    num_graphs: int,
    reduce: str = "sum",
) -> torch.Tensor:
    if edge_rbf.numel() == 0:
        return edge_rbf.new_zeros(num_graphs, edge_rbf.shape[-1] if edge_rbf.dim() > 1 else 1)
    return scatter(edge_rbf, batch_edge, dim=0, dim_size=num_graphs, reduce=reduce)


def edge_count_pool(
    batch_edge: torch.Tensor,
    num_graphs: int,
    edge_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    device = batch_edge.device
    if edge_weight is None:
        weights = torch.ones(batch_edge.shape[0], device=device)
    else:
        weights = edge_weight.to(device)
    return scatter(
        weights, batch_edge, dim=0, dim_size=num_graphs, reduce="sum"
    ).unsqueeze(-1)


# ---------------------------------------------------------------------------
# Feature spec parsing
# ---------------------------------------------------------------------------

_KNOWN_ATOMIC = {"0e", "norm", "rbf_hist", "edge_count"}


def _parse_feature_spec(features: Sequence[str]) -> List[dict]:
    """Parse feature-name list.

    Supports:
      "0e", "rbf_hist", "edge_count", "norm" (all l>0 irrep norms)
      "tp0" — legacy alias for tp_l=1 (kept for backward compat)
      "tp_l1", "tp_l2", ..., "tp_l{N}" — self-contraction of the l=N block
    """
    parsed = []
    for feat in features:
        if feat in _KNOWN_ATOMIC:
            parsed.append({"kind": feat})
        elif feat == "tp0":
            parsed.append({"kind": "tp_l", "l": 1})
        elif feat.startswith("tp_l"):
            l = int(feat[4:])
            if l < 1:
                raise ValueError(f"tp_l requires l>=1, got {feat}")
            parsed.append({"kind": "tp_l", "l": l})
        else:
            raise ValueError(f"Unknown invariant feature: {feat}")
    return parsed


class InvariantPoolingBank(nn.Module):
    """Container that computes and concatenates a configurable set of invariants.

    Args:
        features: list of names — {"0e", "norm", "tp0" (=tp_l1), "tp_l{L}",
            "rbf_hist", "edge_count"}.
        edge_state_dim: dim of the edge_state tensor.
        num_rbf: dim of edge_rbf.
        message_irreps: e3nn Irreps of the message tensor (for norm / tp_l).
            If None, dependent features contribute zero of the expected width.
        pool_reduce: "sum" | "mean". `mean` is a true weight-normalized mean.
    """

    def __init__(
        self,
        features: Sequence[str],
        edge_state_dim: int,
        num_rbf: int,
        message_irreps: Optional[o3.Irreps] = None,
        pool_reduce: str = "sum",
    ):
        super().__init__()
        self.features = list(features)
        self.parsed = _parse_feature_spec(self.features)
        self.edge_state_dim = edge_state_dim
        self.num_rbf = num_rbf
        self.message_irreps = message_irreps
        self.pool_reduce = pool_reduce

        # Per-feature dim
        widths: List[int] = []
        for spec in self.parsed:
            widths.append(self._feature_width(spec))
        self._widths = widths
        self.output_dim = sum(widths)

    def _feature_width(self, spec: dict) -> int:
        k = spec["kind"]
        if k == "0e":
            return self.edge_state_dim
        if k == "rbf_hist":
            return self.num_rbf
        if k == "edge_count":
            return 1
        if k == "norm":
            if self.message_irreps is None:
                return 0
            return sum(mul for mul, ir in self.message_irreps if ir.l > 0)
        if k == "tp_l":
            l = spec["l"]
            if self.message_irreps is None:
                return 0
            return sum(mul for mul, ir in self.message_irreps if ir.l == l)
        raise ValueError(f"Unknown kind: {k}")

    def per_edge_features(
        self,
        edge_state: Optional[torch.Tensor],
        edge_rbf: torch.Tensor,
        edge_weight: torch.Tensor,
        message_features: Optional[torch.Tensor],
        E: int,
        device,
    ) -> torch.Tensor:
        """Return (E, output_dim) BEFORE graph-level reduction."""
        parts = []
        for spec, width in zip(self.parsed, self._widths):
            k = spec["kind"]
            if width == 0:
                continue
            if k == "0e":
                if edge_state is None or edge_state.numel() == 0:
                    parts.append(torch.zeros(E, width, device=device))
                else:
                    parts.append(extract_scalar_channels(edge_state, self.edge_state_dim))
            elif k == "norm":
                if message_features is None:
                    parts.append(torch.zeros(E, width, device=device))
                else:
                    parts.append(extract_irrep_norms(message_features, self.message_irreps))
            elif k == "tp_l":
                if message_features is None:
                    parts.append(torch.zeros(E, width, device=device))
                else:
                    parts.append(irrep_pair_invariant(message_features, self.message_irreps, target_l=spec["l"]))
            elif k == "rbf_hist":
                parts.append(edge_rbf)
            elif k == "edge_count":
                parts.append(torch.ones(E, 1, device=device))
        if not parts:
            return torch.zeros(E, 0, device=device)
        return torch.cat(parts, dim=-1)

    def forward(
        self,
        edge_state: Optional[torch.Tensor],
        edge_rbf: torch.Tensor,
        edge_weight: torch.Tensor,
        batch_edge: torch.Tensor,
        num_graphs: int,
        message_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        E = batch_edge.shape[0]
        device = batch_edge.device
        if E == 0:
            return torch.zeros(num_graphs, self.output_dim, device=device)

        per_edge = self.per_edge_features(
            edge_state, edge_rbf, edge_weight, message_features, E, device
        )  # (E, D)
        weighted = per_edge * edge_weight.unsqueeze(-1)
        return weighted_scatter_reduce(
            weighted, edge_weight, batch_edge, num_graphs, reduce=self.pool_reduce
        )
