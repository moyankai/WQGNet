"""Tensor readout head: produce symmetric rank-2 tensor from node features.

Produces a per-graph 1x0e + 1x2e prediction via:
  - 0e: Linear(scalar_node) -> scalar, then multiplicity-weighted pool
  - 2e: o3.Linear(high_irreps -> 1x2e), then multiplicity-weighted pool

The pooling is equivariant because multiplicity is an invariant scalar.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from e3nn import o3

from .tensor_adapter import SymmetricRank2Adapter


class SymmetricRank2Readout(nn.Module):
    """Readout head for symmetric rank-2 tensor prediction.

    Args:
        scalar_mul: number of scalar (0e) channels.
        high_irreps: high-l irreps string (must contain at least one 2e term).
        adapter: optional SymmetricRank2Adapter instance. Created if None.
    """

    def __init__(
        self,
        scalar_mul: int,
        high_irreps: str,
        adapter: SymmetricRank2Adapter | None = None,
    ):
        super().__init__()

        high = o3.Irreps(high_irreps)
        has_2e = any(ir.l == 2 for _, ir in high)
        if not has_2e:
            raise ValueError(
                f"hidden_irreps must contain at least one 2e term for tensor "
                f"readout, got '{high_irreps}'"
            )

        self.high_irreps = high
        self.scalar_0e_head = nn.Linear(scalar_mul, 1, bias=True)
        self.tensor_2e_head = o3.Linear(high, "1x2e")
        self.adapter = adapter or SymmetricRank2Adapter()

    def forward(
        self,
        scalar_node: torch.Tensor,
        high_l_node: torch.Tensor,
        multiplicity: torch.Tensor,
        batch: torch.Tensor,
        num_graphs: int,
    ) -> dict:
        """Compute per-graph symmetric rank-2 tensor.

        Args:
            scalar_node: (N, scalar_mul) scalar node features.
            high_l_node: (N, dim_high) high-l node features.
            multiplicity: (N,) orbit multiplicities.
            batch: (N,) node-to-graph assignment.
            num_graphs: number of graphs in batch.

        Returns:
            dict with:
                "irreps": (G, 6) irreps-space tensor [0e(1) + 2e(5)]
                "cartesian": (G, 3, 3) symmetric Cartesian tensor
        """
        mult = multiplicity.float()

        # --- 0e: trace / isotropic part ---
        scalar_per_node = self.scalar_0e_head(scalar_node).squeeze(-1)  # (N,)
        pooled_0e = _equivariant_pool(scalar_per_node, mult, batch, num_graphs)

        # --- 2e: deviatoric / traceless-symmetric part ---
        per_node_2e = self.tensor_2e_head(high_l_node)  # (N, 5)
        pooled_2e = _equivariant_pool_vec(per_node_2e, mult, batch, num_graphs)

        irreps_out = torch.cat([pooled_0e.unsqueeze(-1), pooled_2e], dim=-1)  # (G, 6)
        cartesian_out = self.adapter.to_cartesian(irreps_out)  # (G, 3, 3)

        return {
            "irreps": irreps_out,
            "cartesian": cartesian_out,
        }


def _equivariant_pool(
    values: torch.Tensor,
    multiplicity: torch.Tensor,
    batch: torch.Tensor,
    num_graphs: int,
) -> torch.Tensor:
    """Multiplicity-weighted mean pool for scalar values.

    g = sum_p(m_p * v_p) / sum_p(m_p)
    """
    weighted = values * multiplicity
    num = values.new_zeros(num_graphs)
    num.index_add_(0, batch, weighted)
    den = values.new_zeros(num_graphs)
    den.index_add_(0, batch, multiplicity)
    return num / den.clamp(min=1e-8)


def _equivariant_pool_vec(
    values: torch.Tensor,
    multiplicity: torch.Tensor,
    batch: torch.Tensor,
    num_graphs: int,
) -> torch.Tensor:
    """Multiplicity-weighted mean pool for equivariant vector values.

    Same formula as scalar pool but preserves the last dimension.
    """
    mult = multiplicity.unsqueeze(-1)  # (N, 1)
    weighted = values * mult  # (N, D)
    num = values.new_zeros(num_graphs, values.size(-1))
    num.index_add_(0, batch, weighted)
    den = multiplicity.new_zeros(num_graphs, 1)
    den.index_add_(0, batch, multiplicity.unsqueeze(-1))
    return num / den.clamp(min=1e-8)


class PiezoRank3Readout(nn.Module):
    """Readout head for rank-3 piezoelectric tensor prediction.

    The piezoelectric stress tensor e_{ijk} (symmetric in j, k) decomposes as
    ``2x1o + 1x2o + 1x3o`` (18 components).  Two of those sectors, 2o and 3o,
    are absent from the backbone's natural-parity hidden state
    (``0e, 1o, 2e, 3o, ...``): 2o is not a natural-parity sector at all.

    They are recovered here without touching the backbone, using the physical
    decomposition e ~ 1o (x) (0e + 2e), i.e. a tensor product of the existing
    ``1o`` and ``2e`` channels:

        1o (x) 2e  ->  1o + 2o + 3o

    A single ``FullyConnectedTensorProduct`` therefore spans the whole target
    space, and the head stays exactly equivariant under O(3), including parity.

    Args:
        scalar_mul: number of scalar (0e) channels (unused by the tensor path,
            kept for interface symmetry with the rank-2 head).
        high_irreps: high-l irreps string; must contain at least one 1o and
            one 2e term.
        adapter: optional Rank3PiezoAdapter instance. Created if None.
    """

    def __init__(
        self,
        scalar_mul: int,
        high_irreps: str,
        adapter=None,
    ):
        super().__init__()

        high = o3.Irreps(high_irreps)
        self.irreps_1o = o3.Irreps([(mul, ir) for mul, ir in high if ir.l == 1 and ir.p == -1])
        self.irreps_2e = o3.Irreps([(mul, ir) for mul, ir in high if ir.l == 2 and ir.p == 1])
        if self.irreps_1o.dim == 0 or self.irreps_2e.dim == 0:
            raise ValueError(
                f"piezo_rank3 readout needs at least one 1o and one 2e term in "
                f"hidden_irreps, got '{high_irreps}'"
            )

        self.high_irreps = high
        self._slice_1o = self._sector_slice(high, l=1, p=-1)
        self._slice_2e = self._sector_slice(high, l=2, p=1)

        from .tensor_adapter import Rank3PiezoAdapter
        self.adapter = adapter or Rank3PiezoAdapter()
        self.target_irreps = o3.Irreps(self.adapter.irreps_out)

        self.tp = o3.FullyConnectedTensorProduct(
            self.irreps_1o, self.irreps_2e, self.target_irreps,
            shared_weights=True,
        )

    @staticmethod
    def _sector_slice(irreps: o3.Irreps, l: int, p: int) -> slice:
        """Contiguous slice covering every (l, p) sector of ``irreps``.

        e3nn keeps sectors in declaration order, so a hidden state written as
        ``16x1o + 8x2e`` has one contiguous block per sector.
        """
        start = end = None
        offset = 0
        for mul, ir in irreps:
            width = mul * ir.dim
            if ir.l == l and ir.p == p:
                if start is None:
                    start = offset
                end = offset + width
            offset += width
        if start is None:
            raise ValueError(f"irreps {irreps} has no ({l}, {p}) sector")
        return slice(start, end)

    def forward(
        self,
        scalar_node: torch.Tensor,
        high_l_node: torch.Tensor,
        multiplicity: torch.Tensor,
        batch: torch.Tensor,
        num_graphs: int,
    ) -> dict:
        """Compute per-graph rank-3 piezoelectric tensor.

        Args:
            scalar_node: (N, scalar_mul) scalar node features (unused).
            high_l_node: (N, dim_high) high-l node features.
            multiplicity: (N,) orbit multiplicities.
            batch: (N,) node-to-graph assignment.
            num_graphs: number of graphs in batch.

        Returns:
            dict with:
                "irreps": (G, 18) irreps-space tensor
                "voigt": (G, 3, 6) Voigt-form tensor (GMTNet ordering)
                "cartesian": (G, 3, 3, 3) full Cartesian tensor
        """
        x_1o = high_l_node[:, self._slice_1o]
        x_2e = high_l_node[:, self._slice_2e]

        per_node = self.tp(x_1o, x_2e)  # (N, 18)
        mult = multiplicity.to(dtype=per_node.dtype, device=per_node.device)
        pooled = _equivariant_pool_vec(per_node, mult, batch, num_graphs)

        return {
            "irreps": pooled,
            "voigt": self.adapter.irreps_to_voigt(pooled),
            "cartesian": self.adapter.to_cartesian(pooled),
        }
