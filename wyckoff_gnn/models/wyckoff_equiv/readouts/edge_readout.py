"""Edge readout modules for Wyckoff GNN.

Two pooling granularities:
- **Sub-edge pooling** (default): pool all sub-edges directly to graph level.
- **Orbit-pair pooling**: aggregate sub-edges into orbit-pair tokens first,
  then pool orbit-pairs to graph level.

Both produce (num_graphs, edge_pool_dim) that is concatenated with node-level
features in the readout MLP.

Hardening notes (Part C):
- OrbitPairPooling no longer inflates a dense (num_nodes^2, d) tensor. It only
  materializes the pair tokens that actually appear in the batch, which
  removes MLP bias contributions from nonexistent pairs and cuts memory to
  O(num_unique_pairs · d).
- `EdgeInvariantReadout` no longer requires `edge_state`; if the user only
  selects features that need `edge_rbf`/`edge_count`/messages, the pipeline
  works with `edge_state=None`.
- `message_features` (per-edge message tensor) can now be plumbed in for
  `norm` and `tp_l{L}` invariants.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
from torch_scatter import scatter

from wyckoff_gnn.models.wyckoff_equiv.readouts.invariant_pooling import (
    InvariantPoolingBank,
    weighted_scatter_reduce,
)


__all__ = [
    "EdgeInvariantReadout",
    "SubEdgePooling",
    "OrbitPairPooling",
]


class SubEdgePooling(nn.Module):
    """Pool sub-edge invariants directly to graph level."""

    def __init__(self, bank: InvariantPoolingBank):
        super().__init__()
        self.bank = bank

    @property
    def output_dim(self) -> int:
        return self.bank.output_dim

    def forward(
        self,
        edge_state: Optional[torch.Tensor],
        edge_rbf: torch.Tensor,
        edge_weight: torch.Tensor,
        batch_edge: torch.Tensor,
        num_graphs: int,
        message_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.bank(
            edge_state=edge_state,
            edge_rbf=edge_rbf,
            edge_weight=edge_weight,
            batch_edge=batch_edge,
            num_graphs=num_graphs,
            message_features=message_features,
        )


class OrbitPairPooling(nn.Module):
    """Aggregate sub-edges into orbit-pair tokens, then pool orbit-pairs to graph.

    Two-stage reduction:
        G_{pq} = sum_{e ∈ (p,q)} w_e * f_e         (per-pair scatter)
        z_{pq} = MLP(G_{pq})                        (per-pair transform)
        g     = sum_{(p,q) ∈ graph} z_{pq}          (per-graph scatter)

    Compared to the previous implementation, this version:
    * only materializes the pair tokens actually observed in the batch (never
      the dense num_nodes^2 tensor);
    * therefore the per-pair MLP's bias only contributes to pairs that exist,
      eliminating a spurious bias-per-graph contribution proportional to
      num_nodes^2.

    Args:
        bank: InvariantPoolingBank producing per-edge invariant features.
        pair_hidden_dim: intermediate dim of per-pair MLP.
        pair_output_dim: final graph-level output dim.
    """

    def __init__(
        self,
        bank: InvariantPoolingBank,
        pair_hidden_dim: int = 64,
        pair_output_dim: int = 64,
    ):
        super().__init__()
        self.bank = bank
        self.pair_hidden_dim = pair_hidden_dim
        self.pair_output_dim = pair_output_dim
        self.pair_mlp = nn.Sequential(
            nn.Linear(bank.output_dim, pair_hidden_dim),
            nn.SiLU(),
            nn.Linear(pair_hidden_dim, pair_output_dim),
        )

    @property
    def output_dim(self) -> int:
        return self.pair_output_dim

    def forward(
        self,
        edge_state: Optional[torch.Tensor],
        edge_rbf: torch.Tensor,
        edge_weight: torch.Tensor,
        batch_edge: torch.Tensor,
        num_graphs: int,
        geo_edge_index: torch.Tensor,
        num_nodes: int,
        message_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        E = batch_edge.shape[0]
        device = batch_edge.device
        if E == 0:
            return torch.zeros(num_graphs, self.pair_output_dim, device=device)

        per_edge = self.bank.per_edge_features(
            edge_state, edge_rbf, edge_weight, message_features, E, device
        )  # (E, d_inv)
        weighted_edge = per_edge * edge_weight.unsqueeze(-1)

        target = geo_edge_index[0]
        source = geo_edge_index[1]

        # Compute a compact, dense pair index over the pairs that actually
        # appear in this batch. `torch.unique(..., return_inverse=True)` gives
        # us that mapping without dense num_nodes^2 storage.
        pair_key = target * int(num_nodes) + source     # (E,)
        unique_pair_key, pair_local_idx = torch.unique(
            pair_key, return_inverse=True
        )
        n_pairs = int(unique_pair_key.numel())

        # Stage 1: aggregate per-edge features onto their pair.
        pair_features = scatter(
            weighted_edge, pair_local_idx, dim=0,
            dim_size=n_pairs, reduce="sum",
        )                                               # (n_pairs, d_inv)
        pair_weights = scatter(
            edge_weight, pair_local_idx, dim=0,
            dim_size=n_pairs, reduce="sum",
        )                                               # (n_pairs,)

        if self.bank.pool_reduce == "mean":
            pair_features = pair_features / pair_weights.clamp(min=1e-8).unsqueeze(-1)

        # Stage 2: apply MLP to REAL pairs only, then sum per graph.
        pair_emb = self.pair_mlp(pair_features)         # (n_pairs, pair_output_dim)

        # Map each pair to its graph. All sub-edges of a pair belong to the
        # same graph, so scatter-max/first over `batch_edge` gives the graph.
        graph_of_pair = scatter(
            batch_edge, pair_local_idx, dim=0,
            dim_size=n_pairs, reduce="max",
        )                                               # (n_pairs,)

        graph_repr = scatter(
            pair_emb, graph_of_pair, dim=0,
            dim_size=num_graphs, reduce="sum",
        )                                               # (num_graphs, D)
        return graph_repr


class EdgeInvariantReadout(nn.Module):
    """Unified edge readout entry point.

    Dispatches between sub-edge and orbit-pair pooling based on `mode`.

    Args:
        edge_state_dim: dim of edge_state tensor from EdgeStateInit (0 if
            no edge_state is used).
        num_rbf: dim of edge_rbf.
        mode: "subedge" | "orbit_pair" | "both" | "none".
        features: which invariants to use.
        pool_reduce: "sum" | "mean" (mean = TRUE weighted mean).
        pair_hidden_dim, pair_output_dim: only for orbit_pair mode.
        message_irreps: e3nn Irreps of the messages if you plan to use
            "norm"/"tp_l{L}".
    """

    def __init__(
        self,
        edge_state_dim: int,
        num_rbf: int,
        mode: str = "subedge",
        features: Optional[Sequence[str]] = None,
        pool_reduce: str = "sum",
        pair_hidden_dim: int = 64,
        pair_output_dim: int = 64,
        message_irreps=None,
    ):
        super().__init__()
        self.mode = mode
        if features is None:
            features = ["0e", "rbf_hist", "edge_count"]
        self.features = list(features)

        self.bank = InvariantPoolingBank(
            features=self.features,
            edge_state_dim=edge_state_dim,
            num_rbf=num_rbf,
            message_irreps=message_irreps,
            pool_reduce=pool_reduce,
        )

        if mode == "subedge":
            self.pooler = SubEdgePooling(self.bank)
            self._out_dim = self.pooler.output_dim
        elif mode == "orbit_pair":
            self.pooler = OrbitPairPooling(
                self.bank,
                pair_hidden_dim=pair_hidden_dim,
                pair_output_dim=pair_output_dim,
            )
            self._out_dim = self.pooler.output_dim
        elif mode == "both":
            self.subedge_pooler = SubEdgePooling(self.bank)
            self.orbitpair_pooler = OrbitPairPooling(
                self.bank,
                pair_hidden_dim=pair_hidden_dim,
                pair_output_dim=pair_output_dim,
            )
            self._out_dim = (
                self.subedge_pooler.output_dim + self.orbitpair_pooler.output_dim
            )
        elif mode == "none":
            self._out_dim = 0
        else:
            raise ValueError(f"Unknown edge_readout_mode: {mode}")

    @property
    def output_dim(self) -> int:
        return self._out_dim

    def forward(
        self,
        edge_state,
        edge_rbf,
        edge_weight,
        batch_edge,
        num_graphs,
        geo_edge_index=None,
        num_nodes=None,
        message_features=None,
    ) -> torch.Tensor:
        if self.mode == "none":
            return torch.zeros(num_graphs, 0, device=batch_edge.device)

        if self.mode == "subedge":
            return self.pooler(
                edge_state=edge_state,
                edge_rbf=edge_rbf,
                edge_weight=edge_weight,
                batch_edge=batch_edge,
                num_graphs=num_graphs,
                message_features=message_features,
            )

        if self.mode == "orbit_pair":
            assert geo_edge_index is not None and num_nodes is not None
            return self.pooler(
                edge_state=edge_state,
                edge_rbf=edge_rbf,
                edge_weight=edge_weight,
                batch_edge=batch_edge,
                num_graphs=num_graphs,
                geo_edge_index=geo_edge_index,
                num_nodes=num_nodes,
                message_features=message_features,
            )

        assert geo_edge_index is not None and num_nodes is not None
        se = self.subedge_pooler(
            edge_state=edge_state,
            edge_rbf=edge_rbf,
            edge_weight=edge_weight,
            batch_edge=batch_edge,
            num_graphs=num_graphs,
            message_features=message_features,
        )
        op = self.orbitpair_pooler(
            edge_state=edge_state,
            edge_rbf=edge_rbf,
            edge_weight=edge_weight,
            batch_edge=batch_edge,
            num_graphs=num_graphs,
            geo_edge_index=geo_edge_index,
            num_nodes=num_nodes,
            message_features=message_features,
        )
        return torch.cat([se, op], dim=-1)
