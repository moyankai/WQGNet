"""Atom graph builder — builds per-atom radius graphs.

Status: **preprocess only**.  Atom graphs produced by this builder can
be stored in shard caches, but the current ``train_property.py`` only
supports Wyckoff graphs.  Full atom training pipeline will be added in
a later phase.
"""

from __future__ import annotations

from typing import Any, Dict

from wyckoff_gnn.data.graph_builders.base import GraphBuilder
from wyckoff_gnn.data.graph_schema import _as_numpy
from wyckoff_gnn.data.records import StructureRecord


class AtomGraphBuilderWrapper(GraphBuilder):
    """Build per-atom PBC radius graphs from StructureRecords.

    Config keys
    -----------
    cutoff : float (default 5.0)
        Radius cutoff for geometric edges.
    """

    def __init__(self, cutoff: float = 5.0, **kwargs):
        self.cutoff = cutoff

    def build(self, record: StructureRecord) -> Dict[str, Any]:
        from wyckoff_gnn.data.atom_graph import structure_to_atom_graph
        from wyckoff_gnn.data.crystal_to_wyckoff import structure_to_wyckoff_orbits

        # Get standardized cell info for consistent lattice convention.
        _, meta = structure_to_wyckoff_orbits(record.structure, tol=0.1)

        data = structure_to_atom_graph(
            record.structure, meta=meta,
            cutoff=self.cutoff,
            use_standardized_lattice=True,
        )

        # Convert to a simple dict (atom graphs are small — keep full tensors).
        graph: Dict[str, Any] = {
            "material_id": record.material_id,
            "y": float(record.target) if record.target is not None else None,
            "atom_numbers": _as_numpy(data.atom_numbers, "int32"),
            "atom_pos": _as_numpy(data.atom_pos, "float32"),
            "edge_index": _as_numpy(data.edge_index, "int64"),
            "edge_vec": _as_numpy(data.edge_vec, "float32"),
            "edge_shift": _as_numpy(data.edge_shift, "int16"),
            "edge_length": _as_numpy(data.edge_length, "float32"),
            "lattice": _as_numpy(data.lattice, "float32"),
            "num_atoms": int(data.num_nodes),
        }
        return graph


__all__ = ["AtomGraphBuilderWrapper"]
