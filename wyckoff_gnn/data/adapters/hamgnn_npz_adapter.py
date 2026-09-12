"""HamGNN NPZ dataset adapter.

Reads the HamGNN-style ``graph_data.npz`` format, which stores all
structures as a dict of PyG ``Data`` objects with Hamiltonian blocks
(``Hon``, ``Hoff``, ``Son``, ``Soff``) already flattened per edge.

This format is used by the Si-600 / GaAs datasets at::

    /public/home/moyk/HamDeepH/Si-600/graph_data.npz

Layout inside the NPZ::

    graph_data.npz["graph"].item() = {
        0: Data(Hon=(K,n²), Hoff=(E,n²), Son=(K,n²), Soff=(E,n²),
                cell=(3,3), pos=(K,3), z=(K,), edge_index=(2,E),
                cell_shift=(E,3), inv_edge_idx=(E,), node_counts=(1,)),
        1: Data(...),
        ...
    }
    split_idx.npz["train_idx"] = array of int indices
    split_idx.npz["val_idx"]   = ...
    split_idx.npz["test_idx"]  = ...
"""

from __future__ import annotations

import os
from typing import Any, Dict, Iterator, List, Optional

import numpy as np
import torch

from wyckoff_gnn.data.adapters.base import DatasetAdapter
from wyckoff_gnn.data.records import StructureRecord


__all__ = ["HamGNNNpzAdapter"]


class HamGNNNpzAdapter(DatasetAdapter):
    """Adapter for HamGNN-format NPZ datasets (Si-600, GaAs, etc.).

    Args:
        npz_path: path to ``graph_data.npz``.
        split_path: path to ``split_idx.npz`` (optional; if absent all are train).
        max_entries: limit for debugging.
        n_orb_per_atom: orbital count per atom (sqrt of Hon/Hoff's last dim).
    """

    def __init__(
        self,
        npz_path: str,
        split_path: Optional[str] = None,
        max_entries: Optional[int] = None,
        n_orb_per_atom: int = 13,
        **kwargs,
    ):
        self.npz_path = npz_path
        self.split_path = split_path
        self.max_entries = max_entries
        self.n_orb = n_orb_per_atom

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "HamGNNNpzAdapter":
        return cls(**config)

    def iter_records(self) -> Iterator[StructureRecord]:
        from pymatgen.core import Lattice, Structure
        from pymatgen.core.periodic_table import Element

        raw = np.load(self.npz_path, allow_pickle=True)
        graph_dict = raw["graph"].item()

        split_map: Dict[int, str] = {}
        if self.split_path and os.path.exists(self.split_path):
            sp = np.load(self.split_path, allow_pickle=True)
            for idx in sp.get("train_idx", []):
                split_map[int(idx)] = "train"
            for idx in sp.get("val_idx", []):
                split_map[int(idx)] = "val"
            for idx in sp.get("test_idx", []):
                split_map[int(idx)] = "test"

        keys = sorted(graph_dict.keys())
        if self.max_entries:
            keys = keys[: self.max_entries]

        for idx in keys:
            try:
                data = graph_dict[idx]
                record = self._data_to_record(data, idx, split_map.get(idx))
                if record is not None:
                    yield record
            except Exception:
                continue

    def _data_to_record(self, data, idx: int, split: Optional[str]) -> Optional[StructureRecord]:
        from pymatgen.core import Lattice, Structure
        from pymatgen.core.periodic_table import Element

        cell = data.cell.numpy() if hasattr(data.cell, 'numpy') else np.array(data.cell)
        pos = data.pos.numpy() if hasattr(data.pos, 'numpy') else np.array(data.pos)
        z = data.z.numpy() if hasattr(data.z, 'numpy') else np.array(data.z)

        lattice = Lattice(cell)
        elements = [Element.from_Z(int(zi)).symbol for zi in z]
        structure = Structure(lattice, elements, pos, coords_are_cartesian=True)

        n_atoms = len(z)
        n_orb = self.n_orb
        n_orb_sq = n_orb * n_orb

        # Reshape flattened blocks back to (K, n_orb, n_orb) / (E, n_orb, n_orb)
        Hon = data.Hon.numpy() if hasattr(data.Hon, 'numpy') else np.array(data.Hon)
        Hoff = data.Hoff.numpy() if hasattr(data.Hoff, 'numpy') else np.array(data.Hoff)

        # Onsite: (n_atoms, n_orb²) → {(i, i, (0,0,0)): (n_orb, n_orb)}
        hamiltonian_blocks = {}
        for i in range(n_atoms):
            block = Hon[i].reshape(n_orb, n_orb)
            hamiltonian_blocks[(i, i, (0, 0, 0))] = block

        # Offsite: (E, n_orb²) + edge_index + cell_shift
        edge_index = data.edge_index.numpy() if hasattr(data.edge_index, 'numpy') else np.array(data.edge_index)
        cell_shift = data.cell_shift.numpy() if hasattr(data.cell_shift, 'numpy') else np.array(data.cell_shift)

        for e in range(Hoff.shape[0]):
            src = int(edge_index[1, e])
            tgt = int(edge_index[0, e])
            R = tuple(int(x) for x in cell_shift[e])
            block = Hoff[e].reshape(n_orb, n_orb)
            hamiltonian_blocks[(tgt, src, R)] = block

        mid = f"Si600_{idx:06d}"
        return StructureRecord(
            material_id=mid,
            structure=structure,
            target=hamiltonian_blocks,
            target_type="hamiltonian",
            split=split,
            metadata={"source": "hamgnn_npz", "n_orb_per_atom": n_orb, "n_atoms": n_atoms},
        )
