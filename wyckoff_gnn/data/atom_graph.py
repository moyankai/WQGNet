"""Full-atom PBC radius graph builder with standardized lattice.

Independent of Wyckoff orbit compression. Each atom is a node.
Uses the same standardized lattice convention as WyckoffGNN.
"""
from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import Data
from pymatgen.core.structure import Structure


def structure_to_atom_graph(
    struct: Structure,
    meta: dict | None = None,
    cutoff: float = 5.0,
    use_standardized_lattice: bool = True,
) -> Data:
    """Build a full-atom PBC radius graph.

    Args:
        struct: pymatgen Structure.
        meta: Metadata from ``structure_to_wyckoff_orbits``. Must contain
              ``standardized_lattice`` if ``use_standardized_lattice=True``.
        cutoff: Radius cutoff in Å.
        use_standardized_lattice: Use spglib-standardized lattice.

    Returns:
        PyG Data with:
          - atom_numbers: (M,) atomic numbers
          - atom_pos: (M, 3) Cartesian positions (standardized lattice)
          - edge_index: (2, E) directed edges [src, dst]
          - edge_vec: (E, 3) displacement vectors (Cartesian, MIC)
          - edge_length: (E,) distances
          - edge_shift: (E, 3) integer PBC shift
          - lattice: (3, 3) lattice matrix
          - atom_batch: (M,) graph assignment (all zeros for single graph)
          - num_nodes: M
    """
    # Lattice, fractional coords, and species: standardized or original.
    if use_standardized_lattice and meta is not None:
        lat = meta.get("standardized_lattice", struct.lattice.matrix)
        frac_arr = meta.get("standardized_positions", struct.frac_coords)
        numbers_arr = meta.get("standardized_numbers",
                              np.array([s.number for s in struct.species], dtype=np.int32))
    else:
        lat = struct.lattice.matrix
        frac_arr = struct.frac_coords
        numbers_arr = np.array([s.number for s in struct.species], dtype=np.int32)

    M = len(numbers_arr)
    lattice = torch.from_numpy(np.ascontiguousarray(lat)).float()
    frac = torch.from_numpy(np.ascontiguousarray(frac_arr)).float()
    z = torch.from_numpy(numbers_arr.copy()).long()

    # Cartesian: x_cart = r_frac @ A (row convention).
    pos = torch.matmul(frac, lattice)

    # Build directed radius graph with PBC.
    src_list, dst_list, vec_list, shift_list = [], [], [], []
    inv_lattice = torch.inverse(lattice)

    for i in range(M):
        for j in range(M):
            if i == j:
                continue
            v = pos[j] - pos[i]
            v_frac = torch.matmul(v, inv_lattice)
            L = -torch.round(v_frac)
            v_frac_mic = v_frac + L
            v_mic = torch.matmul(v_frac_mic, lattice)
            d = torch.norm(v_mic)
            if d < cutoff:
                src_list.append(i)
                dst_list.append(j)
                vec_list.append(v_mic)
                shift_list.append(L)

    if len(src_list) == 0:
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        edge_vec = torch.zeros(0, 3)
        edge_length = torch.zeros(0)
        edge_shift = torch.zeros(0, 3)
    else:
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        edge_vec = torch.stack(vec_list)
        edge_length = torch.norm(edge_vec, dim=-1)
        edge_shift = torch.stack(shift_list)

    return Data(
        atom_numbers=z,
        atom_pos=pos,
        edge_index=edge_index,
        edge_vec=edge_vec,
        edge_length=edge_length,
        edge_shift=edge_shift,
        lattice=lattice,
        atom_batch=torch.zeros(M, dtype=torch.long),
        num_nodes=M,
    )


__all__ = ["structure_to_atom_graph"]
