"""Readout modules for Wyckoff GNN.

Package structure (Phase 5+):
    invariant_pooling.py       — building blocks (SE(3)-invariant reductions)
    edge_readout.py            — EdgeInvariantReadout (sub-edge + orbit-pair pooling)
    tensor_readout.py          — TensorPropertyHead (arbitrary O(3) irrep output)
    hamiltonian_readout.py     — WyckoffHamiltonianHead (onsite + offsite blocks)
"""

from wyckoff_gnn.models.wyckoff_equiv.readouts.invariant_pooling import (
    InvariantPoolingBank,
    edge_count_pool,
    extract_irrep_norms,
    extract_scalar_channels,
    irrep_pair_invariant,
    rbf_histogram_pool,
)
from wyckoff_gnn.models.wyckoff_equiv.readouts.edge_readout import (
    EdgeInvariantReadout,
    OrbitPairPooling,
    SubEdgePooling,
)
from wyckoff_gnn.models.wyckoff_equiv.readouts.tensor_readout import TensorPropertyHead
from wyckoff_gnn.models.wyckoff_equiv.readouts.hamiltonian_readout import (
    WyckoffHamiltonianHead,
    basis_dim,
    build_bloch_hamiltonian,
    enforce_offsite_hermiticity,
    find_reverse_edge_map,
    hermitize_bloch,
    hermitize_with_reverse_map,
    orbital_rep_matrix,
    symmetrize_onsite_block,
)

__all__ = [
    "InvariantPoolingBank",
    "EdgeInvariantReadout",
    "SubEdgePooling",
    "OrbitPairPooling",
    "TensorPropertyHead",
    "WyckoffHamiltonianHead",
    "basis_dim",
    "build_bloch_hamiltonian",
    "enforce_offsite_hermiticity",
    "find_reverse_edge_map",
    "hermitize_bloch",
    "hermitize_with_reverse_map",
    "orbital_rep_matrix",
    "symmetrize_onsite_block",
    "extract_scalar_channels",
    "extract_irrep_norms",
    "irrep_pair_invariant",
    "rbf_histogram_pool",
    "edge_count_pool",
]
