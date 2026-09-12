"""Site-symmetry projection for Wyckoff orbit features.

For an orbit p with site-symmetry stabilizer group S_p, the subgroup of
space-group operations that leave the representative position invariant:

    S_p = { (S, u) : S @ x_p + u = x_p  (mod 1) }

The orbit's representative features h_p must be invariant under S_p:

    D^{(l)}(S_cart) @ h_p^{(l)} = h_p^{(l)}   for all l and all S ∈ S_p

This means h_p lives in the trivial sub-representation of S_p.  The
projection operator onto this subspace is:

    P_p = (1 / |S_p|) * sum_{S ∈ S_p} D(S_cart)

where S_cart = A.T @ S_frac @ A^{-T}  (column Cartesian convention).

Properties:
- P_p^2 = P_p  (idempotent)
- D(S) @ P_p @ h = P_p @ h = h  for S ∈ S_p (invariance)
- P_p projects polar vectors to zero for centrosymmetric sites
- P_p preserves scalar (l=0) channels (they are always invariant)
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
from e3nn import o3


def compute_site_stabilizer_ops(
    rep_frac: np.ndarray,
    all_rotations: np.ndarray,
    all_translations: np.ndarray,
    tol: float = 1e-4,
) -> Tuple[np.ndarray, np.ndarray]:
    """Find symmetry operations that leave the representative position invariant.

    Vectorized: computes all operations in a single batched matmul.
    """
    # all_rotations: (N_ops, 3, 3), all_translations: (N_ops, 3)
    mapped = np.einsum('nij,j->ni', all_rotations, rep_frac) + all_translations
    diff = mapped - rep_frac
    diff -= np.round(diff)
    norms = np.linalg.norm(diff, axis=1)
    mask = norms < tol

    if not mask.any():
        return np.eye(3, dtype=np.float32)[None], np.zeros((1, 3), dtype=np.float32)

    return all_rotations[mask].astype(np.float32), all_translations[mask].astype(np.float32)


def build_irrep_projection_matrix(
    stabilizer_R_e3nn: torch.Tensor,
    irreps: o3.Irreps,
) -> torch.Tensor:
    """Build the projection matrix P = (1/|S|) * sum_s D(R_s).

    Batched: computes Wigner-D for all stabilizer ops at once per irrep block.
    """
    n_S = stabilizer_R_e3nn.shape[0]
    dim = irreps.dim
    P = torch.zeros(dim, dim)

    idx = 0
    for mul, ir in irreps:
        l = ir.l
        dim_l = ir.dim  # 2l+1
        block_size = mul * dim_l

        if l == 0:
            # Scalar channels: D(R) = I always, so sum = n_S * I
            P[idx:idx + block_size, idx:idx + block_size] = torch.eye(block_size)
        else:
            # Batch Wigner-D for all stabilizer ops at once
            D_batch = ir.D_from_matrix(stabilizer_R_e3nn)  # (n_S, 2l+1, 2l+1)
            D_avg = D_batch.mean(dim=0)  # (2l+1, 2l+1) — average over stabilizer

            # Tile across multiplicities using block_diag structure
            for c in range(mul):
                offset = idx + c * dim_l
                P[offset:offset + dim_l, offset:offset + dim_l] = D_avg

        idx += block_size

    return P


def build_orbit_projections_tensor(
    orbits,
    all_rotations: np.ndarray,
    all_translations: np.ndarray,
    irreps: o3.Irreps,
    lattice: np.ndarray,
) -> torch.Tensor:
    """Build per-orbit projection matrices as a stacked (K, dim, dim) tensor.

    Returns a tensor suitable for ``data.orbit_stabilizer_projections``.
    Orbits with trivial stabilizer (|S|=1) get identity matrices.
    """
    proj_list = build_all_projections(orbits, all_rotations, all_translations,
                                       irreps, lattice)
    return torch.stack(proj_list)  # (K, dim, dim)


def project_orbit_features(
    h: torch.Tensor,
    projection_matrix: torch.Tensor,
) -> torch.Tensor:
    """Apply site-symmetry projection to node features.

    Args:
        h: (K, dim) node features in irrep order.
        projection_matrix: (dim, dim) precomputed projection.

    Returns:
        (K, dim) projected features.
    """
    return torch.matmul(h, projection_matrix.T)


def build_all_projections(
    orbits,
    all_rotations: np.ndarray,
    all_translations: np.ndarray,
    irreps: o3.Irreps,
    lattice: np.ndarray,
) -> List[torch.Tensor]:
    """Build site-symmetry projection matrices for all orbits.

    Optimizations:
    - Vectorized stabilizer computation
    - Batched fractional-to-Cartesian conversion
    - Caches projection matrices for identical stabilizer groups
    """
    A = torch.from_numpy(np.ascontiguousarray(lattice)).float()
    A_T = A.T
    A_inv_T = torch.inverse(A).T

    # Cache: hash stabilizer rotations → projection matrix
    proj_cache = {}
    dim = irreps.dim

    projections = []
    for orb in orbits:
        stab_W, _ = compute_site_stabilizer_ops(
            orb.representative_coord,
            all_rotations, all_translations,
        )

        # Cache key: hash the stabilizer rotation matrices
        cache_key = stab_W.tobytes()
        if cache_key in proj_cache:
            projections.append(proj_cache[cache_key])
            continue

        if stab_W.shape[0] == 1 and np.allclose(stab_W[0], np.eye(3)):
            # Trivial stabilizer (identity only) → P = I
            P = torch.eye(dim)
        else:
            # Batched fractional → Cartesian conversion
            stab_W_t = torch.from_numpy(stab_W).float()  # (|S|, 3, 3)
            stab_R_t = A_T @ stab_W_t @ A_inv_T  # (|S|, 3, 3) via broadcasting

            P = build_irrep_projection_matrix(stab_R_t, irreps)

        proj_cache[cache_key] = P
        projections.append(P)

    return projections


__all__ = [
    "compute_site_stabilizer_ops",
    "build_irrep_projection_matrix",
    "project_orbit_features",
    "build_all_projections",
]
