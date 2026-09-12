"""Neumann-symmetry projection of global Cartesian tensor outputs.

Projects a predicted crystal property tensor onto the subspace left invariant
by the crystal's own point group G:

    Pi_G^(r)(T) = (1/|G|) sum_{R in G} R^{(x)r} T

Concretely, for rank 3:

    Pi(T)_ijk = (1/|G|) sum_R R_ia R_jb R_kc T_abc

G contains both proper and improper operations. No det(R) factor is applied:
the piezoelectric tensor is a polar (true) rank-3 tensor, so an inversion
R = -I contributes (-I)^(x)3 T = -T and a centrosymmetric group therefore
projects to exactly zero. That is Neumann's principle, enforced structurally
rather than learned.

This is a GLOBAL CRYSTAL POINT-GROUP projection. It is deliberately distinct
from, and must not be confused with:
  - site_projection / SiteIrrepProjector  (per-orbit site stabilizer)
  - source-image transport                (per-edge Wigner-D transport)
  - point_group_symmetry.py               (legacy rank-2 module which derives
                                           its group from a dummy one-atom
                                           cell, i.e. the Bravais lattice
                                           group, not the crystal group)
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from wyckoff_gnn.models.unified_equivariant.tensor_adapter import (
    cartesian333_to_voigt36,
    voigt36_to_cartesian333,
)

VOIGT_DIM = 18

_OUT_LETTERS = "ijkl"
_IN_LETTERS = "abcd"


def fractional_to_cartesian_rotations(
    lattice: np.ndarray,
    rotations_frac: np.ndarray,
) -> np.ndarray:
    """Map fractional rotation parts W to Cartesian rotations R.

    Derivation for this project's row-vector lattice convention, where the
    rows of ``lattice`` are the basis vectors a1, a2, a3:

        a Cartesian position is  x = f @ L, i.e.  x^T = L^T f^T
        a symmetry operation acts on fractional coordinates as f -> W f
        hence  x'^T = L^T W f^T = L^T W (L^T)^-1 x^T

        =>  R = L^T W L^-T
    """
    L_T = np.asarray(lattice, dtype=np.float64).T
    L_T_inv = np.linalg.inv(L_T)
    W = np.asarray(rotations_frac, dtype=np.float64)
    return np.einsum("ij,njk,kl->nil", L_T, W, L_T_inv)


def unique_cartesian_rotations(
    rotations_cart: np.ndarray,
    decimals: int = 6,
) -> np.ndarray:
    """Deduplicate rotations that differ only by a lattice translation.

    A global material tensor couples only to the rotational part, so a space
    group operation set must be collapsed to the point group before averaging.
    Weighting a rotation by its translation multiplicity would silently
    average over the wrong group.
    """
    R = np.asarray(rotations_cart, dtype=np.float64)
    keys = np.round(R, decimals).reshape(len(R), 9)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return R[np.sort(idx)]


def _verify_ops_map_structure(
    frac_coords: np.ndarray,
    numbers: np.ndarray,
    rotations_frac: np.ndarray,
    translations: np.ndarray,
    tol: float = 1e-4,
) -> float:
    """Largest residual when mapping the structure onto itself under each op."""
    f = np.asarray(frac_coords, dtype=np.float64)
    z = np.asarray(numbers)
    worst = 0.0
    for W, w in zip(rotations_frac, translations):
        mapped = (W @ f.T).T + w
        for i in range(len(f)):
            d = mapped[i][None, :] - f
            d -= np.round(d)
            dist = np.linalg.norm(d, axis=1)
            dist[z != z[i]] = np.inf
            worst = max(worst, float(dist.min()))
    return worst


def crystal_point_group_rotations(
    lattice: np.ndarray,
    frac_coords: np.ndarray,
    numbers: np.ndarray,
    symprec: float = 1e-5,
    orthogonality_tol: float = 1e-8,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Extract the crystal's own point group as Cartesian rotations.

    The group is taken from the full standardized structure passed in, so the
    returned rotations live in exactly the Cartesian frame of ``lattice`` --
    the same frame the stored targets were rotated into.

    Returns the unique rotations plus an audit dict. Orthogonality and
    determinant errors are reported, never silently normalized away.
    """
    import spglib

    lattice = np.asarray(lattice, dtype=np.float64)
    frac_coords = np.asarray(frac_coords, dtype=np.float64)
    numbers = np.asarray(numbers, dtype=np.int32)

    sym = spglib.get_symmetry((lattice, frac_coords, numbers), symprec=symprec)
    if sym is None:
        info = {
            "n_spacegroup_ops": 0,
            "n_unique_pointgroup_rotations": 1,
            "contains_inversion": False,
            "max_orthogonality_error": 0.0,
            "max_det_error": 0.0,
            "structure_map_residual": float("nan"),
            "spglib_failed": True,
        }
        return np.eye(3)[None, ...], info

    W = np.asarray(sym["rotations"], dtype=np.float64)
    w = np.asarray(sym["translations"], dtype=np.float64)

    R_all = fractional_to_cartesian_rotations(lattice, W)
    R = unique_cartesian_rotations(R_all)

    gram = np.einsum("nij,nkj->nik", R, R)
    orth_err = float(np.abs(gram - np.eye(3)[None, ...]).max())
    dets = np.linalg.det(R)
    det_err = float(np.abs(np.abs(dets) - 1.0).max())
    contains_inversion = bool(
        np.any(np.abs(R + np.eye(3)[None, ...]).max(axis=(1, 2)) < 1e-6)
    )

    info = {
        "n_spacegroup_ops": int(len(W)),
        "n_unique_pointgroup_rotations": int(len(R)),
        "contains_inversion": contains_inversion,
        "max_orthogonality_error": orth_err,
        "max_det_error": det_err,
        "n_proper": int((dets > 0).sum()),
        "n_improper": int((dets < 0).sum()),
        "structure_map_residual": _verify_ops_map_structure(
            frac_coords, numbers, W, w
        ),
        "spglib_failed": False,
        "orthogonality_ok": orth_err < orthogonality_tol,
    }
    return R, info


def project_cartesian_tensor(
    tensor: torch.Tensor,
    rotations_cart: torch.Tensor,
    rank: int,
) -> torch.Tensor:
    """Group-average a rank-r Cartesian tensor over the point group.

    Args:
        tensor: shape (3,) * rank.
        rotations_cart: (n_ops, 3, 3) Cartesian rotations, already deduplicated
            to the point group.
        rank: tensor rank (2, 3, 4 supported).

    Returns:
        Pi_G(tensor), same shape as ``tensor``.
    """
    if rank < 1 or rank > len(_OUT_LETTERS):
        raise ValueError(f"rank must be in 1..{len(_OUT_LETTERS)}, got {rank}")
    if tuple(tensor.shape) != (3,) * rank:
        raise ValueError(
            f"tensor shape {tuple(tensor.shape)} does not match rank {rank}"
        )

    out_idx = _OUT_LETTERS[:rank]
    in_idx = _IN_LETTERS[:rank]
    terms = ",".join(f"n{out_idx[i]}{in_idx[i]}" for i in range(rank))
    eq = f"{terms},{in_idx}->{out_idx}"

    R = rotations_cart.to(dtype=tensor.dtype)
    projected = torch.einsum(eq, *([R] * rank), tensor)
    return projected / R.shape[0]


def build_voigt18_projector(
    rotations_cart: torch.Tensor,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Precompute the (18, 18) matrix P with  y_proj = P @ y  in Voigt space.

    Column n is the image of the n-th Voigt basis vector under
    voigt -> Cartesian -> Pi_G -> Voigt, reusing the verified
    Rank3PiezoAdapter Voigt mapping (VASP order xx, yy, zz, xy, yz, zx, no
    factor of 2). The flattening matches the stored y_tensor layout, i.e.
    row-major reshape of (3, 6).
    """
    R = rotations_cart.to(dtype=dtype)
    cols = []
    for n in range(VOIGT_DIM):
        basis = torch.zeros(VOIGT_DIM, dtype=dtype)
        basis[n] = 1.0
        cart = voigt36_to_cartesian333(basis.reshape(1, 3, 6))[0]
        proj = project_cartesian_tensor(cart, R, rank=3)
        cols.append(cartesian333_to_voigt36(proj.unsqueeze(0))[0].reshape(-1))
    return torch.stack(cols, dim=1)


def build_cartesian_projector_matrix(
    rotations_cart: torch.Tensor,
    rank: int,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Projector matrix acting on the flattened 3**rank Cartesian tensor.

    Used for tensors whose targets are stored as plain Cartesian components
    (rank-2 dielectric: flattened 3x3 -> 9). Rank-3 piezo instead uses
    ``build_voigt18_projector`` because its targets are stored in Voigt form.
    """
    R = rotations_cart.to(dtype=dtype)
    dim = 3 ** rank
    cols = []
    for n in range(dim):
        basis = torch.zeros(dim, dtype=dtype)
        basis[n] = 1.0
        proj = project_cartesian_tensor(basis.reshape((3,) * rank), R, rank)
        cols.append(proj.reshape(-1))
    return torch.stack(cols, dim=1)


def project_voigt_piezo(
    voigt: torch.Tensor,
    rotations_cart: Optional[torch.Tensor] = None,
    projector18: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Project a (3, 6) Voigt piezoelectric tensor onto the invariant subspace.

    Supply either the point group rotations or a precomputed (18, 18) matrix.
    """
    if projector18 is None:
        if rotations_cart is None:
            raise ValueError("need rotations_cart or projector18")
        projector18 = build_voigt18_projector(
            rotations_cart, dtype=voigt.dtype
        )
    flat = voigt.reshape(-1).to(dtype=projector18.dtype)
    return (projector18 @ flat).reshape(3, 6).to(dtype=voigt.dtype)


def crystal_projector_from_symmetry(
    lattice: np.ndarray,
    rotations_frac: np.ndarray,
    rank: int = 3,
    orthogonality_tol: float = 1e-10,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Preprocessing-time point-group projector matrix for a rank-r tensor.

    ``lattice`` and ``rotations_frac`` must both come from the *same*
    standardized cell, in float64. Reusing the symmetry operations that the
    standardization already produced (rather than re-running spglib on a
    float32 round-trip of the cell) keeps the Cartesian rotations at machine
    precision and makes the quotient and P1 caches agree bit-for-bit.

    The matrix acts on the layout the targets are stored in: rank 3 uses the
    18-dimensional Voigt vector, rank 2 the 9-dimensional flattened 3x3.

    For odd rank with a centrosymmetric group, Pi_G is identically zero by
    Neumann's principle: (-I)^(x)r = -Id for odd r, so averaging R and -R
    cancels every component. That zero is written analytically here instead of
    being left as the ~1e-31 residue of a floating-point group average, so no
    downstream epsilon threshold is ever needed. Even rank is untouched by
    inversion and must never be zeroed this way.
    """
    lattice = np.asarray(lattice, dtype=np.float64)
    R_all = fractional_to_cartesian_rotations(lattice, rotations_frac)
    R = unique_cartesian_rotations(R_all)

    gram = np.einsum("nij,nkj->nik", R, R)
    orth_err = float(np.abs(gram - np.eye(3)[None, ...]).max())
    dets = np.linalg.det(R)
    contains_inversion = bool(
        np.any(np.abs(R + np.eye(3)[None, ...]).max(axis=(1, 2)) < 1e-8)
    )

    Rt = torch.tensor(R, dtype=torch.float64)
    if rank == 3:
        dim = VOIGT_DIM
        if contains_inversion:
            P = np.zeros((dim, dim), dtype=np.float64)
        else:
            P = build_voigt18_projector(Rt).numpy()
    else:
        dim = 3 ** rank
        if rank % 2 == 1 and contains_inversion:
            P = np.zeros((dim, dim), dtype=np.float64)
        else:
            P = build_cartesian_projector_matrix(Rt, rank).numpy()

    info = {
        "point_group_order": int(len(R)),
        "n_spacegroup_ops": int(len(rotations_frac)),
        "contains_inversion": contains_inversion,
        "projector_rank": int(np.linalg.matrix_rank(P, tol=1e-9)),
        "projector_dim": int(dim),
        "max_orthogonality_error": orth_err,
        "max_det_error": float(np.abs(np.abs(dets) - 1.0).max()),
        "orthogonality_ok": orth_err < orthogonality_tol,
        "analytic_zero": bool(rank % 2 == 1 and contains_inversion),
    }
    return P, info


def projector_kwargs_from_structure(
    structure: Any, symprec: float, rank: int = 3
) -> Dict[str, Any]:
    """Build the projector graph fields directly from the input structure.

    The group is re-derived here through ``standardize_structure_cell`` rather
    than read out of a builder's metadata. Builder metadata reflects the graph
    construction path -- in particular it degenerates to the identity group
    whenever the quotient builder falls back to P1 -- whereas the point group
    is a property of the crystal. Deriving it independently is what guarantees
    the quotient and P1 caches hold bit-identical projectors, which the
    fairness requirement depends on: the physics constraint must not be a
    quotient-only advantage.
    """
    from wyckoff_gnn.data.crystal_to_wyckoff import standardize_structure_cell

    std = standardize_structure_cell(structure, symprec=symprec)
    lattice = std.get("standardized_lattice")
    rotations = std.get("symmetry_rotations")
    if lattice is None or rotations is None or len(rotations) == 0:
        return {}
    P, info = crystal_projector_from_symmetry(lattice, rotations, rank=rank)
    return {
        "tensor_point_group_projector": P,
        "point_group_order": info["point_group_order"],
        "contains_inversion": info["contains_inversion"],
        "projector_rank": info["projector_rank"],
    }
