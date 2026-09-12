"""Site-symmetry constrained O(3) irrep projection.

For a Wyckoff representative at r_0 in space group G with stabilizer H_{r_0},
this module builds the projection onto the invariant subspace of each
O(3) irrep D^(l,pi):

    P^{l,pi}_H = (1/|H|) sum_{g in H} D^(l,pi)(g)

Projections satisfy:
    P^2 = P              (idempotent)
    D(g) P = P           (H-invariant)
    rank(P) = trace(P)   (# invariant dimensions)

The trivial subspace dimension n_{l,pi}^H = (1/|H|) sum chi_{l,pi}(g) equals
the number of physical degrees of freedom for a node feature of that irrep
sitting at this Wyckoff position.

Wraps existing implementations in wyckoff_gnn.utils.site_symmetry_projection
and adds:
    - allowed_irrep_table_for_site: rank per irrep
    - frac_op_to_cart_op with orthogonality check
    - independent Wigner-D wrapper for testability
    - lattice-independent utilities using standardized cubic frame
      (for canonical stabilizer classes)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from e3nn import o3

from wyckoff_gnn.utils.site_symmetry_projection import (
    build_irrep_projection_matrix as _build_irrep_projection_matrix_e3nn,
    compute_site_stabilizer_ops as _compute_stabilizer,
)


__all__ = [
    "get_stabilizer_ops",
    "frac_op_to_cart_op",
    "irrep_matrix_e3nn",
    "site_projection_matrix",
    "allowed_irrep_table_for_site",
    "verify_projector",
    "parse_irreps_list",
]


# ---------------------------------------------------------------------------
# Stabilizer identification
# ---------------------------------------------------------------------------

def get_stabilizer_ops(
    space_group_ops: List[Tuple[np.ndarray, np.ndarray]],
    rep_frac: np.ndarray,
    atol: float = 1e-4,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Find all (W, w) in G that fix rep_frac (up to lattice translation).

    Condition: W @ rep_frac + w - rep_frac is an integer vector.

    Args:
        space_group_ops: list of (W_frac (3,3), w_frac (3,)) tuples.
        rep_frac: (3,) fractional coords of the Wyckoff representative.
        atol: tolerance for the mod-1 comparison.

    Returns:
        List of (W_frac, w_frac) tuples that fix rep_frac. Always contains
        identity + zero translation (guaranteed by fallback).
    """
    if not space_group_ops:
        return [(np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64))]

    rotations = np.stack([np.asarray(W, dtype=np.float64) for W, _ in space_group_ops])
    translations = np.stack([np.asarray(w, dtype=np.float64) for _, w in space_group_ops])
    rep = np.asarray(rep_frac, dtype=np.float64)

    stab_R, stab_t = _compute_stabilizer(rep, rotations, translations, tol=atol)
    return [(stab_R[i], stab_t[i]) for i in range(stab_R.shape[0])]


# ---------------------------------------------------------------------------
# Fractional -> Cartesian conversion
# ---------------------------------------------------------------------------

def frac_op_to_cart_op(
    W_frac: np.ndarray,
    lattice: np.ndarray,
    ortho_tol: float = 1e-3,
) -> np.ndarray:
    """Convert fractional-basis rotation to Cartesian orthogonal rotation.

    R_cart = A^T @ W_frac @ A^{-T}   (e3nn column convention)

    where A is the lattice matrix with rows as basis vectors.

    Args:
        W_frac: (3, 3) rotation in fractional basis (typically integer).
        lattice: (3, 3) lattice matrix (rows are lattice vectors).
        ortho_tol: max tolerance for R^T @ R - I; raises RuntimeError above.

    Returns:
        (3, 3) Cartesian rotation matrix.
    """
    W = np.asarray(W_frac, dtype=np.float64)
    A = np.asarray(lattice, dtype=np.float64)
    A_T = A.T
    A_inv_T = np.linalg.inv(A).T
    R = A_T @ W @ A_inv_T
    err = np.max(np.abs(R.T @ R - np.eye(3)))
    if err > ortho_tol:
        raise RuntimeError(
            f"frac_op_to_cart_op: R.T @ R - I max error {err:.3e} exceeds {ortho_tol}"
        )
    return R


def frac_ops_to_cart_ops(
    W_frac_batch: np.ndarray,
    lattice: np.ndarray,
) -> np.ndarray:
    """Batched version of frac_op_to_cart_op.

    Args:
        W_frac_batch: (n, 3, 3) rotations in fractional basis.
        lattice: (3, 3) lattice matrix.

    Returns:
        (n, 3, 3) Cartesian rotation matrices.
    """
    W = np.asarray(W_frac_batch, dtype=np.float64)
    A = np.asarray(lattice, dtype=np.float64)
    A_T = A.T
    A_inv_T = np.linalg.inv(A).T
    n = W.shape[0]
    return np.einsum("ij,njk,kl->nil", A_T, W, A_inv_T)


# ---------------------------------------------------------------------------
# Wigner-D matrix
# ---------------------------------------------------------------------------

def irrep_matrix_e3nn(
    R_cart: torch.Tensor,
    irrep: o3.Irrep,
) -> torch.Tensor:
    """Compute Wigner-D matrix D^(l,p)(R) for one or many Cartesian rotations.

    Args:
        R_cart: (3, 3) or (n, 3, 3) Cartesian rotation tensor(s).
        irrep: e3nn o3.Irrep, e.g. o3.Irrep("2e") or o3.Irrep(l, parity).

    Returns:
        For (3,3) input: (dim, dim) where dim = 2l+1.
        For (n,3,3) input: (n, dim, dim).

    Parity handling: irrep.p is +1 for even, -1 for odd. For a rotation (det=+1)
    D is the same as for the corresponding SO(3) irrep. For an improper rotation
    (det=-1), D acquires a factor of parity. e3nn's D_from_matrix handles this
    if the input matrix has det=-1 for improper rotations (which is the case for
    inversion, mirrors, etc. in Cartesian representation).
    """
    if R_cart.dim() == 2:
        R = R_cart.unsqueeze(0)
        D = irrep.D_from_matrix(R)
        return D.squeeze(0)
    return irrep.D_from_matrix(R_cart)


# ---------------------------------------------------------------------------
# Projection matrix
# ---------------------------------------------------------------------------

def site_projection_matrix(
    stabilizer_R_cart: torch.Tensor,
    irrep: o3.Irrep,
) -> Dict[str, Any]:
    """Compute the site-symmetry projection matrix for one irrep.

    P = (1/|H|) sum_{g in H} D_irrep(g)

    Args:
        stabilizer_R_cart: (n_S, 3, 3) Cartesian rotation matrices of the
            stabilizer group H (each row-vector convention consistent with
            frac_op_to_cart_op).
        irrep: e3nn o3.Irrep.

    Returns:
        Dict with:
            P: (dim, dim) projection matrix
            rank: numerical rank of P (int)
            trace: trace(P) (float)
            idempotent_err: max |P^2 - P|
            invariance_err: max |D(g)P - P| over g in H
    """
    dim = irrep.dim
    if stabilizer_R_cart.shape[0] == 0:
        P = torch.eye(dim, dtype=torch.float64)
        return {
            "P": P,
            "rank": dim,
            "trace": float(dim),
            "idempotent_err": 0.0,
            "invariance_err": 0.0,
        }

    R = stabilizer_R_cart.to(torch.float64)
    D_batch = irrep.D_from_matrix(R)  # (n_S, dim, dim)
    P = D_batch.mean(dim=0)  # (dim, dim)

    # Check idempotency: P^2 = P
    idem_err = float((P @ P - P).abs().max().item())

    # Check invariance: D(g) P = P for all g in H
    DP = D_batch @ P  # (n_S, dim, dim)
    inv_err = float((DP - P.unsqueeze(0)).abs().max().item())

    # Rank via singular values
    tr = float(P.diagonal().sum().item())
    # Numerical rank at tol tr(P) is well-defined for a projector
    U, S, Vh = torch.linalg.svd(P)
    numerical_rank = int((S > 1e-6).sum().item())

    return {
        "P": P,
        "rank": numerical_rank,
        "trace": tr,
        "idempotent_err": idem_err,
        "invariance_err": inv_err,
    }


# ---------------------------------------------------------------------------
# Full irrep allowed table for a site
# ---------------------------------------------------------------------------

def parse_irreps_list(lmax: int, parities: str = "e,o") -> List[o3.Irrep]:
    """Enumerate all irreps up to given lmax with specified parities.

    Args:
        lmax: max angular momentum (inclusive).
        parities: comma-separated subset of {"e", "o"}.

    Returns:
        List of o3.Irrep objects.
    """
    parity_set = set(p.strip() for p in parities.split(","))
    irreps = []
    for l in range(lmax + 1):
        if "e" in parity_set:
            irreps.append(o3.Irrep(l, 1))
        if "o" in parity_set:
            irreps.append(o3.Irrep(l, -1))
    return irreps


def allowed_irrep_table_for_site(
    stabilizer_R_cart: torch.Tensor,
    irreps: List[o3.Irrep],
) -> Dict[str, Any]:
    """Compute rank and projection matrix for each irrep at one Wyckoff site.

    Args:
        stabilizer_R_cart: (n_S, 3, 3) Cartesian stabilizer rotations.
        irreps: list of o3.Irrep to enumerate.

    Returns:
        Dict with:
            allowed_ranks: {irrep_str: int rank}
            projectors: {irrep_str: (dim, dim) tensor}
            traces: {irrep_str: float}
            checks: {irrep_str: (idempotent_err, invariance_err)}
            stabilizer_size: int
    """
    result = {
        "allowed_ranks": {},
        "projectors": {},
        "traces": {},
        "checks": {},
        "stabilizer_size": int(stabilizer_R_cart.shape[0]),
    }
    for ir in irreps:
        key = str(ir)
        r = site_projection_matrix(stabilizer_R_cart, ir)
        result["allowed_ranks"][key] = r["rank"]
        result["projectors"][key] = r["P"]
        result["traces"][key] = r["trace"]
        result["checks"][key] = (r["idempotent_err"], r["invariance_err"])
    return result


# ---------------------------------------------------------------------------
# Verification helpers
# ---------------------------------------------------------------------------

def verify_projector(
    P: torch.Tensor,
    stabilizer_R_cart: torch.Tensor,
    irrep: o3.Irrep,
    atol: float = 1e-5,
) -> Dict[str, Any]:
    """Verify P^2=P, D(g)P=P, rank=trace for a projector.

    Returns dict with pass/fail flags and error magnitudes.
    """
    P = P.to(torch.float64)
    R = stabilizer_R_cart.to(torch.float64)
    idem_err = float((P @ P - P).abs().max().item())
    tr = float(P.diagonal().sum().item())
    if R.shape[0] > 0:
        D = irrep.D_from_matrix(R)
        inv_err = float((D @ P - P.unsqueeze(0)).abs().max().item())
    else:
        inv_err = 0.0
    S = torch.linalg.svdvals(P)
    numerical_rank = int((S > 1e-6).sum().item())
    rank_matches_trace = abs(numerical_rank - tr) < atol * max(1, numerical_rank)
    return {
        "idempotent": idem_err < atol,
        "idempotent_err": idem_err,
        "invariant": inv_err < atol,
        "invariance_err": inv_err,
        "rank_eq_trace": rank_matches_trace,
        "rank": numerical_rank,
        "trace": tr,
    }
