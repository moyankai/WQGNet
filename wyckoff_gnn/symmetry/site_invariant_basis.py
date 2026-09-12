"""Numerical site-invariant projector and orthonormal basis in the
e3nn real O(3) irrep basis.

Layer 2 + 3 of the four-layer plan for site-symmetry:

    Layer 1  invariant subspace dimension  m_{l,p}^H     (analytic_wyckoff.py)
    Layer 2  invariant projector           P_{l,p}^H     (this module)
    Layer 3  orthonormal invariant basis   B_{l,p}^H     (this module)
    Layer 4  human-readable basis          symbolic      (out of scope)

Given a site stabilizer H (from :mod:`wyckoff_gnn.symmetry.analytic_wyckoff`)
and an ``e3nn.o3.Irrep(l, parity)``, this module builds

    P = (1/|H|) * sum_{h in H}  D_{l,p}(h)     (Reynolds projector)

in e3nn's real-spherical-harmonic irrep basis, then extracts an orthonormal
basis ``B`` for the invariant subspace via a symmetric-eigendecomposition
of ``0.5 (P + P^T)``. Basis columns are non-unique — any orthogonal
recombination ``B Q`` is also a valid basis for the same subspace — so we
only verify the *subspace* through ``B B^T = P`` and the equivariance
``D(h) B = B``.

Both objects sit in **numerical space**. They are the natural companions to
the analytic multiplicity tables: given ``m_{l,p}^H`` we know ``P`` has
trace ``m`` and ``B`` has ``m`` columns; the numerical build then realises
those objects at a chosen fp precision. All returns are ``torch.float64``.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from e3nn import o3


__all__ = [
    "InvariantBlock",
    "BlockValidation",
    "build_stabilizer_D_matrices",
    "reynolds_projector",
    "orthonormal_invariant_basis",
    "validate_block",
    "build_invariant_block",
]


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class InvariantBlock:
    """One (SG, letter, l, parity) invariant block.

    Attributes:
        sg, letter: site identity.
        l, parity: irrep spec (``parity ∈ {'e', 'o'}``).
        dim_ambient: 2l + 1.
        multiplicity: analytic ``m_{l,p}^H`` (dim of the invariant subspace).
        P: (dim_ambient, dim_ambient) Reynolds projector.
        B: (dim_ambient, multiplicity) orthonormal basis. Empty column count
           when multiplicity == 0. Non-unique up to a right-multiplication by
           an ``O(m)`` matrix.
    """
    sg: int
    letter: str
    l: int
    parity: str
    dim_ambient: int
    multiplicity: int
    P: torch.Tensor
    B: torch.Tensor


@dataclass
class BlockValidation:
    """Result of the 7-way sanity check on one (P, B) block."""
    trace_P: float
    trace_P_minus_multiplicity: float          # |trace(P) - m|
    projector_idempotent_err: float            # ||P^2 - P||_inf
    projector_symmetric_err: float             # ||P - P^T||_inf
    projector_equivariance_err: float          # max_h ||D(h) P - P||_inf
    basis_orthonormal_err: float               # ||B^T B - I_m||_inf (0 if m=0)
    basis_reconstructs_P_err: float            # ||B B^T - P||_inf
    basis_equivariance_err: float              # max_h ||D(h) B - B||_inf
    passed: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_P": self.trace_P,
            "trace_P_minus_multiplicity": self.trace_P_minus_multiplicity,
            "projector_idempotent_err": self.projector_idempotent_err,
            "projector_symmetric_err": self.projector_symmetric_err,
            "projector_equivariance_err": self.projector_equivariance_err,
            "basis_orthonormal_err": self.basis_orthonormal_err,
            "basis_reconstructs_P_err": self.basis_reconstructs_P_err,
            "basis_equivariance_err": self.basis_equivariance_err,
            "passed": self.passed,
        }


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def build_stabilizer_D_matrices(
    W_cart_stack: torch.Tensor,
    l: int,
    parity: str,
) -> torch.Tensor:
    """Return (|H|, 2l+1, 2l+1) real O(3) irrep matrices for stabilizer H.

    ``W_cart_stack`` is a real ``(|H|, 3, 3)`` batch of Cartesian rotations
    (or improper rotations) — one per element of H.
    """
    if parity not in ("e", "o"):
        raise ValueError(f"parity must be 'e' or 'o', got {parity!r}")
    ir = o3.Irrep(l, +1 if parity == "e" else -1)
    W = W_cart_stack.to(torch.float32)
    D = ir.D_from_matrix(W).to(torch.float64)
    return D


def reynolds_projector(D_stack: torch.Tensor) -> torch.Tensor:
    """P = (1/|H|) * sum_h D(h). Symmetrised to be exactly symmetric.

    Note: the *raw* Reynolds sum is not automatically symmetric for real
    irrep matrices even though its trace is real. We symmetrise
    P ← 0.5 (P + P^T) before the eigen-decomposition. This is a projector
    onto the same invariant subspace; symmetrisation only cleans up fp
    noise.
    """
    P = D_stack.mean(dim=0)
    P_sym = 0.5 * (P + P.transpose(-1, -2))
    return P_sym


def orthonormal_invariant_basis(
    P: torch.Tensor,
    expected_multiplicity: int,
    eig_tol: float = 1e-4,
) -> torch.Tensor:
    """Extract an orthonormal basis for the range of the projector P.

    Returns a ``(dim, m)`` matrix whose columns span ``Im(P)``. If
    ``expected_multiplicity == 0`` a ``(dim, 0)`` empty tensor is returned.

    We eig-decompose the symmetric ``P``. Since ``P`` is a projector, its
    eigenvalues are ``≈ 1`` on the invariant subspace and ``≈ 0`` outside;
    we take exactly ``expected_multiplicity`` largest eigenvectors after
    verifying that:
      * they all have eigenvalue within ``eig_tol`` of 1,
      * every other eigenvalue is within ``eig_tol`` of 0.
    Deviations from these are surfaced by :func:`validate_block`.
    """
    dim = P.shape[0]
    if expected_multiplicity == 0:
        return P.new_zeros(dim, 0)

    eigvals, eigvecs = torch.linalg.eigh(P.to(torch.float64))
    # eigh returns eigenvalues in ascending order → take the last m columns.
    m = expected_multiplicity
    top_vals = eigvals[-m:]
    top_vecs = eigvecs[:, -m:]

    # Sanity: top eigenvalues near 1, rest near 0.
    if m < dim:
        max_off = float(eigvals[:-m].abs().max().item())
        if max_off > eig_tol:
            # We do NOT raise here — the caller runs validate_block which
            # records the exact miss. This keeps the function usable in
            # research contexts where you want to see the failure surface.
            pass
    min_top = float(top_vals.min().item())
    if abs(min_top - 1.0) > eig_tol:
        pass

    # Orthonormalise. eigh already returns an orthonormal basis, but the
    # extra QR is a safety net for degenerate eigenvalues where columns can
    # be non-orthogonal beyond fp32 precision.
    Q, _ = torch.linalg.qr(top_vecs)
    # Force the sign of the largest-abs entry in each column to be positive
    # for a stable canonical ordering.
    for c in range(Q.shape[1]):
        i = int(Q[:, c].abs().argmax().item())
        if Q[i, c] < 0:
            Q[:, c] = -Q[:, c]
    return Q


def validate_block(
    P: torch.Tensor,
    B: torch.Tensor,
    D_stack: torch.Tensor,
    expected_multiplicity: int,
    tol: float = 1e-4,
) -> BlockValidation:
    """Run the 7 checks recommended for numerical projector/basis blocks.

    Tolerances default to 1e-4 because e3nn's ``D_from_matrix`` internally
    goes through fp32 spherical harmonics; the practical worst case for
    l=8, |H|=48 on this repo is around 3e-5.
    """
    dim = P.shape[0]
    m = B.shape[1]

    trace_P = float(P.diagonal().sum().item())
    trace_diff = abs(trace_P - expected_multiplicity)

    P2 = P @ P
    proj_idem = float((P2 - P).abs().max().item())
    proj_sym = float((P - P.transpose(-1, -2)).abs().max().item())

    if D_stack.numel() > 0:
        DP = torch.einsum("oij,jk->oik", D_stack, P)
        proj_equiv = float((DP - P.unsqueeze(0)).abs().max().item())
    else:
        proj_equiv = 0.0

    if m > 0:
        I_m = torch.eye(m, dtype=B.dtype, device=B.device)
        basis_orth = float((B.transpose(-1, -2) @ B - I_m).abs().max().item())
        BBt = B @ B.transpose(-1, -2)
        basis_reconstruct = float((BBt - P).abs().max().item())
        if D_stack.numel() > 0:
            DB = torch.einsum("oij,jk->oik", D_stack, B)
            basis_equiv = float((DB - B.unsqueeze(0)).abs().max().item())
        else:
            basis_equiv = 0.0
    else:
        basis_orth = 0.0
        # ||B B^T - P|| with B (dim, 0) reduces to ||P||
        basis_reconstruct = float(P.abs().max().item())
        basis_equiv = 0.0

    passed = (
        trace_diff < tol
        and proj_idem < tol
        and proj_sym < tol
        and proj_equiv < tol
        and basis_orth < tol
        and basis_reconstruct < tol
        and basis_equiv < tol
    )
    return BlockValidation(
        trace_P=trace_P,
        trace_P_minus_multiplicity=trace_diff,
        projector_idempotent_err=proj_idem,
        projector_symmetric_err=proj_sym,
        projector_equivariance_err=proj_equiv,
        basis_orthonormal_err=basis_orth,
        basis_reconstructs_P_err=basis_reconstruct,
        basis_equivariance_err=basis_equiv,
        passed=passed,
    )


# ---------------------------------------------------------------------------
# Convenience: build (P, B, validation) from a stabilizer record + irrep
# ---------------------------------------------------------------------------

def build_invariant_block(
    W_cart_stack: torch.Tensor,
    l: int,
    parity: str,
    expected_multiplicity: int,
    sg: int = -1,
    letter: str = "",
    tol: float = 1e-4,
) -> Tuple[InvariantBlock, BlockValidation]:
    """One-shot: from stabilizer Cartesian rotations, produce (P, B) + report."""
    D_stack = build_stabilizer_D_matrices(W_cart_stack, l, parity)
    P = reynolds_projector(D_stack)
    B = orthonormal_invariant_basis(P, expected_multiplicity)
    val = validate_block(P, B, D_stack, expected_multiplicity, tol=tol)
    block = InvariantBlock(
        sg=sg,
        letter=letter,
        l=l,
        parity=parity,
        dim_ambient=P.shape[0],
        multiplicity=expected_multiplicity,
        P=P,
        B=B,
    )
    return block, val
