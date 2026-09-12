"""Expand Wyckoff-orbit vectors/forces to atom-level using symmetry operations.

For a strictly symmetric crystal, each equivalent atom (p, a) of orbit p is
related to the representative by a symmetry operation (R_{p,a}, t_{p,a}).
Vectors transform as:

    polar (force, position, electric field):
        y_{p,a} = R_{p,a} @ y_p

    axial (magnetic moment, angular momentum):
        y_{p,a} = det(R_{p,a}) * R_{p,a} @ y_p

GRADIENT SCALING (critical):
    The model energy E is a function of representative coordinates x_p.
    The gradient from autograd gives::

        g_p = dE / dx_p

    For a symmetry-constrained system where each atom's position is::

        x_{p,a} = R_{p,a} @ x_p + t_{p,a}

    the chain rule gives::

        g_p = sum_a R_{p,a}^T @ (dE / dx_{p,a})
            = sum_a R_{p,a}^T @ (-F_{p,a})
            = sum_a R_{p,a}^T @ (-R_{p,a} @ F_p^rep)
            = - sum_a F_p^rep
            = - m_p * F_p^rep

    Therefore::

        F_p^rep = -g_p / m_p  (NOT just -g_p!)

    and then expand::

        F_{p,a} = R_{p,a} @ F_p^rep

IMPORTANT: This expansion is ONLY valid for strictly symmetric structures.
For MD, phonons, defects, or surface reconstructions, atoms within one
Wyckoff orbit are NOT symmetry-equivalent — use P1 fallback instead.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch


# ---------------------------------------------------------------------------
# Core expansion function
# ---------------------------------------------------------------------------


def expand_orbit_vectors_to_atoms(
    orbit_vectors: torch.Tensor,
    orbit_sym_ops_rotations: torch.Tensor,
    orbit_mult_mask: torch.Tensor,
    atom_to_orbit: torch.Tensor,
    atom_image_index: torch.Tensor,
    vector_type: str = "polar",
) -> torch.Tensor:
    """Expand per-orbit vectors to per-atom vectors using symmetry operations.

    Args:
        orbit_vectors: (K, D) per-orbit representative vectors.
            D=3 for Cartesian vectors, or any dimension for generic features.
        orbit_sym_ops_rotations: (K, max_mult, 3, 3) padded rotation matrices
            (Cartesian, NOT fractional).
        orbit_mult_mask: (K, max_mult) bool mask; True for valid entries.
        atom_to_orbit: (M,) long — maps each atom to its orbit index p.
        atom_image_index: (M,) long — maps each atom to its equivalent index a.
        vector_type: ``"polar"`` or ``"axial"``.

    Returns:
        (M, D) per-atom vectors.

    Raises:
        ValueError: if any atom_image_index exceeds the valid multiplicity
            for its orbit, or if vector_type is unknown.
        RuntimeError: if orbit_vectors are not 3D (for rotation-based expansion)
            and D != 3 (future: general tensor expansion).
    """
    M = atom_to_orbit.shape[0]
    D = orbit_vectors.shape[-1]
    device = orbit_vectors.device

    if atom_image_index.shape[0] != M:
        raise ValueError(
            f"atom_to_orbit ({M}) and atom_image_index "
            f"({atom_image_index.shape[0]}) must have same length."
        )

    # Validate image indices.
    for p in range(orbit_vectors.shape[0]):
        mask_p = (atom_to_orbit == p)
        if mask_p.any():
            max_img = atom_image_index[mask_p].max().item()
            valid_count = int(orbit_mult_mask[p].sum().item())
            if max_img >= valid_count:
                raise ValueError(
                    f"Orbit {p}: atom_image_index max={max_img} >= "
                    f"valid multiplicity={valid_count}. "
                    f"Padding entries must not be used."
                )

    # Fetch rotation matrix for each atom: R_{p, a}  [M, 3, 3].
    R_flat = orbit_sym_ops_rotations[atom_to_orbit, atom_image_index]  # (M, 3, 3)

    if vector_type == "polar":
        # y_{p,a} = R_{p,a} @ y_p   (polar: rotates with the structure)
        y_rep = orbit_vectors[atom_to_orbit]  # (M, D)
        if D == 3:
            atom_vectors = torch.bmm(
                y_rep.unsqueeze(1), R_flat.transpose(1, 2)
            ).squeeze(1)  # (M, 3) = y @ R^T (row convention)
        else:
            raise NotImplementedError(
                f"General D={D} vector expansion not yet implemented. "
                f"Use expand_orbit_tensors_to_atoms for non-Cartesian vectors."
            )
    elif vector_type == "axial":
        det = torch.det(R_flat).unsqueeze(-1).unsqueeze(-1)  # (M, 1, 1)
        R_axial = det * R_flat  # (M, 3, 3)
        y_rep = orbit_vectors[atom_to_orbit]
        if D == 3:
            atom_vectors = torch.bmm(
                y_rep.unsqueeze(1), R_axial.transpose(1, 2)
            ).squeeze(1)
        else:
            raise NotImplementedError(f"Axial expansion only supports D=3 currently.")
    else:
        raise ValueError(
            f"Unknown vector_type='{vector_type}'. Use 'polar' or 'axial'."
        )

    return atom_vectors


# ---------------------------------------------------------------------------
# Gradient to representative force scaling
# ---------------------------------------------------------------------------


def force_rep_from_symmetric_energy_gradient(
    grad_rep: torch.Tensor,
    orbit_multiplicity: torch.Tensor,
) -> torch.Tensor:
    """Compute per-orbit representative force from energy gradient.

    For a strictly symmetric structure where each atom position is::

        x_{p,a} = R_{p,a} @ x_p + t_{p,a}

    The chain rule gives::

        dE/dx_p = sum_a R_{p,a}^T @ (dE/dx_{p,a}) = - m_p * F_p^rep

    Therefore::

        F_p^rep = - (1 / m_p) * dE/dx_p

    Args:
        grad_rep: (K, 3) gradient of energy w.r.t. representative Cartesian
            positions: ``grad = autograd.grad(E, pos_cart)[0]``.
        orbit_multiplicity: (K,) multiplicity m_p for each orbit.

    Returns:
        (K, 3) representative forces.
    """
    return -grad_rep / orbit_multiplicity.unsqueeze(-1).clamp(min=1)


# ---------------------------------------------------------------------------
# Symmetry validation
# ---------------------------------------------------------------------------


def validate_symmetric_atom_vectors(
    atom_vectors: torch.Tensor,
    atom_to_orbit: torch.Tensor,
    atom_image_index: torch.Tensor,
    orbit_sym_ops_rotations: torch.Tensor,
    orbit_mult_mask: torch.Tensor,
    tol: float = 1e-3,
    vector_type: str = "polar",
) -> Tuple[bool, str]:
    """Check that per-atom vectors satisfy orbit symmetry constraints.

    For each orbit p, verifies that for all equivalent atoms a:

        y_{p,a} ≈ R_{p,a} @ y_{p,rep}    (polar)
        y_{p,a} ≈ det(R_{p,a}) R_{p,a} @ y_{p,rep}  (axial)

    where y_{p,rep} is the vector at the representative (image_index=0).

    This can be used to validate DFT force labels before feeding them
    to a Wyckoff-compressed model.

    Args:
        atom_vectors: (M, D) per-atom vectors (e.g. DFT forces).
        atom_to_orbit: (M,) orbit index per atom.
        atom_image_index: (M,) equivalent-image index per atom.
        orbit_sym_ops_rotations: (K, max_mult, 3, 3) padded rotations.
        orbit_mult_mask: (K, max_mult) valid-entry mask.
        tol: Relative tolerance for the symmetry check.
        vector_type: ``"polar"`` or ``"axial"``.

    Returns:
        (passed, message) — True if all symmetry checks pass.
    """
    K = orbit_sym_ops_rotations.shape[0]
    max_err = 0.0
    worst_orbit = 0
    worst_atom = 0

    for p in range(K):
        mask_p = (atom_to_orbit == p)
        if not mask_p.any():
            continue
        rep_idx = torch.where(mask_p & (atom_image_index == 0))[0]
        if len(rep_idx) == 0:
            return False, f"Orbit {p}: no representative atom (image_index=0) found."
        y_rep = atom_vectors[rep_idx[0]]  # (D,)

        for idx in torch.where(mask_p)[0]:
            a = int(atom_image_index[idx])
            if a == 0:
                continue
            R = orbit_sym_ops_rotations[p, a]  # (3, 3)
            if not orbit_mult_mask[p, a]:
                continue

            expected = R @ y_rep if vector_type == "polar" else torch.det(R) * R @ y_rep
            actual = atom_vectors[idx]
            err = (expected - actual).norm() / expected.norm().clamp(min=1e-8)
            if err > max_err:
                max_err = float(err)
                worst_orbit = p
                worst_atom = a

    if max_err > tol:
        return (
            False,
            f"Symmetry violated: max rel err={max_err:.6f} "
            f"at orbit={worst_orbit}, atom_image={worst_atom}",
        )
    return True, f"All symmetry checks passed (max err={max_err:.6f})."


# ---------------------------------------------------------------------------
# Tensor expansion (placeholder)
# ---------------------------------------------------------------------------


def expand_orbit_tensors_to_atoms(
    orbit_tensors,
    orbit_sym_ops_rotations,
    orbit_mult_mask,
    atom_to_orbit,
    atom_image_index,
    rank=2,
):
    """Expand per-orbit rank-l tensors to per-atom.

    NOT YET IMPLEMENTED.  Will use Wigner-D matrices for l>1 expansion.
    For now, raises NotImplementedError with a message pointing to the
    planned implementation strategy.
    """
    raise NotImplementedError(
        "expand_orbit_tensors_to_atoms: rank>1 tensor expansion not yet "
        "implemented.  For rank-2 (stress-like) outputs, each orbit's "
        "representative tensor contribution must be expanded via "
        "D^{(2)}(R_{p,a}) @ sigma_p @ D^{(2)}(R_{p,a})^T."
    )


__all__ = [
    "expand_orbit_vectors_to_atoms",
    "force_rep_from_symmetric_energy_gradient",
    "validate_symmetric_atom_vectors",
    "expand_orbit_tensors_to_atoms",
]
