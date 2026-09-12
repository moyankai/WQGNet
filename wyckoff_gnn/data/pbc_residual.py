"""Periodic boundary condition minimum Cartesian residual.

Computes the true minimum-image Cartesian distance between two fractional
coordinates under periodic boundary conditions.  For non-orthogonal lattices,
component-wise ``round(diff_frac)`` does NOT always yield the nearest
Cartesian image, so we enumerate a 3×3×3 neighborhood of integer shifts
around the rounded value and pick the one with the smallest Cartesian norm.

Lattice convention: row-vector, ``x_cart = x_frac @ A`` where rows of ``A``
are the lattice vectors.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np


class PBCResidual(NamedTuple):
    """Result of a minimum-image PBC residual computation."""

    residual_cart: float
    """Cartesian distance in Ångströms."""

    residual_frac: np.ndarray
    """The fractional difference that achieves the minimum (shape (3,))."""

    lattice_shift: np.ndarray
    """The integer lattice shift ``n`` that was selected (shape (3,), int)."""


def minimum_image_cartesian_residual(
    transformed_frac: np.ndarray,
    target_frac: np.ndarray,
    lattice: np.ndarray,
) -> PBCResidual:
    """Compute the minimum-image Cartesian residual under PBC.

    Finds::

        min_{n ∈ Z^3} || (transformed_frac - target_frac - n) @ A ||_2

    by enumerating a 3×3×3 neighborhood of integer shifts around
    ``round(transformed_frac - target_frac)``.

    Args:
        transformed_frac: (3,) fractional coordinate after symmetry operation.
        target_frac: (3,) target fractional coordinate.
        lattice: (3, 3) lattice matrix (row vectors), in Ångströms.

    Returns:
        PBCResidual with residual_cart (Å), residual_frac, lattice_shift.
    """
    diff_frac = np.asarray(transformed_frac, dtype=np.float64) - np.asarray(
        target_frac, dtype=np.float64
    )
    A = np.asarray(lattice, dtype=np.float64)

    n_center = np.round(diff_frac).astype(np.int64)

    best_cart = float("inf")
    best_frac = None
    best_shift = None

    for di in range(-1, 2):
        for dj in range(-1, 2):
            for dk in range(-1, 2):
                n = n_center + np.array([di, dj, dk], dtype=np.int64)
                frac_diff = diff_frac - n.astype(np.float64)
                cart_diff = frac_diff @ A
                d = float(np.linalg.norm(cart_diff))
                if d < best_cart:
                    best_cart = d
                    best_frac = frac_diff.copy()
                    best_shift = n.copy()

    return PBCResidual(
        residual_cart=best_cart,
        residual_frac=best_frac,
        lattice_shift=best_shift,
    )


def batch_minimum_image_cartesian_residual(
    transformed_frac: np.ndarray,
    target_frac: np.ndarray,
    lattice: np.ndarray,
) -> PBCResidual:
    """Vectorized version for batched inputs.

    Args:
        transformed_frac: (B, 3) or (3,) fractional coordinates.
        target_frac: (B, 3) or (3,) fractional coordinates.
        lattice: (3, 3) lattice matrix (row vectors).

    Returns:
        PBCResidual with batched arrays (or scalars for single input).
    """
    diff = np.asarray(transformed_frac, dtype=np.float64) - np.asarray(
        target_frac, dtype=np.float64
    )
    A = np.asarray(lattice, dtype=np.float64)

    single = diff.ndim == 1
    if single:
        diff = diff[np.newaxis, :]

    B = diff.shape[0]
    n_center = np.round(diff).astype(np.int64)

    shifts = np.array(
        [[di, dj, dk] for di in range(-1, 2) for dj in range(-1, 2) for dk in range(-1, 2)],
        dtype=np.int64,
    )
    n_candidates = shifts.shape[0]

    n_all = n_center[:, None, :] + shifts[None, :, :]
    frac_diff_all = diff[:, None, :] - n_all.astype(np.float64)
    cart_diff_all = frac_diff_all @ A
    dists_all = np.linalg.norm(cart_diff_all, axis=-1)

    best_idx = np.argmin(dists_all, axis=-1)

    best_carts = dists_all[np.arange(B), best_idx]
    best_fracs = frac_diff_all[np.arange(B), best_idx]
    best_shifts = n_all[np.arange(B), best_idx]

    if single:
        return PBCResidual(
            residual_cart=float(best_carts[0]),
            residual_frac=best_fracs[0],
            lattice_shift=best_shifts[0],
        )
    return PBCResidual(
        residual_cart=best_carts,
        residual_frac=best_fracs,
        lattice_shift=best_shifts,
    )


__all__ = [
    "PBCResidual",
    "minimum_image_cartesian_residual",
    "batch_minimum_image_cartesian_residual",
]
