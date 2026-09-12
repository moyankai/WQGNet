"""Cartesian tensor frame transformations (rank 2 and rank 3).

Transforms symmetric rank-2 tensors (e.g. dielectric) from one Cartesian
frame to another under a proper rotation Q::

    T_std = Q @ T_input @ Q^T

and rank-3 piezoelectric tensors, symmetric in their last two indices::

    T_std[i,j,k] = Q[i,a] Q[j,b] Q[k,c] T_input[a,b,c]

All computation is done in float64 for numerical safety.  Outputs are
returned as float32 for storage.

The module enforces:
- Q must be a proper rotation (Q @ Q^T = I, det(Q) = +1)
- T must have the expected index symmetry (violation below tolerance)
- Invariants (trace, Frobenius norm, det, eigenvalues) are preserved
- Roundtrip recovers the input tensor
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np


__all__ = [
    "validate_rotation_matrix",
    "transform_symmetric_rank2_cartesian",
    "transform_piezo_rank3_voigt",
    "tensor_invariants",
]


def validate_rotation_matrix(
    Q: np.ndarray,
    atol: float = 1e-8,
) -> None:
    """Raise ``ValueError`` if *Q* is not a proper rotation matrix.

    Checks:
    - Shape is (3, 3)
    - All entries are finite
    - Q @ Q^T ≈ I  (orthogonality)
    - det(Q) ≈ +1  (proper rotation, not reflection)
    """
    if not isinstance(Q, np.ndarray):
        raise ValueError(f"Q must be a numpy ndarray, got {type(Q).__name__}")
    if Q.ndim != 2 or Q.shape != (3, 3):
        raise ValueError(f"Q must be (3, 3), got shape {Q.shape}")
    if not np.all(np.isfinite(Q)):
        raise ValueError("Q contains non-finite values (NaN or Inf)")

    QtQ = Q @ Q.T
    orth_err = float(np.max(np.abs(QtQ - np.eye(3))))
    if orth_err > atol:
        raise ValueError(
            f"Q is not orthogonal: max|Q@Q^T - I| = {orth_err:.2e} > {atol:.2e}"
        )

    det_Q = float(np.linalg.det(Q))
    if abs(det_Q - 1.0) > atol:
        raise ValueError(
            f"det(Q) = {det_Q:.10f} is not +1 (deviation {abs(det_Q - 1.0):.2e})"
        )


def tensor_invariants(T: np.ndarray) -> Dict[str, Any]:
    """Compute rotation-invariant properties of a 3×3 symmetric tensor.

    Returns dict with keys:
        trace, frobenius_norm, det, eigenvalues (sorted ascending)
    """
    T = np.asarray(T, dtype=np.float64)
    if T.shape != (3, 3):
        raise ValueError(f"Expected (3, 3) tensor, got {T.shape}")

    eigvals = np.linalg.eigvalsh(T)

    return {
        "trace": float(np.trace(T)),
        "frobenius_norm": float(np.linalg.norm(T, ord="fro")),
        "det": float(np.linalg.det(T)),
        "eigenvalues": eigvals.tolist(),
    }


def transform_symmetric_rank2_cartesian(
    tensor_input: np.ndarray,
    rotation_input_to_standardized: np.ndarray,
    *,
    symmetry_tol: float = 1e-6,
    orthogonality_tol: float = 1e-8,
    strict: bool = True,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Rotate a symmetric rank-2 tensor into the standardized Cartesian frame.

    Parameters
    ----------
    tensor_input : ndarray, shape (3, 3) or (9,)
        Dielectric (or similar) tensor in the input Cartesian frame.
    rotation_input_to_standardized : ndarray, shape (3, 3)
        Proper rotation Q such that ``T_std = Q @ T @ Q^T``.
    symmetry_tol : float
        Maximum allowed Frobenius norm of the antisymmetric part.
    orthogonality_tol : float
        Tolerance for rotation matrix validation.
    strict : bool
        If True, raise on any validation failure.  If False, return
        best-effort result with warnings in the audit dict.

    Returns
    -------
    tensor_std : ndarray, shape (3, 3), dtype float32
        Rotated tensor in the standardized frame.
    audit : dict
        Diagnostic information including invariant preservation errors,
        roundtrip error, and rotation properties.
    """
    Q = np.asarray(rotation_input_to_standardized, dtype=np.float64)
    T = np.asarray(tensor_input, dtype=np.float64).reshape(3, 3)

    audit: Dict[str, Any] = {}

    # --- Validate Q ---
    try:
        validate_rotation_matrix(Q, atol=orthogonality_tol)
        audit["rotation_valid"] = True
    except ValueError as e:
        audit["rotation_valid"] = False
        audit["rotation_error"] = str(e)
        if strict:
            raise

    # --- Symmetrize input ---
    anti = 0.5 * (T - T.T)
    anti_norm = float(np.linalg.norm(anti, ord="fro"))
    audit["antisymmetric_norm_before"] = anti_norm

    if anti_norm > symmetry_tol:
        msg = (
            f"Input tensor has significant antisymmetric part: "
            f"||anti||_F = {anti_norm:.2e} > {symmetry_tol:.2e}"
        )
        if strict:
            raise ValueError(msg)
        audit["symmetrization_warning"] = msg

    T_sym = 0.5 * (T + T.T)

    # --- Invariants before ---
    inv_before = tensor_invariants(T_sym)
    audit["trace_before"] = inv_before["trace"]
    audit["frobenius_before"] = inv_before["frobenius_norm"]
    audit["det_before"] = inv_before["det"]
    audit["eigenvalues_before"] = inv_before["eigenvalues"]

    # --- Coordinate transformation ---
    T_std = Q @ T_sym @ Q.T
    T_std = 0.5 * (T_std + T_std.T)

    # --- Invariants after ---
    inv_after = tensor_invariants(T_std)
    audit["trace_after"] = inv_after["trace"]
    audit["frobenius_after"] = inv_after["frobenius_norm"]
    audit["det_after"] = inv_after["det"]
    audit["eigenvalues_after"] = inv_after["eigenvalues"]

    audit["trace_abs_error"] = abs(inv_after["trace"] - inv_before["trace"])
    audit["frobenius_abs_error"] = abs(
        inv_after["frobenius_norm"] - inv_before["frobenius_norm"]
    )
    audit["det_abs_error"] = abs(inv_after["det"] - inv_before["det"])
    eig_before = np.array(inv_before["eigenvalues"])
    eig_after = np.array(inv_after["eigenvalues"])
    audit["eigenvalue_max_abs_error"] = float(
        np.max(np.abs(eig_after - eig_before))
    )

    # --- Rotation properties ---
    det_Q = float(np.linalg.det(Q))
    audit["det_Q"] = det_Q
    orth_err = float(np.max(np.abs(Q @ Q.T - np.eye(3))))
    audit["orthogonality_error"] = orth_err

    cos_angle = (np.trace(Q) - 1.0) / 2.0
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    audit["rotation_angle_deg"] = float(np.degrees(np.arccos(cos_angle)))

    # --- Roundtrip check ---
    T_roundtrip = Q.T @ T_std @ Q
    roundtrip_err = float(np.max(np.abs(T_roundtrip - T_sym)))
    audit["roundtrip_max_abs_error"] = roundtrip_err

    # --- Tensor change (Cartesian representation change, not physical) ---
    audit["tensor_change_frobenius"] = float(
        np.linalg.norm(T_std - T_sym, ord="fro")
    )

    # --- Antisymmetric norm after ---
    anti_after = 0.5 * (T_std - T_std.T)
    audit["antisymmetric_norm_after"] = float(np.linalg.norm(anti_after, ord="fro"))

    return T_std.astype(np.float32), audit


# Voigt column -> (i, k) Cartesian pair, GMTNet / VASP PIEZO ordering:
#   xx, yy, zz, xy, yz, zx.  No factor of 2 is applied.
_PIEZO_VOIGT_PAIRS = ((0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (0, 2))


def _piezo_voigt_to_cartesian(v: np.ndarray) -> np.ndarray:
    """(3, 6) Voigt -> (3, 3, 3) Cartesian, symmetric in the last two axes."""
    out = np.zeros((3, 3, 3), dtype=np.float64)
    for col, (i, k) in enumerate(_PIEZO_VOIGT_PAIRS):
        out[:, i, k] = v[:, col]
        if i != k:
            out[:, k, i] = v[:, col]
    return out


def _piezo_cartesian_to_voigt(t: np.ndarray) -> np.ndarray:
    """(3, 3, 3) Cartesian -> (3, 6) Voigt (inverse of the above)."""
    return np.stack([t[:, i, k] for i, k in _PIEZO_VOIGT_PAIRS], axis=-1)


def transform_piezo_rank3_voigt(
    tensor_input: np.ndarray,
    rotation_input_to_standardized: np.ndarray,
    *,
    symmetry_tol: float = 1e-6,
    orthogonality_tol: float = 1e-8,
    strict: bool = True,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Rotate a rank-3 piezoelectric tensor into the standardized frame.

    The tensor is supplied and returned in (3, 6) Voigt form using the
    GMTNet / VASP PIEZO column ordering ``xx, yy, zz, xy, yz, zx`` with no
    factor of 2.  Internally it is expanded to the full (3, 3, 3) Cartesian
    tensor, rotated as

        T_std[i,j,k] = Q[i,a] Q[j,b] Q[k,c] T_input[a,b,c],

    and contracted back to Voigt form.

    Parameters
    ----------
    tensor_input : ndarray, shape (3, 6) or (18,)
        Piezoelectric tensor in the input Cartesian frame, Voigt form.
    rotation_input_to_standardized : ndarray, shape (3, 3)
        Proper rotation Q taking the input frame to the standardized frame.
    symmetry_tol : float
        Maximum allowed violation of the last-two-index symmetry after the
        Voigt round trip.
    orthogonality_tol : float
        Tolerance for rotation matrix validation.
    strict : bool
        If True, raise on any validation failure.  If False, return a
        best-effort result with warnings recorded in the audit dict.

    Returns
    -------
    tensor_std : ndarray, shape (3, 6), dtype float32
        Rotated tensor in the standardized frame, Voigt form.
    audit : dict
        Diagnostic information: Frobenius norm before/after (a rotation
        invariant), roundtrip error, and rotation properties.
    """
    Q = np.asarray(rotation_input_to_standardized, dtype=np.float64)
    V = np.asarray(tensor_input, dtype=np.float64).reshape(3, 6)

    audit: Dict[str, Any] = {}

    try:
        validate_rotation_matrix(Q, atol=orthogonality_tol)
        audit["rotation_valid"] = True
    except ValueError as e:
        audit["rotation_valid"] = False
        audit["rotation_error"] = str(e)
        if strict:
            raise

    T = _piezo_voigt_to_cartesian(V)

    # The Voigt form cannot represent an asymmetric last-index pair, so a
    # round trip is exact by construction; check it anyway to catch a
    # mis-specified input shape.
    voigt_roundtrip_err = float(
        np.max(np.abs(_piezo_cartesian_to_voigt(T) - V))
    )
    audit["voigt_roundtrip_max_abs_error"] = voigt_roundtrip_err
    if voigt_roundtrip_err > symmetry_tol:
        msg = (
            f"Voigt round trip changed the tensor by {voigt_roundtrip_err:.2e} "
            f"> {symmetry_tol:.2e}; check the input layout."
        )
        if strict:
            raise ValueError(msg)
        audit["voigt_roundtrip_warning"] = msg

    frob_before = float(np.sqrt((T ** 2).sum()))
    audit["frobenius_before"] = frob_before

    T_std = np.einsum("ia,jb,kc,abc->ijk", Q, Q, Q, T, optimize=True)

    frob_after = float(np.sqrt((T_std ** 2).sum()))
    audit["frobenius_after"] = frob_after
    audit["frobenius_abs_error"] = abs(frob_after - frob_before)

    sym_err = float(np.max(np.abs(T_std - np.swapaxes(T_std, 1, 2))))
    audit["last_index_symmetry_error_after"] = sym_err
    if sym_err > symmetry_tol:
        msg = (
            f"Rotated tensor lost last-index symmetry: {sym_err:.2e} "
            f"> {symmetry_tol:.2e}"
        )
        if strict:
            raise ValueError(msg)
        audit["symmetry_warning"] = msg

    det_Q = float(np.linalg.det(Q))
    audit["det_Q"] = det_Q
    audit["orthogonality_error"] = float(np.max(np.abs(Q @ Q.T - np.eye(3))))
    cos_angle = np.clip((np.trace(Q) - 1.0) / 2.0, -1.0, 1.0)
    audit["rotation_angle_deg"] = float(np.degrees(np.arccos(cos_angle)))

    T_roundtrip = np.einsum("ai,bj,ck,abc->ijk", Q, Q, Q, T_std, optimize=True)
    audit["roundtrip_max_abs_error"] = float(np.max(np.abs(T_roundtrip - T)))

    audit["tensor_change_frobenius"] = float(np.sqrt(((T_std - T) ** 2).sum()))

    V_std = _piezo_cartesian_to_voigt(T_std)
    return V_std.astype(np.float32), audit


def validate_spglib_standardization_frame(
    input_lattice: np.ndarray,
    refined_lattice: np.ndarray,
    transformation_matrix: np.ndarray,
    std_rotation_matrix: np.ndarray,
    dataset_std_lattice: np.ndarray,
    atol: float = 1e-6,
) -> Dict[str, Any]:
    """Validate the spglib standardization relation between lattices.

    For row-vector lattices, the expected relation is::

        std_before = (input_lattice.T @ inv(transformation_matrix)).T
        std_after_reconstructed = std_before @ std_rotation_matrix.T

    And ``std_after_reconstructed ≈ dataset_std_lattice ≈ refined_lattice``.

    Returns a dict with reconstruction errors and pass/fail flags.
    """
    P = np.asarray(transformation_matrix, dtype=np.float64)
    Q = np.asarray(std_rotation_matrix, dtype=np.float64)
    L_in = np.asarray(input_lattice, dtype=np.float64)
    L_ref = np.asarray(refined_lattice, dtype=np.float64)
    L_ds = np.asarray(dataset_std_lattice, dtype=np.float64)

    P_inv = np.linalg.inv(P)
    std_before = (L_in.T @ P_inv).T
    std_after = std_before @ Q.T

    result = {
        "std_before": std_before,
        "std_after_reconstructed": std_after,
        "lattice_reconstruction_error": float(np.max(np.abs(std_after - L_ds))),
        "refine_vs_dataset_error": float(np.max(np.abs(L_ref - L_ds))),
        "lattice_reconstruction_pass": bool(
            np.allclose(std_after, L_ds, atol=atol)
        ),
        "refine_vs_dataset_pass": bool(
            np.allclose(L_ref, L_ds, atol=atol)
        ),
    }
    return result
