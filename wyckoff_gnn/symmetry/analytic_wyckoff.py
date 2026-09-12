"""Analytic Wyckoff site-symmetry irrep multiplicity.

For each Wyckoff position of every space group, this module computes the
symbolic site stabilizer H (in the space-group operation set) using exact
rational arithmetic on the affine form of the representative,

    r(u) = a + A u,      u ∈ free parameters ⊂ {x, y, z}

with a ∈ Q^3 and A ∈ Z^{3×3}. A space-group operation g = (W, w) is in H iff

    W A = A                      (exact integer equality; independent of u)
    W a + w - a ∈ Z^3            (exact rational equality mod 1)

Both conditions are decided in **exact arithmetic** — no floating-point
tolerance, no sampled representative.

For each H we then compute the O(3) character-formula multiplicities

    m_{l, p}^H = (1/|H|) Σ_{g ∈ H} χ_{l, p}(g)

where the character χ_{l, p} depends only on (l, parity p, rotation angle θ,
det W_cart):

    proper   (det = +1):  χ_l(θ) = sin((l + 1/2) θ) / sin(θ / 2)
    improper (det = -1):  χ_{l, p} = p · χ_l(θ_proper),  θ_proper = angle of −W_cart

This is the standard O(3) character; it does not care about the specific
Wyckoff embedding beyond the rotation angles + parities in H.

The angles are computed **exactly** in ``sympy`` — every crystallographic
proper rotation has an angle in ``{0, π/3, π/2, 2π/3, π}``, so
``m_{l, p}`` is guaranteed to be an exact non-negative integer.

Outputs of a run consumed by :func:`scripts.generate_analytic_wyckoff` are
strict JSON with rational entries serialised as ``"num/den"``. See
``scripts/generate_analytic_wyckoff.py`` for the driver.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Dict, List, Optional, Tuple

import numpy as np
import sympy as sp


__all__ = [
    "SymbolicRep",
    "StabilizerRecord",
    "affine_from_wp_ops0",
    "sg_operations_exact",
    "compute_site_stabilizer",
    "validate_group_axioms",
    "cartesianize_stabilizer",
    "conjugacy_classes_from_cartesian",
    "character_l_proper_exact",
    "character_lp_exact",
    "multiplicity_lp_exact",
    "wp_letter_to_free_params",
]


# ---------------------------------------------------------------------------
# Rational helpers
# ---------------------------------------------------------------------------

_DENOM_LIMIT = 24  # 1/24 covers 1/2, 1/3, 1/4, 1/6, 1/8, 1/12; safe for crystallography

def _to_fraction(x, denom_limit: int = _DENOM_LIMIT, tol: float = 1e-6) -> Fraction:
    """Convert a numpy/pyxtal float to an exact Fraction, checked for rounding.

    Raises ValueError if the round-trip error exceeds ``tol``; callers must
    then decide whether to reject the space-group entry.
    """
    if isinstance(x, Fraction):
        return x
    if isinstance(x, (int, np.integer)):
        return Fraction(int(x))
    f = Fraction(float(x)).limit_denominator(denom_limit)
    if abs(float(f) - float(x)) > tol:
        raise ValueError(
            f"Cannot round-trip {x!r} to a rational within tol={tol}; "
            f"closest is {f} (error {abs(float(f) - float(x)):.3e})"
        )
    return f


def _mat_to_fractions(M, tol: float = 1e-6) -> List[List[Fraction]]:
    A = np.asarray(M)
    return [[_to_fraction(A[i, j], tol=tol) for j in range(A.shape[1])] for i in range(A.shape[0])]


def _vec_to_fractions(v, tol: float = 1e-6) -> List[Fraction]:
    a = np.asarray(v).flatten()
    return [_to_fraction(x, tol=tol) for x in a]


def _matmul_int(A: List[List[Fraction]], B: List[List[Fraction]]) -> List[List[Fraction]]:
    n, m, p = len(A), len(A[0]), len(B[0])
    out = [[Fraction(0) for _ in range(p)] for _ in range(n)]
    for i in range(n):
        for k in range(m):
            aik = A[i][k]
            if aik == 0:
                continue
            for j in range(p):
                out[i][j] += aik * B[k][j]
    return out


def _matvec(A: List[List[Fraction]], v: List[Fraction]) -> List[Fraction]:
    return [sum((A[i][j] * v[j] for j in range(len(v))), Fraction(0))
            for i in range(len(A))]


def _mat_equal(A, B) -> bool:
    if len(A) != len(B):
        return False
    for i in range(len(A)):
        if len(A[i]) != len(B[i]):
            return False
        for j in range(len(A[i])):
            if A[i][j] != B[i][j]:
                return False
    return True


def _is_integer_vector(v: List[Fraction]) -> bool:
    return all(x.denominator == 1 for x in v)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class SymbolicRep:
    """Symbolic representative of a Wyckoff position: r(u) = a + A u.

    Attributes:
        sg: space group number 1..230.
        letter: Wyckoff letter, e.g. 'a', 'b', ...
        multiplicity: WP multiplicity as reported by pyxtal.
        A: (3, 3) integer matrix in {-1, 0, +1}. Free-parameter columns are
            those with any non-zero entry.
        a: length-3 rational offset.
        free_params: subset of ['x', 'y', 'z'] identifying the free columns of A.
    """
    sg: int
    letter: str
    multiplicity: int
    A: List[List[Fraction]]
    a: List[Fraction]
    free_params: List[str]


@dataclass
class StabilizerRecord:
    """The result of exact stabilizer computation for a Wyckoff position."""
    sg: int
    letter: str
    order: int
    ops_indices: List[int]                # indices into sg_operations_exact
    axioms_ok: bool
    axioms_report: Dict[str, bool] = field(default_factory=dict)
    # Conjugacy classes (populated by cartesianize_stabilizer). Each class
    # is stored as a dict with keys: count, det (+1/-1), theta (sympy expr,
    # for proper rotations), theta_proper (for improper, angle of -W_cart),
    # trace_frac (rational trace of W_frac).
    conj_classes: List[Dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# A: Data source
# ---------------------------------------------------------------------------

_FREE_PARAM_NAMES = ["x", "y", "z"]


def wp_letter_to_free_params(A: List[List[Fraction]]) -> List[str]:
    """Return which of {'x','y','z'} appear as free parameters of A."""
    out = []
    for c in range(3):
        if any(A[r][c] != 0 for r in range(3)):
            out.append(_FREE_PARAM_NAMES[c])
    return out


def affine_from_wp_ops0(sg: int, letter: str, wp, tol: float = 1e-6) -> SymbolicRep:
    """Extract SymbolicRep(a, A) from a pyxtal Wyckoff_position instance.

    The pyxtal ``wp.ops[0]`` encodes the WP's canonical representative as
    an affine map on (x, y, z): ``r(u) = R u + t`` with R ∈ {-1,0,+1}^{3x3}
    (with a few 1/n entries in rare non-orthogonal SGs) and t ∈ Q^3.

    Raises ValueError if any entry cannot be represented as a small rational,
    which is our contract with the "no floats" requirement.
    """
    R = wp.ops[0].rotation_matrix
    t = wp.ops[0].translation_vector
    A = _mat_to_fractions(R, tol=tol)
    a = _vec_to_fractions(t, tol=tol)
    # We require A ∈ Z^{3×3} — pyxtal's WP linear part is always integer.
    for i in range(3):
        for j in range(3):
            if A[i][j].denominator != 1:
                raise ValueError(
                    f"SG{sg} letter {letter}: A[{i}][{j}] = {A[i][j]} is not integer"
                )
    free_params = wp_letter_to_free_params(A)
    return SymbolicRep(
        sg=sg, letter=letter,
        multiplicity=int(wp.multiplicity),
        A=A, a=a, free_params=free_params,
    )


def sg_operations_exact(sg: int, tol: float = 1e-6) -> List[Tuple[List[List[Fraction]], List[Fraction]]]:
    """Return the SG's operations as a list of (W, w) with rational entries.

    Uses ``get_space_group_operations`` (which resolves the Hall-number
    lookup correctly). Every W is guaranteed integer, every w rational
    with denominator ≤ 24.
    """
    from wyckoff_gnn.utils.symmetry import get_space_group_operations
    R, t = get_space_group_operations(sg)
    ops = []
    for k in range(R.shape[0]):
        Wk = _mat_to_fractions(R[k], tol=tol)
        wk = _vec_to_fractions(t[k], tol=tol)
        for i in range(3):
            for j in range(3):
                if Wk[i][j].denominator != 1:
                    raise ValueError(f"SG{sg} op #{k}: non-integer rotation entry")
        ops.append((Wk, wk))
    return ops


# ---------------------------------------------------------------------------
# C: Exact site stabilizer
# ---------------------------------------------------------------------------

def compute_site_stabilizer(
    rep: SymbolicRep,
    sg_ops: List[Tuple[List[List[Fraction]], List[Fraction]]],
) -> StabilizerRecord:
    """Find all g = (W, w) in sg_ops fixing rep for generic free parameters.

    The generic-parameter fixing condition is:

        (i)  W · A = A          (equality of integer matrices — u-independent)
        (ii) W · a + w - a ∈ Z^3

    (i) alone guarantees ``W r(u) - r(u) = W a + w - a`` for every free u,
    so (i) & (ii) together imply generic invariance modulo lattice.
    """
    indices: List[int] = []
    for k, (W, w) in enumerate(sg_ops):
        WA = _matmul_int(W, rep.A)
        if not _mat_equal(WA, rep.A):
            continue
        Wa_w_a = [_ for _ in _matvec(W, rep.a)]
        shift = [Wa_w_a[i] + w[i] - rep.a[i] for i in range(3)]
        if _is_integer_vector(shift):
            indices.append(k)
    return StabilizerRecord(
        sg=rep.sg,
        letter=rep.letter,
        order=len(indices),
        ops_indices=indices,
        axioms_ok=False,
    )


# ---------------------------------------------------------------------------
# D: Group axioms
# ---------------------------------------------------------------------------

def _op_compose(g1, g2, exact: bool = True):
    """Compose g1 ∘ g2 as (W, w) → (W1 W2, W1 w2 + w1) with fractions."""
    W1, w1 = g1
    W2, w2 = g2
    W = _matmul_int(W1, W2)
    w = [_matvec(W1, w2)[i] + w1[i] for i in range(3)]
    return W, w


def _det3(W: List[List[Fraction]]) -> Fraction:
    a, b, c = W[0][0], W[0][1], W[0][2]
    d, e, f = W[1][0], W[1][1], W[1][2]
    g, h, i = W[2][0], W[2][1], W[2][2]
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)


def _op_inverse_rot_only(W: List[List[Fraction]]) -> List[List[Fraction]]:
    """Exact 3x3 rational inverse via adjugate / det.

    Space-group W has det = ±1 and integer entries, so W^{-1} is also
    integer. But this routine works for any invertible rational 3x3.
    """
    a, b, c = W[0][0], W[0][1], W[0][2]
    d, e, f = W[1][0], W[1][1], W[1][2]
    g, h, i = W[2][0], W[2][1], W[2][2]
    det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    if det == 0:
        raise ValueError("Singular rotation matrix (det=0)")
    # Cofactor matrix
    cof = [
        [ (e * i - f * h), -(d * i - f * g),  (d * h - e * g)],
        [-(b * i - c * h),  (a * i - c * g), -(a * h - b * g)],
        [ (b * f - c * e), -(a * f - c * d),  (a * e - b * d)],
    ]
    # Adjugate = transpose of cofactor matrix
    inv = [[cof[j][i] / det for j in range(3)] for i in range(3)]
    return inv


def _canonical_frac(v: List[Fraction]) -> Tuple[Fraction, ...]:
    """Reduce translation modulo 1 to a canonical rep in [0, 1)."""
    return tuple((x - Fraction(int(x.numerator // x.denominator))) % 1 for x in v)


def _op_key(g) -> Tuple:
    W, w = g
    W_key = tuple(tuple(row) for row in W)
    w_key = _canonical_frac(w)
    return (W_key, w_key)


def validate_group_axioms(
    rec: StabilizerRecord,
    sg_ops: List[Tuple[List[List[Fraction]], List[Fraction]]],
) -> StabilizerRecord:
    """Check identity, closure, inverse, and divisibility of the stabilizer.

    Results are written into rec.axioms_report; rec.axioms_ok reflects the
    conjunction.
    """
    idx = rec.ops_indices
    H_ops = [sg_ops[i] for i in idx]
    H_keys = {_op_key(g): pos for pos, g in enumerate(H_ops)}

    # 1. identity
    identity = ([[Fraction(1) if i == j else Fraction(0)
                   for j in range(3)] for i in range(3)],
                [Fraction(0)] * 3)
    has_identity = _op_key(identity) in H_keys

    # 2. closure
    closed = True
    for i, g1 in enumerate(H_ops):
        for j, g2 in enumerate(H_ops):
            k = _op_key(_op_compose(g1, g2))
            if k not in H_keys:
                closed = False
                break
        if not closed:
            break

    # 3. inverse-existence  (for space-group ops modulo lattice)
    has_inverse = True
    for g in H_ops:
        W, w = g
        W_inv = _op_inverse_rot_only(W)
        w_inv = [-_matvec(W_inv, w)[i] for i in range(3)]
        k = _op_key((W_inv, w_inv))
        if k not in H_keys:
            has_inverse = False
            break

    # 4. order divides |G|
    order = len(idx)
    G_order = len(sg_ops)
    divides = (G_order % max(order, 1)) == 0

    rec.axioms_report = {
        "has_identity": has_identity,
        "closed": closed,
        "has_inverse": has_inverse,
        "divides_group_order": divides,
    }
    rec.axioms_ok = has_identity and closed and has_inverse and divides
    return rec


# ---------------------------------------------------------------------------
# E: Character formula (analytic, integer)
# ---------------------------------------------------------------------------

# Standardized-lattice choice for characters. Since Wigner-D characters are
# lattice-invariant (they only see the rotation matrix in Cartesian axes),
# we use a lattice that respects the crystal system's holohedry.

def _standard_lattice_for_sg(sg: int) -> sp.Matrix:
    """Return a hexagonal-vs-cartesian standardized lattice as a sympy Matrix.

    For SGs 143-194 (trigonal/hexagonal families) we use the standard hex cell:
        a = 1, c = 1, γ = 120°.
    All other SGs get identity (cubic-orthogonal representation).

    This is used solely to Cartesianize integer W matrices from spglib.
    """
    if 143 <= sg <= 194:
        # Hex: a1 = (1,0,0), a2 = (-1/2, sqrt(3)/2, 0), a3 = (0,0,1)
        return sp.Matrix([
            [sp.Integer(1), sp.Rational(-1, 2), sp.Integer(0)],
            [sp.Integer(0), sp.sqrt(3) / 2,     sp.Integer(0)],
            [sp.Integer(0), sp.Integer(0),      sp.Integer(1)],
        ])
    return sp.eye(3)


def cartesianize_stabilizer(
    rec: StabilizerRecord,
    sg_ops: List[Tuple[List[List[Fraction]], List[Fraction]]],
) -> StabilizerRecord:
    """Cartesianize each W ∈ H and classify by (det, rotation angle).

    Cartesian W_cart = L · W_frac · L^{-1}, where L is a lattice matrix with
    rows = crystallographic axes. Since we only need the *angle* (via
    trace(W_cart_proper)), the lattice choice matters only up to the crystal
    system. See ``_standard_lattice_for_sg``.

    Each conjugacy class is aggregated into rec.conj_classes with keys:
        count      — number of H elements with this (det, θ)
        det        — +1 or −1
        theta      — sympy expression for the rotation angle in [0, π]
        cos_theta  — sympy exact cos θ (rational)
        trace_cart_proper — trace of the proper part (equals 1 + 2 cos θ)
    """
    L = _standard_lattice_for_sg(rec.sg)
    L_inv = L.inv()

    class_bins: Dict[Tuple, Dict] = {}
    for op_index in rec.ops_indices:
        W_frac, _ = sg_ops[op_index]
        Wf = sp.Matrix([[sp.Rational(W_frac[i][j].numerator, W_frac[i][j].denominator)
                         for j in range(3)] for i in range(3)])
        Wc = L * Wf * L_inv           # Cartesian rotation
        det = sp.simplify(Wc.det())
        det_int = int(det)
        if det_int not in (-1, 1):
            raise ValueError(
                f"SG{rec.sg} letter {rec.letter} op #{op_index}: det(W_cart)={det}, "
                f"expected ±1"
            )
        # Proper-rotation angle: for det=+1, trace = 1 + 2cosθ.
        # For det=-1, trace(-W) = 1 + 2cos(θ_proper).
        Wc_proper = Wc if det_int == 1 else -Wc
        tr = sp.simplify(Wc_proper.trace())
        cos_theta = sp.nsimplify((tr - 1) / 2, rational=False)
        # Clamp float noise to the canonical crystallographic values.
        # cos θ ∈ {-1, -1/2, 0, 1/2, 1}.
        for cand in (sp.Integer(1), sp.Rational(1, 2), sp.Integer(0),
                     sp.Rational(-1, 2), sp.Integer(-1)):
            if sp.simplify(cos_theta - cand) == 0:
                cos_theta = cand
                break
        else:
            # Could be from a hex axis; allow +/- sqrt(3)/2 etc, but that's the
            # angle of an improper op combining hex with a mirror. We keep
            # the exact sympy expression and let acos handle it.
            pass
        theta = sp.acos(cos_theta)
        key = (det_int, sp.simplify(cos_theta))
        bin_ = class_bins.setdefault(key, {
            "count": 0, "det": det_int, "theta": theta,
            "cos_theta": cos_theta,
            "trace_cart_proper": tr,
        })
        bin_["count"] += 1

    rec.conj_classes = [
        {
            "count": v["count"],
            "det": v["det"],
            "theta": str(v["theta"]),
            "cos_theta": str(v["cos_theta"]),
            "trace_cart_proper": str(v["trace_cart_proper"]),
        }
        for v in class_bins.values()
    ]
    # Also keep the sympy values (not serialisable) for the character formula.
    rec._sym_classes = list(class_bins.values())  # type: ignore[attr-defined]
    return rec


def character_l_proper_exact(l: int, cos_theta: sp.Expr) -> sp.Expr:
    """Return χ_l(θ) = sin((l+1/2)θ) / sin(θ/2) as an exact sympy Rational.

    The character for a proper rotation has a well-known closed form. To
    avoid trig instability for θ = 0 we handle it specially.
    """
    if cos_theta == 1:
        return sp.Integer(2 * l + 1)
    # For crystallographic cosθ ∈ {1, 1/2, 0, -1/2, -1}, use the direct sum.
    theta = sp.acos(cos_theta)
    val = sp.sin((sp.Rational(2 * l + 1, 2)) * theta) / sp.sin(theta / 2)
    val = sp.simplify(val)
    val = sp.nsimplify(val, rational=True)
    return val


def character_lp_exact(l: int, parity: str, det: int, cos_theta: sp.Expr) -> sp.Expr:
    """O(3) character χ_{l,p}(g) with parity p ∈ {'e', 'o'}.

    For det = +1: χ = χ_l(θ).
    For det = −1: χ = ±χ_l(θ_proper), where θ_proper is the angle of −W_cart;
    the sign is +1 if parity is 'e' and (−1)^{...} otherwise. In the standard
    convention adopted throughout e3nn/wyckoff_gnn: parity 'e' means the
    irrep of ``O(3)`` on which the inversion acts as +1, parity 'o' as −1.

    Implementation note: for parity 'e' (even), χ_{l,e}(g) = +χ_l(θ_proper);
    for parity 'o' (odd), χ_{l,o}(g) = −χ_l(θ_proper) for improper g.
    """
    if det == 1:
        return character_l_proper_exact(l, cos_theta)
    sign = sp.Integer(1) if parity == "e" else sp.Integer(-1)
    return sign * character_l_proper_exact(l, cos_theta)


def multiplicity_lp_exact(rec: StabilizerRecord, l: int, parity: str) -> int:
    """Compute m_{l,p}^H = (1/|H|) Σ_g χ_{l,p}(g) exactly.

    Requires cartesianize_stabilizer(rec, ...) to have populated rec._sym_classes.
    """
    if not hasattr(rec, "_sym_classes"):
        raise RuntimeError("Call cartesianize_stabilizer(...) first.")
    total = sp.Integer(0)
    for cls in rec._sym_classes:  # type: ignore[attr-defined]
        chi = character_lp_exact(l, parity, cls["det"], cls["cos_theta"])
        total += cls["count"] * chi
    m = sp.simplify(total / rec.order)
    m = sp.nsimplify(m, rational=True)
    if not (m.is_Integer and m >= 0):
        raise ValueError(
            f"Non-integer multiplicity for SG{rec.sg} {rec.letter} l={l} p={parity}: "
            f"got {m} = {sp.N(m)}. This indicates a stabilizer error."
        )
    return int(m)
