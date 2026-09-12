"""Irrep <-> Cartesian tensor conversions for property prediction.

Provides utilities to convert between e3nn irrep representations and their
Cartesian tensor equivalents:

    0e   <->  scalar                    (rank 0)
    1o   <->  polar vector              (rank 1, odd parity)
    1e   <->  axial vector              (pseudovector, rank 1, even parity)
    0e+2e <->  symmetric rank-2 tensor  (trace + traceless symmetric)
    0e+1e+2e <->  general rank-2 tensor (trace + antisymmetric + traceless)

For rank > 2 the interface is reserved but not implemented; the general recipe
is to construct the tensor via successive tensor products of vectors and
project onto the desired irreducible representations.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from e3nn import o3


__all__ = [
    "irreps_1o_to_cartesian",
    "cartesian_to_irreps_1o",
    "irreps_1e_to_cartesian",
    "cartesian_to_irreps_1e",
    "irreps_0e2e_to_symmetric_tensor",
    "symmetric_tensor_to_irreps_0e2e",
    "irreps_0e1e2e_to_general_tensor",
    "general_tensor_to_irreps_0e1e2e",
    "get_irreps_for_tensor",
]


# ---------------------------------------------------------------------------
# Rank-1: vector conversions
# ---------------------------------------------------------------------------
# e3nn uses the y, z, x order for l=1 (spherical harmonics: Y_{1,-1}=y, Y_{1,0}=z, Y_{1,1}=x).
# So to convert to Cartesian (x, y, z) we need to reorder.

_L1_TO_CART_PERM = torch.tensor([2, 0, 1], dtype=torch.long)  # [x, y, z] indices in irrep order
_CART_TO_L1_PERM = torch.tensor([1, 2, 0], dtype=torch.long)  # [y_irrep, z_irrep, x_irrep] indices in cart order


def irreps_1o_to_cartesian(vec_1o: torch.Tensor) -> torch.Tensor:
    """Convert 1o irrep (y, z, x order) to Cartesian polar vector (x, y, z).

    Args:
        vec_1o: (..., 3) tensor in e3nn l=1 spherical order.

    Returns:
        (..., 3) tensor in Cartesian (x, y, z) order.
    """
    return vec_1o[..., _L1_TO_CART_PERM]


def cartesian_to_irreps_1o(vec_cart: torch.Tensor) -> torch.Tensor:
    """Convert Cartesian polar vector (x, y, z) to 1o irrep (y, z, x order).

    Args:
        vec_cart: (..., 3) tensor in Cartesian order.

    Returns:
        (..., 3) tensor in e3nn l=1 order.
    """
    return vec_cart[..., _CART_TO_L1_PERM]


def irreps_1e_to_cartesian(vec_1e: torch.Tensor) -> torch.Tensor:
    """Convert 1e irrep (axial vector) to Cartesian pseudovector.

    Same reordering as 1o (only parity differs, coordinates layout is same).
    """
    return irreps_1o_to_cartesian(vec_1e)


def cartesian_to_irreps_1e(vec_cart: torch.Tensor) -> torch.Tensor:
    """Convert Cartesian pseudovector to 1e irrep."""
    return cartesian_to_irreps_1o(vec_cart)


# ---------------------------------------------------------------------------
# Rank-2: symmetric tensor (0e + 2e)
# ---------------------------------------------------------------------------
# A symmetric 3x3 tensor T decomposes into:
#   trace part:      0e =  T_xx + T_yy + T_zz    (1 component)
#   traceless part:  2e = symmetric traceless    (5 components)
#
# e3nn's l=2 basis is a specific linear combination of Cartesian symmetric traceless
# components. The transformation is:
#   Y_{2, -2} = (x*y + y*x) / 2 = xy                       (basis Y_1)
#   Y_{2, -1} = (y*z + z*y) / 2 = yz                       (basis Y_2)
#   Y_{2, 0}  = (3z^2 - r^2) / 2                            (basis Y_3)
#   Y_{2, +1} = (x*z + z*x) / 2 = xz                       (basis Y_4)
#   Y_{2, +2} = (x^2 - y^2) / 2                            (basis Y_5)
#
# We use the standard Racah normalization consistent with e3nn.


def _get_l2_cartesian_basis() -> torch.Tensor:
    """Return the 5x9 matrix M such that l=2 components = M @ vec(T_symtraceless).

    T_symtraceless is stored as (T_xx, T_xy, T_xz, T_yx, T_yy, T_yz, T_zx, T_zy, T_zz)
    (flattened 3x3 row-major).

    e3nn l=2 spherical harmonics in standard order:
        m=-2: xy
        m=-1: yz
        m= 0: (2z^2 - x^2 - y^2) / (2 sqrt(3))    # with e3nn normalization
        m=+1: xz
        m=+2: (x^2 - y^2) / 2

    We construct M as change-of-basis. Rather than deriving analytically (which
    e3nn also handles internally), we use e3nn's basis directly:
    """
    # The matrix maps the 9-dim vec of a symmetric 3x3 tensor to l=0+l=2 irrep coords.
    # We construct it by matching e3nn's l=2 basis.

    # Using standard 5-basis for l=2 symmetric traceless part (real spherical harmonics form):
    # b_m2 (xy):     1/sqrt(2) * (e_x e_y + e_y e_x)
    # b_m1 (yz):     1/sqrt(2) * (e_y e_z + e_z e_y)
    # b_0  (z^2):    1/sqrt(6) * (2 e_z e_z - e_x e_x - e_y e_y)   # normalized
    # b_p1 (xz):     1/sqrt(2) * (e_x e_z + e_z e_x)
    # b_p2 (x2-y2):  1/sqrt(2) * (e_x e_x - e_y e_y)               # normalized

    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    inv_sqrt6 = 1.0 / np.sqrt(6.0)

    # M is (5, 9) — rows are the 5 l=2 basis vectors flattened as 3x3.
    # 9-dim vec: [xx, xy, xz, yx, yy, yz, zx, zy, zz]
    #             0    1   2   3   4   5   6   7   8
    M = np.zeros((5, 9))

    # m=-2 (xy) : 1/sqrt(2) * (xy + yx)
    M[0, 1] = inv_sqrt2
    M[0, 3] = inv_sqrt2

    # m=-1 (yz) : 1/sqrt(2) * (yz + zy)
    M[1, 5] = inv_sqrt2
    M[1, 7] = inv_sqrt2

    # m=0 (z^2-...) : 1/sqrt(6) * (2 zz - xx - yy)
    M[2, 0] = -inv_sqrt6
    M[2, 4] = -inv_sqrt6
    M[2, 8] = 2.0 * inv_sqrt6

    # m=+1 (xz) : 1/sqrt(2) * (xz + zx)
    M[3, 2] = inv_sqrt2
    M[3, 6] = inv_sqrt2

    # m=+2 (x^2 - y^2) : 1/sqrt(2) * (xx - yy)
    M[4, 0] = inv_sqrt2
    M[4, 4] = -inv_sqrt2

    return torch.from_numpy(M).float()


_L2_CART_BASIS = _get_l2_cartesian_basis()  # (5, 9)


def symmetric_tensor_to_irreps_0e2e(T: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Decompose a symmetric 3x3 tensor into 0e (trace/3) and 2e (traceless).

    Args:
        T: (..., 3, 3) symmetric tensor.

    Returns:
        (scalar_0e, tensor_2e):
            scalar_0e: (..., 1) trace / sqrt(3)
            tensor_2e: (..., 5) traceless symmetric part in e3nn l=2 order
    """
    shape = T.shape[:-2]
    T_flat = T.reshape(-1, 9)  # (N, 9)

    # Trace part: (Txx + Tyy + Tzz) / sqrt(3)
    trace = (T_flat[:, 0] + T_flat[:, 4] + T_flat[:, 8]) / np.sqrt(3.0)  # (N,)
    scalar_0e = trace.unsqueeze(-1)  # (N, 1)

    # Traceless symmetric part: T_st = T - trace/3 * I, then project to 5 l=2 basis
    T_st = T_flat.clone()
    T_st[:, 0] -= T_flat[:, 0] / 3.0 + T_flat[:, 4] / 3.0 + T_flat[:, 8] / 3.0
    T_st[:, 4] -= T_flat[:, 0] / 3.0 + T_flat[:, 4] / 3.0 + T_flat[:, 8] / 3.0
    T_st[:, 8] -= T_flat[:, 0] / 3.0 + T_flat[:, 4] / 3.0 + T_flat[:, 8] / 3.0

    # Project to l=2 basis
    M = _L2_CART_BASIS.to(T.device, T.dtype)  # (5, 9)
    tensor_2e = T_st @ M.T  # (N, 5)

    return scalar_0e.reshape(*shape, 1), tensor_2e.reshape(*shape, 5)


def irreps_0e2e_to_symmetric_tensor(
    scalar_0e: torch.Tensor,
    tensor_2e: torch.Tensor,
) -> torch.Tensor:
    """Reconstruct a symmetric 3x3 tensor from 0e + 2e irreps.

    Args:
        scalar_0e: (..., 1) scalar irrep.
        tensor_2e: (..., 5) traceless symmetric part in e3nn l=2 order.

    Returns:
        (..., 3, 3) symmetric tensor.
    """
    shape = scalar_0e.shape[:-1]
    s_flat = scalar_0e.reshape(-1, 1)  # (N, 1)
    t_flat = tensor_2e.reshape(-1, 5)  # (N, 5)

    # 0e -> trace/sqrt(3) * I
    trace_component = s_flat * np.sqrt(3.0)  # (N, 1)
    I_flat = torch.zeros(t_flat.shape[0], 9, device=scalar_0e.device, dtype=scalar_0e.dtype)
    I_flat[:, 0] = I_flat[:, 4] = I_flat[:, 8] = 1.0 / 3.0
    trace_part = trace_component * I_flat  # (N, 9)

    # 2e -> traceless part via M^T (pseudo-inverse of M)
    M = _L2_CART_BASIS.to(scalar_0e.device, scalar_0e.dtype)  # (5, 9)
    # M @ M.T should be diagonal (5,5) since basis vectors are orthonormal wrt Frobenius
    # For orthonormal M: reconstruction is t_flat @ M
    traceless_part = t_flat @ M  # (N, 9)

    T_flat = trace_part + traceless_part
    return T_flat.reshape(*shape, 3, 3)


# ---------------------------------------------------------------------------
# Rank-2: general tensor (0e + 1e + 2e)
# ---------------------------------------------------------------------------

def general_tensor_to_irreps_0e1e2e(T: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decompose a general 3x3 tensor into 0e (trace), 1e (antisymm axial vector), 2e (traceless sym).

    Args:
        T: (..., 3, 3) general tensor.

    Returns:
        (scalar_0e, vector_1e, tensor_2e):
            scalar_0e: (..., 1)
            vector_1e: (..., 3) — axial vector from antisymmetric part
            tensor_2e: (..., 5)
    """
    # Symmetric + antisymmetric split
    T_sym = 0.5 * (T + T.transpose(-1, -2))
    T_asym = 0.5 * (T - T.transpose(-1, -2))

    # Symmetric -> 0e + 2e
    scalar_0e, tensor_2e = symmetric_tensor_to_irreps_0e2e(T_sym)

    # Antisymmetric 3x3 has 3 independent components → axial vector
    # A = [[0, a, b], [-a, 0, c], [-b, -c, 0]] → axial = (c, -b, a) in Cartesian (x, y, z)
    # We want vector_1e in e3nn irrep order (y, z, x). See irreps_1e_to_cartesian for perm.
    a_xy = T_asym[..., 0, 1]  # A[0,1] = a
    a_xz = T_asym[..., 0, 2]  # A[0,2] = b
    a_yz = T_asym[..., 1, 2]  # A[1,2] = c
    # Cartesian axial vector: (c, -b, a) = (yz, -xz, xy)
    axial_cart = torch.stack([a_yz, -a_xz, a_xy], dim=-1)  # (..., 3)
    vector_1e = cartesian_to_irreps_1e(axial_cart)

    return scalar_0e, vector_1e, tensor_2e


def irreps_0e1e2e_to_general_tensor(
    scalar_0e: torch.Tensor,
    vector_1e: torch.Tensor,
    tensor_2e: torch.Tensor,
) -> torch.Tensor:
    """Reconstruct a general 3x3 tensor from (0e, 1e, 2e) irreps.

    Args:
        scalar_0e: (..., 1) trace/sqrt(3).
        vector_1e: (..., 3) axial vector (e3nn 1e order).
        tensor_2e: (..., 5) traceless symmetric part (e3nn l=2 order).

    Returns:
        (..., 3, 3) general tensor.
    """
    # Symmetric part
    T_sym = irreps_0e2e_to_symmetric_tensor(scalar_0e, tensor_2e)

    # Antisymmetric part from axial vector
    axial_cart = irreps_1e_to_cartesian(vector_1e)  # (..., 3)
    a_yz = axial_cart[..., 0]
    a_xz = -axial_cart[..., 1]
    a_xy = axial_cart[..., 2]

    T_asym = torch.zeros_like(T_sym)
    T_asym[..., 0, 1] = a_xy
    T_asym[..., 1, 0] = -a_xy
    T_asym[..., 0, 2] = a_xz
    T_asym[..., 2, 0] = -a_xz
    T_asym[..., 1, 2] = a_yz
    T_asym[..., 2, 1] = -a_yz

    return T_sym + T_asym


# ---------------------------------------------------------------------------
# Convenience: irreps for common tensor types
# ---------------------------------------------------------------------------

def get_irreps_for_tensor(tensor_type: str) -> o3.Irreps:
    """Return the e3nn Irreps for a common tensor type.

    Recognized types:
        "scalar" -> 0e
        "pseudoscalar" -> 0o
        "polar_vector" / "vector" -> 1o
        "axial_vector" / "pseudovector" -> 1e
        "symmetric_tensor" -> 0e+2e
        "general_tensor" -> 0e+1e+2e
    """
    mapping = {
        "scalar": "1x0e",
        "pseudoscalar": "1x0o",
        "polar_vector": "1x1o",
        "vector": "1x1o",
        "axial_vector": "1x1e",
        "pseudovector": "1x1e",
        "symmetric_tensor": "1x0e+1x2e",
        "general_tensor": "1x0e+1x1e+1x2e",
    }
    if tensor_type not in mapping:
        raise ValueError(
            f"Unknown tensor_type '{tensor_type}'. "
            f"Available: {list(mapping.keys())}"
        )
    return o3.Irreps(mapping[tensor_type])
