"""Symmetric rank-2 tensor adapter: 3x3 <-> irreps (1x0e + 1x2e).

Wraps e3nn's CartesianTensor("ij=ji") for batched training use.
All basis transformations are verified against e3nn's official API —
no hand-written formulas.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from e3nn.io import CartesianTensor


class SymmetricRank2Adapter(nn.Module):
    """Convert between symmetric 3x3 tensors and irreps (1x0e + 1x2e).

    The e3nn CartesianTensor("ij=ji") provides a fixed linear basis for
    the 6-dimensional space of symmetric 3x3 matrices, decomposed as:
        - 1x0e (trace / isotropic part)
        - 1x2e (traceless symmetric / deviatoric part)

    This adapter registers the transformation matrices as buffers so that
    batched forward passes use simple matrix multiplications.

    Attributes:
        irreps_out: "1x0e + 1x2e" (str)
        cart_dim: 6 (number of independent components in irreps space)
    """

    irreps_out = "1x0e + 1x2e"

    def __init__(self):
        super().__init__()
        self._ct = CartesianTensor("ij=ji")

        cart_dim = self._ct.dim  # 6
        self.cart_dim = cart_dim

        # Build the fixed transformation matrices by probing with basis
        # vectors. This is exact and avoids relying on internal APIs.
        # from_cartesian: R^9 -> R^6 (projects symmetric part + change of basis)
        # to_cartesian: R^6 -> R^9 (reconstructs symmetric 3x3)
        # We extract the linear maps as matrices.

        # Probe from_cartesian: symmetric 3x3 -> 6-dim irreps
        # Use the 6 independent symmetric basis matrices
        A_rows = []
        sym_basis = _symmetric_basis()  # (6, 3, 3)
        for i in range(6):
            A_rows.append(self._ct.from_cartesian(sym_basis[i]))
        # from_cartesian acts on flattened 3x3, so we need the full 9->6 matrix
        eye9 = torch.eye(9, dtype=torch.float64)
        B_rows = []
        for i in range(9):
            B_rows.append(self._ct.from_cartesian(eye9[i].reshape(3, 3)))
        self.register_buffer(
            "to_irreps_matrix",
            torch.stack(B_rows, dim=0).float(),  # (9, 6)
        )

        # Probe to_cartesian: 6-dim irreps -> symmetric 3x3 (as flat 9)
        eye6 = torch.eye(6, dtype=torch.float64)
        C_rows = []
        for i in range(6):
            recon = self._ct.to_cartesian(eye6[i])
            C_rows.append(recon.reshape(9))
        self.register_buffer(
            "from_irreps_matrix",
            torch.stack(C_rows, dim=0).float(),  # (6, 9)
        )

        # Symmetrizer: projects arbitrary 3x3 to symmetric (as flat 9)
        # S = 0.5 * (I + P_{01,10}) where P swaps (i,j) and (j,i)
        sym_proj = torch.zeros(9, 9, dtype=torch.float32)
        for i in range(3):
            for j in range(3):
                idx_ij = i * 3 + j
                idx_ji = j * 3 + i
                sym_proj[idx_ij, idx_ij] += 0.5
                sym_proj[idx_ij, idx_ji] += 0.5
        self.register_buffer("symmetrizer", sym_proj)  # (9, 9)

    def from_cartesian(self, tensor_3x3: torch.Tensor) -> torch.Tensor:
        """Convert symmetric 3x3 tensor to irreps representation.

        Args:
            tensor_3x3: (*, 3, 3) symmetric tensor(s). If not perfectly
                symmetric, the symmetric part is used.

        Returns:
            (*, 6) irreps coefficients [1x0e (1) + 1x2e (5)]
        """
        flat = tensor_3x3.reshape(tensor_3x3.shape[:-2] + (9,))
        mat = self.to_irreps_matrix.to(dtype=flat.dtype)
        return flat @ mat  # (*, 6)

    def to_cartesian(self, irreps_coeff: torch.Tensor) -> torch.Tensor:
        """Convert irreps representation back to symmetric 3x3 tensor.

        Args:
            irreps_coeff: (*, 6) irreps coefficients

        Returns:
            (*, 3, 3) symmetric tensor
        """
        flat = irreps_coeff @ self.from_irreps_matrix.to(dtype=irreps_coeff.dtype)  # (*, 9)
        return flat.reshape(irreps_coeff.shape[:-1] + (3, 3))

    def D_from_matrix(self, Q: torch.Tensor) -> torch.Tensor:
        """Get the 6x6 representation matrix for O(3) transformation Q.

        For a symmetric rank-2 tensor T, the action of Q is:
            T' = Q T Q^T
        This method returns D(Q) such that:
            from_cartesian(Q T Q^T) = D(Q) @ from_cartesian(T)

        Args:
            Q: (*, 3, 3) orthogonal matrix/matrices

        Returns:
            (*, 6, 6) representation matrix/matrices
        """
        return self._ct.D_from_matrix(Q)

    def symmetrize(self, tensor_3x3: torch.Tensor) -> torch.Tensor:
        """Symmetrize a 3x3 tensor: T_sym = 0.5 * (T + T^T).

        Args:
            tensor_3x3: (*, 3, 3)

        Returns:
            (*, 3, 3) symmetrized tensor
        """
        return 0.5 * (tensor_3x3 + tensor_3x3.transpose(-1, -2))

    def antisymmetric_norm(self, tensor_3x3: torch.Tensor) -> torch.Tensor:
        """Compute the Frobenius norm of the antisymmetric part.

        A = 0.5 * (T - T^T), returns ||A||_F

        Args:
            tensor_3x3: (*, 3, 3)

        Returns:
            (*,) Frobenius norm of antisymmetric part
        """
        anti = 0.5 * (tensor_3x3 - tensor_3x3.transpose(-1, -2))
        return anti.pow(2).sum(dim=(-1, -2)).sqrt()


def _symmetric_basis() -> torch.Tensor:
    """Return 6 independent symmetric 3x3 basis matrices (float64)."""
    basis = torch.zeros(6, 3, 3, dtype=torch.float64)
    # Diagonal
    basis[0, 0, 0] = 1.0  # e_11
    basis[1, 1, 1] = 1.0  # e_22
    basis[2, 2, 2] = 1.0  # e_33
    # Off-diagonal (symmetric)
    basis[3, 0, 1] = basis[3, 1, 0] = 1.0  # e_12 + e_21
    basis[4, 0, 2] = basis[4, 2, 0] = 1.0  # e_13 + e_31
    basis[5, 1, 2] = basis[5, 2, 1] = 1.0  # e_23 + e_32
    return basis


# Voigt column index -> (i, k) Cartesian pair, in GMTNet / VASP PIEZO ordering:
#   xx, yy, zz, xy, yz, zx
# i.e. columns 0..5 map to (0,0), (1,1), (2,2), (0,1), (1,2), (0,2).
# No factor of 2 or 1/2 is applied (GMTNet data.py / transformer.py).
_VOIGT_PAIRS = [(0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (0, 2)]


def voigt36_to_cartesian333(v: torch.Tensor) -> torch.Tensor:
    """Convert a (..., 3, 6) Voigt matrix to a (..., 3, 3, 3) Cartesian tensor.

    The piezoelectric stress tensor e_{ijk} is stored with rows = j (the
    polarization direction, axis 0) and columns = the symmetric (i, k) Voigt
    pair occupying the LAST TWO Cartesian axes.  Both (i, k) and (k, i)
    entries of the reconstructed tensor receive the same value, with no
    factor of 2 (GMTNet data.py / transformer.py convention).
    """
    cart = torch.zeros(v.shape[:-2] + (3, 3, 3), dtype=v.dtype, device=v.device)
    for col, (i, k) in enumerate(_VOIGT_PAIRS):
        cart[..., :, i, k] = v[..., :, col]
        if i != k:
            cart[..., :, k, i] = v[..., :, col]
    return cart


def cartesian333_to_voigt36(t: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`voigt36_to_cartesian333`."""
    return torch.stack([t[..., :, i, k] for i, k in _VOIGT_PAIRS], dim=-1)


class Rank3PiezoAdapter(nn.Module):
    """Adapter between rank-3 piezoelectric tensors and irreps.

    Wraps e3nn's CartesianTensor("ijk=ikj"), whose irreps are
        2x1o + 1x2o + 1x3o   (18 dimensions),
    and provides lossless Voigt (3, 6) <-> Cartesian (3, 3, 3) conversion
    using the GMTNet / VASP PIEZO column ordering with no factor of 2.

    All basis transformations are probed from e3nn's official API.
    """

    irreps_out = "2x1o + 1x2o + 1x3o"

    def __init__(self):
        super().__init__()
        self._ct = CartesianTensor("ijk=ikj")
        self.cart_dim = self._ct.dim  # 18

        # A single shared ReducedTensorProducts is REQUIRED. e3nn rebuilds one
        # per call when ``rtp=None``, and that default path does not use the
        # same basis as ``D_from_matrix`` for irreps with multiplicity > 1
        # (here 2x1o), which silently breaks equivariance. Probing every basis
        # vector through one shared rtp keeps the two consistent.
        rtp = self._ct.reduced_tensor_products()

        eye_cart = torch.eye(27, dtype=torch.float32).reshape(27, 3, 3, 3)
        rows = [self._ct.from_cartesian(eye_cart[i], rtp=rtp) for i in range(27)]
        self.register_buffer(
            "to_irreps_matrix", torch.stack(rows, dim=0).float())  # (27, 18)

        eye_irr = torch.eye(18, dtype=torch.float32)
        cols = [self._ct.to_cartesian(eye_irr[i], rtp=rtp).reshape(27)
                for i in range(18)]
        self.register_buffer(
            "from_irreps_matrix", torch.stack(cols, dim=0).float())  # (18, 27)

    def from_cartesian(self, tensor_3x3x3: torch.Tensor) -> torch.Tensor:
        """(..., 3, 3, 3) -> (..., 18) irreps coefficients."""
        flat = tensor_3x3x3.reshape(tensor_3x3x3.shape[:-3] + (27,))
        return flat @ self.to_irreps_matrix.to(dtype=flat.dtype)

    def to_cartesian(self, irreps_coeff: torch.Tensor) -> torch.Tensor:
        """(..., 18) irreps -> (..., 3, 3, 3) Cartesian tensor."""
        flat = irreps_coeff @ self.from_irreps_matrix.to(dtype=irreps_coeff.dtype)
        return flat.reshape(irreps_coeff.shape[:-1] + (3, 3, 3))

    def voigt_to_irreps(self, v: torch.Tensor) -> torch.Tensor:
        """(..., 3, 6) Voigt -> (..., 18) irreps."""
        return self.from_cartesian(voigt36_to_cartesian333(v))

    def irreps_to_voigt(self, irreps_coeff: torch.Tensor) -> torch.Tensor:
        """(..., 18) irreps -> (..., 3, 6) Voigt."""
        return cartesian333_to_voigt36(self.to_cartesian(irreps_coeff))

    def D_from_matrix(self, Q: torch.Tensor) -> torch.Tensor:
        """Representation matrix for an O(3) transformation.

        Shapes mirror the input: a single ``(3, 3)`` rotation returns
        ``(18, 18)``, and a batch ``(B, 3, 3)`` returns ``(B, 18, 18)``.
        e3nn prepends a singleton batch axis for unbatched input, which
        silently broadcasts if a caller writes ``D.T``; that is normalized
        away here.
        """
        D = self._ct.D_from_matrix(Q)
        if Q.ndim == 2:
            D = D.reshape(self.cart_dim, self.cart_dim)
        return D
