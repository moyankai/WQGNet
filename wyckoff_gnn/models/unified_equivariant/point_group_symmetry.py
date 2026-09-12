"""Point group symmetry enforcement for tensor predictions.

Applies group averaging to enforce crystal point group symmetry constraints:
  ε_sym = (1/|G|) Σ_{g∈G} D(g) ε D(g)^T

where G is the point group, D(g) is the rotation matrix for operation g.
This ensures predicted tensors satisfy crystal symmetry (Table 2 & 3 in GMTNet).

LEGACY -- do not use for new work; use crystal_tensor_projection.py instead.

An audit found that ``_get_point_group_rotations`` builds a dummy one-atom cell
from the lattice alone, so spglib returns the *Bravais lattice* point group
rather than the crystal point group, and the ``space_group`` argument this
module receives is never used. Every Bravais lattice is centrosymmetric
(measured: 60/60 sampled crystals, versus 11/60 that are genuinely
centrosymmetric), so this over-constrains rank-2 tensors and would collapse
every crystal to zero at odd rank. Behaviour is frozen as-is so existing
dielectric results stay reproducible; it is disabled by default in all
shipped configs.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import spglib


class PointGroupSymmetryEnforcement(nn.Module):
    """Enforce point group symmetry on predicted tensors via group averaging.

    Args:
        enabled: Whether to apply symmetry enforcement (default True).
        tolerance: Tolerance for symmetry operation extraction (default 1e-5).
    """

    def __init__(self, enabled: bool = True, tolerance: float = 1e-5):
        super().__init__()
        self.enabled = enabled
        self.tolerance = tolerance

    def forward(
        self,
        tensor_pred: torch.Tensor,
        lattice: torch.Tensor,
        space_group: torch.Tensor,
    ) -> torch.Tensor:
        """Apply point group symmetry enforcement.

        Args:
            tensor_pred: (G, 3, 3) predicted symmetric tensors.
            lattice: (G, 3, 3) lattice matrices (row vectors).
            space_group: (G,) space group numbers.

        Returns:
            (G, 3, 3) symmetry-enforced tensors.
        """
        if not self.enabled:
            return tensor_pred

        batch_size = tensor_pred.size(0)
        device = tensor_pred.device
        dtype = tensor_pred.dtype

        enforced = torch.zeros_like(tensor_pred)

        for i in range(batch_size):
            sg = space_group[i].item()
            lat = lattice[i].detach().cpu().numpy()

            # Get point group rotation matrices in Cartesian frame
            rotations_cart = self._get_point_group_rotations(sg, lat, device, dtype)

            # Group averaging: ε_sym = (1/|G|) Σ R ε R^T
            tensor_i = tensor_pred[i]  # (3, 3)
            sym_tensor = torch.zeros_like(tensor_i)

            for R in rotations_cart:
                sym_tensor += R @ tensor_i @ R.T

            sym_tensor /= len(rotations_cart)
            enforced[i] = sym_tensor

        return enforced

    def _get_point_group_rotations(
        self,
        space_group: int,
        lattice: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> list[torch.Tensor]:
        """Extract point group rotation matrices in Cartesian frame.

        Args:
            space_group: Space group number (1-230).
            lattice: (3, 3) lattice matrix (row vectors).
            device: Target device.
            dtype: Target dtype.

        Returns:
            List of (3, 3) rotation matrices in Cartesian frame.
        """
        # Get symmetry operations from spglib
        # Use a dummy structure since we only need the space group operations
        dummy_positions = [[0.0, 0.0, 0.0]]
        dummy_numbers = [1]
        cell = (lattice, dummy_positions, dummy_numbers)

        try:
            symmetry_ops = spglib.get_symmetry(cell, symprec=self.tolerance)
            if symmetry_ops is None:
                # Fallback to identity
                return [torch.eye(3, device=device, dtype=dtype)]

            rotations_frac = symmetry_ops['rotations']  # (N, 3, 3) fractional
        except Exception:
            # Fallback to identity
            return [torch.eye(3, device=device, dtype=dtype)]

        # Convert fractional rotations to Cartesian
        # R_cart = L^T @ R_frac @ L^{-T}
        # where L is lattice matrix (row vectors)
        lat_t = lattice.T  # (3, 3)
        lat_inv_t = torch.inverse(lat_t)  # (3, 3)

        rotations_cart = []
        for R_frac in rotations_frac:
            R_frac_t = torch.tensor(R_frac, device=device, dtype=dtype)
            R_cart = lat_t @ R_frac_t @ lat_inv_t
            rotations_cart.append(R_cart)

        return rotations_cart


def apply_point_group_symmetry(
    tensor_pred: torch.Tensor,
    lattice: torch.Tensor,
    space_group: torch.Tensor,
    enabled: bool = True,
) -> torch.Tensor:
    """Functional interface for point group symmetry enforcement.

    Args:
        tensor_pred: (G, 3, 3) predicted symmetric tensors.
        lattice: (G, 3, 3) lattice matrices.
        space_group: (G,) space group numbers.
        enabled: Whether to apply enforcement.

    Returns:
        (G, 3, 3) symmetry-enforced tensors.
    """
    if not enabled:
        return tensor_pred

    module = PointGroupSymmetryEnforcement(enabled=True)
    return module(tensor_pred, lattice, space_group)
