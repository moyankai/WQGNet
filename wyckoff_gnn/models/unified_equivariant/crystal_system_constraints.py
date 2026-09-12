"""Hard symmetry constraints for dielectric tensor by crystal system.

Instead of post-hoc group averaging, this module restricts the model output
to only predict the independent parameters allowed by the crystal system.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import numpy as np


def get_crystal_system(space_group: int) -> str:
    """Map space group number to crystal system."""
    if space_group <= 2:
        return "triclinic"
    elif space_group <= 15:
        return "monoclinic"
    elif space_group <= 74:
        return "orthorhombic"
    elif space_group <= 142:
        return "tetragonal"
    elif space_group <= 162:
        return "trigonal"
    elif space_group <= 194:
        return "hexagonal"
    else:
        return "cubic"


def get_independent_params_count(space_group: int) -> int:
    """Get number of independent parameters for dielectric tensor.

    Dielectric tensor is symmetric 3x3, with constraints by crystal system:
    - Cubic: 1 (ε11=ε22=ε33)
    - Hexagonal/Tetragonal: 2 (ε11=ε22, ε33)
    - Trigonal: 3 (ε11=ε22, ε33, ε12)
    - Orthorhombic: 3 (ε11, ε22, ε33)
    - Monoclinic: 4 (ε11, ε22, ε33, ε13)
    - Triclinic: 6 (full symmetric)
    """
    system = get_crystal_system(space_group)
    counts = {
        "cubic": 1,
        "hexagonal": 2,
        "tetragonal": 2,
        "trigonal": 3,
        "orthorhombic": 3,
        "monoclinic": 4,
        "triclinic": 6,
    }
    return counts[system]


def reconstruct_tensor(
    params: torch.Tensor, space_group: int
) -> torch.Tensor:
    """Reconstruct 3x3 symmetric tensor from independent parameters.

    Args:
        params: (B, N) where N is number of independent params for this SG
        space_group: Space group number (1-230)

    Returns:
        tensor: (B, 3, 3) symmetric tensor
    """
    system = get_crystal_system(space_group)
    B = params.shape[0]
    device = params.device
    dtype = params.dtype

    if system == "cubic":
        # ε = diag(a, a, a)
        a = params[:, 0]
        tensor = torch.zeros(B, 3, 3, device=device, dtype=dtype)
        tensor[:, 0, 0] = a
        tensor[:, 1, 1] = a
        tensor[:, 2, 2] = a

    elif system in ("hexagonal", "tetragonal"):
        # ε = diag(a, a, c)
        a, c = params[:, 0], params[:, 1]
        tensor = torch.zeros(B, 3, 3, device=device, dtype=dtype)
        tensor[:, 0, 0] = a
        tensor[:, 1, 1] = a
        tensor[:, 2, 2] = c

    elif system == "trigonal":
        # ε = [[a, d, 0], [d, a, 0], [0, 0, c]] (hexagonal setting)
        a, c, d = params[:, 0], params[:, 1], params[:, 2]
        tensor = torch.zeros(B, 3, 3, device=device, dtype=dtype)
        tensor[:, 0, 0] = a
        tensor[:, 1, 1] = a
        tensor[:, 2, 2] = c
        tensor[:, 0, 1] = d
        tensor[:, 1, 0] = d

    elif system == "orthorhombic":
        # ε = diag(a, b, c)
        a, b, c = params[:, 0], params[:, 1], params[:, 2]
        tensor = torch.zeros(B, 3, 3, device=device, dtype=dtype)
        tensor[:, 0, 0] = a
        tensor[:, 1, 1] = b
        tensor[:, 2, 2] = c

    elif system == "monoclinic":
        # ε = [[a, 0, d], [0, b, 0], [d, 0, c]] (unique axis b)
        a, b, c, d = params[:, 0], params[:, 1], params[:, 2], params[:, 3]
        tensor = torch.zeros(B, 3, 3, device=device, dtype=dtype)
        tensor[:, 0, 0] = a
        tensor[:, 1, 1] = b
        tensor[:, 2, 2] = c
        tensor[:, 0, 2] = d
        tensor[:, 2, 0] = d

    elif system == "triclinic":
        # Full symmetric: [[a, d, e], [d, b, f], [e, f, c]]
        a, b, c, d, e, f = (
            params[:, 0], params[:, 1], params[:, 2],
            params[:, 3], params[:, 4], params[:, 5],
        )
        tensor = torch.zeros(B, 3, 3, device=device, dtype=dtype)
        tensor[:, 0, 0] = a
        tensor[:, 1, 1] = b
        tensor[:, 2, 2] = c
        tensor[:, 0, 1] = tensor[:, 1, 0] = d
        tensor[:, 0, 2] = tensor[:, 2, 0] = e
        tensor[:, 1, 2] = tensor[:, 2, 1] = f

    else:
        raise ValueError(f"Unknown crystal system: {system}")

    return tensor


class CrystalSystemTensorReadout(nn.Module):
    """Tensor readout with hard symmetry constraints by crystal system.

    Instead of always predicting 6 parameters (full symmetric tensor),
    this module predicts only the independent parameters allowed by
    the crystal system, then reconstructs the full tensor.
    """

    def __init__(self, scalar_dim: int, max_params: int = 6):
        """Initialize.

        Args:
            scalar_dim: Dimension of input scalar features
            max_params: Maximum number of independent parameters (default 6)
        """
        super().__init__()
        self.max_params = max_params
        # Always output max_params, but only use the first N based on crystal system
        self.param_head = nn.Linear(scalar_dim, max_params)

    def forward(
        self,
        scalar_features: torch.Tensor,
        batch_idx: torch.Tensor,
        space_groups: torch.Tensor,
        multiplicity: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with crystal system constraints.

        Args:
            scalar_features: (N, D) node scalar features
            batch_idx: (N,) batch index for each node
            space_groups: (B,) space group for each graph in batch
            multiplicity: (N,) optional multiplicity for weighted pooling

        Returns:
            dict with 'params' (B, max_params) and 'cartesian' (B, 3, 3)
        """
        # Predict all parameters
        node_params = self.param_head(scalar_features)  # (N, 6)

        # Pool to graph level
        if multiplicity is not None:
            # Multiplicity-weighted pooling
            weighted = node_params * multiplicity.unsqueeze(1)
            graph_params = torch.zeros(
                space_groups.shape[0], self.max_params,
                device=node_params.device, dtype=node_params.dtype,
            )
            graph_params.index_add_(0, batch_idx, weighted)
            norm = torch.zeros(
                space_groups.shape[0], 1,
                device=node_params.device, dtype=node_params.dtype,
            )
            norm.index_add_(0, batch_idx, multiplicity.unsqueeze(1))
            graph_params = graph_params / (norm + 1e-8)
        else:
            # Simple sum pooling
            graph_params = torch.zeros(
                space_groups.shape[0], self.max_params,
                device=node_params.device, dtype=node_params.dtype,
            )
            graph_params.index_add_(0, batch_idx, node_params)

        # Reconstruct tensor for each graph based on its space group
        tensors = []
        for i in range(space_groups.shape[0]):
            sg = space_groups[i].item()
            n_params = get_independent_params_count(sg)
            # Only use first n_params, zero out the rest
            params_i = graph_params[i, :n_params]
            tensor_i = reconstruct_tensor(params_i.unsqueeze(0), sg)
            tensors.append(tensor_i)

        cartesian = torch.cat(tensors, dim=0)  # (B, 3, 3)

        return {
            "params": graph_params,  # (B, 6) - all predicted params
            "cartesian": cartesian,  # (B, 3, 3) - reconstructed with constraints
        }


__all__ = [
    "get_crystal_system",
    "get_independent_params_count",
    "reconstruct_tensor",
    "CrystalSystemTensorReadout",
]
