"""Multi-property loss functions for WyckoffGNN.

Dispatches the correct loss function based on ``property_type``, so the
training loop does not need type-specific branching. Each loss returns a
scalar tensor suitable for ``.backward()``.

Supported property types and their losses:

    graph_scalar_intensive / graph_scalar_extensive:
        - mae (L1)
        - mse (L2)
        - huber (smooth L1, delta=0.1)

    graph_vector:
        - mse (component-wise MSE)
        - cosine (1 - cosine similarity, plus L2 on magnitudes)

    graph_tensor:
        - mse (component-wise MSE in irrep basis)
        - frobenius (Frobenius norm of difference after Cartesian reconstruction)

    atom_scalar / atom_vector / atom_tensor:
        - Same as graph counterparts, but applied per-atom and averaged.

    hamiltonian:
        - block_frobenius (sum of Frobenius norms of (pred - ref) blocks)
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

import torch
import torch.nn.functional as F

from wyckoff_gnn.data.property_types import PropertyType, validate_property_type


__all__ = ["compute_loss", "get_loss_fn"]


def compute_loss(
    pred: Union[torch.Tensor, Dict[str, torch.Tensor]],
    batch: Any,
    property_type: str,
    loss_name: str = "mae",
    loss_config: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    """Unified loss dispatch.

    Args:
        pred: model output (shape depends on property_type).
        batch: PyG Batch object carrying targets.
        property_type: canonical property type string.
        loss_name: loss function name (mae, mse, huber, cosine, frobenius, block_frobenius).
        loss_config: optional extra kwargs (e.g. huber delta, crystal_system_constraints).

    Returns:
        Scalar loss tensor.
    """
    pt = validate_property_type(property_type)
    loss_config = loss_config or {}

    if pt in (PropertyType.GRAPH_SCALAR_INTENSIVE, PropertyType.GRAPH_SCALAR_EXTENSIVE):
        target = batch.y.view(-1).float()
        pred_flat = pred.view(-1)
        return _scalar_loss(pred_flat, target, loss_name, loss_config)

    if pt == PropertyType.GRAPH_VECTOR:
        target = _get_target_field(batch, "y_vector", pred.shape)
        return _vector_loss(pred, target, loss_name, loss_config)

    if pt == PropertyType.GRAPH_TENSOR:
        if isinstance(pred, dict):
            tensor_out = pred["tensor"]
            if isinstance(tensor_out, dict):
                # Rank-3 heads expose "voigt" (G,3,6), matching the stored
                # y_tensor layout; rank-2 heads only have "cartesian".
                pred_cart = tensor_out.get("voigt", tensor_out.get("cartesian"))
            else:
                pred_cart = tensor_out
            pred_flat = pred_cart.reshape(pred_cart.size(0), -1)
            target = _get_target_field(batch, "y_tensor", pred_flat.shape)

            # Check if crystal system constraints are enabled
            if loss_config.get("crystal_system_constraints", False):
                space_groups = getattr(batch, "space_group", None)
                if space_groups is not None:
                    return _tensor_loss_crystal_system(
                        pred_cart, target.reshape(-1, 3, 3), space_groups, loss_name, loss_config
                    )

            return _tensor_loss(pred_flat, target, loss_name, loss_config)
        target = _get_target_field(batch, "y_tensor", pred.shape)
        return _tensor_loss(pred, target, loss_name, loss_config)

    if pt == PropertyType.ATOM_SCALAR:
        target = _get_target_field(batch, "y_atom_scalar", pred.shape)
        return _scalar_loss(pred.view(-1), target.view(-1), loss_name, loss_config)

    if pt == PropertyType.ATOM_VECTOR:
        target = _get_target_field(batch, "y_atom_vector", pred.shape)
        return _vector_loss(pred, target, loss_name, loss_config)

    if pt == PropertyType.ATOM_TENSOR:
        target = _get_target_field(batch, "y_atom_tensor", pred.shape)
        return _tensor_loss(pred, target, loss_name, loss_config)

    if pt == PropertyType.HAMILTONIAN:
        return _hamiltonian_loss(pred, batch, loss_name, loss_config)

    raise ValueError(f"No loss defined for property_type={pt}")


def get_loss_fn(property_type: str, loss_name: str = "mae"):
    """Return a callable loss(pred, batch) for the given property type."""
    pt = validate_property_type(property_type)

    def loss_fn(pred, batch):
        return compute_loss(pred, batch, pt, loss_name)

    return loss_fn


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_target_field(batch, field_name: str, expected_shape) -> torch.Tensor:
    """Retrieve a non-scalar target from the batch, with a helpful error."""
    target = getattr(batch, field_name, None)
    if target is None:
        raise AttributeError(
            f"Batch is missing '{field_name}'. Ensure the preprocess step "
            f"stored this field. Expected shape compatible with {expected_shape}."
        )
    return target.to(dtype=torch.float32)


def _scalar_loss(pred, target, loss_name, cfg) -> torch.Tensor:
    if loss_name == "mae":
        return F.l1_loss(pred, target)
    if loss_name == "mse":
        return F.mse_loss(pred, target)
    if loss_name == "huber":
        delta = cfg.get("huber_delta", 0.1)
        return F.huber_loss(pred, target, delta=delta)
    raise ValueError(f"Unknown scalar loss: {loss_name}")


def _vector_loss(pred, target, loss_name, cfg) -> torch.Tensor:
    if loss_name in ("mae", "mse"):
        return F.mse_loss(pred, target)
    if loss_name == "cosine":
        cos = F.cosine_similarity(pred, target, dim=-1)
        angle_loss = (1 - cos).mean()
        mag_loss = F.mse_loss(pred.norm(dim=-1), target.norm(dim=-1))
        alpha = cfg.get("cosine_mag_weight", 0.1)
        return angle_loss + alpha * mag_loss
    raise ValueError(f"Unknown vector loss: {loss_name}")


def _tensor_loss(pred, target, loss_name, cfg) -> torch.Tensor:
    if loss_name in ("mae", "frobenius"):
        return F.l1_loss(pred, target)
    if loss_name == "mse":
        return F.mse_loss(pred, target)
    raise ValueError(f"Unknown tensor loss: {loss_name}")


def _tensor_loss_crystal_system(
    pred_cart: torch.Tensor,
    target_cart: torch.Tensor,
    space_groups: torch.Tensor,
    loss_name: str,
    cfg: Dict[str, Any],
) -> torch.Tensor:
    """Compute tensor loss only on independent parameters per crystal system.

    For elements that should be equal by symmetry (e.g., ε11=ε22 in hexagonal),
    we average them before computing loss.

    Args:
        pred_cart: (B, 3, 3) predicted symmetric tensors
        target_cart: (B, 3, 3) target symmetric tensors
        space_groups: (B,) space group numbers
        loss_name: "mae" or "mse"
        cfg: loss config dict

    Returns:
        Scalar loss computed only on independent tensor elements.
    """
    from wyckoff_gnn.models.unified_equivariant.crystal_system_constraints import (
        get_crystal_system,
    )

    B = pred_cart.shape[0]
    device = pred_cart.device

    # Collect independent parameters for each sample
    all_pred_params = []
    all_target_params = []

    for i in range(B):
        sg = space_groups[i].item()
        system = get_crystal_system(sg)

        # Extract independent parameters, averaging equal elements
        if system == "cubic":
            # ε11 = ε22 = ε33, average them
            pred_val = (pred_cart[i, 0, 0] + pred_cart[i, 1, 1] + pred_cart[i, 2, 2]) / 3
            target_val = (target_cart[i, 0, 0] + target_cart[i, 1, 1] + target_cart[i, 2, 2]) / 3
            pred_params = pred_val.unsqueeze(0)
            target_params = target_val.unsqueeze(0)

        elif system in ("hexagonal", "tetragonal"):
            # ε11 = ε22, ε33 independent
            pred_a = (pred_cart[i, 0, 0] + pred_cart[i, 1, 1]) / 2
            pred_c = pred_cart[i, 2, 2]
            target_a = (target_cart[i, 0, 0] + target_cart[i, 1, 1]) / 2
            target_c = target_cart[i, 2, 2]
            pred_params = torch.stack([pred_a, pred_c])
            target_params = torch.stack([target_a, target_c])

        elif system == "trigonal":
            # ε11 = ε22, ε33, ε12 (in hexagonal setting)
            pred_a = (pred_cart[i, 0, 0] + pred_cart[i, 1, 1]) / 2
            pred_c = pred_cart[i, 2, 2]
            pred_d = pred_cart[i, 0, 1]
            target_a = (target_cart[i, 0, 0] + target_cart[i, 1, 1]) / 2
            target_c = target_cart[i, 2, 2]
            target_d = target_cart[i, 0, 1]
            pred_params = torch.stack([pred_a, pred_c, pred_d])
            target_params = torch.stack([target_a, target_c, target_d])

        elif system == "orthorhombic":
            # ε11, ε22, ε33 all independent (diagonal)
            pred_params = torch.stack([
                pred_cart[i, 0, 0], pred_cart[i, 1, 1], pred_cart[i, 2, 2]
            ])
            target_params = torch.stack([
                target_cart[i, 0, 0], target_cart[i, 1, 1], target_cart[i, 2, 2]
            ])

        elif system == "monoclinic":
            # ε11, ε22, ε33, ε13 (unique axis b)
            pred_params = torch.stack([
                pred_cart[i, 0, 0], pred_cart[i, 1, 1], pred_cart[i, 2, 2], pred_cart[i, 0, 2]
            ])
            target_params = torch.stack([
                target_cart[i, 0, 0], target_cart[i, 1, 1], target_cart[i, 2, 2], target_cart[i, 0, 2]
            ])

        elif system == "triclinic":
            # Full symmetric: ε11, ε22, ε33, ε12, ε13, ε23
            pred_params = torch.stack([
                pred_cart[i, 0, 0], pred_cart[i, 1, 1], pred_cart[i, 2, 2],
                pred_cart[i, 0, 1], pred_cart[i, 0, 2], pred_cart[i, 1, 2]
            ])
            target_params = torch.stack([
                target_cart[i, 0, 0], target_cart[i, 1, 1], target_cart[i, 2, 2],
                target_cart[i, 0, 1], target_cart[i, 0, 2], target_cart[i, 1, 2]
            ])

        else:
            raise ValueError(f"Unknown crystal system: {system}")

        all_pred_params.append(pred_params)
        all_target_params.append(target_params)

    # Stack all samples - pad to max params (6 for triclinic)
    max_params = 6
    pred_padded = torch.zeros(B, max_params, device=device)
    target_padded = torch.zeros(B, max_params, device=device)
    mask = torch.zeros(B, max_params, device=device)

    for i in range(B):
        n = all_pred_params[i].shape[0]
        pred_padded[i, :n] = all_pred_params[i]
        target_padded[i, :n] = all_target_params[i]
        mask[i, :n] = 1.0

    # Compute loss only on valid (independent) parameters
    if loss_name in ("mae", "frobenius"):
        per_element_loss = torch.abs(pred_padded - target_padded)
    elif loss_name == "mse":
        per_element_loss = (pred_padded - target_padded) ** 2
    else:
        raise ValueError(f"Unknown tensor loss: {loss_name}")

    # Mask and average
    masked_loss = per_element_loss * mask
    total_loss = masked_loss.sum() / mask.sum().clamp(min=1.0)

    return total_loss


def _hamiltonian_loss(pred, batch, loss_name, cfg) -> torch.Tensor:
    """Loss for Hamiltonian block predictions.

    pred is a dict {"onsite": (K, n, n), "offsite": (E, n, n)}.
    batch must have:
        - y_hamiltonian_onsite: (K, n, n)
        - y_hamiltonian_offsite: (E, n, n)
    """
    loss = torch.tensor(0.0, device=next(iter(pred.values())).device)

    if "onsite" in pred:
        target_on = getattr(batch, "y_hamiltonian_onsite", None)
        if target_on is not None:
            loss = loss + F.mse_loss(pred["onsite"], target_on.float())

    if "offsite" in pred:
        target_off = getattr(batch, "y_hamiltonian_offsite", None)
        if target_off is not None:
            loss = loss + F.mse_loss(pred["offsite"], target_off.float())

    return loss
