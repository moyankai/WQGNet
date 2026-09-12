"""Lightweight graph dict schema for shard-based caching.

Converts the heavy PyG Data object produced by :class:`WyckoffGraphBuilder`
into a compact pure-Python dict suitable for serialization into shards.

Design rules
------------
- ``orbit_sym_ops_W_frac`` stored as int8 (W are integer rotation matrices).
- ``geo_edge_shift`` stored as int16 (small integer PBC offsets).
- Token / index fields stored as int16 or int32.
- Dynamic (recomputable) fields are NOT stored:
  edge_sh, edge_rbf, source_rotations_cart, edge_attr (legacy RBF),
  orbit_stabilizer_projections, orbit_param_basis_frac.
- Cartesian rotations ``orbit_sym_ops_rotations`` are NOT stored — they
  are recomputed from ``W_frac + lattice`` at load time.
- Aliases (rep_coords_frac, atomic_numbers) are NOT duplicated.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch
from torch_geometric.data import Data


class WyckoffData(Data):
    """PyG Data subclass with correct batching semantics for multi-target fields.

    Graph-level targets (y_tensor, y_vector) have shape (1, D) per graph and
    must be concatenated along dim=0 when batching. Per-atom targets
    (y_atom_scalar, y_atom_vector) follow node count and cat along dim=0
    naturally. This class ensures PyG's Batch.from_data_list does the right
    thing without silent mis-batching.

    angle_pair_index (shape (2, P)) indexes into the geo edge list, so each
    entry must be offset by the running edge count when batching. Similarly
    angle_pair_target (shape (P,)) indexes into orbits, so offset by orbit
    count.
    """

    _GRAPH_LEVEL_TARGETS = {"y_tensor", "y_vector", "y"}

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key in self._GRAPH_LEVEL_TARGETS:
            return 0
        if key == "angle_pair_index":
            return 1  # (2, P), concat along P
        return super().__cat_dim__(key, value, *args, **kwargs)

    def __inc__(self, key, value, *args, **kwargs):
        if key == "angle_pair_index":
            # Entries index into geo_edge_index columns; offset by edge count.
            return int(self.geo_edge_index.size(1)) if hasattr(self, "geo_edge_index") else 0
        if key == "angle_pair_target":
            # Entries index into orbit list.
            return int(self.num_nodes) if hasattr(self, "num_nodes") else 0
        return super().__inc__(key, value, *args, **kwargs)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def pyg_data_to_light_dict(
    data: Data,
    material_id: Optional[str] = None,
    y: Optional[float] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Convert a PyG Data (from WyckoffGraphBuilder.build) to a lightweight dict.

    Args:
        data: PyG Data object as returned by ``WyckoffGraphBuilder.build()``.
        material_id: Optional material identifier (if not already on ``data``).
        y: Optional scalar target value (if not already on ``data``).
        **kwargs: Extra fields to store (y_tensor, hamiltonian_blocks, target_type, etc.).

    Returns:
        A pure-Python dict suitable for ``torch.save``.
    """
    # --- Node features ---
    orbit_element = _as_numpy(data.orbit_element, "int32")
    orbit_rep_frac = _as_numpy(data.orbit_rep_frac, "float32")
    orbit_letter_in_sg = _as_numpy(data.orbit_letter_in_sg, "int16")
    orbit_site_sym = _as_numpy(data.orbit_site_sym, "int16")
    letter_in_sg_token = _as_numpy(data.letter_in_sg_token, "int32")
    multiplicity = _as_numpy(data.multiplicity, "float32")

    # --- Geometric sub-edges ---
    geo_edge_index = _as_numpy(data.geo_edge_index, "int64")
    geo_edge_source_image = _as_numpy(data.geo_edge_source_image, "int32")
    # source_frac kept as float32 — needed for differentiable vec reconstruction.
    geo_edge_source_frac = _as_numpy(data.geo_edge_source_frac, "float32")
    geo_edge_shift = _as_numpy(data.geo_edge_shift, "int16")
    # distance is optional — can be recomputed, but cheap to store.
    geo_edge_distance = _as_numpy(data.geo_edge_distance, "float32")
    geo_edge_weight = _as_numpy(data.geo_edge_weight, "float32")

    # --- Angle pair index (optional, empty if angle MP disabled during build) ---
    if hasattr(data, "angle_pair_index") and data.angle_pair_index is not None:
        angle_pair_index = _as_numpy(data.angle_pair_index, "int32")
    else:
        angle_pair_index = np.zeros((2, 0), dtype=np.int32)
    if hasattr(data, "angle_pair_target") and data.angle_pair_target is not None:
        angle_pair_target = _as_numpy(data.angle_pair_target, "int32")
    else:
        angle_pair_target = np.zeros((0,), dtype=np.int32)

    # --- Symmetry edges ---
    sym_edge_index = _as_numpy(data.sym_edge_index, "int64")
    sym_edge_attr = _as_numpy(data.sym_edge_attr, "float32")

    # --- Per-orbit symmetry operations (fractional, NOT Cartesian) ---
    has_W = (
        hasattr(data, "orbit_sym_ops_W_frac")
        and data.orbit_sym_ops_W_frac is not None
    )
    has_w = (
        hasattr(data, "orbit_sym_ops_w_frac")
        and data.orbit_sym_ops_w_frac is not None
    )
    orbit_sym_ops_W_frac = (
        _as_numpy(data.orbit_sym_ops_W_frac, "int8") if has_W else None
    )
    orbit_sym_ops_w_frac = (
        _as_numpy(data.orbit_sym_ops_w_frac, "float32") if has_w else None
    )
    orbit_mult_mask = _as_numpy(data.orbit_mult_mask, "bool")

    # --- Per-orbit site stabilizer (full H_p, not just image generators) ---
    has_stab_W = (
        hasattr(data, "orbit_stabilizer_W_frac")
        and data.orbit_stabilizer_W_frac is not None
    )
    has_stab_w = (
        hasattr(data, "orbit_stabilizer_w_frac")
        and data.orbit_stabilizer_w_frac is not None
    )
    has_stab_mask = (
        hasattr(data, "orbit_stabilizer_mask")
        and data.orbit_stabilizer_mask is not None
    )
    orbit_stabilizer_W_frac = (
        _as_numpy(data.orbit_stabilizer_W_frac, "int8") if has_stab_W else None
    )
    orbit_stabilizer_w_frac = (
        _as_numpy(data.orbit_stabilizer_w_frac, "float32") if has_stab_w else None
    )
    orbit_stabilizer_mask = (
        _as_numpy(data.orbit_stabilizer_mask, "bool") if has_stab_mask else None
    )

    # --- Atom-level mapping ---
    atom_to_orbit = None
    if hasattr(data, "atom_to_orbit") and data.atom_to_orbit is not None:
        atom_to_orbit = _as_numpy(data.atom_to_orbit, "int32")
    atom_image_index = None
    if hasattr(data, "atom_image_index") and data.atom_image_index is not None:
        atom_image_index = _as_numpy(data.atom_image_index, "int32")

    # --- DOF ---
    orbit_dof = _as_numpy(data.orbit_dof, "int16")

    # --- Global ---
    lattice = _as_numpy(data.lattice, "float32")
    space_group = int(data.space_group.item()) if hasattr(data, "space_group") else 1

    # --- Metadata ---
    mid = material_id
    if mid is None and hasattr(data, "material_id"):
        raw = data.material_id
        if isinstance(raw, (list, np.ndarray)):
            mid = str(raw[0])
        else:
            mid = str(raw)
    target = y
    if target is None and hasattr(data, "y") and data.y is not None:
        target = float(data.y.item()) if data.y.numel() == 1 else float(data.y[0])

    graph: Dict[str, Any] = {
        # Node
        "orbit_element": orbit_element,
        "orbit_rep_frac": orbit_rep_frac,
        "orbit_letter_in_sg": orbit_letter_in_sg,
        "orbit_site_sym": orbit_site_sym,
        "letter_in_sg_token": letter_in_sg_token,
        "multiplicity": multiplicity,
        # Geo edges
        "geo_edge_index": geo_edge_index,
        "geo_edge_source_image": geo_edge_source_image,
        "geo_edge_source_frac": geo_edge_source_frac,
        "geo_edge_shift": geo_edge_shift,
        "geo_edge_distance": geo_edge_distance,
        "geo_edge_weight": geo_edge_weight,
        # Angle pair index (optional, shape (2, P) and (P,))
        "angle_pair_index": angle_pair_index,
        "angle_pair_target": angle_pair_target,
        # Sym edges
        "sym_edge_index": sym_edge_index,
        "sym_edge_attr": sym_edge_attr,
        # Sym ops (fractional)
        "orbit_sym_ops_W_frac": orbit_sym_ops_W_frac,
        "orbit_sym_ops_w_frac": orbit_sym_ops_w_frac,
        "orbit_mult_mask": orbit_mult_mask,
        # Site stabilizer (full H_p, for projector)
        "orbit_stabilizer_W_frac": orbit_stabilizer_W_frac,
        "orbit_stabilizer_w_frac": orbit_stabilizer_w_frac,
        "orbit_stabilizer_mask": orbit_stabilizer_mask,
        # Atom mapping
        "atom_to_orbit": atom_to_orbit,
        "atom_image_index": atom_image_index,
        # DOF
        "orbit_dof": orbit_dof,
        # Global
        "lattice": lattice,
        "space_group": space_group,
        # Meta
        "material_id": mid,
        "y": target,
    }

    # Optional multi-type target fields (stored alongside scalar 'y' for
    # property types that need vector/tensor/per-atom labels).
    for field in ("y_tensor", "y_vector", "y_atom_scalar", "y_atom_vector"):
        if field in kwargs:
            val = kwargs[field]
            if hasattr(val, 'numpy'):
                val = val.detach().cpu().numpy()
            graph[field] = np.asarray(val, dtype=np.float32)
    if "hamiltonian_blocks" in kwargs:
        graph["hamiltonian_blocks"] = kwargs["hamiltonian_blocks"]
    if "target_type" in kwargs:
        graph["target_type"] = kwargs["target_type"]
    if "tensor_frame_rotation" in kwargs:
        graph["tensor_frame_rotation"] = np.asarray(
            kwargs["tensor_frame_rotation"], dtype=np.float32
        )
    if "tensor_frame_transformed" in kwargs:
        graph["tensor_frame_transformed"] = bool(kwargs["tensor_frame_transformed"])
    # Crystal point-group (Neumann) projector for global tensor outputs.
    # Kept in float64: it encodes an exact symmetry constraint and is
    # analytically zero for centrosymmetric odd-rank tensors.
    if "tensor_point_group_projector" in kwargs:
        graph["tensor_point_group_projector"] = np.asarray(
            kwargs["tensor_point_group_projector"], dtype=np.float64
        )
    for field in ("point_group_order", "projector_rank"):
        if field in kwargs:
            graph[field] = int(kwargs[field])
    if "contains_inversion" in kwargs:
        graph["contains_inversion"] = bool(kwargs["contains_inversion"])

    return graph


def light_dict_to_pyg_data(graph: Dict[str, Any]) -> Data:
    """Convert a lightweight graph dict back to a PyG Data object.

    Recomputes ``orbit_sym_ops_rotations`` (Cartesian R_e3nn) from
    the stored fractional ``W_frac`` and ``lattice``.

    Args:
        graph: Dict as produced by :func:`pyg_data_to_light_dict`.

    Returns:
        PyG Data ready for model consumption.
    """
    lattice = torch.from_numpy(np.ascontiguousarray(graph["lattice"])).float()

    # --- Node features ---
    orbit_element = _to_tensor(graph["orbit_element"], torch.long)
    orbit_rep_frac = _to_tensor(graph["orbit_rep_frac"], torch.float32)
    orbit_letter_in_sg = _to_tensor(graph["orbit_letter_in_sg"], torch.long)
    orbit_site_sym = _to_tensor(graph["orbit_site_sym"], torch.long)
    letter_in_sg_token = _to_tensor(graph["letter_in_sg_token"], torch.long)
    multiplicity = _to_tensor(graph["multiplicity"], torch.float32)

    # --- Geo edges ---
    geo_edge_index = _to_tensor(graph["geo_edge_index"], torch.long)
    geo_edge_source_image = _to_tensor(graph["geo_edge_source_image"], torch.long)
    geo_edge_source_frac = _to_tensor(graph["geo_edge_source_frac"], torch.float32)
    geo_edge_shift = _to_tensor(graph["geo_edge_shift"], torch.float32)
    geo_edge_distance = _to_tensor(graph["geo_edge_distance"], torch.float32)
    geo_edge_weight = _to_tensor(graph["geo_edge_weight"], torch.float32)

    # --- Angle pair index (optional, backward-compat) ---
    if "angle_pair_index" in graph and graph["angle_pair_index"] is not None:
        angle_pair_index = _to_tensor(graph["angle_pair_index"], torch.long)
        if angle_pair_index.dim() == 1:  # flattened accidentally
            angle_pair_index = angle_pair_index.view(2, -1)
    else:
        angle_pair_index = torch.zeros(2, 0, dtype=torch.long)
    if "angle_pair_target" in graph and graph["angle_pair_target"] is not None:
        angle_pair_target = _to_tensor(graph["angle_pair_target"], torch.long)
    else:
        angle_pair_target = torch.zeros(0, dtype=torch.long)

    # --- Sym edges ---
    sym_edge_index = _to_tensor(graph["sym_edge_index"], torch.long)
    sym_edge_attr = _to_tensor(graph["sym_edge_attr"], torch.float32)

    # --- Sym ops: recompute Cartesian R_e3nn from fractional W_frac ---
    W_frac_arr = graph.get("orbit_sym_ops_W_frac")
    w_frac_arr = graph.get("orbit_sym_ops_w_frac")
    mult_mask = _to_tensor(graph["orbit_mult_mask"], torch.bool)

    if W_frac_arr is not None:
        W_frac = torch.from_numpy(np.ascontiguousarray(W_frac_arr)).float()
        K, M = W_frac.shape[:2]
        # R_e3nn = A.T @ W @ A^{-T}
        A = lattice  # (3,3) for single graph
        A_T = A.T
        A_inv_T = torch.inverse(A).T
        W_flat = W_frac.reshape(K * M, 3, 3)
        R_flat = torch.bmm(torch.bmm(A_T.unsqueeze(0).expand(K * M, 3, 3), W_flat),
                           A_inv_T.unsqueeze(0).expand(K * M, 3, 3))
        orbit_sym_ops_rotations = R_flat.reshape(K, M, 3, 3)
    else:
        orbit_sym_ops_rotations = None
        K, M = mult_mask.shape[0], mult_mask.shape[1]

    if w_frac_arr is not None:
        orbit_sym_ops_w_frac = torch.from_numpy(
            np.ascontiguousarray(w_frac_arr)
        ).float()
    else:
        orbit_sym_ops_w_frac = torch.zeros(K, M, 3)

    # --- Site stabilizer (full H_p) ---
    stab_W_arr = graph.get("orbit_stabilizer_W_frac")
    stab_w_arr = graph.get("orbit_stabilizer_w_frac")
    stab_mask_arr = graph.get("orbit_stabilizer_mask")

    if stab_W_arr is not None:
        orbit_stabilizer_W_frac = torch.from_numpy(
            np.ascontiguousarray(stab_W_arr)
        ).float()
    else:
        orbit_stabilizer_W_frac = None
    if stab_w_arr is not None:
        orbit_stabilizer_w_frac = torch.from_numpy(
            np.ascontiguousarray(stab_w_arr)
        ).float()
    else:
        orbit_stabilizer_w_frac = None
    if stab_mask_arr is not None:
        orbit_stabilizer_mask = torch.from_numpy(
            np.ascontiguousarray(stab_mask_arr)
        ).bool()
    else:
        orbit_stabilizer_mask = None

    # --- Atom mapping ---
    atom_to_orbit = None
    if graph.get("atom_to_orbit") is not None:
        atom_to_orbit = _to_tensor(graph["atom_to_orbit"], torch.long)
    atom_image_index = None
    if graph.get("atom_image_index") is not None:
        atom_image_index = _to_tensor(graph["atom_image_index"], torch.long)

    # --- DOF ---
    orbit_dof = _to_tensor(graph["orbit_dof"], torch.long)

    num_orbits = orbit_element.shape[0]

    # --- Legacy edge_attr (RBF + unit_dir): placeholder ---
    E = geo_edge_index.shape[1]
    edge_attr = torch.zeros(E, 0)  # empty, model recomputes from source_frac+shift

    data = WyckoffData(
        orbit_element=orbit_element,
        orbit_rep_frac=orbit_rep_frac,
        orbit_letter_in_sg=orbit_letter_in_sg,
        orbit_site_sym=orbit_site_sym,
        letter_in_sg_token=letter_in_sg_token,
        rep_coords_frac=orbit_rep_frac,
        atomic_numbers=orbit_element,
        geo_edge_index=geo_edge_index,
        geo_edge_source_image=geo_edge_source_image,
        geo_edge_source_frac=geo_edge_source_frac,
        geo_edge_shift=geo_edge_shift,
        geo_edge_distance=geo_edge_distance,
        geo_edge_weight=geo_edge_weight,
        geo_edge_attr=edge_attr,
        angle_pair_index=angle_pair_index,
        angle_pair_target=angle_pair_target,
        multiplicity=multiplicity,
        atom_to_orbit=atom_to_orbit,
        atom_image_index=atom_image_index,
        sym_edge_index=sym_edge_index,
        sym_edge_attr=sym_edge_attr,
        orbit_sym_ops_rotations=orbit_sym_ops_rotations,
        orbit_sym_ops_W_frac=(
            torch.from_numpy(np.ascontiguousarray(W_frac_arr)).float()
            if W_frac_arr is not None else None
        ),
        orbit_sym_ops_w_frac=orbit_sym_ops_w_frac,
        orbit_mult_mask=mult_mask,
        orbit_stabilizer_W_frac=orbit_stabilizer_W_frac,
        orbit_stabilizer_w_frac=orbit_stabilizer_w_frac,
        orbit_stabilizer_mask=orbit_stabilizer_mask,
        orbit_dof=orbit_dof,
        space_group=torch.tensor([graph["space_group"]]),
        lattice=lattice,
        num_nodes=num_orbits,
        num_orbits=num_orbits,
    )

    # Attach material_id and target.
    mid = graph.get("material_id")
    if mid is not None:
        data.material_id = mid
    y_val = graph.get("y")
    if y_val is not None:
        data.y = torch.tensor([float(y_val)], dtype=torch.float32)

    # Optional multi-type targets.
    for field in ("y_tensor", "y_vector"):
        arr = graph.get(field)
        if arr is not None:
            t = torch.from_numpy(
                np.ascontiguousarray(arr) if isinstance(arr, np.ndarray) else np.array(arr)
            ).float()
            # Graph-level targets must be (1, D) for correct PyG batching
            # (cat_dim=0 concatenates along the graph axis).
            t = t.reshape(1, -1)
            setattr(data, field, t)
    for field in ("y_atom_scalar", "y_atom_vector"):
        arr = graph.get(field)
        if arr is not None:
            setattr(data, field, torch.from_numpy(
                np.ascontiguousarray(arr) if isinstance(arr, np.ndarray) else np.array(arr)
            ).float())

    tfr = graph.get("tensor_frame_rotation")
    if tfr is not None:
        data.tensor_frame_rotation = torch.from_numpy(
            np.ascontiguousarray(tfr) if isinstance(tfr, np.ndarray)
            else np.array(tfr)
        ).float()
    tft = graph.get("tensor_frame_transformed")
    if tft is not None:
        data.tensor_frame_transformed = bool(tft)

    # Stored with a leading singleton axis so PyG batching concatenates along
    # dim 0 and yields (B, 18, 18): one projector per graph, never broadcast.
    proj = graph.get("tensor_point_group_projector")
    if proj is not None:
        data.tensor_point_group_projector = torch.from_numpy(
            np.ascontiguousarray(np.asarray(proj, dtype=np.float64))
        ).unsqueeze(0)
    for field in ("point_group_order", "projector_rank"):
        val = graph.get(field)
        if val is not None:
            setattr(data, field, torch.tensor([int(val)], dtype=torch.long))
    ci = graph.get("contains_inversion")
    if ci is not None:
        data.contains_inversion = torch.tensor([bool(ci)], dtype=torch.bool)

    return data


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _as_numpy(tensor: torch.Tensor, dtype: str) -> np.ndarray:
    """Convert a torch tensor to a numpy array with the requested dtype."""
    arr = tensor.detach().cpu().numpy()
    if arr.dtype != np.dtype(dtype):
        arr = arr.astype(dtype)
    return arr


def _to_tensor(arr: np.ndarray, dtype: torch.dtype) -> torch.Tensor:
    """Convert a numpy array to a torch tensor."""
    return torch.from_numpy(np.ascontiguousarray(arr)).to(dtype)


__all__ = [
    "pyg_data_to_light_dict",
    "light_dict_to_pyg_data",
]
