"""Wyckoff graph builder — wraps existing Wyckoff construction pipeline."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from wyckoff_gnn.data.graph_builders.base import GraphBuilder
from wyckoff_gnn.data.records import StructureRecord


class WyckoffGraphBuilderWrapper(GraphBuilder):
    """Build Wyckoff orbit-level graphs from StructureRecords.

    Wraps the existing pipeline::

        structure_to_wyckoff_orbits → WyckoffGraphBuilder.build
        → pyg_data_to_light_dict

    Config keys
    -----------
    cutoff : float (default 5.0)
        Max inter-orbit distance for geometric edges.
    symprec : float (default 0.1)
        Symmetry tolerance for spglib.
    fallback_p1 : bool (default True)
        If True, fall back to P1 when symmetry detection fails.
    global_max_mult : int (default 192)
        Padding size for per-orbit sym_ops tensors.
    use_site_projection : bool (default False)
        Whether to build site-stabilizer projection matrices.
    """

    def __init__(
        self,
        cutoff: float = 5.0,
        symprec: float = 0.1,
        fallback_p1: bool = True,
        global_max_mult: int = 192,
        use_site_projection: bool = False,
        angle_pair_top_k: int = 0,
        rotate_tensor_to_std: bool = False,
        symop_match_tol_cart: Optional[float] = None,
        site_stabilizer_tol_cart: Optional[float] = None,
        symmetry_match_failure: str = "fallback_p1",
        **kwargs,
    ):
        self.cutoff = cutoff
        self.symprec = symprec
        self.fallback_p1 = fallback_p1
        self.global_max_mult = global_max_mult
        self.use_site_projection = use_site_projection
        self.angle_pair_top_k = angle_pair_top_k
        self.rotate_tensor_to_std = rotate_tensor_to_std
        self.symop_match_tol_cart = symop_match_tol_cart
        self.site_stabilizer_tol_cart = site_stabilizer_tol_cart
        self.symmetry_match_failure = symmetry_match_failure

    def build(self, record: StructureRecord) -> Dict[str, Any]:
        from wyckoff_gnn.data.crystal_to_wyckoff import structure_to_wyckoff_orbits
        from wyckoff_gnn.data.wyckoff_graph import WyckoffGraphBuilder
        from wyckoff_gnn.data.graph_schema import pyg_data_to_light_dict

        orbits, meta = structure_to_wyckoff_orbits(
            record.structure, tol=self.symprec,
            fallback_p1=self.fallback_p1,
            symop_match_tol_cart=self.symop_match_tol_cart,
            site_stabilizer_tol_cart=self.site_stabilizer_tol_cart,
            symmetry_match_failure=self.symmetry_match_failure,
        )
        if len(orbits) == 0:
            raise RuntimeError(
                f"No Wyckoff orbits found for {record.material_id}"
            )

        # Ensure proper padding size.
        max_mult = max(o.multiplicity for o in orbits)
        gm = max(self.global_max_mult, max_mult)

        builder = WyckoffGraphBuilder(
            cutoff_radius=self.cutoff,
            global_max_mult=gm,
            angle_pair_top_k=self.angle_pair_top_k,
        )
        std_lattice = meta.get("standardized_lattice")
        data = builder.build(
            orbits, std_lattice,
            atom_to_orbit=meta.get("atom_to_orbit"),
            atom_image_index=meta.get("atom_image_index"),
        )

        # Attach material_id and target (multi-type aware).
        data.material_id = record.material_id
        extra_kwargs = {}

        if record.target is not None:
            import numpy as np
            target_type = getattr(record, "target_type", None)

            if target_type == "hamiltonian":
                # Hamiltonian targets stored as a separate field, not in data.y
                extra_kwargs["hamiltonian_blocks"] = record.target
                data.y = torch.tensor([0.0], dtype=torch.float32)
            elif isinstance(record.target, np.ndarray):
                target = record.target
                if self.rotate_tensor_to_std and target.ndim == 2:
                    Q = meta.get("std_rotation_matrix")
                    if Q is not None and target.shape == (3, 3):
                        from wyckoff_gnn.data.tensor_frame import (
                            transform_symmetric_rank2_cartesian,
                        )
                        target, audit = transform_symmetric_rank2_cartesian(
                            target, Q, strict=True, symmetry_tol=1e-3,
                        )
                        extra_kwargs["tensor_frame_rotation"] = Q.astype(
                            np.float32
                        )
                        extra_kwargs["tensor_frame_transformed"] = True
                    elif Q is not None and target.shape == (3, 6):
                        from wyckoff_gnn.data.tensor_frame import (
                            transform_piezo_rank3_voigt,
                        )
                        target, audit = transform_piezo_rank3_voigt(
                            target, Q, strict=True, symmetry_tol=1e-3,
                        )
                        extra_kwargs["tensor_frame_rotation"] = Q.astype(
                            np.float32
                        )
                        extra_kwargs["tensor_frame_transformed"] = True
                _tensor_rank = {(3, 6): 3, (3, 3): 2}.get(
                    getattr(target, "shape", None))
                if _tensor_rank is not None:
                    from wyckoff_gnn.models.unified_equivariant.crystal_tensor_projection import (  # noqa: E501
                        projector_kwargs_from_structure,
                    )
                    extra_kwargs.update(projector_kwargs_from_structure(
                        record.structure, self.symprec, rank=_tensor_rank))
                extra_kwargs["y_tensor"] = target
                # Also set scalar y to trace/mean for manifest compatibility
                data.y = torch.tensor([float(target.mean())], dtype=torch.float32)
            elif isinstance(record.target, (int, float)):
                data.y = torch.tensor([float(record.target)], dtype=torch.float32)
            else:
                # Fallback: try float
                try:
                    data.y = torch.tensor([float(record.target)], dtype=torch.float32)
                except (TypeError, ValueError):
                    data.y = torch.tensor([0.0], dtype=torch.float32)

        if hasattr(record, "target_type") and record.target_type:
            extra_kwargs["target_type"] = record.target_type

        # Convert to light dict.
        light = pyg_data_to_light_dict(data, **extra_kwargs)

        # Override with record-level split if available.
        # (The split is attached in the manifest entry, not the graph dict.)
        return light


__all__ = ["WyckoffGraphBuilderWrapper"]
