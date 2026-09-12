"""P1 graph builder — strict matched full-atom baseline.

Produces Wyckoff-schema graphs (same format as the quotient builder) from the
**same standardized conventional cell** used by WyckoffGraphBuilderWrapper.

The only difference from Wyckoff quotient:
  - P1: every standardized atom is its own orbit (K=M, multiplicity=1, SG=1)
  - Wyckoff: symmetry-equivalent atoms compressed into one orbit (K<M)

Both share:
  - identical standardized conventional cell (from spglib.refine_cell)
  - identical cutoff and periodic-image enumeration (WyckoffGraphBuilder)
  - identical graph schema (Wyckoff-schema Data object)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np
import torch

from wyckoff_gnn.data.graph_builders.base import GraphBuilder
from wyckoff_gnn.data.records import StructureRecord

logger = logging.getLogger(__name__)


class P1GraphBuilderWrapper(GraphBuilder):
    """Build strict matched P1 graphs from StructureRecords.

    Every standardized atom becomes its own Wyckoff orbit with multiplicity 1
    and space group P1. The resulting graph uses the same schema as the
    standard Wyckoff quotient graph, so the existing encoder and training
    pipeline can consume it directly.

    **Critical**: P1 uses the **same standardized conventional cell** as
    Wyckoff, ensuring N_P1 = N_std_atoms = sum(multiplicity_p) for quotient.

    Config keys
    -----------
    cutoff : float (default 5.0)
        Radius cutoff for geometric edges.
    symprec : float (default 0.1)
        Symmetry tolerance for spglib (must match Wyckoff builder).
    angle_pair_top_k : int (default 0)
        Number of nearest edges per target for angular MP (0=disabled).
    """

    def __init__(
        self,
        cutoff: float = 5.0,
        symprec: float = 0.1,
        angle_pair_top_k: int = 0,
        rotate_tensor_to_std: bool = False,
        symop_match_tol_cart: Optional[float] = None,
        site_stabilizer_tol_cart: Optional[float] = None,
        symmetry_match_failure: str = "fallback_p1",
        **kwargs,
    ):
        self.cutoff = cutoff
        self.symprec = symprec
        self.angle_pair_top_k = angle_pair_top_k
        self.rotate_tensor_to_std = rotate_tensor_to_std
        self.symop_match_tol_cart = symop_match_tol_cart
        self.site_stabilizer_tol_cart = site_stabilizer_tol_cart
        self.symmetry_match_failure = symmetry_match_failure

        # Log quotient-only params that are ignored for explicit P1
        ignored_params = []
        if symop_match_tol_cart is not None:
            ignored_params.append("symop_match_tol_cart")
        if site_stabilizer_tol_cart is not None:
            ignored_params.append("site_stabilizer_tol_cart")
        if symmetry_match_failure != "fallback_p1":
            ignored_params.append("symmetry_match_failure")
        if ignored_params:
            logger.info(
                "P1 builder: ignoring quotient-only params %s "
                "(explicit P1 does not perform symmetry matching)",
                ignored_params,
            )

    def build(self, record: StructureRecord) -> Dict[str, Any]:
        from wyckoff_gnn.data.crystal_to_wyckoff import (
            standardize_structure_cell,
            _p1_fallback,
        )
        from wyckoff_gnn.data.wyckoff_graph import WyckoffGraphBuilder
        from wyckoff_gnn.data.graph_schema import pyg_data_to_light_dict

        # Step 1: Standardize cell (no quotient side effects)
        std_info = standardize_structure_cell(record.structure, symprec=self.symprec)

        std_lattice = std_info["standardized_lattice"]
        std_positions = std_info["standardized_positions"]
        std_numbers = std_info["standardized_numbers"]
        std_rotation_matrix = std_info["std_rotation_matrix"]

        if std_lattice is None or std_positions is None or std_numbers is None:
            raise RuntimeError(
                f"Failed to obtain standardized cell for {record.material_id}. "
                "Cannot build matched P1 baseline without standardized cell."
            )

        # Step 2: Build P1 orbits directly from standardized cell
        p1_orbits, p1_meta = _p1_fallback(
            record.structure,
            std_lattice,
            std_positions,
            std_numbers,
            std_rotation_matrix=std_rotation_matrix,
            fallback_stage="explicit_p1",
            detected_sg_number=std_info["input_spg_number"],
            detected_international_symbol=std_info["input_international_symbol"],
        )

        # Step 3: Correctness assertions (P1-only, no quotient dependency)
        self._assert_p1_correctness(p1_orbits, p1_meta, std_info)

        # Step 4: Build graph using WyckoffGraphBuilder (same as quotient)
        builder = WyckoffGraphBuilder(
            cutoff_radius=self.cutoff,
            subedge_aggregation="sum",
            angle_pair_top_k=self.angle_pair_top_k,
        )
        data = builder.build(
            p1_orbits,
            std_lattice,
            atom_to_orbit=p1_meta.get("atom_to_orbit"),
            atom_image_index=p1_meta.get("atom_image_index"),
        )

        # Step 5: Attach material_id and target (aligned with Wyckoff builder)
        data.material_id = record.material_id
        extra_kwargs = {}

        if record.target is not None:
            target_type = getattr(record, "target_type", None)

            if target_type == "hamiltonian":
                extra_kwargs["hamiltonian_blocks"] = record.target
                data.y = torch.tensor([0.0], dtype=torch.float32)
            elif isinstance(record.target, np.ndarray):
                target = record.target
                if self.rotate_tensor_to_std and target.ndim == 2:
                    Q = std_info.get("std_rotation_matrix")
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
                data.y = torch.tensor([float(target.mean())], dtype=torch.float32)
            elif isinstance(record.target, (int, float)):
                data.y = torch.tensor([float(record.target)], dtype=torch.float32)
            else:
                try:
                    data.y = torch.tensor([float(record.target)], dtype=torch.float32)
                except (TypeError, ValueError):
                    data.y = torch.tensor([0.0], dtype=torch.float32)

        if hasattr(record, "target_type") and record.target_type:
            extra_kwargs["target_type"] = record.target_type

        # Step 6: Convert to light dict
        light = pyg_data_to_light_dict(data, **extra_kwargs)
        return light

    def _assert_p1_correctness(
        self,
        p1_orbits: list,
        p1_meta: dict,
        std_info: dict,
    ) -> None:
        """Verify P1 correctness assertions without quotient dependency."""
        std_lattice = std_info["standardized_lattice"]
        std_positions = std_info["standardized_positions"]
        std_numbers = std_info["standardized_numbers"]

        # 1. P1 node count equals standardized atom count
        assert len(p1_orbits) == len(std_numbers), (
            f"P1 node count {len(p1_orbits)} != std atom count {len(std_numbers)}"
        )
        assert len(p1_orbits) == len(std_positions), (
            f"P1 node count {len(p1_orbits)} != std position count {len(std_positions)}"
        )

        # 2. All P1 orbits have multiplicity=1, SG=1, site_symmetry="1"
        for i, orbit in enumerate(p1_orbits):
            assert orbit.multiplicity == 1, (
                f"P1 orbit {i} has multiplicity {orbit.multiplicity} != 1"
            )
            assert orbit.space_group == 1, (
                f"P1 orbit {i} has SG {orbit.space_group} != 1"
            )
            assert orbit.site_symmetry == "1", (
                f"P1 orbit {i} has site_symmetry '{orbit.site_symmetry}' != '1'"
            )
            assert orbit.num_free_params == 3, (
                f"P1 orbit {i} has num_free_params {orbit.num_free_params} != 3"
            )

        # 3. atom_to_orbit is identity mapping
        atom_to_orbit = p1_meta["atom_to_orbit"]
        expected = np.arange(len(std_numbers), dtype=np.int32)
        assert np.array_equal(atom_to_orbit, expected), (
            "P1 atom_to_orbit is not identity mapping"
        )

        # 4. atom_image_index is all zeros
        atom_image_index = p1_meta["atom_image_index"]
        assert np.all(atom_image_index == 0), (
            "P1 atom_image_index is not all zeros"
        )

        # 5. Standardized metadata matches
        assert np.allclose(
            p1_meta["standardized_lattice"],
            std_info["standardized_lattice"],
            atol=1e-10,
        ), "P1 standardized_lattice != std_info standardized_lattice"

        assert np.allclose(
            p1_meta["standardized_positions"],
            std_info["standardized_positions"],
            atol=1e-10,
        ), "P1 standardized_positions != std_info standardized_positions"

        assert np.array_equal(
            p1_meta["standardized_numbers"],
            std_info["standardized_numbers"],
        ), "P1 standardized_numbers != std_info standardized_numbers"


__all__ = ["P1GraphBuilderWrapper"]
