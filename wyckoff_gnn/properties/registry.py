"""Property registry — single source of truth for physical property definitions.

Every property that WyckoffGNN can predict is registered here with its full
physical specification. The model factory, training loop, and evaluation
pipeline must all read from this registry — never from target shape or
heuristic guessing.

Design principles:
  - A shape=(3,) target is NEVER automatically interpreted as a vector.
  - The same data key (e.g. "epsx") can appear in different PropertySpec
    entries with different physical_type depending on what's actually available.
  - output_head is determined by physical_type, not by target shape.
  - status explicitly marks what is validated vs experimental vs unsupported.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


__all__ = [
    "PropertySpec",
    "PhysicalType",
    "Status",
    "PROPERTY_REGISTRY",
    "get_property_spec",
    "list_properties",
    "get_supported_properties",
]


class PhysicalType:
    """Canonical physical type strings."""
    GRAPH_SCALAR_INTENSIVE = "graph_scalar_intensive"
    GRAPH_SCALAR_EXTENSIVE = "graph_scalar_extensive"
    GRAPH_RANK2_SYMMETRIC_TENSOR = "graph_rank2_symmetric_tensor"
    GRAPH_RANK2_DIAGONAL_CANONICAL = "graph_rank2_diagonal_canonical"
    GRAPH_RANK4_ELASTIC_TENSOR = "graph_rank4_elastic_tensor"
    GRAPH_RANK3_PIEZO_TENSOR = "graph_rank3_piezo_tensor"
    GRAPH_VECTOR_POLAR = "graph_vector_polar"
    GRAPH_VECTOR_AXIAL = "graph_vector_axial"
    ATOM_SCALAR = "atom_scalar"
    ATOM_VECTOR_POLAR = "atom_vector_polar"
    ATOM_VECTOR_AXIAL = "atom_vector_axial"
    ATOM_RANK2_SYMMETRIC_TENSOR = "atom_rank2_symmetric_tensor"
    HAMILTONIAN = "hamiltonian"


class Status:
    """Implementation status."""
    SUPPORTED = "supported"
    BENCHMARK_ONLY = "benchmark_only"
    SMOKE_ONLY = "smoke_only"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class PropertySpec:
    """Complete physical specification of a predictable property.

    Attributes:
        name: unique registry key (e.g. "formation_energy_peratom").
        source: data source (e.g. "jarvis", "deeph", "mp").
        level: prediction granularity: graph / atom / orbit / pair / hamiltonian.
        physical_type: one of PhysicalType constants.
        target_shape: expected shape per sample (tuple), e.g. (1,), (3,3), (6,6).
        target_irreps: e3nn irreps string for the equivariant output head.
            None for scalar heads. Must match the physical_type exactly.
        required_label_components: list of raw data keys needed to construct
            the target (e.g. ["epsx", "epsy", "epsz"] for diagonal dielectric).
        output_head: name of the output head class to use.
        pooling: graph-level aggregation method ("mean", "sum", "none").
        notes: human-readable explanation of caveats.
        status: one of Status constants.
    """
    name: str
    source: str
    level: str
    physical_type: str
    target_shape: Tuple
    target_irreps: Optional[str]
    required_label_components: List[str]
    output_head: str
    pooling: str
    notes: str = ""
    status: str = Status.SUPPORTED


# ---------------------------------------------------------------------------
# JARVIS DFT-3D properties
# ---------------------------------------------------------------------------

_JARVIS_SCALARS = [
    PropertySpec(
        name="formation_energy_peratom",
        source="jarvis",
        level="graph",
        physical_type=PhysicalType.GRAPH_SCALAR_INTENSIVE,
        target_shape=(1,),
        target_irreps=None,
        required_label_components=["formation_energy_peratom"],
        output_head="ScalarReadout",
        pooling="mean",
        notes="Per-atom formation energy. Intensive: independent of cell size.",
        status=Status.SUPPORTED,
    ),
    PropertySpec(
        name="optb88vdw_bandgap",
        source="jarvis",
        level="graph",
        physical_type=PhysicalType.GRAPH_SCALAR_INTENSIVE,
        target_shape=(1,),
        target_irreps=None,
        required_label_components=["optb88vdw_bandgap"],
        output_head="ScalarReadout",
        pooling="mean",
        notes="DFT PBE+vdW band gap. Intensive.",
        status=Status.SUPPORTED,
    ),
    PropertySpec(
        name="ehull",
        source="jarvis",
        level="graph",
        physical_type=PhysicalType.GRAPH_SCALAR_INTENSIVE,
        target_shape=(1,),
        target_irreps=None,
        required_label_components=["ehull"],
        output_head="ScalarReadout",
        pooling="mean",
        notes="Energy above convex hull. Intensive (per-atom-like stability measure).",
        status=Status.SUPPORTED,
    ),
    PropertySpec(
        name="optb88vdw_total_energy",
        source="jarvis",
        level="graph",
        physical_type=PhysicalType.GRAPH_SCALAR_EXTENSIVE,
        target_shape=(1,),
        target_irreps=None,
        required_label_components=["optb88vdw_total_energy"],
        output_head="ScalarReadout",
        pooling="sum",
        notes="Total energy. Extensive: scales with num_atoms.",
        status=Status.SUPPORTED,
    ),
    PropertySpec(
        name="magmom_oszicar",
        source="jarvis",
        level="graph",
        physical_type=PhysicalType.GRAPH_SCALAR_INTENSIVE,
        target_shape=(1,),
        target_irreps=None,
        required_label_components=["magmom_oszicar"],
        output_head="ScalarReadout",
        pooling="mean",
        notes="Total magnetic moment (μB) from OSZICAR. JARVIS values do NOT scale "
              "with num_atoms (corr≈0.18; 78% are zero), so intensive mean pooling "
              "is the correct readout despite the 'total' name.",
        status=Status.SUPPORTED,
    ),
    PropertySpec(
        name="bulk_modulus_kv",
        source="jarvis",
        level="graph",
        physical_type=PhysicalType.GRAPH_SCALAR_INTENSIVE,
        target_shape=(1,),
        target_irreps=None,
        required_label_components=["bulk_modulus_kv"],
        output_head="ScalarReadout",
        pooling="mean",
        notes="Voigt-Reuss-Hill bulk modulus. Scalar reduction of elastic tensor. "
              "NOT a tensor head — it's already a rotationally invariant scalar.",
        status=Status.SUPPORTED,
    ),
    PropertySpec(
        name="shear_modulus_gv",
        source="jarvis",
        level="graph",
        physical_type=PhysicalType.GRAPH_SCALAR_INTENSIVE,
        target_shape=(1,),
        target_irreps=None,
        required_label_components=["shear_modulus_gv"],
        output_head="ScalarReadout",
        pooling="mean",
        notes="Voigt-Reuss-Hill shear modulus. Scalar reduction of elastic tensor.",
        status=Status.SUPPORTED,
    ),
    PropertySpec(
        name="mbj_bandgap",
        source="jarvis",
        level="graph",
        physical_type=PhysicalType.GRAPH_SCALAR_INTENSIVE,
        target_shape=(1,),
        target_irreps=None,
        required_label_components=["mbj_bandgap"],
        output_head="ScalarReadout",
        pooling="mean",
        notes="mBJ band gap (more accurate than PBE).",
        status=Status.SUPPORTED,
    ),
    PropertySpec(
        name="exfoliation_energy",
        source="jarvis",
        level="graph",
        physical_type=PhysicalType.GRAPH_SCALAR_INTENSIVE,
        target_shape=(1,),
        target_irreps=None,
        required_label_components=["exfoliation_energy"],
        output_head="ScalarReadout",
        pooling="mean",
        notes="Exfoliation energy for 2D materials. Intensive.",
        status=Status.SUPPORTED,
    ),
    PropertySpec(
        name="slme",
        source="jarvis",
        level="graph",
        physical_type=PhysicalType.GRAPH_SCALAR_INTENSIVE,
        target_shape=(1,),
        target_irreps=None,
        required_label_components=["slme"],
        output_head="ScalarReadout",
        pooling="mean",
        notes="Spectroscopic limited maximum efficiency (solar cell metric).",
        status=Status.SUPPORTED,
    ),
    PropertySpec(
        name="dfpt_piezo_max_dij",
        source="jarvis",
        level="graph",
        physical_type=PhysicalType.GRAPH_SCALAR_INTENSIVE,
        target_shape=(1,),
        target_irreps=None,
        required_label_components=["dfpt_piezo_max_dij"],
        output_head="ScalarReadout",
        pooling="mean",
        notes="Maximum piezoelectric strain coefficient d_ij. "
              "This is a SCALAR (max over all components), NOT a vector or tensor. "
              "The name contains 'piezo' but the label is the scalar maximum value.",
        status=Status.SUPPORTED,
    ),
    PropertySpec(
        name="max_efg",
        source="jarvis",
        level="graph",
        physical_type=PhysicalType.GRAPH_SCALAR_INTENSIVE,
        target_shape=(1,),
        target_irreps=None,
        required_label_components=["max_efg"],
        output_head="ScalarReadout",
        pooling="mean",
        notes="Maximum electric field gradient in the crystal. "
              "This is a GRAPH-LEVEL SCALAR (maximum over atoms), NOT a per-atom "
              "tensor. JARVIS does not provide per-atom EFG tensors.",
        status=Status.SUPPORTED,
    ),
    PropertySpec(
        name="density",
        source="jarvis",
        level="graph",
        physical_type=PhysicalType.GRAPH_SCALAR_INTENSIVE,
        target_shape=(1,),
        target_irreps=None,
        required_label_components=["density"],
        output_head="ScalarReadout",
        pooling="mean",
        status=Status.SUPPORTED,
    ),
]

_JARVIS_TENSORS = [
    PropertySpec(
        name="dielectric_diagonal_canonical",
        source="jarvis",
        level="graph",
        physical_type=PhysicalType.GRAPH_RANK2_DIAGONAL_CANONICAL,
        target_shape=(3,),
        target_irreps=None,
        required_label_components=["epsx", "epsy", "epsz"],
        output_head="DiagonalCanonicalHead",
        pooling="mean",
        notes="JARVIS provides ONLY the diagonal components (εx, εy, εz) in the "
              "database canonical frame (aligned to standardized lattice). This is "
              "NOT a rotationally covariant tensor output — predicting 3 scalars in a "
              "fixed frame. Do NOT claim SO(3) equivariance for this head. "
              "For full tensor equivariance, use 'dielectric_full_tensor' which requires "
              "all 6 independent components.",
        status=Status.BENCHMARK_ONLY,
    ),
    PropertySpec(
        name="dielectric_full_tensor",
        source="jarvis",
        level="graph",
        physical_type=PhysicalType.GRAPH_RANK2_SYMMETRIC_TENSOR,
        target_shape=(3, 3),
        target_irreps="1x0e+1x2e",
        required_label_components=["eps_xx", "eps_xy", "eps_xz", "eps_yy", "eps_yz", "eps_zz"],
        output_head="TensorPropertyHead",
        pooling="sum",
        notes="Full symmetric dielectric tensor decomposed into trace (0e) + traceless "
              "symmetric (2e) = 6 components in irrep basis. Requires ALL 6 independent "
              "tensor components. JARVIS DFT-3D does NOT provide these — this spec is "
              "for future datasets that do (e.g. Materials Project DFPT).",
        status=Status.UNSUPPORTED,
    ),
    PropertySpec(
        name="dielectric_tensor",
        source="gmtnet",
        level="graph",
        physical_type=PhysicalType.GRAPH_RANK2_SYMMETRIC_TENSOR,
        target_shape=(3, 3),
        target_irreps="1x0e+1x2e",
        required_label_components=["dielectric"],
        output_head="TensorPropertyHead",
        pooling="sum",
        notes="Full symmetric dielectric tensor from GMTNet/JARVIS-DFT dataset. "
              "4713 samples with 3x3 tensor targets. Used for benchmark comparison.",
        status=Status.SUPPORTED,
    ),
    PropertySpec(
        name="elastic_tensor",
        source="jarvis",
        level="graph",
        physical_type=PhysicalType.GRAPH_RANK4_ELASTIC_TENSOR,
        target_shape=(6, 6),
        target_irreps="2x0e+2x2e+1x4e",
        required_label_components=["elastic_tensor"],
        output_head="ElasticTensorHead",
        pooling="sum",
        notes="Full 4th-rank elasticity tensor C_{ijkl} stored as 6x6 Voigt matrix. "
              "Correct irrep decomposition: 2×A₁g(0e) + 2×Eg(2e) + 1×T₂g(4e) = "
              "2 + 10 + 9 = 21 independent components. "
              "FORBIDDEN: do NOT use rank-2 TensorPropertyHead(1x0e+1x2e) for this — "
              "that discards 15 of 21 independent components and violates the tensor rank. "
              "ElasticTensorHead is not yet implemented.",
        status=Status.UNSUPPORTED,
    ),
    PropertySpec(
        name="piezoelectric_tensor",
        source="gmtnet",
        level="graph",
        physical_type=PhysicalType.GRAPH_RANK3_PIEZO_TENSOR,
        target_shape=(3, 6),
        target_irreps="2x1o+1x2o+1x3o",
        required_label_components=["piezoelectric_C_m2"],
        output_head="PiezoRank3Readout",
        pooling="sum",
        notes="Full 3rd-rank piezoelectric tensor e_{ijk} in C/m^2, stored as "
              "3x6 Voigt with VASP column order (xx, yy, zz, xy, yz, zx) and "
              "no factor of 2. The bare JARVIS record only exposes the scalar "
              "dfpt_piezo_max_eij, but the GMTNet benchmark pkl ships the full "
              "tensor under 'piezoelectric_C_m2'; use the gmtnet_piezo adapter.",
        status=Status.SUPPORTED,
    ),
]

_JARVIS_ATOM_LEVEL = [
    PropertySpec(
        name="atom_bader_charge",
        source="jarvis",
        level="atom",
        physical_type=PhysicalType.ATOM_SCALAR,
        target_shape=(-1, 1),
        target_irreps="1x0e",
        required_label_components=["bader_charges"],
        output_head="TensorPropertyHead",
        pooling="none",
        notes="Per-atom Bader charge. JARVIS does NOT provide this label. "
              "Would need Materials Project or separate Bader analysis.",
        status=Status.UNSUPPORTED,
    ),
    PropertySpec(
        name="atom_born_effective_charge",
        source="generic",
        level="atom",
        physical_type=PhysicalType.ATOM_VECTOR_POLAR,
        target_shape=(-1, 3),
        target_irreps="1x1o",
        required_label_components=["born_effective_charges"],
        output_head="TensorPropertyHead",
        pooling="none",
        notes="Per-atom Born effective charge vector (polar, 1o). "
              "Static property in Wyckoff representation — NOT force. "
              "Requires DFPT data (not in JARVIS).",
        status=Status.UNSUPPORTED,
    ),
    PropertySpec(
        name="atom_magnetic_moment",
        source="generic",
        level="atom",
        physical_type=PhysicalType.ATOM_VECTOR_AXIAL,
        target_shape=(-1, 3),
        target_irreps="1x1e",
        required_label_components=["magnetic_moments"],
        output_head="TensorPropertyHead",
        pooling="none",
        notes="Per-atom magnetic moment direction (axial/pseudovector, 1e). "
              "Static property.",
        status=Status.UNSUPPORTED,
    ),
    PropertySpec(
        name="atom_efg_tensor",
        source="generic",
        level="atom",
        physical_type=PhysicalType.ATOM_RANK2_SYMMETRIC_TENSOR,
        target_shape=(-1, 3, 3),
        target_irreps="1x0e+1x2e",
        required_label_components=["efg_tensors"],
        output_head="TensorPropertyHead",
        pooling="none",
        notes="Per-atom electric field gradient tensor (symmetric, traceless → 1x2e). "
              "JARVIS max_efg is graph-level max, NOT per-atom tensors.",
        status=Status.UNSUPPORTED,
    ),
]

_HAMILTONIAN = [
    PropertySpec(
        name="hamiltonian",
        source="deeph",
        level="hamiltonian",
        physical_type=PhysicalType.HAMILTONIAN,
        target_shape=None,
        target_irreps=None,
        required_label_components=["hamiltonian_blocks"],
        output_head="WyckoffHamiltonianHead",
        pooling="none",
        notes="Tight-binding Hamiltonian H_{ij}(R) blocks. "
              "Required checks: Hermiticity H_ji(-R)=H_ij(R)†, "
              "onsite block site-symmetry, orbital convention.",
        status=Status.SUPPORTED,
    ),
]

# ---------------------------------------------------------------------------
# Full registry
# ---------------------------------------------------------------------------

PROPERTY_REGISTRY: Dict[str, PropertySpec] = {}
for spec in _JARVIS_SCALARS + _JARVIS_TENSORS + _JARVIS_ATOM_LEVEL + _HAMILTONIAN:
    if spec.name in PROPERTY_REGISTRY:
        raise ValueError(f"Duplicate property name: {spec.name}")
    PROPERTY_REGISTRY[spec.name] = spec


# ---------------------------------------------------------------------------
# Access functions
# ---------------------------------------------------------------------------

def get_property_spec(name: str) -> PropertySpec:
    """Look up a PropertySpec by name. Raises KeyError if not found."""
    if name not in PROPERTY_REGISTRY:
        raise KeyError(
            f"Unknown property '{name}'. Available: {list(PROPERTY_REGISTRY.keys())}"
        )
    return PROPERTY_REGISTRY[name]


def list_properties(source: Optional[str] = None, status: Optional[str] = None) -> List[PropertySpec]:
    """List properties, optionally filtered by source and/or status."""
    results = list(PROPERTY_REGISTRY.values())
    if source is not None:
        results = [p for p in results if p.source == source]
    if status is not None:
        results = [p for p in results if p.status == status]
    return results


def get_supported_properties(source: Optional[str] = None) -> List[PropertySpec]:
    """Return only properties with status=SUPPORTED or BENCHMARK_ONLY."""
    return [
        p for p in list_properties(source=source)
        if p.status in (Status.SUPPORTED, Status.BENCHMARK_ONLY)
    ]
