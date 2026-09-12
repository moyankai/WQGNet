"""Property-type enumeration for WyckoffGNN multi-target support.

Defines the canonical property types the system can predict, along with
metadata about each (intensive/extensive, irrep representation, per-atom
vs per-graph scope, Cartesian shape, etc.).

This module is the single source of truth for property-type strings
throughout the codebase: config parsing, data records, model factory,
training loop, and evaluation all reference these constants.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


__all__ = [
    "PropertyType",
    "PROPERTY_TYPE_INFO",
    "validate_property_type",
    "is_graph_level",
    "is_atom_level",
    "is_tensor_type",
    "default_pool_for_property_type",
    "default_target_irreps_for_property_type",
]


# Canonical property-type strings.
class PropertyType:
    GRAPH_SCALAR_INTENSIVE = "graph_scalar_intensive"
    GRAPH_SCALAR_EXTENSIVE = "graph_scalar_extensive"
    GRAPH_VECTOR = "graph_vector"
    GRAPH_TENSOR = "graph_tensor"
    ATOM_SCALAR = "atom_scalar"
    ATOM_VECTOR = "atom_vector"
    ATOM_TENSOR = "atom_tensor"
    HAMILTONIAN = "hamiltonian"

    ALL = (
        GRAPH_SCALAR_INTENSIVE,
        GRAPH_SCALAR_EXTENSIVE,
        GRAPH_VECTOR,
        GRAPH_TENSOR,
        ATOM_SCALAR,
        ATOM_VECTOR,
        ATOM_TENSOR,
        HAMILTONIAN,
    )


@dataclass(frozen=True)
class PropertyTypeInfo:
    name: str
    scope: str                      # "graph" or "atom" or "edge"
    intensive: bool                 # True → per-atom normalised (pool=mean)
    default_pool: str               # "mean" | "sum" | "none"
    target_irreps: Optional[str]    # e3nn irreps (None for scalar / hamiltonian)
    cartesian_dim: Optional[int]    # output dim in Cartesian space (None = variable)
    description: str


PROPERTY_TYPE_INFO = {
    PropertyType.GRAPH_SCALAR_INTENSIVE: PropertyTypeInfo(
        name="graph_scalar_intensive",
        scope="graph",
        intensive=True,
        default_pool="mean",
        target_irreps=None,
        cartesian_dim=1,
        description="Graph-level intensive scalar (bandgap, formation energy/atom, ehull)",
    ),
    PropertyType.GRAPH_SCALAR_EXTENSIVE: PropertyTypeInfo(
        name="graph_scalar_extensive",
        scope="graph",
        intensive=False,
        default_pool="sum",
        target_irreps=None,
        cartesian_dim=1,
        description="Graph-level extensive scalar (total energy, ∝ num_atoms)",
    ),
    PropertyType.GRAPH_VECTOR: PropertyTypeInfo(
        name="graph_vector",
        scope="graph",
        intensive=True,
        default_pool="sum",
        target_irreps="1x1o",
        cartesian_dim=3,
        description="Graph-level polar vector (e.g. piezo dipole). "
                    "Forbidden for centrosymmetric crystals.",
    ),
    PropertyType.GRAPH_TENSOR: PropertyTypeInfo(
        name="graph_tensor",
        scope="graph",
        intensive=True,
        default_pool="sum",
        target_irreps="1x0e+1x2e",
        cartesian_dim=6,
        description="Graph-level symmetric rank-2 tensor (dielectric, elastic Cij). "
                    "Default irreps = 1x0e+1x2e (trace + traceless symmetric); "
                    "override with target_irreps in config for general rank-2 or higher.",
    ),
    PropertyType.ATOM_SCALAR: PropertyTypeInfo(
        name="atom_scalar",
        scope="atom",
        intensive=True,
        default_pool="none",
        target_irreps=None,
        cartesian_dim=1,
        description="Per-atom scalar (Bader charge, Mulliken population)",
    ),
    PropertyType.ATOM_VECTOR: PropertyTypeInfo(
        name="atom_vector",
        scope="atom",
        intensive=True,
        default_pool="none",
        target_irreps="1x1o",
        cartesian_dim=3,
        description="Per-atom static vector (Born effective charge, magnetic moment direction). "
                    "NOT force — see theory doc for why autograd ∂E/∂pos ≠ force in Wyckoff.",
    ),
    PropertyType.ATOM_TENSOR: PropertyTypeInfo(
        name="atom_tensor",
        scope="atom",
        intensive=True,
        default_pool="none",
        target_irreps="1x0e+1x2e",
        cartesian_dim=6,
        description="Per-atom symmetric rank-2 tensor (EFG, thermal ellipsoid)",
    ),
    PropertyType.HAMILTONIAN: PropertyTypeInfo(
        name="hamiltonian",
        scope="edge",
        intensive=True,
        default_pool="none",
        target_irreps=None,
        cartesian_dim=None,
        description="Tight-binding Hamiltonian H_{ij}(R) blocks. Special head.",
    ),
}


def validate_property_type(pt: str) -> str:
    """Validate and canonicalise a property_type string.

    Raises ValueError for unknown types.
    """
    if pt in PropertyType.ALL:
        return pt
    # Legacy aliases
    _ALIASES = {
        "scalar": PropertyType.GRAPH_SCALAR_INTENSIVE,
        "graph_scalar": PropertyType.GRAPH_SCALAR_INTENSIVE,
        "intensive": PropertyType.GRAPH_SCALAR_INTENSIVE,
        "extensive": PropertyType.GRAPH_SCALAR_EXTENSIVE,
        "vector": PropertyType.GRAPH_VECTOR,
        "tensor": PropertyType.GRAPH_TENSOR,
    }
    if pt in _ALIASES:
        return _ALIASES[pt]
    raise ValueError(
        f"Unknown property_type: {pt!r}. Valid: {PropertyType.ALL}"
    )


def is_graph_level(pt: str) -> bool:
    return PROPERTY_TYPE_INFO[validate_property_type(pt)].scope == "graph"


def is_atom_level(pt: str) -> bool:
    return PROPERTY_TYPE_INFO[validate_property_type(pt)].scope == "atom"


def is_tensor_type(pt: str) -> bool:
    info = PROPERTY_TYPE_INFO[validate_property_type(pt)]
    return info.target_irreps is not None


def default_pool_for_property_type(pt: str) -> str:
    return PROPERTY_TYPE_INFO[validate_property_type(pt)].default_pool


def default_target_irreps_for_property_type(pt: str) -> Optional[str]:
    return PROPERTY_TYPE_INFO[validate_property_type(pt)].target_irreps
