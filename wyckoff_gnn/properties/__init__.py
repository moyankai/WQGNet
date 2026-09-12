"""Properties package — physical property definitions and head dispatch."""
from wyckoff_gnn.properties.registry import (
    PropertySpec,
    PhysicalType,
    Status,
    PROPERTY_REGISTRY,
    get_property_spec,
    list_properties,
    get_supported_properties,
)

__all__ = [
    "PropertySpec",
    "PhysicalType",
    "Status",
    "PROPERTY_REGISTRY",
    "get_property_spec",
    "list_properties",
    "get_supported_properties",
]
