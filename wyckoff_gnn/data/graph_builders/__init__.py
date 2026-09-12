"""Graph builders — convert StructureRecords into light graph dicts."""

from wyckoff_gnn.data.graph_builders.base import GraphBuilder
from wyckoff_gnn.data.graph_builders.wyckoff_builder import WyckoffGraphBuilderWrapper
from wyckoff_gnn.data.graph_builders.atom_builder import AtomGraphBuilderWrapper
from wyckoff_gnn.data.graph_builders.composition_builder import CompositionGraphBuilder
from wyckoff_gnn.data.graph_builders.registry import (
    GRAPH_BUILDER_REGISTRY,
    create_graph_builder,
    list_graph_builders,
)

__all__ = [
    "GraphBuilder",
    "WyckoffGraphBuilderWrapper",
    "AtomGraphBuilderWrapper",
    "CompositionGraphBuilder",
    "GRAPH_BUILDER_REGISTRY",
    "create_graph_builder",
    "list_graph_builders",
]
