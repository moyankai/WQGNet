"""Graph builder registry."""

from __future__ import annotations

from typing import Any, Dict, Type

from wyckoff_gnn.data.graph_builders.base import GraphBuilder
from wyckoff_gnn.data.graph_builders.wyckoff_builder import WyckoffGraphBuilderWrapper
from wyckoff_gnn.data.graph_builders.atom_builder import AtomGraphBuilderWrapper
from wyckoff_gnn.data.graph_builders.composition_builder import CompositionGraphBuilder
from wyckoff_gnn.data.graph_builders.p1_builder import P1GraphBuilderWrapper

GRAPH_BUILDER_REGISTRY: Dict[str, Type[GraphBuilder]] = {
    "wyckoff": WyckoffGraphBuilderWrapper,
    "atom": AtomGraphBuilderWrapper,
    "composition": CompositionGraphBuilder,
    "p1": P1GraphBuilderWrapper,
}

GRAPH_BUILDER_STATUS: Dict[str, str] = {
    "wyckoff": "full — preprocess + train supported",
    "atom": "preprocess only — train_property does not yet support atom graphs",
    "composition": "preprocess only — train_property does not yet support composition",
    "p1": "full — preprocess + train supported (no symmetry compression)",
}


def create_graph_builder(config: Dict[str, Any]) -> GraphBuilder:
    """Create a GraphBuilder from a config dict.

    Supports nested configs::

        {"graph": {"type": "wyckoff", "cutoff": 5.0}}
        {"type": "wyckoff", "cutoff": 5.0}
    """
    if "graph" in config:
        config = config["graph"]

    gtype = config.get("type")
    if gtype is None:
        available = list(GRAPH_BUILDER_REGISTRY.keys())
        raise ValueError(
            f"Config missing 'graph.type' key. Available: {available}"
        )

    cls = GRAPH_BUILDER_REGISTRY.get(gtype)
    if cls is None:
        available = list(GRAPH_BUILDER_REGISTRY.keys())
        raise ValueError(
            f"Unknown graph type '{gtype}'. Available: {available}"
        )

    kwargs = {k: v for k, v in config.items() if k != "type"}
    return cls.from_config(kwargs)


def list_graph_builders() -> Dict[str, str]:
    """Return a dict of builder names and their status."""
    return dict(GRAPH_BUILDER_STATUS)


__all__ = [
    "GRAPH_BUILDER_REGISTRY",
    "GRAPH_BUILDER_STATUS",
    "create_graph_builder",
    "list_graph_builders",
]
