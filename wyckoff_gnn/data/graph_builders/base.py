"""Abstract base class for graph builders."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

from wyckoff_gnn.data.records import StructureRecord


class GraphBuilder(ABC):
    """Convert a :class:`StructureRecord` into a light graph dict.

    Subclasses implement ``build``.  They may override ``from_config``
    for config-driven construction.

    Builders MUST NOT:
    - Read raw data files (CSV, CIF, JSON, etc.)
    - Write shard caches
    - Normalize targets
    - Handle split generation
    """

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "GraphBuilder":
        """Create a builder from a config dict (graph sub-block)."""
        return cls(**config)

    @abstractmethod
    def build(self, record: StructureRecord) -> Dict[str, Any]:
        """Build a light graph dict for one material.

        Returns a dict compatible with :class:`ShardWriter` and
        :func:`pyg_data_to_light_dict`.
        """
        ...


__all__ = ["GraphBuilder"]
