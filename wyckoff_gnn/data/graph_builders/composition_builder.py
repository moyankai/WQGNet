"""Composition feature builder for baseline models.

Extracts per-element counts and formula information from StructureRecords.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict

from wyckoff_gnn.data.graph_builders.base import GraphBuilder
from wyckoff_gnn.data.records import StructureRecord


class CompositionGraphBuilder(GraphBuilder):
    """Build composition feature dicts from StructureRecords.

    Extracts element counts, formula, and target value.  This builder
    does NOT produce graph tensors — it produces a feature dict suitable
    for composition baselines (ridge regression, random forest, etc.).
    """

    def __init__(self):
        pass

    def build(self, record: StructureRecord) -> Dict[str, Any]:
        struct = record.structure
        elements: Dict[str, int] = {}
        for site in struct:
            el = str(site.specie)
            elements[el] = elements.get(el, 0) + 1

        return {
            "material_id": record.material_id,
            "y": float(record.target) if record.target is not None else None,
            "formula": struct.composition.reduced_formula
                if hasattr(struct, "composition") else "unknown",
            "num_atoms": len(struct),
            "elements": dict(elements),
            "metadata": record.metadata,
        }


__all__ = ["CompositionGraphBuilder"]
