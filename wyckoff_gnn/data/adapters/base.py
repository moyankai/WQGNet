"""Abstract base class for dataset adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Iterable, List

from wyckoff_gnn.data.records import StructureRecord


class DatasetAdapter(ABC):
    """Convert a raw data source into an iterable of :class:`StructureRecord`.

    Subclasses must implement ``iter_records``.  They may optionally
    override ``from_config`` for config-driven construction.

    Adapters MUST NOT:
    - Call WyckoffGraphBuilder
    - Write shard caches
    - Normalise targets
    - Produce PyG Data objects
    """

    @classmethod
    @abstractmethod
    def from_config(cls, config: Dict[str, Any]) -> "DatasetAdapter":
        """Create an adapter instance from a configuration dict.

        The config dict is typically the ``adapter_kwargs`` sub-block
        of a dataset config.
        """
        ...

    @abstractmethod
    def iter_records(self) -> Iterable[StructureRecord]:
        """Yield :class:`StructureRecord` objects one at a time.

        Implementations SHOULD be lazy (generator-based) so that large
        datasets are not fully loaded into memory.
        """
        ...

    def list_records(self) -> List[StructureRecord]:
        """Convenience: collect all records into a list."""
        return list(self.iter_records())

    @staticmethod
    def _resolve_path(base_dir: Path, rel: str) -> Path:
        """Resolve a relative path against a base directory."""
        p = Path(rel)
        if p.is_absolute():
            return p
        return (base_dir / p).resolve()


__all__ = ["DatasetAdapter"]
