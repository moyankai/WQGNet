"""Adapter registry — maps adapter names to classes and provides a factory."""

from __future__ import annotations

from typing import Any, Dict, Type

from wyckoff_gnn.data.adapters.base import DatasetAdapter
from wyckoff_gnn.data.adapters.csv_adapter import CSVAdapter
from wyckoff_gnn.data.adapters.cif_dir_adapter import CIFDirAdapter
from wyckoff_gnn.data.adapters.jarvis_adapter import JARVISAdapter
from wyckoff_gnn.data.adapters.deeph_adapter import DeepHAdapter
from wyckoff_gnn.data.adapters.hamgnn_npz_adapter import HamGNNNpzAdapter
from wyckoff_gnn.data.adapters.matbench_adapter import MatbenchAdapter
from wyckoff_gnn.data.adapters.gmtnet_dielectric_adapter import GMTNetDielectricAdapter
from wyckoff_gnn.data.adapters.gmtnet_piezo_adapter import GMTNetPiezoAdapter

ADAPTER_REGISTRY: Dict[str, Type[DatasetAdapter]] = {
    "csv": CSVAdapter,
    "cif_dir": CIFDirAdapter,
    "jarvis": JARVISAdapter,
    "deeph": DeepHAdapter,
    "hamgnn_npz": HamGNNNpzAdapter,
    "matbench": MatbenchAdapter,
    "gmtnet_dielectric": GMTNetDielectricAdapter,
    "gmtnet_piezo": GMTNetPiezoAdapter,
}


def create_adapter(config: Dict[str, Any]) -> DatasetAdapter:
    """Create a DatasetAdapter from a config dict.

    The config must contain an ``"adapter"`` key with the adapter name,
    and optionally ``"adapter_kwargs"`` with constructor arguments.

    Supports nested configs::

        {"dataset": {"adapter": "csv", "adapter_kwargs": {...}}}
        {"adapter": "csv", "adapter_kwargs": {...}}
    """
    # Support nested "dataset" key.
    if "dataset" in config:
        config = config["dataset"]

    adapter_name = config.get("adapter")
    if adapter_name is None:
        available = list(ADAPTER_REGISTRY.keys())
        raise ValueError(
            f"Config missing 'adapter' key. Available adapters: {available}"
        )

    cls = ADAPTER_REGISTRY.get(adapter_name)
    if cls is None:
        available = list(ADAPTER_REGISTRY.keys())
        raise ValueError(
            f"Unknown adapter '{adapter_name}'. Available: {available}"
        )

    kwargs = config.get("adapter_kwargs", {})
    return cls.from_config(kwargs)


def list_adapters() -> Dict[str, str]:
    """Return a dict of adapter names and their docstrings."""
    return {name: cls.__doc__ or "(no docstring)" for name, cls in ADAPTER_REGISTRY.items()}


__all__ = ["ADAPTER_REGISTRY", "create_adapter", "list_adapters"]
