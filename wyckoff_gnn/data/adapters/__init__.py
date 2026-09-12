"""Dataset adapters — convert raw data sources into StructureRecords."""

from wyckoff_gnn.data.adapters.base import DatasetAdapter
from wyckoff_gnn.data.adapters.csv_adapter import CSVAdapter
from wyckoff_gnn.data.adapters.cif_dir_adapter import CIFDirAdapter
from wyckoff_gnn.data.adapters.jarvis_adapter import JARVISAdapter
from wyckoff_gnn.data.adapters.matbench_adapter import MatbenchAdapter
from wyckoff_gnn.data.adapters.registry import ADAPTER_REGISTRY, create_adapter

__all__ = [
    "DatasetAdapter",
    "CSVAdapter",
    "CIFDirAdapter",
    "JARVISAdapter",
    "MatbenchAdapter",
    "ADAPTER_REGISTRY",
    "create_adapter",
]
