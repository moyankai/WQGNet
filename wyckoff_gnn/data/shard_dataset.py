"""Lazy LMDB-backed Dataset for Wyckoff graphs.

Loads graphs on-demand from LMDB via mmap. Per-sample random access,
multi-worker safe, no first-epoch warmup cost.
"""

from __future__ import annotations

import json
import os
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from wyckoff_gnn.data.graph_schema import light_dict_to_pyg_data
from wyckoff_gnn.data.lmdb_cache import LMDBReader


class WyckoffDataset(Dataset):
    """Lazy-loading dataset from a manifest + LMDB cache.

    Args:
        manifest_path: Path to ``manifest.jsonl``.
        data_dir: Directory containing ``data.lmdb`` and ``manifest.jsonl``.
        split: If provided, only return samples with this split label.
        transform: Optional callable ``(Data) -> Data`` applied on load.
        global_max_mult: Pad sym_ops tensors to this multiplicity for batching.
        normalizer: Optional TargetNormalizer for z-score normalization.
        entries_override: If provided, bypass reading manifest.jsonl and use
            these entries directly. Useful for injecting an official
            leaderboard-based split whose ``split``/``target`` fields have
            been overwritten upstream. When set, ``split`` is ignored.
    """

    def __init__(
        self,
        manifest_path: str,
        data_dir: str,
        split: Optional[str] = None,
        transform: Optional[Callable[[Data], Data]] = None,
        global_max_mult: int = 192,
        normalizer: Optional[Any] = None,
        entries_override: Optional[List[Dict[str, Any]]] = None,
    ):
        super().__init__()
        self.manifest_path = manifest_path
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self._global_max_mult = global_max_mult
        self._normalizer = normalizer

        self._entries: List[Dict[str, Any]] = []
        if entries_override is not None:
            # Caller has already filtered/prepared the entries.
            self._entries = list(entries_override)
        else:
            with open(manifest_path, "r") as f:
                for line in f:
                    entry = json.loads(line)
                    if split is not None and entry.get("split") != split:
                        continue
                    self._entries.append(entry)

        self.reader = LMDBReader(data_dir)

    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, idx: int) -> Data:
        entry = self._entries[idx]
        key = entry.get("lmdb_key", entry["material_id"])
        graph_dict = self.reader.get(key)
        data = light_dict_to_pyg_data(graph_dict)

        data.material_id = entry["material_id"]
        if entry.get("target") is not None:
            raw_y = float(entry["target"])
            if self._normalizer is not None:
                data.y_raw = torch.tensor([raw_y], dtype=torch.float32)
                data.y = torch.tensor(
                    [self._normalizer.normalize(raw_y)], dtype=torch.float32,
                )
            else:
                data.y = torch.tensor([raw_y], dtype=torch.float32)

        data = self._pad_to_global_max_mult(data)
        if self.transform is not None:
            data = self.transform(data)
        return data

    def _pad_to_global_max_mult(self, data: Data) -> Data:
        """Pad orbit_sym_ops_* and orbit_mult_mask to uniform second dim."""
        global_max = self._global_max_mult
        fields_to_pad = [
            "orbit_sym_ops_rotations",
            "orbit_sym_ops_W_frac",
            "orbit_sym_ops_w_frac",
            "orbit_mult_mask",
        ]
        for field in fields_to_pad:
            tensor = getattr(data, field, None)
            if tensor is None:
                continue
            K, M = tensor.shape[0], tensor.shape[1]
            if M >= global_max:
                continue
            shape = (K, global_max) + tuple(tensor.shape[2:])
            padded = torch.zeros(shape, dtype=tensor.dtype)
            padded[:, :M, ...] = tensor
            setattr(data, field, padded)
        return data

    @property
    def material_ids(self) -> List[str]:
        return [e["material_id"] for e in self._entries]

    @property
    def targets(self) -> np.ndarray:
        """Target values; NaN for entries with missing targets."""
        return np.array([
            float(e["target"]) if e.get("target") is not None else float("nan")
            for e in self._entries
        ], dtype=np.float32)

    @property
    def num_atoms_list(self) -> List[int]:
        return [int(e.get("num_atoms", 0)) for e in self._entries]

    @property
    def num_orbits_list(self) -> List[int]:
        return [int(e.get("num_orbits", 0)) for e in self._entries]

    @property
    def splits(self) -> List[str]:
        return [e.get("split", "train") for e in self._entries]

    def entry(self, idx: int) -> Dict[str, Any]:
        """Access manifest entry by index."""
        return self._entries[idx]

    @property
    def entries(self) -> List[Dict[str, Any]]:
        """Read-only access to manifest entries."""
        return self._entries

    def truncate(self, n: int) -> None:
        """Truncate dataset to first n entries (for debug/limit modes)."""
        self._entries = self._entries[:n]

    def filter_valid_targets(self) -> int:
        """Remove entries with null targets. Returns count removed."""
        before = len(self._entries)
        self._entries = [e for e in self._entries if e.get("target") is not None]
        return before - len(self._entries)


# Backward compatibility alias
WyckoffShardDataset = WyckoffDataset

__all__ = ["WyckoffDataset", "WyckoffShardDataset"]
