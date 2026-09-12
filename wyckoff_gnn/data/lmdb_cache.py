"""LMDB-based graph cache for fast random-access data loading.

Provides LMDBWriter and LMDBReader as drop-in alternatives to
ShardWriter and ShardReader. Uses msgpack serialization with
raw numpy byte storage for maximum read speed.

Benefits over .pt shards:
- Per-sample random access (no need to load 1000-sample shard)
- mmap-based (OS manages page cache, no manual caching needed)
- Multi-process safe (num_workers > 0 just works)
- No first-epoch warmup cost
- Scales to millions of samples without OOM
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import lmdb
import msgpack
import numpy as np

from wyckoff_gnn.utils.io import save_json


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def _serialize_graph(graph_dict: Dict[str, Any]) -> bytes:
    """Serialize a light graph dict for LMDB storage."""
    packed = {}
    for k, v in graph_dict.items():
        packed[k] = _serialize_value(v)
    return msgpack.packb(packed, use_bin_type=True)


def _serialize_value(v: Any) -> Any:
    """Recursively serialize a value for msgpack."""
    if isinstance(v, np.ndarray):
        return {
            b"__np__": True,
            b"dtype": v.dtype.str.encode(),
            b"shape": list(v.shape),
            b"data": v.tobytes(),
        }
    elif isinstance(v, dict):
        out = {}
        for dk, dv in v.items():
            str_key = str(dk) if not isinstance(dk, str) else dk
            out[str_key] = _serialize_value(dv)
        return {b"__dict__": True, b"entries": out}
    else:
        return v


def _deserialize_graph(data: bytes) -> Dict[str, Any]:
    """Deserialize a light graph dict from LMDB."""
    packed = msgpack.unpackb(data, raw=True)
    result = {}
    for k, v in packed.items():
        key = k.decode() if isinstance(k, bytes) else k
        result[key] = _deserialize_value(v)
    return result


def _deserialize_value(v: Any) -> Any:
    """Recursively deserialize a value from msgpack."""
    if isinstance(v, dict):
        is_np = v.get(b"__np__") or v.get("__np__")
        is_dict = v.get(b"__dict__") or v.get("__dict__")
        if is_np:
            dtype_raw = v.get(b"dtype") or v.get("dtype")
            dtype = dtype_raw.decode() if isinstance(dtype_raw, bytes) else dtype_raw
            shape = v.get(b"shape") or v.get("shape")
            raw_data = v.get(b"data") or v.get("data")
            if raw_data is not None:
                return np.frombuffer(raw_data, dtype=dtype).reshape(shape).copy()
            else:
                return np.zeros(shape, dtype=dtype)
        elif is_dict:
            entries = v.get(b"entries") or v.get("entries")
            if entries is None:
                return {}
            out = {}
            for ek, ev in entries.items():
                out_key = ek.decode() if isinstance(ek, bytes) else ek
                out[out_key] = _deserialize_value(ev)
            return out
        else:
            decoded = {}
            for dk, dv in v.items():
                decoded[dk.decode() if isinstance(dk, bytes) else dk] = _deserialize_value(dv)
            return decoded
    else:
        if isinstance(v, bytes):
            try:
                return v.decode()
            except UnicodeDecodeError:
                return v
        return v


# ---------------------------------------------------------------------------
# LMDBWriter
# ---------------------------------------------------------------------------

class LMDBWriter:
    """Write graph dicts to a single LMDB file.

    Drop-in replacement for ShardWriter with the same public API.

    Args:
        output_dir: Root directory for LMDB file and metadata.
        map_size: Maximum LMDB map size in bytes (default 1TB virtual).
    """

    def __init__(self, output_dir: str, map_size: int = 2**40):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self._lmdb_path = os.path.join(output_dir, "data.lmdb")
        self._env = lmdb.open(
            self._lmdb_path,
            map_size=map_size,
            subdir=False,
            lock=True,
            readahead=False,
            meminit=False,
        )
        self._txn = self._env.begin(write=True)
        self._write_count = 0
        self._commit_every = 500

        self._manifest_entries: List[Dict[str, Any]] = []
        self._failed_entries: List[Dict[str, Any]] = []
        self._stats: Dict[str, Any] = {
            "n_total": 0,
            "n_ok": 0,
            "n_failed": 0,
            "target_min": float("inf"),
            "target_max": float("-inf"),
            "target_sum": 0.0,
            "target_sum_sq": 0.0,
            "avg_atoms": 0.0,
            "avg_orbits": 0.0,
            "avg_edges": 0.0,
            "min_orbits": float("inf"),
            "max_orbits": float("-inf"),
            "space_group_counts": {},
        }

    def add_ok(
        self,
        graph_dict: Dict[str, Any],
        material_id: str,
        split: str = "train",
    ) -> None:
        """Add a successfully built graph to LMDB."""
        key = material_id.encode()
        value = _serialize_graph(graph_dict)
        self._txn.put(key, value)
        self._write_count += 1

        if self._write_count % self._commit_every == 0:
            self._txn.commit()
            self._txn = self._env.begin(write=True)

        if "orbit_element" in graph_dict:
            n_orbits = int(graph_dict["orbit_element"].shape[0])
            atom_to_orbit = graph_dict.get("atom_to_orbit")
            n_atoms = int(atom_to_orbit.shape[0]) if atom_to_orbit is not None else n_orbits
            n_edges = int(graph_dict["geo_edge_index"].shape[1])
            compression_ratio = n_atoms / max(n_orbits, 1)
        else:
            n_atoms = int(graph_dict["num_atoms"])
            n_orbits = n_atoms
            n_edges = int(graph_dict["edge_index"].shape[1])
            compression_ratio = 1.0
        y_val = graph_dict.get("y")
        sg = graph_dict.get("space_group", 1)
        target_type = graph_dict.get("target_type")

        # For manifest: always store a scalar-like target value.
        # Non-scalar targets (tensor/hamiltonian) store a scalar summary (mean/0).
        manifest_target = None
        if y_val is not None:
            try:
                manifest_target = float(y_val)
            except (TypeError, ValueError):
                manifest_target = 0.0

        manifest_entry = {
            "material_id": material_id,
            "split": split,
            "target": manifest_target,
            "lmdb_key": material_id,
            "num_atoms": n_atoms,
            "num_orbits": n_orbits,
            "space_group": sg,
            "num_geo_edges": n_edges,
            "compression_ratio": compression_ratio,
        }
        if target_type:
            manifest_entry["target_type"] = target_type
        self._manifest_entries.append(manifest_entry)

        self._stats["n_ok"] += 1
        self._stats["n_total"] += 1
        self._stats["avg_atoms"] += n_atoms
        self._stats["avg_orbits"] += n_orbits
        self._stats["avg_edges"] += n_edges
        self._stats["min_orbits"] = min(self._stats["min_orbits"], n_orbits)
        self._stats["max_orbits"] = max(self._stats["max_orbits"], n_orbits)
        sg_key = str(sg)
        self._stats["space_group_counts"][sg_key] = (
            self._stats["space_group_counts"].get(sg_key, 0) + 1
        )
        if manifest_target is not None:
            yf = float(manifest_target)
            self._stats["target_min"] = min(self._stats["target_min"], yf)
            self._stats["target_max"] = max(self._stats["target_max"], yf)
            self._stats["target_sum"] += yf
            self._stats["target_sum_sq"] += yf * yf

    def add_failed(self, material_id: str, reason: str) -> None:
        """Log a failed sample."""
        self._failed_entries.append({"material_id": material_id, "reason": reason})
        self._stats["n_failed"] += 1
        self._stats["n_total"] += 1

    def finalize(self) -> None:
        """Commit remaining writes and write metadata files."""
        self._txn.commit()
        self._env.close()

        n = max(self._stats["n_ok"], 1)
        self._stats["avg_atoms"] /= n
        self._stats["avg_orbits"] /= n
        self._stats["avg_edges"] /= n
        self._stats["target_mean"] = self._stats["target_sum"] / max(n, 1)
        variance = (self._stats["target_sum_sq"] / max(n, 1)
                    - self._stats["target_mean"] ** 2)
        self._stats["target_std"] = max(float(np.sqrt(max(variance, 0))), 1e-8)
        self._stats["format"] = "lmdb"

        manifest_path = os.path.join(self.output_dir, "manifest.jsonl")
        with open(manifest_path, "w") as f:
            for entry in self._manifest_entries:
                f.write(json.dumps(entry) + "\n")

        save_json(os.path.join(self.output_dir, "stats.json"), self._stats)

        if self._failed_entries:
            failed_path = os.path.join(self.output_dir, "failed.jsonl")
            with open(failed_path, "w") as f:
                for entry in self._failed_entries:
                    f.write(json.dumps(entry) + "\n")

    @property
    def manifest_entries(self) -> List[Dict[str, Any]]:
        return self._manifest_entries

    @property
    def stats(self) -> Dict[str, Any]:
        return dict(self._stats)

    @classmethod
    def resume_from_existing(cls, output_dir: str, **kwargs) -> "LMDBWriter":
        """Create an LMDBWriter pre-loaded with existing manifest entries."""
        writer = cls(output_dir=output_dir, **kwargs)
        manifest_path = os.path.join(output_dir, "manifest.jsonl")
        if not os.path.exists(manifest_path):
            return writer
        with open(manifest_path, "r") as f:
            for line in f:
                entry = json.loads(line)
                writer._manifest_entries.append(entry)
        return writer


# ---------------------------------------------------------------------------
# LMDBReader
# ---------------------------------------------------------------------------

class LMDBReader:
    """Read individual graph dicts from LMDB.

    Drop-in replacement for ShardReader. Uses mmap for fast access
    without loading the entire database into memory.

    Each process gets its own lmdb.Environment to avoid cross-process
    sharing issues with DataLoader workers.
    """

    def __init__(self, shard_dir: str):
        lmdb_path = os.path.join(shard_dir, "data.lmdb")
        self._abs_path = os.path.abspath(lmdb_path)
        self._env: Optional[lmdb.Environment] = None

    def _get_env(self) -> lmdb.Environment:
        """Lazy initialization of LMDB environment per process."""
        if self._env is None:
            self._env = lmdb.open(
                self._abs_path,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
                subdir=False,
            )
        return self._env

    def get(self, key: str, index: int = 0) -> Dict[str, Any]:
        """Load one graph dict by its LMDB key (material_id).

        Args:
            key: Material ID string (used as LMDB key).
            index: Ignored (kept for interface compatibility with ShardReader).
        """
        with self._get_env().begin(write=False) as txn:
            raw = txn.get(key.encode())
        if raw is None:
            raise KeyError(f"LMDB key not found: {key}")
        return _deserialize_graph(raw)


__all__ = ["LMDBWriter", "LMDBReader"]
