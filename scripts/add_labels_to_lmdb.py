#!/usr/bin/env python
"""Add new property labels to existing LMDB without rebuilding graphs.

Usage::

    python scripts/add_labels_to_lmdb.py \
        --src_lmdb data/processed/jarvis_lmdb_bandgap \
        --dst_lmdb data/processed/jarvis_lmdb_density \
        --target_key density \
        --jarvis_json data/jarvis_raw/jdft_3d-9-24-2025.json

This copies graph structures from src_lmdb and adds new labels from JARVIS JSON.
Much faster than full preprocessing (no symmetry analysis, no graph construction).
"""

import argparse
import json
import lmdb
import os
import shutil
from pathlib import Path
from typing import Dict, Any

import numpy as np
import msgpack


# ---------------------------------------------------------------------------
# Serialization (copied from wyckoff_gnn.data.lmdb_cache)
# ---------------------------------------------------------------------------

def _serialize_value(v: Any) -> Any:
    """Recursively serialize a value for msgpack."""
    if isinstance(v, np.ndarray):
        return {
            b"__np__": True,
            b"dtype": v.dtype.str.encode(),
            b"shape": list(v.shape),
            b"data": v.tobytes(),
        }
    if isinstance(v, dict):
        return {b"__dict__": True, b"data": {k: _serialize_value(x) for k, x in v.items()}}
    if isinstance(v, (list, tuple)):
        return [_serialize_value(x) for x in v]
    if isinstance(v, (int, float, str, bool, bytes)) or v is None:
        return v
    return str(v)


def _serialize_graph(graph_dict: Dict[str, Any]) -> bytes:
    """Serialize a light graph dict for LMDB storage."""
    packed = {}
    for k, v in graph_dict.items():
        packed[k] = _serialize_value(v)
    return msgpack.packb(packed, use_bin_type=True)


def _deserialize_value(v: Any) -> Any:
    """Recursively deserialize a value from msgpack."""
    if isinstance(v, dict):
        is_np = v.get(b"__np__") or v.get("__np__")
        is_dict = v.get(b"__dict__") or v.get("__dict__")
        if is_np:
            dtype = v[b"dtype"] if b"dtype" in v else v["dtype"]
            shape = v[b"shape"] if b"shape" in v else v["shape"]
            data = v[b"data"] if b"data" in v else v["data"]
            return np.frombuffer(data, dtype=np.dtype(dtype)).reshape(shape).copy()
        if is_dict:
            raw = v[b"data"] if b"data" in v else v["data"]
            return {
                (k.decode() if isinstance(k, bytes) else k): _deserialize_value(x)
                for k, x in raw.items()
            }
    if isinstance(v, list):
        return [_deserialize_value(x) for x in v]
    return v


def _deserialize_graph(data: bytes) -> Dict[str, Any]:
    """Deserialize a light graph dict from LMDB."""
    packed = msgpack.unpackb(data, raw=True)
    result = {}
    for k, v in packed.items():
        key = k.decode() if isinstance(k, bytes) else k
        result[key] = _deserialize_value(v)
    return result


def load_jarvis_targets(json_path: str, target_key: str) -> Dict[str, float]:
    """Load target values from JARVIS JSON."""
    print(f"Loading {target_key} from {json_path}...")
    with open(json_path) as f:
        data = json.load(f)
    
    targets = {}
    n_missing = 0
    for entry in data:
        jid = entry["jid"]
        val = entry.get(target_key)
        # Skip None, 'na' strings, and NaN
        if val is None or val == 'na' or (isinstance(val, float) and np.isnan(val)):
            n_missing += 1
            continue
        try:
            targets[jid] = float(val)
        except (ValueError, TypeError):
            n_missing += 1
    
    print(f"  Loaded {len(targets)} valid targets ({n_missing} missing/invalid)")
    return targets


def copy_lmdb_with_new_labels(
    src_lmdb: str,
    dst_lmdb: str,
    targets: Dict[str, float],
) -> Dict[str, Any]:
    """Copy LMDB, replacing target values."""
    src_path = Path(src_lmdb)
    dst_path = Path(dst_lmdb)
    
    if dst_path.exists():
        print(f"Removing existing {dst_path}")
        shutil.rmtree(dst_path)
    dst_path.mkdir(parents=True)
    
    # Open source LMDB (data.lmdb is a single file, not a directory)
    src_env = lmdb.open(
        str(src_path / "data.lmdb"),
        readonly=True,
        lock=False,
        subdir=False,
    )
    
    # Create destination LMDB
    dst_env = lmdb.open(
        str(dst_path / "data.lmdb"),
        map_size=10 * 1024**3,  # 10 GB
        subdir=False,
    )
    
    # Copy manifest
    shutil.copy(src_path / "manifest.jsonl", dst_path / "manifest.jsonl")
    
    n_updated = 0
    n_missing = 0
    n_total = 0
    
    with src_env.begin() as src_txn, dst_env.begin(write=True) as dst_txn:
        cursor = src_txn.cursor()
        for key, value in cursor:
            n_total += 1
            record = _deserialize_graph(value)
            
            # Update target (material_id may be bytes or string)
            mid = record["material_id"]
            if isinstance(mid, bytes):
                mid = mid.decode()
            
            if mid in targets:
                record["y"] = targets[mid]
                n_updated += 1
            else:
                record["y"] = None
                n_missing += 1
            
            # Write to destination
            dst_txn.put(key, _serialize_graph(record))
    
    src_env.close()
    dst_env.close()
    
    # Update manifest with new target stats
    update_manifest_targets(dst_path, targets)
    
    return {
        "n_total": n_total,
        "n_updated": n_updated,
        "n_missing": n_missing,
    }


def update_manifest_targets(
    lmdb_dir: Path,
    targets: Dict[str, float],
) -> None:
    """Update manifest.jsonl with new target values."""
    manifest_path = lmdb_dir / "manifest.jsonl"
    
    # Read existing manifest
    entries = []
    with open(manifest_path) as f:
        for line in f:
            entries.append(json.loads(line))
    
    # Update targets
    for entry in entries:
        mid = entry["material_id"]
        if mid in targets:
            entry["target"] = targets[mid]
        else:
            entry["target"] = None
    
    # Write back
    with open(manifest_path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")
    
    print(f"Updated {len(entries)} manifest entries")


def main():
    parser = argparse.ArgumentParser(
        description="Add new property labels to existing LMDB"
    )
    parser.add_argument(
        "--src_lmdb",
        required=True,
        help="Source LMDB directory (e.g., jarvis_lmdb_bandgap)",
    )
    parser.add_argument(
        "--dst_lmdb",
        required=True,
        help="Destination LMDB directory (new property)",
    )
    parser.add_argument(
        "--target_key",
        required=True,
        help="Target property key in JARVIS JSON (e.g., density, ehull)",
    )
    parser.add_argument(
        "--jarvis_json",
        default="data/jarvis_raw/jdft_3d-9-24-2025.json",
        help="Path to JARVIS JSON file",
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Add Labels to LMDB")
    print("=" * 70)
    print(f"Source:      {args.src_lmdb}")
    print(f"Destination: {args.dst_lmdb}")
    print(f"Target:      {args.target_key}")
    print()
    
    # Load targets
    targets = load_jarvis_targets(args.jarvis_json, args.target_key)
    
    # Copy and update
    stats = copy_lmdb_with_new_labels(args.src_lmdb, args.dst_lmdb, targets)
    
    print()
    print("=" * 70)
    print("Done!")
    print(f"  Total records:   {stats['n_total']}")
    print(f"  Updated:         {stats['n_updated']}")
    print(f"  Missing targets: {stats['n_missing']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
