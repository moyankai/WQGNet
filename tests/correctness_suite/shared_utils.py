"""Shared utilities for the correctness suite.

Key helpers:
- build_p1_view: flatten orbits into one-atom-per-orbit P1 graph
- load_paper_config: exact train.yaml params for the paper model
- dump_json: timestamped JSON audit dump
- benchmark_structures: sample real structures from LMDB
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from pymatgen.core.structure import Structure

from wyckoff_gnn.data.crystal_to_wyckoff import (
    WyckoffOrbit,
    structure_to_wyckoff_orbits,
    orbits_to_node_features,
)
from wyckoff_gnn.data.wyckoff_graph import WyckoffGraphBuilder


# ---------------------------------------------------------------------------
# P1 view: flatten every atom into its own orbit
# ---------------------------------------------------------------------------


def build_p1_view(
    struct: Structure,
    cutoff: float = 5.0,
    standardized_lattice: Optional[np.ndarray] = None,
    standardized_positions: Optional[np.ndarray] = None,
    standardized_numbers: Optional[np.ndarray] = None,
) -> Tuple["torch_geometric.data.Data", list, dict]:
    """Build a P1 graph where every atom is its own orbit (mult=1).

    Directly calls _p1_fallback to bypass spglib — every atom becomes
    its own orbit with trivial stabilizer (|H|=1).

    Args:
        struct: pymatgen Structure (used if standardized_* not provided).
        cutoff: Neighbor cutoff radius.
        standardized_lattice: If provided, use this lattice instead of
            the structure's original lattice. This ensures the P1 view
            uses the same cell as the quotient view.
        standardized_positions: Fractional coordinates in the std cell.
        standardized_numbers: Atomic numbers in the std cell ordering.

    Returns:
        data: WyckoffData with all fields populated.
        orbits: List of WyckoffOrbit (one per atom).
        meta: Metadata dict.
    """
    from wyckoff_gnn.data.crystal_to_wyckoff import _p1_fallback

    if standardized_lattice is not None:
        lat = np.array(standardized_lattice, dtype=np.float64)
        pos = np.array(standardized_positions, dtype=np.float64)
        nums = np.array(standardized_numbers, dtype=np.int32)
    else:
        lat = struct.lattice.matrix.copy()
        pos = struct.frac_coords.copy()
        nums = np.array([s.Z for s in struct.species], dtype=np.int32)

    orbits, meta = _p1_fallback(struct, lat, pos, nums)

    builder = WyckoffGraphBuilder(
        cutoff_radius=cutoff, subedge_aggregation="sum"
    )
    data = builder.build(
        orbits, lat,
        atom_to_orbit=meta.get("atom_to_orbit"),
        atom_image_index=meta.get("atom_image_index"),
    )
    data.batch = torch.zeros(data.num_nodes, dtype=torch.long)
    data.multiplicity = torch.tensor(
        [o.multiplicity for o in orbits], dtype=torch.float32
    )
    feats = orbits_to_node_features(orbits)
    data.orbit_sym_ops_rotations = feats["orbit_sym_ops_rotations"]
    data.orbit_mult_mask = feats["orbit_mult_mask"]
    if "orbit_stabilizer_W_frac" in feats:
        data.orbit_stabilizer_W_frac = feats["orbit_stabilizer_W_frac"]
        data.orbit_stabilizer_w_frac = feats["orbit_stabilizer_w_frac"]
        data.orbit_stabilizer_mask = feats["orbit_stabilizer_mask"]

    return data, orbits, meta


def build_quotient_graph(
    struct: Structure,
    cutoff: float = 5.0,
    tol: float = 1e-3,
) -> Tuple["torch_geometric.data.Data", list, dict]:
    """Build the standard Wyckoff quotient graph (compressed).

    Returns:
        data: WyckoffData with orbit-level compression.
        orbits: List of WyckoffOrbit.
        meta: Metadata dict.
    """
    orbits, meta = structure_to_wyckoff_orbits(struct, tol=tol)
    std_lat = meta.get("standardized_lattice", struct.lattice.matrix)

    builder = WyckoffGraphBuilder(
        cutoff_radius=cutoff, subedge_aggregation="sum"
    )
    data = builder.build(
        orbits, std_lat,
        atom_to_orbit=meta.get("atom_to_orbit"),
        atom_image_index=meta.get("atom_image_index"),
    )
    data.batch = torch.zeros(data.num_nodes, dtype=torch.long)
    data.multiplicity = torch.tensor(
        [o.multiplicity for o in orbits], dtype=torch.float32
    )
    feats = orbits_to_node_features(orbits)
    data.orbit_sym_ops_rotations = feats["orbit_sym_ops_rotations"]
    data.orbit_mult_mask = feats["orbit_mult_mask"]
    if "orbit_stabilizer_W_frac" in feats:
        data.orbit_stabilizer_W_frac = feats["orbit_stabilizer_W_frac"]
        data.orbit_stabilizer_w_frac = feats["orbit_stabilizer_w_frac"]
        data.orbit_stabilizer_mask = feats["orbit_stabilizer_mask"]

    return data, orbits, meta


# ---------------------------------------------------------------------------
# Paper config
# ---------------------------------------------------------------------------


def load_paper_config() -> dict:
    """Return the exact model config from configs/train.yaml.

    Overrides use_site_projection=True (factory default may be False).
    """
    return {
        "model_type": "wyckoff_gnn",
        "init_irreps": "128x0e",
        "hidden_irreps": "512x0e+16x1o",
        "num_layers": 4,
        "num_rbf": 16,
        "cutoff": 5.0,
        "tp_mode": "dynamic_v2",
        "radial_mlp_width": 32,
        "radial_gate_mode": "per_type",
        "use_edge_state": True,
        "edge_state_dim": 128,
        "edge_state_layers": 4,
        "edge_readout": True,
        "edge_pool_mode": "normalized",
        "use_site_projection": True,
        "dropout": 0.2,
    }


# ---------------------------------------------------------------------------
# JSON dump helpers
# ---------------------------------------------------------------------------

_RUN_TIMESTAMP: Optional[str] = None


def get_run_timestamp() -> str:
    """Return a single timestamp for the entire suite run."""
    global _RUN_TIMESTAMP
    if _RUN_TIMESTAMP is None:
        from datetime import datetime
        _RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    return _RUN_TIMESTAMP


def set_run_timestamp(ts: str):
    """Override the run timestamp (for follow-up jobs writing to same dir)."""
    global _RUN_TIMESTAMP
    _RUN_TIMESTAMP = ts


def dump_json(
    part_name: str,
    records: list | dict,
    output_root: str = "results/correctness_suite",
) -> Path:
    """Write JSON audit dump to results/correctness_suite/<ts>/<part>.json."""
    ts = get_run_timestamp()
    out_dir = Path(output_root) / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{part_name}.json"
    with open(path, "w") as f:
        json.dump(records, f, indent=2, default=_json_default)
    return path


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy().tolist()
    raise TypeError(f"Cannot JSON-encode {type(obj)}")


# ---------------------------------------------------------------------------
# Benchmark structure sampling
# ---------------------------------------------------------------------------


def sample_benchmark_structures(
    limit: int = 20,
    shard_dir: str = "data/processed/jarvis_lmdb_formation_energy_peratom",
    manifest_path: str = "data/processed/jarvis_lmdb_formation_energy_peratom/manifest.jsonl",
    seed: int = 42,
) -> List[Structure]:
    """Sample structures from LMDB for benchmark tests.

    Loads from the preprocessed LMDB cache, reconstructing pymatgen Structures
    from the stored graph dicts.
    """
    from wyckoff_gnn.data.lmdb_cache import LMDBReader

    manifest_entries = []
    with open(manifest_path) as f:
        for line in f:
            entry = json.loads(line.strip())
            if entry.get("split") == "train":
                manifest_entries.append(entry)

    rng = random.Random(seed)
    sampled = rng.sample(manifest_entries, min(limit, len(manifest_entries)))

    reader = LMDBReader(shard_dir)
    structures = []
    for entry in sampled:
        graph_dict = reader.get(entry["lmdb_key"])
        struct = _graph_dict_to_structure(graph_dict)
        if struct is not None:
            structures.append(struct)
    return structures


def iter_benchmark_entries(
    shard_dir: str = "data/processed/jarvis_lmdb_formation_energy_peratom",
    manifest_path: str = "data/processed/jarvis_lmdb_formation_energy_peratom/manifest.jsonl",
    split: str = "train",
    limit: int = -1,
):
    """Yield graph dicts from LMDB one at a time (memory-efficient).

    Args:
        shard_dir: Directory containing data.lmdb.
        manifest_path: Path to manifest.jsonl.
        split: Which split to iterate ("train", "val", "test").
        limit: Max entries to yield (-1 = all).
    """
    from wyckoff_gnn.data.lmdb_cache import LMDBReader

    reader = LMDBReader(shard_dir)
    count = 0
    with open(manifest_path) as f:
        for line in f:
            entry = json.loads(line.strip())
            if entry.get("split") != split:
                continue
            try:
                gd = reader.get(entry["lmdb_key"])
                yield gd
            except Exception:
                continue
            count += 1
            if 0 < limit <= count:
                return


def _graph_dict_to_structure(gd: dict) -> Optional[Structure]:
    """Reconstruct a pymatgen Structure from a serialized graph dict."""
    try:
        from pymatgen.core.lattice import Lattice

        lat = np.array(gd["lattice"])
        numbers = np.array(gd.get("atomic_numbers", gd.get("orbit_element", [])))
        frac = np.array(gd.get("orbit_rep_frac", gd.get("rep_coords_frac", [])))

        if len(numbers) == 0 or len(frac) == 0:
            return None

        lattice = Lattice(lat)
        return Structure(lattice, numbers.tolist(), frac.tolist())
    except Exception:
        return None
