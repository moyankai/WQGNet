#!/usr/bin/env python3
"""Full-data graph-diff audit: OLD (stored LMDB) vs FIXED (fresh rebuild).

Parallelized with multiprocessing.  For each structure in the JARVIS raw JSON,
rebuild the quotient and P1 graphs with the fixed builder and compare the
physical edge multiset against the stored LMDB.

Outputs:
    results/graph_diff_audit/wq_changed_ids.txt
    results/graph_diff_audit/p1_changed_ids.txt
    results/graph_diff_audit/summary.json
    docs/periodic_builder_fix_audit.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from multiprocessing import Pool
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

sys.path.insert(0, "/public/home/moyk/wyckoff_gnn")

from wyckoff_gnn.data.lmdb_cache import LMDBReader
from wyckoff_gnn.data.records import StructureRecord
from wyckoff_gnn.data.graph_builders.wyckoff_builder import WyckoffGraphBuilderWrapper
from wyckoff_gnn.data.graph_builders.p1_builder import P1GraphBuilderWrapper
from wyckoff_gnn.data.adapters.jarvis_adapter import _atoms_to_pymatgen_structure


def edge_set(g: Dict[str, Any]) -> Set[Tuple]:
    """Physical edge set: (target, rounded vector, dist)."""
    lat = np.array(g["lattice"], dtype=np.float64)
    rep = np.array(g["orbit_rep_frac"], dtype=np.float64)
    sfrac = np.array(g["geo_edge_source_frac"], dtype=np.float64)
    sh = np.array(g["geo_edge_shift"], dtype=np.float64)
    ei = np.array(g["geo_edge_index"])
    edges = set()
    for e in range(ei.shape[1]):
        t = int(ei[0, e])
        v = (sfrac[e] + sh[e] - rep[t]) @ lat
        d = np.linalg.norm(v)
        edges.add((t, tuple(np.round(v, 4)), round(float(d), 4)))
    return edges


# Global objects (initialized once per worker)
_raw_by_id: Optional[Dict[str, Any]] = None
_qb: Optional[WyckoffGraphBuilderWrapper] = None
_pb: Optional[P1GraphBuilderWrapper] = None
_wq_reader: Optional[LMDBReader] = None
_p1_reader: Optional[LMDBReader] = None


def _init_worker():
    global _raw_by_id, _qb, _pb, _wq_reader, _p1_reader
    raw = json.load(open("data/jarvis_raw/jdft_3d-9-24-2025.json"))
    _raw_by_id = {r["jid"]: r for r in raw}
    _qb = WyckoffGraphBuilderWrapper(cutoff=5.0, symprec=0.1, angle_pair_top_k=0)
    _pb = P1GraphBuilderWrapper(cutoff=5.0, symprec=0.1, angle_pair_top_k=0)
    _wq_reader = LMDBReader("data/processed/jarvis_lmdb_bandgap")
    _p1_reader = LMDBReader("data/processed/jarvis_lmdb_bandgap_p1_std")


def _check_one(mid: str) -> Dict[str, Any]:
    """Check one structure: return diff stats."""
    if mid not in _raw_by_id:
        return {"mid": mid, "skip": True}
    r = _raw_by_id[mid]
    try:
        rec = StructureRecord(
            material_id=mid,
            structure=_atoms_to_pymatgen_structure(r["atoms"], mid),
            target=None,
        )
    except Exception:
        return {"mid": mid, "skip": True}

    result = {"mid": mid, "skip": False}

    # WQGNet (quotient)
    try:
        g_old = _wq_reader.get(mid)
        g_new = _qb.build(rec)
        es_old = edge_set(g_old)
        es_new = edge_set(g_new)
        result["wq_total"] = len(es_old)
        result["wq_changed"] = es_old != es_new
        result["wq_diff"] = len(es_old.symmetric_difference(es_new))
    except Exception:
        result["wq_total"] = 0
        result["wq_changed"] = False
        result["wq_diff"] = 0

    # P1
    try:
        g_old = _p1_reader.get(mid)
        g_new = _pb.build(rec)
        es_old = edge_set(g_old)
        es_new = edge_set(g_new)
        result["p1_total"] = len(es_old)
        result["p1_changed"] = es_old != es_new
        result["p1_diff"] = len(es_old.symmetric_difference(es_new))
    except Exception:
        result["p1_total"] = 0
        result["p1_changed"] = False
        result["p1_diff"] = 0

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-workers", type=int, default=28)
    parser.add_argument("--output-dir", type=str, default="results/graph_diff_audit")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # get all material IDs from the manifest
    entries = [
        json.loads(l)
        for l in open("data/processed/jarvis_lmdb_bandgap/manifest.jsonl")
    ]
    mids = [e["material_id"] for e in entries]
    print(f"Total structures: {len(mids)}")

    with Pool(args.num_workers, initializer=_init_worker) as pool:
        results = pool.map(_check_one, mids, chunksize=64)

    # aggregate
    wq_total = sum(r.get("wq_total", 0) for r in results)
    wq_changed = sum(1 for r in results if r.get("wq_changed", False))
    wq_diff = sum(r.get("wq_diff", 0) for r in results)
    p1_total = sum(r.get("p1_total", 0) for r in results)
    p1_changed = sum(1 for r in results if r.get("p1_changed", False))
    p1_diff = sum(r.get("p1_diff", 0) for r in results)
    n_skip = sum(1 for r in results if r.get("skip", False))

    wq_changed_ids = [r["mid"] for r in results if r.get("wq_changed", False)]
    p1_changed_ids = [r["mid"] for r in results if r.get("p1_changed", False)]

    summary = {
        "total_structures": len(mids),
        "skipped": n_skip,
        "wq": {
            "total_edges": wq_total,
            "changed_structures": wq_changed,
            "changed_edges": wq_diff,
            "changed_fraction": wq_changed / max(len(mids) - n_skip, 1),
            "edge_fraction": wq_diff / max(wq_total, 1),
        },
        "p1": {
            "total_edges": p1_total,
            "changed_structures": p1_changed,
            "changed_edges": p1_diff,
            "changed_fraction": p1_changed / max(len(mids) - n_skip, 1),
            "edge_fraction": p1_diff / max(p1_total, 1),
        },
    }

    # save
    with open(os.path.join(args.output_dir, "wq_changed_ids.txt"), "w") as f:
        for mid in wq_changed_ids:
            f.write(mid + "\n")
    with open(os.path.join(args.output_dir, "p1_changed_ids.txt"), "w") as f:
        for mid in p1_changed_ids:
            f.write(mid + "\n")
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # write audit doc
    doc = f"""# Periodic Builder Fix Audit (OLD vs FIXED)

## Summary

| Dataset | Total structures | Changed structures | Changed edges | Edge fraction |
|---------|-----------------|-------------------|---------------|---------------|
| WQGNet (quotient) | {len(mids) - n_skip} | {wq_changed} ({wq_changed / max(len(mids) - n_skip, 1):.4%}) | {wq_diff} / {wq_total} | {wq_diff / max(wq_total, 1):.6%} |
| matched P1 | {len(mids) - n_skip} | {p1_changed} ({p1_changed / max(len(mids) - n_skip, 1):.4%}) | {p1_diff} / {p1_total} | {p1_diff / max(p1_total, 1):.6%} |

Skipped: {n_skip}

## Changed structure IDs

- WQGNet: `results/graph_diff_audit/wq_changed_ids.txt` ({len(wq_changed_ids)} structures)
- P1: `results/graph_diff_audit/p1_changed_ids.txt` ({len(p1_changed_ids)} structures)

## Interpretation

- **CASE A** (WQGNet 0 changed, P1 changed): existing WQGNet headline/multiseed/ablation
  results remain valid; only matched P1 accuracy/efficiency/bootstrap need rerun.
- **CASE B** (WQGNet any non-zero change): report change fraction and decide which
  headline experiments must be rerun with fixed LMDB.
"""
    with open("docs/periodic_builder_fix_audit.md", "w") as f:
        f.write(doc)

    print(f"\n=== Summary ===")
    print(f"WQGNet: {wq_changed}/{len(mids) - n_skip} changed ({wq_changed / max(len(mids) - n_skip, 1):.4%}), {wq_diff}/{wq_total} edges ({wq_diff / max(wq_total, 1):.6%})")
    print(f"P1: {p1_changed}/{len(mids) - n_skip} changed ({p1_changed / max(len(mids) - n_skip, 1):.4%}), {p1_diff}/{p1_total} edges ({p1_diff / max(p1_total, 1):.6%})")
    print(f"Skipped: {n_skip}")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
