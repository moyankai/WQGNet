"""Paired scaling benchmark: same materials in Wyckoff vs P1 graphs.

For each material in the val split, load its graph from both LMDBs,
measure single-graph forward time and peak memory. Bucket by atom count.
"""
import json
import os
import sys
import time

import torch

sys.path.insert(0, "/public/home/moyk/wyckoff_gnn")

from torch_geometric.data import Batch
from wyckoff_gnn.data.lmdb_cache import LMDBReader
from wyckoff_gnn.data.graph_schema import light_dict_to_pyg_data
from wyckoff_gnn.models.factory import create_model

WYCKOFF_DIR = "data/processed/jarvis_lmdb_bandgap"
P1_DIR = "data/processed/jarvis_lmdb_bandgap_p1_std"


def load_graph(reader, mid):
    g = reader.get(mid)
    data = light_dict_to_pyg_data(g)
    if g.get("y") is not None:
        y = g["y"]
        data.y = torch.tensor([float(y[0]) if isinstance(y, (list, tuple)) else float(y)])
    return data


def main():
    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    model = create_model({
        "model_type": "unified_quotient_equivariant",
        "hidden_irreps": "128x0e + 4x1o",
        "block_type": "dynamic_tp",
        "property_type": "graph_scalar_intensive",
    }).to(device)
    model.eval()
    print(f"Params: {sum(p.numel() for p in model.parameters())}")

    # val split ids from both manifests (same official split)
    def manifest_ids(path):
        ids = []
        with open(path) as f:
            for line in f:
                e = json.loads(line)
                if e.get("split") == "val":
                    ids.append(e["material_id"])
        return ids

    wy_ids = set(manifest_ids(f"{WYCKOFF_DIR}/manifest.jsonl"))
    p1_ids = set(manifest_ids(f"{P1_DIR}/manifest.jsonl"))
    common = sorted(wy_ids & p1_ids)
    print(f"common val ids: {len(common)}")

    # atom counts from P1 manifest (== Wyckoff conventional cell atoms)
    atoms_of = {}
    with open(f"{P1_DIR}/manifest.jsonl") as f:
        for line in f:
            e = json.loads(line)
            if e["material_id"] in common:
                atoms_of[e["material_id"]] = e["num_atoms"]

    buckets = {"small<=10": (1, 10), "medium 11-30": (11, 30),
               "large 31-60": (31, 60), "very_large>60": (61, 10**9)}
    picked = {}
    for name, (lo, hi) in buckets.items():
        cands = [m for m in common if lo <= atoms_of[m] <= hi]
        cands.sort(key=lambda m: atoms_of[m])
        n = min(10, len(cands))
        # spread across size range
        step = max(1, len(cands) // n)
        picked[name] = cands[::step][:n]
        print(f"  {name}: {len(cands)} candidates, picked {len(picked[name])}")

    wy_reader = LMDBReader(WYCKOFF_DIR)
    p1_reader = LMDBReader(P1_DIR)

    results = {"gpu": torch.cuda.get_device_name(0), "buckets": {}}

    for name, mids in picked.items():
        bucket = {"wyckoff": {"count": 0, "atoms": [], "nodes": [], "edges": [],
                              "fwd_ms": [], "mem_mb": []},
                  "p1": {"count": 0, "atoms": [], "nodes": [], "edges": [],
                         "fwd_ms": [], "mem_mb": []}}
        for mid in mids:
            for tag, reader in [("wyckoff", wy_reader), ("p1", p1_reader)]:
                data = load_graph(reader, mid)
                batch = Batch.from_data_list([data]).to(device)
                nodes = batch.num_nodes
                edges = batch.geo_edge_index.shape[1]
                atoms = int(batch.multiplicity.sum().item())
                times = []
                mems = []
                with torch.no_grad():
                    for _ in range(3):
                        torch.cuda.reset_peak_memory_stats()
                        torch.cuda.synchronize()
                        t0 = time.perf_counter()
                        model(batch)
                        torch.cuda.synchronize()
                        times.append((time.perf_counter() - t0) * 1000)
                        mems.append(torch.cuda.max_memory_allocated() / 1e6)
                times.sort()
                bucket[tag]["count"] += 1
                bucket[tag]["atoms"].append(atoms)
                bucket[tag]["nodes"].append(nodes)
                bucket[tag]["edges"].append(edges)
                bucket[tag]["fwd_ms"].append(times[len(times) // 2])
                bucket[tag]["mem_mb"].append(sum(mems) / len(mems))
        # aggregate
        agg = {}
        for tag in ("wyckoff", "p1"):
            b = bucket[tag]
            agg[tag] = {
                "count": b["count"],
                "mean_atoms": sum(b["atoms"]) / max(1, len(b["atoms"])),
                "mean_nodes": sum(b["nodes"]) / max(1, len(b["nodes"])),
                "mean_edges": sum(b["edges"]) / max(1, len(b["edges"])),
                "median_fwd_ms": sorted(b["fwd_ms"])[len(b["fwd_ms"]) // 2],
                "mean_fwd_ms": sum(b["fwd_ms"]) / max(1, len(b["fwd_ms"])),
                "mean_mem_mb": sum(b["mem_mb"]) / max(1, len(b["mem_mb"])),
            }
        speedup = agg["p1"]["median_fwd_ms"] / max(agg["wyckoff"]["median_fwd_ms"], 1e-9)
        mem_red = 1 - agg["wyckoff"]["mean_mem_mb"] / max(agg["p1"]["mean_mem_mb"], 1e-9)
        agg["speedup"] = speedup
        agg["mem_reduction"] = mem_red
        results["buckets"][name] = agg
        print(f"\n{name}: speedup={speedup:.2f}x, mem_reduction={100*mem_red:.0f}%")
        for tag in ("wyckoff", "p1"):
            a = agg[tag]
            print(f"  {tag:<8} atoms={a['mean_atoms']:>6.1f} nodes={a['mean_nodes']:>6.1f} "
                  f"edges={a['mean_edges']:>7.0f} fwd={a['median_fwd_ms']:.2f}ms mem={a['mean_mem_mb']:.0f}MB")

    os.makedirs("results/bench_efficiency", exist_ok=True)
    with open("results/bench_efficiency/scaling_paired.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved: results/bench_efficiency/scaling_paired.json")


if __name__ == "__main__":
    main()
