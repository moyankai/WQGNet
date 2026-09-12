"""Batch-level scaling: pack N graphs per bucket into one batch, measure forward."""
import json, os, sys, time
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
    return data

def main():
    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    model = create_model({"model_type": "unified_quotient_equivariant",
                          "hidden_irreps": "128x0e + 4x1o", "block_type": "dynamic_tp",
                          "property_type": "graph_scalar_intensive"}).to(device)
    model.eval()
    print(f"Params: {sum(p.numel() for p in model.parameters())}")

    def manifest(path):
        ids = []
        with open(path) as f:
            for line in f:
                e = json.loads(line)
                if e.get("split") == "val":
                    ids.append((e["material_id"], e["num_atoms"]))
        return ids

    wy_ids = dict(manifest(f"{WYCKOFF_DIR}/manifest.jsonl"))
    p1_ids = dict(manifest(f"{P1_DIR}/manifest.jsonl"))
    common = sorted(set(wy_ids) & set(p1_ids))
    atoms_of = {m: p1_ids[m] for m in common}

    buckets = {"small<=10": (1, 10), "medium 11-30": (11, 30),
               "large 31-60": (31, 60), "very_large>60": (61, 10**9)}
    wy_reader, p1_reader = LMDBReader(WYCKOFF_DIR), LMDBReader(P1_DIR)
    results = {"gpu": torch.cuda.get_device_name(0), "buckets": {}}
    PER_BATCH = 16

    for name, (lo, hi) in buckets.items():
        cands = sorted([m for m in common if lo <= atoms_of[m] <= hi], key=lambda m: atoms_of[m])
        if len(cands) < PER_BATCH:
            cands = (cands * (PER_BATCH // max(1, len(cands)) + 1))[:PER_BATCH]
        mids = cands[:PER_BATCH]
        print(f"\n{name}: {PER_BATCH} graphs (candidates {len(cands)})")
        agg = {}
        for tag, reader in [("wyckoff", wy_reader), ("p1", p1_reader)]:
            data_list = [load_graph(reader, m) for m in mids]
            batch = Batch.from_data_list(data_list).to(device)
            nodes, edges = batch.num_nodes, batch.geo_edge_index.shape[1]
            atoms = int(batch.multiplicity.sum().item())
            times, mems = [], []
            with torch.no_grad():
                for _ in range(5):
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    model(batch)
                    torch.cuda.synchronize()
                    times.append((time.perf_counter() - t0) * 1000)
                    mems.append(torch.cuda.max_memory_allocated() / 1e6)
            times.sort()
            agg[tag] = {"graphs": PER_BATCH, "atoms": atoms, "nodes": nodes,
                        "edges": edges, "median_fwd_ms": times[len(times)//2],
                        "mean_mem_mb": sum(mems)/len(mems)}
            print(f"  {tag:<8} atoms={atoms:>5} nodes={nodes:>6} edges={edges:>7} "
                  f"fwd={agg[tag]['median_fwd_ms']:.2f}ms mem={agg[tag]['mean_mem_mb']:.0f}MB")
        w, p = agg["wyckoff"], agg["p1"]
        agg["speedup"] = p["median_fwd_ms"] / max(w["median_fwd_ms"], 1e-9)
        agg["mem_reduction"] = 1 - w["mean_mem_mb"] / max(p["mean_mem_mb"], 1e-9)
        print(f"  -> speedup={agg['speedup']:.2f}x mem_reduction={100*agg['mem_reduction']:.0f}%")
        results["buckets"][name] = agg

    os.makedirs("results/bench_efficiency", exist_ok=True)
    with open("results/bench_efficiency/scaling_batch.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved: results/bench_efficiency/scaling_batch.json")

if __name__ == "__main__":
    main()
