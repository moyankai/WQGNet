"""WyckoffGNN Efficiency / Scaling Benchmark.

Compares Wyckoff quotient graph vs P1 full-atom graph on the same GPU:
  1. forward inference timing
  2. training step timing
  3. full epoch timing (3 epochs)
  4. memory breakdown
  5. scaling by atom-count buckets

Usage:
    python scripts/bench_efficiency.py --model-type wyckoff|p1 --data-dir DIR \
        --checkpoint PATH [--no-checkpoint]
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict

import torch

sys.path.insert(0, "/public/home/moyk/wyckoff_gnn")

from wyckoff_gnn.data.datamodule import PropertyDataModule
from wyckoff_gnn.models.unified_equivariant import UnifiedQuotientEquivariantGNN
from wyckoff_gnn.models.factory import create_model


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def time_fn(fn, warmup=20, timed=100, device=None):
    for _ in range(warmup):
        fn()
    sync()
    times = []
    for _ in range(timed):
        sync()
        t0 = time.perf_counter()
        fn()
        sync()
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return {
        "median_ms": times[len(times) // 2],
        "min_ms": times[0],
        "max_ms": times[-1],
        "mean_ms": sum(times) / len(times),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-type", choices=["wyckoff", "p1"], required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--hidden", default="128x0e + 4x1o")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Model type: {args.model_type}, data: {args.data_dir}")

    split_zip = "data/jarvis_leaderboard/dft_3d_optb88vdw_bandgap.json.zip"
    if not os.path.exists(split_zip):
        split_zip = None  # non-JARVIS datasets use their own manifest splits
    dm = PropertyDataModule(
        manifest_path=f"{args.data_dir}/manifest.jsonl",
        shard_dir=args.data_dir,
        leaderboard_split_zip=split_zip,
        batch_size=args.batch_size,
        val_batch_size=args.batch_size,
        num_workers=args.num_workers,
        global_max_mult=192,
    ).setup()
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()

    model = create_model({
        "model_type": "unified_quotient_equivariant",
        "hidden_irreps": args.hidden,
        "block_type": "dynamic_tp",
        "property_type": "graph_scalar_intensive",
    }).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Params: {n_params}")

    if args.checkpoint and os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        sd = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(sd, strict=False)
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        print("No checkpoint (random init)")

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-5)

    # Reference batch stats
    ref_batch = next(iter(train_loader)).to(device)
    ref_nodes = ref_batch.num_nodes
    ref_edges = ref_batch.geo_edge_index.shape[1]
    ref_graphs = ref_batch.num_graphs
    ref_atoms = int(ref_batch.multiplicity.sum().item())
    print(f"Ref batch: {ref_graphs} graphs, {ref_nodes} nodes, {ref_edges} edges, {ref_atoms} atoms")
    print(f"Nodes/graph: {ref_nodes/ref_graphs:.1f}, Edges/graph: {ref_edges/ref_graphs:.1f}")

    results = {"params": n_params, "gpu": torch.cuda.get_device_name(0)}

    # ---------------- Benchmark 1: forward inference ----------------
    print("\n=== Benchmark 1: forward inference (eval mode) ===")
    model.eval()
    with torch.no_grad():
        t = time_fn(lambda: model(ref_batch), warmup=20, timed=100)
    results["forward_ms"] = t
    results["forward_samples_per_s"] = ref_graphs / (t["median_ms"] / 1000)
    results["forward_nodes_per_s"] = ref_nodes / (t["median_ms"] / 1000)
    results["forward_atoms_per_s"] = ref_atoms / (t["median_ms"] / 1000)
    results["forward_edges_per_s"] = ref_edges / (t["median_ms"] / 1000)
    print(f"forward: median {t['median_ms']:.2f} ms/batch, {results['forward_samples_per_s']:.0f} samples/s")
    print(f"  peak alloc: {torch.cuda.max_memory_allocated()/1e6:.0f} MB, "
          f"reserved: {torch.cuda.max_memory_reserved()/1e6:.0f} MB")
    results["forward_peak_alloc_mb"] = torch.cuda.max_memory_allocated() / 1e6
    results["forward_peak_reserved_mb"] = torch.cuda.max_memory_reserved() / 1e6
    torch.cuda.reset_peak_memory_stats()

    # ---------------- Benchmark 2: training step ----------------
    print("\n=== Benchmark 2: training step ===")
    model.train()

    def train_step():
        optimizer.zero_grad()
        out = model(ref_batch)
        loss = torch.nn.functional.l1_loss(out, ref_batch.y.view(-1))
        loss.backward()
        optimizer.step()

    # forward-only part of step
    def fwd_only():
        out = model(ref_batch)
        return out

    t_fwd = time_fn(fwd_only, warmup=20, timed=100)
    torch.cuda.reset_peak_memory_stats()
    t_step = time_fn(train_step, warmup=20, timed=100)
    print(f"train step: median {t_step['median_ms']:.2f} ms (fwd~{t_fwd['median_ms']:.2f})")
    print(f"  samples/s: {ref_graphs/(t_step['median_ms']/1000):.0f}")
    results["train_step_ms"] = t_step
    results["train_samples_per_s"] = ref_graphs / (t_step["median_ms"] / 1000)
    results["train_peak_alloc_mb"] = torch.cuda.max_memory_allocated() / 1e6
    results["train_peak_reserved_mb"] = torch.cuda.max_memory_reserved() / 1e6
    print(f"  peak alloc (train): {results['train_peak_alloc_mb']:.0f} MB, "
          f"reserved: {results['train_peak_reserved_mb']:.0f} MB")
    torch.cuda.reset_peak_memory_stats()

    # ---------------- Benchmark 3: full epochs (3) ----------------
    print("\n=== Benchmark 3: full epochs (3) ===")
    epoch_times = []
    for ep in range(3):
        sync()
        t0 = time.perf_counter()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)
            loss = torch.nn.functional.l1_loss(out, batch.y.view(-1))
            loss.backward()
            optimizer.step()
        sync()
        dt = time.perf_counter() - t0
        epoch_times.append(dt)
        print(f"  epoch {ep+1}: {dt:.1f} s")
    epoch_times.sort()
    results["epoch_times_s"] = epoch_times
    results["epoch_median_s"] = epoch_times[len(epoch_times) // 2]
    results["epoch_samples_per_s"] = dm.train_size / results["epoch_median_s"]
    results["train_size"] = dm.train_size
    print(f"median epoch: {results['epoch_median_s']:.1f} s, {results['epoch_samples_per_s']:.0f} samples/s")

    # val MAE (3-epoch smoke)
    model.eval()
    errs, n = 0.0, 0
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            out = model(batch)
            errs += torch.abs(out - batch.y.view(-1)).sum().item()
            n += batch.num_graphs
    val_mae = errs / n
    results["val_mae_3epoch"] = val_mae
    print(f"val MAE after 3 epochs (smoke): {val_mae:.4f}")

    # ---------------- Benchmark 4: scaling by atom count ----------------
    print("\n=== Scaling by atom count (P1 atoms) ===")
    model.eval()
    buckets = {"small": (1, 10), "medium": (11, 30), "large": (31, 60), "very_large": (61, 10**9)}
    bucket_stats = defaultdict(lambda: {"count": 0, "nodes": [], "edges": [], "times": [], "mem": []})
    # Per-graph approach via single-graph batches
    single_loader = PropertyDataModule(
        manifest_path=f"{args.data_dir}/manifest.jsonl",
        shard_dir=args.data_dir,
        leaderboard_split_zip="data/jarvis_leaderboard/dft_3d_optb88vdw_bandgap.json.zip",
        batch_size=1, val_batch_size=1, num_workers=0, global_max_mult=192,
    ).setup().val_dataloader()
    with torch.no_grad():
        for i, batch in enumerate(single_loader):
            if i >= 60:
                break
            batch = batch.to(device)
            atoms = int(batch.multiplicity.sum().item())
            nodes = batch.num_nodes
            edges = batch.geo_edge_index.shape[1]
            for name, (lo, hi) in buckets.items():
                if lo <= atoms <= hi:
                    torch.cuda.reset_peak_memory_stats()
                    sync()
                    t0 = time.perf_counter()
                    model(batch)
                    sync()
                    dt = (time.perf_counter() - t0) * 1000
                    bucket_stats[name]["count"] += 1
                    bucket_stats[name]["nodes"].append(nodes)
                    bucket_stats[name]["edges"].append(edges)
                    bucket_stats[name]["times"].append(dt)
                    bucket_stats[name]["mem"].append(torch.cuda.max_memory_allocated() / 1e6)
                    break
    scaling = {}
    for name, s in bucket_stats.items():
        if s["count"] == 0:
            continue
        scaling[name] = {
            "count": s["count"],
            "mean_atoms": sum(nodes for nodes in s["nodes"]) / s["count"],
            "mean_nodes": sum(nodes for nodes in s["nodes"]) / s["count"],
            "mean_edges": sum(edges for edges in s["edges"]) / s["count"],
            "median_fwd_ms": sorted(s["times"])[len(s["times"]) // 2],
            "mean_fwd_ms": sum(s["times"]) / len(s["times"]),
            "mean_peak_mem_mb": sum(s["mem"]) / len(s["mem"]),
        }
        print(f"  {name:<12} n={s['count']:>3} atoms={scaling[name]['mean_atoms']:>6.1f} "
              f"nodes={scaling[name]['mean_nodes']:>6.1f} edges={scaling[name]['mean_edges']:>8.0f} "
              f"fwd={scaling[name]['median_fwd_ms']:.2f}ms mem={scaling[name]['mean_peak_mem_mb']:.0f}MB")
    results["scaling"] = scaling

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved: {args.out}")


if __name__ == "__main__":
    main()
