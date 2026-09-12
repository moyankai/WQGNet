#!/usr/bin/env python3
"""Stage 3: real-material computational benchmark.

Answers: for real crystals whose PRIMITIVE cell is already large but which have
few independent crystallographic degrees of freedom, does WQGNet still beat a
conventional full-atom GNN that already uses the primitive cell?

Three representations per material:
  A. P1-primitive : standardized primitive cell, every atom = one P1 node
  B. P1-matched   : same standardized/model cell as WQGNet, every atom = node
  C. WQGNet       : current fixed quotient builder

Two compression metrics kept strictly separate:
  C_prim  = N_atom_prim / N_orbit_prim              (intrinsic, primitive cell)
  C_model = P1matched_nodes / WQ_nodes              (matched-model-cell control)
"""
from __future__ import annotations
import argparse, csv, gc, json, os, subprocess, sys, time
import numpy as np
import torch

sys.path.insert(0, "/data-storage/home/huwei/moyk/wyckoff_gnn")

RES = "results/real_material_benchmark"
CUTOFF = 5.0
SYMPREC_MAIN = 0.1
SYMPREC_SCAN = [0.01, 0.03, 0.05, 0.10]
BATCHES = [1, 8, 32, 64]
WARMUP = 30
ITERS = 100

MATERIALS = [
    {"key": "Ho3Al5O12",     "source": "jarvis", "id": "JVASP-88967",          "sg": 230},
    {"key": "Ca3Al2O6",      "source": "mp",     "id": "mb-mp-e-form-129036",  "sg": 205},
    {"key": "Rb3Sc2(AsO4)3", "source": "mp",     "id": "mb-mp-e-form-103011",  "sg": 205},
    {"key": "Ba8Ge43",       "source": "mp",     "id": "mb-mp-e-form-021196",  "sg": 230},
    {"key": "MOF5",          "source": "qmof",   "id": "qmof-a2d95c3",         "sg": 225},
]

MODEL_CFG = {
    "model_type": "unified_quotient_equivariant",
    "hidden_irreps": "128x0e + 4x1o",
    "block_type": "dynamic_tp",
    "property_type": "graph_scalar_intensive",
    "use_site_irrep_projection": False,
    "num_layers": 5,
    "num_rbf": 32,
    "rbf_max": 8.0,
}


# ---------------------------------------------------------------- structures
def load_structures():
    from pymatgen.core import Structure, Lattice
    out = {}
    need_j = {m["id"] for m in MATERIALS if m["source"] == "jarvis"}
    need_m = {m["id"] for m in MATERIALS if m["source"] == "mp"}
    need_q = {m["id"] for m in MATERIALS if m["source"] == "qmof"}

    if need_j:
        raw = json.load(open("data/jarvis_raw/jdft_3d-9-24-2025.json"))
        for r in raw:
            if r["jid"] in need_j:
                a = r["atoms"]
                out[r["jid"]] = Structure(
                    Lattice(list(a["lattice_mat"])), list(a["elements"]),
                    list(a["coords"]),
                    coords_are_cartesian=bool(a.get("cartesian", False)))
    if need_m:
        from matbench.bench import MatbenchBenchmark
        mb = MatbenchBenchmark(autoload=False, subset=["matbench_mp_e_form"])
        t = getattr(mb, "matbench_mp_e_form")
        t.load()
        for idx, row in t.df.iterrows():
            if str(idx) in need_m:
                out[str(idx)] = row["structure"]
    for qid in need_q:
        out[qid] = Structure.from_file(f"data/mof/qmof_cifs/{qid}.cif")
    return out


def prim_stats(structure, symprec):
    """Primitive-cell atom/orbit counts at a given symprec."""
    import spglib
    cell = (np.array(structure.lattice.matrix, dtype=np.float64),
            np.array(structure.frac_coords, dtype=np.float64),
            np.array(structure.atomic_numbers, dtype=np.int32))
    prim = spglib.find_primitive(cell, symprec=symprec)
    if prim is None:
        prim = cell
    plat, ppos, pnum = prim
    ds = spglib.get_symmetry_dataset(prim, symprec=symprec)
    if ds is None:
        return None
    equiv = ds["equivalent_atoms"]
    orb = {(int(equiv[i]), int(pnum[i])) for i in range(len(pnum))}
    return {
        "N_atom_prim": len(pnum), "N_orbit_prim": len(orb),
        "sg": int(ds["number"]), "sg_symbol": ds["international"],
        "C_prim": round(len(pnum) / max(len(orb), 1), 4),
        "prim_cell": (plat, ppos, pnum),
    }


# ---------------------------------------------------------------- graphs
def build_p1_from_cell(lattice, positions, numbers, mid):
    """P1 graph directly from given atoms; bypasses conventional refinement."""
    from wyckoff_gnn.data.crystal_to_wyckoff import _p1_fallback
    from wyckoff_gnn.data.wyckoff_graph import WyckoffGraphBuilder
    from wyckoff_gnn.data.graph_schema import light_dict_to_pyg_data, pyg_data_to_light_dict
    orbits, meta = _p1_fallback(
        None, np.asarray(lattice, dtype=np.float64),
        np.asarray(positions, dtype=np.float64),
        np.asarray(numbers, dtype=np.int32),
        std_rotation_matrix=np.eye(3, dtype=np.float64),
        fallback_stage="explicit_p1",
        detected_sg_number=1, detected_international_symbol="P1")
    b = WyckoffGraphBuilder(cutoff_radius=CUTOFF, subedge_aggregation="sum",
                            angle_pair_top_k=0)
    data = b.build(orbits, np.asarray(lattice, dtype=np.float64),
                   atom_to_orbit=meta["atom_to_orbit"],
                   atom_image_index=meta["atom_image_index"])
    data.material_id = mid
    return light_dict_to_pyg_data(pyg_data_to_light_dict(data))


def build_three_reps(structure, mid, prim_cell):
    """Return (graphs, build_times). Graphs: p1prim, p1matched, wq."""
    from wyckoff_gnn.data.records import StructureRecord
    from wyckoff_gnn.data.graph_builders.wyckoff_builder import WyckoffGraphBuilderWrapper
    from wyckoff_gnn.data.graph_builders.p1_builder import P1GraphBuilderWrapper
    from wyckoff_gnn.data.graph_schema import light_dict_to_pyg_data

    graphs, times = {}, {}

    # A. P1-primitive
    plat, ppos, pnum = prim_cell
    t0 = time.perf_counter()
    graphs["p1prim"] = build_p1_from_cell(plat, ppos, pnum, f"{mid}_prim")
    times["p1prim"] = time.perf_counter() - t0

    rec = StructureRecord(material_id=mid, structure=structure, target=None)

    # B. P1-matched (standardized/model cell, same as WQ pipeline)
    pb = P1GraphBuilderWrapper(cutoff=CUTOFF, symprec=SYMPREC_MAIN, angle_pair_top_k=0)
    t0 = time.perf_counter()
    graphs["p1matched"] = light_dict_to_pyg_data(pb.build(rec))
    times["p1matched"] = time.perf_counter() - t0

    # C. WQGNet
    qb = WyckoffGraphBuilderWrapper(cutoff=CUTOFF, symprec=SYMPREC_MAIN, angle_pair_top_k=0)
    t0 = time.perf_counter()
    graphs["wq"] = light_dict_to_pyg_data(qb.build(rec))
    times["wq"] = time.perf_counter() - t0

    return graphs, times


# ---------------------------------------------------------------- timing
def make_batch(data, bs, device):
    from torch_geometric.data import Batch
    import copy
    return Batch.from_data_list([copy.deepcopy(data) for _ in range(bs)]).to(device)


def time_forward(model, batch):
    times = []
    with torch.no_grad():
        for _ in range(WARMUP):
            model(batch)
        torch.cuda.synchronize()
        for _ in range(ITERS):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(batch)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
    return np.array(times)


def time_train_step(model, batch, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    times = []
    for _ in range(WARMUP):
        opt.zero_grad(set_to_none=True)
        out = model(batch)
        loss = out.float().pow(2).mean()
        loss.backward()
        opt.step()
    torch.cuda.synchronize()
    for _ in range(ITERS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        out = model(batch)
        loss = out.float().pow(2).mean()
        loss.backward()
        opt.step()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    return np.array(times)


def measure(model, data, bs, device, mode):
    """Return dict with timing stats + peak memory, or oom flag."""
    try:
        gc.collect(); torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        batch = make_batch(data, bs, device)
        if mode == "forward":
            model.eval()
            t = time_forward(model, batch)
        else:
            model.train()
            t = time_train_step(model, batch)
        mem_alloc = torch.cuda.max_memory_allocated() / 1e6
        mem_res = torch.cuda.max_memory_reserved() / 1e6
        del batch
        gc.collect(); torch.cuda.empty_cache()
        return {"oom": False, "median_ms": float(np.median(t)),
                "mean_ms": float(t.mean()), "std_ms": float(t.std()),
                "mem_alloc_mb": mem_alloc, "mem_reserved_mb": mem_res}
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if "out of memory" in str(e).lower():
            gc.collect(); torch.cuda.empty_cache()
            return {"oom": True, "median_ms": None, "mean_ms": None,
                    "std_ms": None, "mem_alloc_mb": None, "mem_reserved_mb": None}
        raise


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="graphs only, no GPU")
    args = ap.parse_args()

    os.makedirs(RES, exist_ok=True)
    os.makedirs(f"{RES}/figures", exist_ok=True)

    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        commit = "unknown"

    print("loading structures...", flush=True)
    structs = load_structures()
    for m in MATERIALS:
        assert m["id"] in structs, f"missing structure {m['id']}"
    print(f"loaded {len(structs)}", flush=True)

    # ---------- 1. symprec robustness ----------
    print("\n=== symprec robustness ===", flush=True)
    sym_rows = []
    for m in MATERIALS:
        s = structs[m["id"]]
        vals = []
        for sp in SYMPREC_SCAN:
            st = prim_stats(s, sp)
            if st is None:
                sym_rows.append({"key": m["key"], "id": m["id"], "symprec": sp,
                                 "N_atom_prim": None, "N_orbit_prim": None,
                                 "sg": None, "C_prim": None, "note": "symmetry failed"})
                continue
            vals.append(st)
            sym_rows.append({"key": m["key"], "id": m["id"], "symprec": sp,
                             "N_atom_prim": st["N_atom_prim"],
                             "N_orbit_prim": st["N_orbit_prim"],
                             "sg": st["sg"], "C_prim": st["C_prim"], "note": ""})
            print(f"  {m['key']:>16} symprec={sp:>5} prim={st['N_atom_prim']:>4}/"
                  f"{st['N_orbit_prim']:>3} SG={st['sg']:>3} C_prim={st['C_prim']}", flush=True)
        orbs = {v["N_orbit_prim"] for v in vals}
        stable = len(orbs) == 1
        for r in sym_rows:
            if r["key"] == m["key"]:
                r["orbit_stable_across_symprec"] = stable
        print(f"  -> {m['key']}: orbit count {'STABLE' if stable else 'VARIES ' + str(sorted(orbs))}", flush=True)

    with open(f"{RES}/symprec_robustness.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["key", "id", "symprec", "N_atom_prim",
                                          "N_orbit_prim", "sg", "C_prim",
                                          "orbit_stable_across_symprec", "note"],
                           extrasaction="ignore")
        w.writeheader()
        for r in sym_rows:
            w.writerow(r)

    # ---------- 2. graphs ----------
    print("\n=== building three representations (symprec=0.1) ===", flush=True)
    graph_rows, all_graphs, prep_rows = [], {}, []
    for m in MATERIALS:
        s = structs[m["id"]]
        st = prim_stats(s, SYMPREC_MAIN)
        graphs, btimes = build_three_reps(s, m["id"], st["prim_cell"])
        all_graphs[m["key"]] = graphs

        g = {k: {"nodes": int(v.num_nodes),
                 "edges": int(v.geo_edge_index.shape[1])} for k, v in graphs.items()}
        row = {
            "key": m["key"], "id": m["id"], "source": m["source"],
            "SG": st["sg"], "SG_symbol": st["sg_symbol"],
            "N_atom_prim": st["N_atom_prim"], "N_orbit_prim": st["N_orbit_prim"],
            "C_prim": st["C_prim"],
            "P1prim_nodes": g["p1prim"]["nodes"], "P1prim_edges": g["p1prim"]["edges"],
            "P1matched_nodes": g["p1matched"]["nodes"], "P1matched_edges": g["p1matched"]["edges"],
            "WQ_nodes": g["wq"]["nodes"], "WQ_subedges": g["wq"]["edges"],
            "C_model": round(g["p1matched"]["nodes"] / max(g["wq"]["nodes"], 1), 4),
            "model_cell_vs_prim": round(g["p1matched"]["nodes"] / max(st["N_atom_prim"], 1), 4),
            "intrinsic_node_ratio": round(g["p1prim"]["nodes"] / max(g["wq"]["nodes"], 1), 4),
            "matched_node_ratio": round(g["p1matched"]["nodes"] / max(g["wq"]["nodes"], 1), 4),
            "intrinsic_edge_ratio": round(g["p1prim"]["edges"] / max(g["wq"]["edges"], 1), 4),
            "matched_edge_ratio": round(g["p1matched"]["edges"] / max(g["wq"]["edges"], 1), 4),
        }
        graph_rows.append(row)
        prep_rows.append({"key": m["key"], "id": m["id"],
                          "p1prim_build_s": round(btimes["p1prim"], 4),
                          "p1matched_build_s": round(btimes["p1matched"], 4),
                          "wq_build_s": round(btimes["wq"], 4),
                          "wq_overhead_vs_p1prim_s": round(btimes["wq"] - btimes["p1prim"], 4),
                          "wq_overhead_vs_p1matched_s": round(btimes["wq"] - btimes["p1matched"], 4)})
        print(f"  {m['key']:>16} prim {st['N_atom_prim']}/{st['N_orbit_prim']} C_prim={st['C_prim']} | "
              f"P1prim {g['p1prim']['nodes']}n/{g['p1prim']['edges']}e | "
              f"P1matched {g['p1matched']['nodes']}n/{g['p1matched']['edges']}e | "
              f"WQ {g['wq']['nodes']}n/{g['wq']['edges']}e", flush=True)

    with open(f"{RES}/graph_statistics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(graph_rows[0].keys()))
        w.writeheader()
        for r in graph_rows:
            w.writerow(r)
    with open(f"{RES}/preprocessing_timing.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(prep_rows[0].keys()))
        w.writeheader()
        for r in prep_rows:
            w.writerow(r)

    if args.dry_run:
        print("\ndry-run: graphs only, stopping before GPU.", flush=True)
        return

    # ---------- 3. GPU benchmark ----------
    device = torch.device("cuda")
    from wyckoff_gnn.models.factory import create_model
    model = create_model(dict(MODEL_CFG)).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    env = {"gpu": torch.cuda.get_device_name(0),
           "gpu_mem_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1),
           "cuda": torch.version.cuda, "torch": torch.__version__,
           "git_commit": commit, "params": n_params,
           "warmup": WARMUP, "iters": ITERS, "cutoff": CUTOFF, "symprec": SYMPREC_MAIN}
    print(f"\n=== GPU benchmark ===\n{json.dumps(env, indent=2)}", flush=True)
    json.dump(env, open(f"{RES}/environment.json", "w"), indent=2)

    raw_rows, sum_rows = [], []
    for m in MATERIALS:
        graphs = all_graphs[m["key"]]
        for bs in BATCHES:
            res = {}
            for rep in ("p1prim", "p1matched", "wq"):
                for mode in ("forward", "train"):
                    r = measure(model, graphs[rep], bs, device, mode)
                    res[(rep, mode)] = r
                    raw_rows.append({"key": m["key"], "id": m["id"], "batch": bs,
                                     "representation": rep, "mode": mode, **r})
                    tag = "OOM" if r["oom"] else f"{r['median_ms']:.2f}ms mem={r['mem_alloc_mb']:.0f}MB"
                    print(f"  {m['key']:>16} bs={bs:>3} {rep:>10} {mode:>7}: {tag}", flush=True)

            def g(rep, mode, k):
                return res[(rep, mode)][k]

            def ratio(a, b):
                return round(a / b, 4) if (a and b) else None

            def red(wq, ref):
                return round(1 - wq / ref, 4) if (wq and ref) else None

            sum_rows.append({
                "key": m["key"], "id": m["id"], "batch": bs,
                "p1prim_fwd_ms": g("p1prim", "forward", "median_ms"),
                "p1matched_fwd_ms": g("p1matched", "forward", "median_ms"),
                "wq_fwd_ms": g("wq", "forward", "median_ms"),
                "speedup_practical_fwd": ratio(g("p1prim", "forward", "median_ms"),
                                               g("wq", "forward", "median_ms")),
                "speedup_matched_fwd": ratio(g("p1matched", "forward", "median_ms"),
                                             g("wq", "forward", "median_ms")),
                "p1prim_train_ms": g("p1prim", "train", "median_ms"),
                "p1matched_train_ms": g("p1matched", "train", "median_ms"),
                "wq_train_ms": g("wq", "train", "median_ms"),
                "speedup_practical_train": ratio(g("p1prim", "train", "median_ms"),
                                                 g("wq", "train", "median_ms")),
                "speedup_matched_train": ratio(g("p1matched", "train", "median_ms"),
                                               g("wq", "train", "median_ms")),
                "p1prim_train_mem_mb": g("p1prim", "train", "mem_alloc_mb"),
                "p1matched_train_mem_mb": g("p1matched", "train", "mem_alloc_mb"),
                "wq_train_mem_mb": g("wq", "train", "mem_alloc_mb"),
                "mem_reduction_practical": red(g("wq", "train", "mem_alloc_mb"),
                                               g("p1prim", "train", "mem_alloc_mb")),
                "mem_reduction_matched": red(g("wq", "train", "mem_alloc_mb"),
                                             g("p1matched", "train", "mem_alloc_mb")),
                "any_oom": any(res[k]["oom"] for k in res),
            })

    with open(f"{RES}/timing_raw.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(raw_rows[0].keys()))
        w.writeheader()
        for r in raw_rows:
            w.writerow(r)
    with open(f"{RES}/timing_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sum_rows[0].keys()))
        w.writeheader()
        for r in sum_rows:
            w.writerow(r)

    # ---------- 4. figures ----------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    b64 = {r["key"]: r for r in sum_rows if r["batch"] == 64}
    gmap = {r["key"]: r for r in graph_rows}
    keys = [m["key"] for m in MATERIALS]
    cprim = [gmap[k]["C_prim"] for k in keys]

    def scatter(xs, ys, xl, yl, title, fn):
        fig, ax = plt.subplots(figsize=(7, 5))
        for x, y, k in zip(xs, ys, keys):
            if x is None or y is None:
                continue
            ax.scatter(x, y, s=70)
            ax.annotate(k, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=9)
        ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_title(title)
        ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(f"{RES}/figures/{fn}", dpi=150); plt.close(fig)

    scatter(cprim, [b64[k]["mem_reduction_practical"] for k in keys],
            "C_prim (N_atom_prim / N_orbit_prim)", "practical memory reduction",
            "A: intrinsic redundancy vs memory reduction (batch=64, train)",
            "figA_cprim_vs_memreduction.png")
    scatter(cprim, [b64[k]["speedup_practical_fwd"] for k in keys],
            "C_prim", "practical forward speedup (x)",
            "B: intrinsic redundancy vs forward speedup (batch=64)",
            "figB_cprim_vs_speedup.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(keys)); w = 0.35
    ax.bar(x - w/2, [gmap[k]["P1prim_nodes"] for k in keys], w, label="P1-primitive nodes")
    ax.bar(x + w/2, [gmap[k]["WQ_nodes"] for k in keys], w, label="WQGNet nodes")
    ax.set_xticks(x); ax.set_xticklabels(keys, rotation=20, ha="right")
    ax.set_ylabel("graph nodes"); ax.set_yscale("log")
    ax.set_title("C: P1-primitive vs WQGNet node count"); ax.legend(); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(f"{RES}/figures/figC_nodes.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w/2, [b64[k]["p1prim_train_mem_mb"] or 0 for k in keys], w, label="P1-primitive")
    ax.bar(x + w/2, [b64[k]["wq_train_mem_mb"] or 0 for k in keys], w, label="WQGNet")
    ax.set_xticks(x); ax.set_xticklabels(keys, rotation=20, ha="right")
    ax.set_ylabel("peak allocated memory (MB), train step")
    ax.set_title("D: peak memory at batch=64"); ax.legend(); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(f"{RES}/figures/figD_memory.png", dpi=150); plt.close(fig)

    # ---------- 5. summary ----------
    with open(f"{RES}/summary.md", "w") as f:
        f.write("# Real-material computational benchmark\n\n")
        f.write("Question: for real crystals whose **primitive** cell is already large but which\n")
        f.write("have few independent crystallographic degrees of freedom, does WQGNet still show\n")
        f.write("a significant computational advantage over a full-atom GNN that *already uses the\n")
        f.write("primitive cell*?\n\n")
        f.write("Metrics kept strictly separate:\n\n")
        f.write("- `C_prim = N_atom_prim / N_orbit_prim` — intrinsic crystallographic redundancy.\n")
        f.write("- `C_model = P1matched_nodes / WQ_nodes` — matched-model-cell controlled ratio.\n")
        f.write("  For materials with `model_cell_vs_prim = 2`, the factor-2 part is conventional-cell\n")
        f.write("  centering, **not** intrinsic redundancy, and must not be quoted as such.\n\n")
        f.write(f"Environment: {env['gpu']} ({env['gpu_mem_gb']} GB), torch {env['torch']}, "
                f"CUDA {env['cuda']}, commit `{env['git_commit'][:10]}`, "
                f"{env['params']:,} params, warmup {WARMUP}, {ITERS} timed iters.\n\n")

        f.write("## Graph statistics\n\n")
        f.write("| material | ID | SG | prim atoms/orbits | C_prim | P1prim n/e | P1matched n/e | WQ n/e | C_model | cell infl |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|\n")
        for r in graph_rows:
            f.write(f"| {r['key']} | {r['id']} | {r['SG']} | {r['N_atom_prim']}/{r['N_orbit_prim']} | "
                    f"{r['C_prim']} | {r['P1prim_nodes']}/{r['P1prim_edges']} | "
                    f"{r['P1matched_nodes']}/{r['P1matched_edges']} | {r['WQ_nodes']}/{r['WQ_subedges']} | "
                    f"{r['C_model']} | {r['model_cell_vs_prim']}x |\n")

        f.write("\n## Batch=64 results\n\n")
        f.write("| material | fwd P1prim | fwd P1matched | fwd WQ | speedup practical | speedup matched | "
                "train P1prim | train P1matched | train WQ | speedup practical | mem P1prim | mem P1matched | mem WQ | "
                "mem red practical | mem red matched |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
        for k in keys:
            r = b64[k]
            def fmt(v, u=""):
                return "OOM" if v is None else f"{v:.2f}{u}"
            f.write(f"| {k} | {fmt(r['p1prim_fwd_ms'])} | {fmt(r['p1matched_fwd_ms'])} | {fmt(r['wq_fwd_ms'])} | "
                    f"{fmt(r['speedup_practical_fwd'],'x')} | {fmt(r['speedup_matched_fwd'],'x')} | "
                    f"{fmt(r['p1prim_train_ms'])} | {fmt(r['p1matched_train_ms'])} | {fmt(r['wq_train_ms'])} | "
                    f"{fmt(r['speedup_practical_train'],'x')} | "
                    f"{fmt(r['p1prim_train_mem_mb'])} | {fmt(r['p1matched_train_mem_mb'])} | {fmt(r['wq_train_mem_mb'])} | "
                    f"{fmt(r['mem_reduction_practical'])} | {fmt(r['mem_reduction_matched'])} |\n")

        f.write("\n## Preprocessing cost\n\n")
        f.write("| material | P1prim build (s) | P1matched build (s) | WQ build (s) | WQ overhead vs P1prim (s) |\n")
        f.write("|---|---|---|---|---|\n")
        for r in prep_rows:
            f.write(f"| {r['key']} | {r['p1prim_build_s']} | {r['p1matched_build_s']} | "
                    f"{r['wq_build_s']} | {r['wq_overhead_vs_p1prim_s']} |\n")

        f.write("\n### Rough break-even\n\n")
        f.write("Training steps needed for the extra WQ preprocessing cost to be repaid by the\n")
        f.write("per-step runtime saving, at batch=64 (vs P1-primitive):\n\n")
        f.write("| material | WQ overhead (s) | train-step saving (ms) | break-even steps |\n|---|---|---|---|\n")
        for r in prep_rows:
            b = b64[r["key"]]
            if b["p1prim_train_ms"] and b["wq_train_ms"]:
                save = b["p1prim_train_ms"] - b["wq_train_ms"]
                be = (r["wq_overhead_vs_p1prim_s"] * 1000 / save) if save > 0 else None
                f.write(f"| {r['key']} | {r['wq_overhead_vs_p1prim_s']} | {save:.2f} | "
                        f"{('%.0f' % be) if be and be > 0 else 'n/a (WQ slower)'} |\n")

        f.write("\n## symprec robustness\n\n")
        f.write("| material | symprec 0.01 | 0.03 | 0.05 | 0.10 | orbit count stable |\n|---|---|---|---|---|---|\n")
        for m in MATERIALS:
            rs = {r["symprec"]: r for r in sym_rows if r["key"] == m["key"]}
            cells = []
            for sp in SYMPREC_SCAN:
                r = rs.get(sp)
                cells.append("n/a" if not r or r["N_atom_prim"] is None
                             else f"{r['N_atom_prim']}/{r['N_orbit_prim']} (C={r['C_prim']})")
            stable = rs.get(0.10, {}).get("orbit_stable_across_symprec", "?")
            f.write(f"| {m['key']} | " + " | ".join(cells) + f" | {stable} |\n")

        oom_any = [r for r in sum_rows if r["any_oom"]]
        f.write(f"\n## OOM\n\n{'No OOM at any batch size.' if not oom_any else str(len(oom_any)) + ' (material,batch) combos hit OOM; see timing_raw.csv.'}\n")

    print(f"\nDone. Output in {RES}/", flush=True)


if __name__ == "__main__":
    main()
