#!/usr/bin/env python3
"""Screen crystallographic redundancy across JARVIS and Materials Project.

For each structure:
  1. Standardize to primitive cell via spglib
  2. Get symmetry dataset (equivalent_atoms)
  3. N_atom_prim = number of atoms in primitive cell
  4. N_orbit_prim = number of unique equivalent_atoms labels
  5. C_crys = N_atom_prim / N_orbit_prim
  6. node_reduction_proxy = 1 - N_orbit_prim / N_atom_prim

Outputs:
  results/crystallographic_redundancy/jarvis_redundancy.csv
  results/crystallographic_redundancy/mp_redundancy.csv
  results/crystallographic_redundancy/jarvis_pareto.csv
  results/crystallographic_redundancy/mp_pareto.csv
  results/crystallographic_redundancy/top_candidates.csv
  results/crystallographic_redundancy/failed_structures.csv
  results/crystallographic_redundancy/summary.md
  results/crystallographic_redundancy/figures/
"""
from __future__ import annotations
import argparse, csv, json, os, sys, time
from multiprocessing import Pool
import numpy as np

sys.path.insert(0, "/data-storage/home/huwei/moyk/wyckoff_gnn")

SYMPREC = 0.1
CRYSTAL_SYSTEMS = [
    ("triclinic", 1, 2), ("monoclinic", 3, 15), ("orthorhombic", 16, 74),
    ("tetragonal", 75, 142), ("trigonal", 143, 167), ("hexagonal", 168, 194),
    ("cubic", 195, 230),
]

def crystal_system(sg):
    for name, lo, hi in CRYSTAL_SYSTEMS:
        if lo <= sg <= hi:
            return name
    return "unknown"

def analyze_one(args):
    """Analyze a single structure. Returns dict of fields or None on failure."""
    mid, atoms_or_structure, source = args

    try:
        if source == "jarvis":
            from pymatgen.core import Structure as PmgStruct, Lattice
            lat, coords, elements, cart = atoms_or_structure
            structure = PmgStruct(Lattice(lat), elements, coords, coords_are_cartesian=cart)
        else:
            structure = atoms_or_structure  # already a pymatgen Structure

        import spglib
        cell = (
            np.array(structure.lattice.matrix, dtype=np.float64),
            np.array(structure.frac_coords, dtype=np.float64),
            np.array(structure.atomic_numbers, dtype=np.int32),
        )

        # Primitive cell
        prim = spglib.find_primitive(cell, symprec=SYMPREC)
        if prim is None:
            prim = cell  # fallback: use input cell

        prim_lat, prim_pos, prim_nums = prim
        n_atom_prim = len(prim_nums)

        # Symmetry dataset on primitive cell
        ds = spglib.get_symmetry_dataset(prim, symprec=SYMPREC)
        if ds is None:
            return None

        sg = ds["number"]
        equiv = ds["equivalent_atoms"]
        # N_orbit = unique equivalent_atoms labels, but MUST respect species
        # (spglib already respects species, but double-check)
        species_orbit = set()
        for i in range(n_atom_prim):
            species_orbit.add((int(equiv[i]), int(prim_nums[i])))
        n_orbit_prim = len(species_orbit)

        if n_orbit_prim < 1 or n_atom_prim < n_orbit_prim:
            return {"id": mid, "error": f"invalid: n_atom={n_atom_prim} n_orbit={n_orbit_prim}", "source": source}

        c_crys = n_atom_prim / n_orbit_prim
        node_red = 1.0 - n_orbit_prim / n_atom_prim

        # volume and density
        vol_prim = float(np.linalg.det(prim_lat))
        density = 0.0
        try:
            density = float(structure.density)
        except Exception:
            pass

        return {
            "id": mid,
            "formula": structure.composition.reduced_formula,
            "space_group_number": sg,
            "space_group_symbol": ds["international"],
            "crystal_system": crystal_system(sg),
            "N_atom_input": len(structure),
            "N_atom_prim": n_atom_prim,
            "N_orbit_prim": n_orbit_prim,
            "C_crys": round(c_crys, 4),
            "node_reduction_proxy": round(node_red, 4),
            "volume_prim": round(vol_prim, 4),
            "density": round(density, 4),
            "primitive_conversion_success": prim is not cell,
            "symmetry_success": True,
            "source": source,
        }
    except Exception as e:
        return {"id": mid, "error": str(e)[:200], "source": source}


def load_jarvis():
    raw = json.load(open("data/jarvis_raw/jdft_3d-9-24-2025.json"))
    items = []
    for r in raw:
        a = r["atoms"]
        lat = list(a.get("lattice_mat", []))
        coords = list(a.get("coords", []))
        elements = list(a.get("elements", []))
        cart = bool(a.get("cartesian", False))
        items.append((r["jid"], (lat, coords, elements, cart), "jarvis"))
    return items

def load_mp():
    from matbench.bench import MatbenchBenchmark
    mb = MatbenchBenchmark(autoload=False, subset=['matbench_mp_e_form'])
    t = getattr(mb, 'matbench_mp_e_form')
    t.load()
    items = []
    for idx, row in t.df.iterrows():
        items.append((str(idx), row['structure'], "mp"))
    return items

def pareto_front(entries):
    """2D Pareto front: maximize N_atom_prim and C_crys."""
    pts = sorted(entries, key=lambda e: (-e["N_atom_prim"], -e["C_crys"]))
    front = []
    max_c = -1
    for e in pts:
        if e["C_crys"] > max_c:
            front.append(e)
            max_c = e["C_crys"]
    return front

def write_csv(path, rows, fields):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["jarvis", "mp", "both"], default="both")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    out_dir = "results/crystallographic_redundancy"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(f"{out_dir}/figures", exist_ok=True)

    fields = ["id", "formula", "space_group_number", "space_group_symbol",
              "crystal_system", "N_atom_input", "N_atom_prim", "N_orbit_prim",
              "C_crys", "node_reduction_proxy", "volume_prim", "density",
              "primitive_conversion_success", "symmetry_success", "source"]

    all_failed = []
    all_results = {}

    for src in (["jarvis", "mp"] if args.source == "both" else [args.source]):
        print(f"\n=== Loading {src} ===", flush=True)
        if src == "jarvis":
            items = load_jarvis()
        else:
            items = load_mp()
        print(f"  {len(items)} structures", flush=True)

        t0 = time.time()
        with Pool(args.workers) as pool:
            results = pool.map(analyze_one, items, chunksize=64)
        elapsed = time.time() - t0
        print(f"  analyzed in {elapsed:.0f}s", flush=True)

        ok = [r for r in results if r and "error" not in r]
        fail = [r for r in results if r and "error" in r]
        print(f"  ok: {len(ok)}, failed: {len(fail)}", flush=True)

        write_csv(f"{out_dir}/{src}_redundancy.csv", ok, fields)
        write_csv(f"{out_dir}/failed_structures.csv", fail, ["id", "error", "source"])
        all_failed.extend(fail)
        all_results[src] = ok

    # Pareto fronts
    for src, results in all_results.items():
        pf = pareto_front(results)
        write_csv(f"{out_dir}/{src}_pareto.csv", pf, fields)
        print(f"\n{src} Pareto front: {len(pf)} structures", flush=True)

    # Top candidates (Tier A/B/C)
    all_ok = []
    for results in all_results.values():
        all_ok.extend(results)

    tier_a = [r for r in all_ok if r["N_atom_prim"] >= 80 and r["C_crys"] >= 6]
    tier_b = [r for r in all_ok if r["N_atom_prim"] >= 60 and r["C_crys"] >= 4 and r not in tier_a]
    tier_c = [r for r in all_ok if r["N_atom_prim"] >= 40 and r["C_crys"] >= 4 and r not in tier_a and r not in tier_b]

    tier_a.sort(key=lambda r: (-r["N_atom_prim"], -r["C_crys"]))
    tier_b.sort(key=lambda r: (-r["N_atom_prim"], -r["C_crys"]))
    tier_c.sort(key=lambda r: (-r["N_atom_prim"], -r["C_crys"]))

    top_all = tier_a + tier_b + tier_c
    write_csv(f"{out_dir}/top_candidates.csv", top_all, fields)

    # Figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for src, results in all_results.items():
        nats = [r["N_atom_prim"] for r in results]
        ccrys = [r["C_crys"] for r in results]
        nred = [r["node_reduction_proxy"] for r in results]

        # A: scatter N_atom_prim vs C_crys
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(nats, ccrys, s=1, alpha=0.3)
        ax.set_xlabel("N_atom_prim")
        ax.set_ylabel("C_crys")
        ax.set_title(f"{src}: Primitive cell atoms vs Crystallographic redundancy")
        ax.axhline(y=4, color="r", linestyle="--", alpha=0.5)
        ax.axvline(x=40, color="r", linestyle="--", alpha=0.5)
        fig.savefig(f"{out_dir}/figures/{src}_scatter_natoms_vs_ccrys.png", dpi=150)
        plt.close(fig)

        # B: scatter N_atom_prim vs node_reduction_proxy
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(nats, nred, s=1, alpha=0.3)
        ax.set_xlabel("N_atom_prim")
        ax.set_ylabel("node_reduction_proxy")
        ax.set_title(f"{src}: Primitive cell atoms vs Node reduction proxy")
        fig.savefig(f"{out_dir}/figures/{src}_scatter_natoms_vs_nodered.png", dpi=150)
        plt.close(fig)

        # C: histogram C_crys
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(ccrys, bins=100, range=(1, 20), alpha=0.7)
        ax.set_xlabel("C_crys")
        ax.set_ylabel("count")
        ax.set_title(f"{src}: C_crys distribution")
        fig.savefig(f"{out_dir}/figures/{src}_hist_ccrys.png", dpi=150)
        plt.close(fig)

        # D: crystal system vs C_crys
        fig, ax = plt.subplots(figsize=(10, 6))
        systems = sorted(set(r["crystal_system"] for r in results))
        data = [[r["C_crys"] for r in results if r["crystal_system"] == s] for s in systems]
        ax.boxplot(data, labels=systems)
        ax.set_ylabel("C_crys")
        ax.set_title(f"{src}: Crystal system vs C_crys")
        fig.savefig(f"{out_dir}/figures/{src}_crystal_system_vs_ccrys.png", dpi=150)
        plt.close(fig)

    # Summary
    def stats(results):
        nats = np.array([r["N_atom_prim"] for r in results])
        ccrys = np.array([r["C_crys"] for r in results])
        s = {
            "n_total": len(results),
            "median_nat": float(np.median(nats)),
            "mean_nat": float(np.mean(nats)),
            "median_ccrys": float(np.median(ccrys)),
            "mean_ccrys": float(np.mean(ccrys)),
            "nat_ge20": int((nats >= 20).sum()),
            "nat_ge40": int((nats >= 40).sum()),
            "nat_ge60": int((nats >= 60).sum()),
            "nat_ge80": int((nats >= 80).sum()),
            "nat_ge100": int((nats >= 100).sum()),
            "ccrys_ge2": int((ccrys >= 2).sum()),
            "ccrys_ge4": int((ccrys >= 4).sum()),
            "ccrys_ge6": int((ccrys >= 6).sum()),
            "ccrys_ge8": int((ccrys >= 8).sum()),
            "ccrys_ge10": int((ccrys >= 10).sum()),
            "nat40_ccrys4": int(((nats >= 40) & (ccrys >= 4)).sum()),
            "nat60_ccrys4": int(((nats >= 60) & (ccrys >= 4)).sum()),
            "nat80_ccrys4": int(((nats >= 80) & (ccrys >= 4)).sum()),
            "nat80_ccrys6": int(((nats >= 80) & (ccrys >= 6)).sum()),
            "nat100_ccrys6": int(((nats >= 100) & (ccrys >= 6)).sum()),
        }
        return s

    with open(f"{out_dir}/summary.md", "w") as f:
        f.write("# Crystallographic Redundancy Screening\n\n")
        for src, results in all_results.items():
            s = stats(results)
            f.write(f"## {src.upper()}\n\n")
            f.write(f"- Total structures analyzed: {s['n_total']}\n")
            f.write(f"- Median N_atom_prim: {s['median_nat']:.1f}\n")
            f.write(f"- Mean N_atom_prim: {s['mean_nat']:.1f}\n")
            f.write(f"- Median C_crys: {s['median_ccrys']:.2f}\n")
            f.write(f"- Mean C_crys: {s['mean_ccrys']:.2f}\n\n")
            f.write(f"| Threshold | Count |\n|-----------|-------|\n")
            for k in ["nat_ge20","nat_ge40","nat_ge60","nat_ge80","nat_ge100"]:
                f.write(f"| {k} | {s[k]} |\n")
            f.write(f"| ccrys_ge2 | {s['ccrys_ge2']} |\n")
            f.write(f"| ccrys_ge4 | {s['ccrys_ge4']} |\n")
            f.write(f"| ccrys_ge6 | {s['ccrys_ge6']} |\n")
            f.write(f"| ccrys_ge8 | {s['ccrys_ge8']} |\n")
            f.write(f"| ccrys_ge10 | {s['ccrys_ge10']} |\n\n")
            f.write(f"| Cross condition | Count |\n|-----------------|-------|\n")
            for k in ["nat40_ccrys4","nat60_ccrys4","nat80_ccrys4","nat80_ccrys6","nat100_ccrys6"]:
                f.write(f"| {k} | {s[k]} |\n")
            f.write("\n")

            # Top 15
            top15 = sorted(results, key=lambda r: (-r["N_atom_prim"], -r["C_crys"]))[:15]
            f.write(f"### {src.upper()} Top 15 candidates\n\n")
            f.write("| ID | Formula | SG | N_atom_prim | N_orbit_prim | C_crys | node_reduction |\n")
            f.write("|-----|---------|-----|-------------|--------------|--------|----------------|\n")
            for r in top15:
                f.write(f"| {r['id']} | {r['formula']} | {r['space_group_number']} | {r['N_atom_prim']} | {r['N_orbit_prim']} | {r['C_crys']} | {r['node_reduction_proxy']} |\n")
            f.write("\n")

        f.write(f"## Pareto front\n\n")
        for src, results in all_results.items():
            pf = pareto_front(results)
            f.write(f"- {src}: {len(pf)} structures on Pareto front\n")

        f.write(f"\n## Tier candidates\n\n")
        f.write(f"- Tier A (N_atom_prim>=80, C_crys>=6): {len(tier_a)}\n")
        f.write(f"- Tier B (N_atom_prim>=60, C_crys>=4): {len(tier_b)}\n")
        f.write(f"- Tier C (N_atom_prim>=40, C_crys>=4): {len(tier_c)}\n")

    print(f"\nDone. Output in {out_dir}/", flush=True)

if __name__ == "__main__":
    main()
