"""Compute Table 2 (dataset stats) and Table 4 (robustness) values.

Run with ``--mode stats`` (default) to compute the mean edge / sub-edge
ratios and node-compression statistics used in Table~\\ref{tab:dataset_stats}.
Run with ``--mode robustness`` to compute the space-group / representative-
count / failure sensitivity to spglib ``symprec`` used in
Table~\\ref{tab:robustness}.

Stats mode: for 500 random structures sampled from each dataset's manifest,
we build:
  - the quotient graph via WyckoffGraphBuilder (float32, from LMDB shards)
  - the full-atom graph via the pure-numpy periodic enumerator that Part 1
    of the correctness suite uses.

Ratio = num_full_atom_edges / num_quotient_sub_edges, aggregated as the mean
over structures. Also reports mean N_atom, N_rep, gamma_node = N_atom / N_rep.

Robustness mode: sample N JARVIS-bandgap structures, build orbits at
``symprec=1e-3`` as reference, then rebuild at ``{1e-5, 1e-4, 1e-2, 1e-1}``
and record the fraction of structures whose space-group number, representative
count or graph size changes, plus the failure rate. Also compares primitive
versus conventional cell handling via pymatgen's ``get_primitive_structure``.

Output: ``results/table2_dataset_stats.json`` (stats mode) or
``results/table4_robustness.json`` (robustness mode). Both modes also print a
LaTeX row fragment ready to paste into ``main.tex``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from wyckoff_gnn.data.adapters import JARVISAdapter, MatbenchAdapter
from wyckoff_gnn.data.crystal_to_wyckoff import structure_to_wyckoff_orbits
from tests.correctness_suite.shared_utils import (
    build_full_atom_neighbor_list,
    enumerate_quotient_subedges_f64,
)


DATASETS = [
    {
        "name": "Matbench mp_e_form",
        "adapter": "matbench",
        "kwargs": {"task_name": "matbench_mp_e_form", "fold": 0},
        "target_filter": None,
    },
    {
        "name": "Matbench mp_gap",
        "adapter": "matbench",
        "kwargs": {"task_name": "matbench_mp_gap", "fold": 0},
        "target_filter": None,
    },
    {
        "name": "JARVIS formation energy",
        "adapter": "jarvis",
        "kwargs": {
            "json_path": "data/jarvis_raw/jdft_3d-9-24-2025.json",
            "target_key": "formation_energy_peratom",
        },
        "target_filter": "drop_nan",
    },
    {
        "name": "JARVIS band gap (optB88vdW)",
        "adapter": "jarvis",
        "kwargs": {
            "json_path": "data/jarvis_raw/jdft_3d-9-24-2025.json",
            "target_key": "optb88vdw_bandgap",
        },
        "target_filter": "drop_nan",
    },
]


def _record_target_ok(rec) -> bool:
    val = getattr(rec, "target", None)
    if val is None:
        return False
    try:
        v = float(val)
    except Exception:
        return False
    return v == v  # False for NaN


def stats_for_dataset(name: str, adapter, cutoff: float, sample_size: int,
                       seed: int, target_filter: str | None) -> Dict[str, Any]:
    all_records = list(adapter.iter_records())
    if target_filter == "drop_nan":
        records = [r for r in all_records if _record_target_ok(r)]
    else:
        records = all_records

    rng = np.random.default_rng(seed)
    if sample_size >= len(records):
        indices = np.arange(len(records))
    else:
        indices = rng.choice(len(records), size=sample_size, replace=False)
    indices.sort()

    n_atoms_list: List[int] = []
    n_orbits_list: List[int] = []
    n_sub_list: List[int] = []
    n_full_list: List[int] = []
    p1_count = 0
    failed = 0

    for idx in indices:
        try:
            struct = records[int(idx)].structure
            orbits, meta = structure_to_wyckoff_orbits(struct, tol=1e-3)
        except Exception:
            failed += 1
            continue
        sub = enumerate_quotient_subedges_f64(
            orbits, meta["standardized_lattice"], cutoff=cutoff, meta=meta,
        )
        full = build_full_atom_neighbor_list(meta, cutoff=cutoff)
        n_atoms_list.append(int(len(meta["standardized_numbers"])))
        n_orbits_list.append(int(len(orbits)))
        n_sub_list.append(int(sub["vec_cart"].shape[0]))
        n_full_list.append(int(full["vec_cart"].shape[0]))
        if int(meta["sg_number"]) == 1:
            p1_count += 1

    n_atoms = np.asarray(n_atoms_list, dtype=np.float64)
    n_orbits = np.asarray(n_orbits_list, dtype=np.float64)
    n_sub = np.asarray(n_sub_list, dtype=np.float64)
    n_full = np.asarray(n_full_list, dtype=np.float64)

    mean_atoms = float(n_atoms.mean()) if len(n_atoms) else 0.0
    mean_orbits = float(n_orbits.mean()) if len(n_orbits) else 0.0

    return {
        "name": name,
        "cutoff": cutoff,
        "sample_size": int(len(n_atoms_list)),
        "total_available": int(len(records)),
        "total_raw": int(len(all_records)),
        "failed": failed,
        "mean_atoms": mean_atoms,
        "mean_orbits": mean_orbits,
        # paper reports mean(N_atom)/mean(N_rep); we also keep the alternative
        # per-structure ratio for reference.
        "gamma_node": mean_atoms / mean_orbits if mean_orbits > 0 else 0.0,
        "gamma_node_per_struct": float((n_atoms / n_orbits).mean()) if len(n_atoms) else 0.0,
        "mean_subedges": float(n_sub.mean()) if len(n_sub) else 0.0,
        "mean_full_edges": float(n_full.mean()) if len(n_full) else 0.0,
        "edge_ratio": float((n_full / n_sub.clip(min=1)).mean()) if len(n_sub) else 0.0,
        "p1_fraction": float(p1_count / max(len(n_atoms_list), 1)),
    }


def _load_paper_model(checkpoint_path: Path, dtype: torch.dtype = torch.float32):
    """Load the paper's ``wyckoff_gnn`` config and set weights from a
    trained-checkpoint state_dict. Returns the model in eval mode.
    """
    import torch as _torch
    from tests.correctness_suite.part4_invariance import _instantiate_paper_model

    model = _instantiate_paper_model(seed=0)
    state = _torch.load(str(checkpoint_path), map_location="cpu",
                        weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"  [warn] missing={len(missing)} unexpected={len(unexpected)} "
              f"first_missing={missing[:2] if missing else None} "
              f"first_unexpected={unexpected[:2] if unexpected else None}")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _forward_scalar(model, struct, tol: float) -> float:
    """Build the graph at the given symprec and return the scalar output."""
    from tests.correctness_suite.part4_invariance import _build_data_for_model
    data, _, _ = _build_data_for_model(struct, cutoff=5.0, symprec=tol)
    with torch.no_grad():
        f = model(data).detach().flatten()[0]
    return float(f.item())


def run_robustness(cutoff: float, sample_size: int, seed: int,
                    output_path: Path,
                    checkpoint_path: Path | None = None
                    ) -> List[Dict[str, Any]]:
    """Symprec / cell-convention sensitivity sweep for Table 4.

    For each of 300 sampled JARVIS-DFT bandgap structures we compute the
    reference orbits at ``symprec=1e-3`` and then at four alternative
    tolerances plus a primitive-cell comparison. We report the fraction of
    structures whose spglib space-group number or representative-site count
    differs from the reference, and the fraction where standardisation raises
    an exception. When ``checkpoint_path`` is provided, we additionally load
    the trained model and record the graph-scalar output for every setting,
    reporting median and 95th-percentile prediction drift.
    """
    adapter = JARVISAdapter(
        json_path="data/jarvis_raw/jdft_3d-9-24-2025.json",
        target_key="optb88vdw_bandgap",
    )
    records = list(adapter.iter_records())
    records = [r for r in records if _record_target_ok(r)]
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(records), size=min(sample_size, len(records)),
                      replace=False)
    idx.sort()
    sampled = [records[int(i)] for i in idx]

    ref_symprec = 1.0e-3
    reference: List[Dict[str, Any] | None] = []
    for rec in sampled:
        try:
            orbits, meta = structure_to_wyckoff_orbits(rec.structure,
                                                        tol=ref_symprec)
            reference.append({
                "sg": int(meta["sg_number"]),
                "n_rep": len(orbits),
            })
        except Exception:
            reference.append(None)

    # Load model + compute reference predictions once.
    model = None
    f_ref: List[float | None] = [None] * len(sampled)
    if checkpoint_path is not None:
        print(f"  loading checkpoint: {checkpoint_path}")
        model = _load_paper_model(checkpoint_path)
        for i, (rec, ref) in enumerate(zip(sampled, reference)):
            if ref is None:
                continue
            try:
                f_ref[i] = _forward_scalar(model, rec.structure, ref_symprec)
            except Exception:
                f_ref[i] = None

    def _drift_stats(drifts: List[float]) -> Dict[str, float]:
        if not drifts:
            return {"median_drift": 0.0, "p95_drift": 0.0, "n_drift": 0}
        arr = np.asarray(drifts)
        return {
            "median_drift": float(np.median(arr)),
            "p95_drift": float(np.percentile(arr, 95)),
            "n_drift": int(arr.size),
        }

    def _sweep(setting_label: str, mutate) -> Dict[str, Any]:
        """``mutate`` maps ``struct -> (struct_variant, tol_for_variant)``."""
        sg_change = 0
        nrep_change = 0
        failed = 0
        compared = 0
        drifts: List[float] = []
        for i, (rec, ref) in enumerate(zip(sampled, reference)):
            if ref is None:
                continue
            try:
                variant, tol = mutate(rec.structure)
                orbits, meta = structure_to_wyckoff_orbits(variant, tol=tol)
                if int(meta["sg_number"]) != ref["sg"]:
                    sg_change += 1
                if len(orbits) != ref["n_rep"]:
                    nrep_change += 1
                compared += 1
                if model is not None and f_ref[i] is not None:
                    try:
                        f_setting = _forward_scalar(model, variant, tol)
                        drifts.append(abs(f_setting - f_ref[i]))
                    except Exception:
                        pass
            except Exception:
                failed += 1
        row = {
            "setting": setting_label,
            "sg_change_fraction": sg_change / max(compared, 1),
            "nrep_change_fraction": nrep_change / max(compared, 1),
            "failure_rate": failed / len(sampled),
            "n_compared": compared,
        }
        row.update(_drift_stats(drifts))
        return row

    rows: List[Dict[str, Any]] = []
    for tol in (1e-5, 1e-4, 1e-3, 1e-2, 1e-1):
        exp = int(round(np.log10(tol)))
        rows.append(_sweep(f"symprec=1e{exp}",
                            lambda s, t=tol: (s, t)))
    rows.append(_sweep("Primitive versus conventional",
                        lambda s: (s.get_primitive_structure(), ref_symprec)))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({
        "sample_size": len(sampled),
        "reference_symprec": ref_symprec,
        "reference_valid": sum(1 for r in reference if r is not None),
        "checkpoint": str(checkpoint_path) if checkpoint_path else None,
        "rows": rows,
    }, indent=2))
    print(f"\nWrote {output_path}")

    print("\n--- LaTeX rows for Table 4 (Table \\ref{tab:robustness}) ---")

    def _fmt_frac(x: float) -> str:
        if x <= 0:
            return "0\\%"
        return f"{100 * x:.1f}\\%"

    def _fmt_drift(x: float, has_model: bool) -> str:
        if not has_model:
            return "\\TBD{}"
        if x <= 0.0:
            return "$<10^{-6}$~eV"
        if x < 1e-3:
            return f"{x*1e6:.1f}~\\textmu eV"
        if x < 1.0:
            return f"{x*1e3:.1f}~meV"
        return f"{x:.3f}~eV"

    has_model = model is not None
    for row in rows:
        setting = row["setting"]
        med = _fmt_drift(row.get("median_drift", 0.0), has_model)
        p95 = _fmt_drift(row.get("p95_drift", 0.0), has_model)
        if setting == "symprec=1e-3":
            print(f"$\\mathrm{{symprec}}=10^{{-3}}$ & reference & reference & "
                  f"reference & reference & {_fmt_frac(row['failure_rate'])} \\\\")
            continue
        if setting.startswith("symprec="):
            exp = setting.replace("symprec=1e", "")
            print(f"$\\mathrm{{symprec}}=10^{{{exp}}}$ & "
                  f"{_fmt_frac(row['sg_change_fraction'])} & "
                  f"{_fmt_frac(row['nrep_change_fraction'])} & "
                  f"{med} & {p95} & {_fmt_frac(row['failure_rate'])} \\\\")
        else:
            print(f"{setting} & "
                  f"{_fmt_frac(row['sg_change_fraction'])} & "
                  f"{_fmt_frac(row['nrep_change_fraction'])} & "
                  f"{med} & {p95} & {_fmt_frac(row['failure_rate'])} \\\\")

    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("stats", "robustness"), default="stats")
    parser.add_argument("--cutoff", type=float, default=5.0)
    parser.add_argument("--sample-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Optional trained checkpoint for prediction-drift "
                             "columns in robustness mode.")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if args.mode == "robustness":
        out = args.output or (REPO / "results" / "table4_robustness.json")
        run_robustness(args.cutoff, args.sample_size, args.seed, out,
                        checkpoint_path=args.checkpoint)
        return 0

    # ---- stats mode (existing behaviour) ----
    out = args.output or (REPO / "results" / "table2_dataset_stats.json")
    all_stats: List[Dict[str, Any]] = []
    for spec in DATASETS:
        print(f"\n=== {spec['name']} ===")
        if spec["adapter"] == "jarvis":
            adapter = JARVISAdapter(**spec["kwargs"])
        else:
            adapter = MatbenchAdapter(**spec["kwargs"])

        stats = stats_for_dataset(
            spec["name"], adapter, args.cutoff, args.sample_size, args.seed,
            spec.get("target_filter"),
        )
        all_stats.append(stats)
        print(f"  n={stats['sample_size']}/{stats['total_available']} "
              f"(raw {stats['total_raw']}), "
              f"atoms={stats['mean_atoms']:.1f}, orbits={stats['mean_orbits']:.2f}, "
              f"γ_node={stats['gamma_node']:.2f}, edge_ratio={stats['edge_ratio']:.2f}, "
              f"P1={100*stats['p1_fraction']:.1f}%")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(all_stats, indent=2))
    print(f"\nWrote {out}")

    print("\n--- LaTeX rows for Table 2 ---")
    for s in all_stats:
        p1 = 100 * s["p1_fraction"]
        print(f"{s['name']} & {s['total_available']:,} & {s['mean_atoms']:.1f} "
              f"& {s['mean_orbits']:.2f} & {s['gamma_node']:.2f} & "
              f"{s['edge_ratio']:.2f} & {p1:.1f}\\% \\\\")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
