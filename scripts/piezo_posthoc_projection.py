#!/usr/bin/env python3
"""P4: post-hoc crystal point-group projection of trained piezo predictions.

No retraining. Loads existing predictions, builds each test structure's true
crystal point group from the standardized conventional cell stored in the
LMDB, projects the predictions, and reports official + grouped metrics.

Official metric is the V18 Frobenius norm over the 18 stored Voigt components
(fingerprint-confirmed against the GMTNet paper), never the C27 expansion.

Usage:
    python scripts/piezo_posthoc_projection.py
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/public/home/moyk/wyckoff_gnn")

from wyckoff_gnn.data.lmdb_cache import LMDBReader
from wyckoff_gnn.models.unified_equivariant.crystal_tensor_projection import (
    build_voigt18_projector,
    crystal_point_group_rotations,
    project_cartesian_tensor,
)
from wyckoff_gnn.models.unified_equivariant.tensor_adapter import (
    voigt36_to_cartesian333,
)

P1_CACHE = "data/processed/gmtnet_piezo_p1_stdframe"
WQ_CACHE = "data/processed/gmtnet_piezo_wyckoff_stdframe"
RUNS = {
    "WQ": "results/wqgnet_piezo_wyckoff_s2",
    "P1": "results/p1_piezo_s2",
}
OUT = Path("results/piezo_posthoc_projection")
THRESHOLDS = (0.25, 0.10, 0.05, 0.02)


def v18(x):
    """Official Frobenius norm: over the 18 stored Voigt components."""
    return np.linalg.norm(np.asarray(x).reshape(len(x), -1), axis=1)


def load_predictions(run_dir):
    rows = list(csv.DictReader(open(Path(run_dir) / "predictions.csv")))
    out = {}
    for r in rows:
        out[r["material_id"]] = (
            np.array(json.loads(r["y_true_components"]), dtype=np.float64).reshape(3, 6),
            np.array(json.loads(r["y_pred_components"]), dtype=np.float64).reshape(3, 6),
        )
    return out


def ewt(err, label_norm, policy):
    """Two candidate EwT policies; the fingerprint decides which is official."""
    out = {}
    for t in THRESHOLDS:
        if policy == "exclude_zero":
            m = label_norm > 0
            out[t] = float((err[m] / label_norm[m] < t).mean())
        elif policy == "include_zero":
            # pass iff err <= t*||label||; a zero label demands an exact zero
            out[t] = float((err <= t * label_norm).mean())
        else:
            raise ValueError(policy)
    return out


def fmt_ewt(d, n):
    return "  ".join(f"@{int(t*100)}%={100*v:5.2f}%" for t, v in d.items()) + f"   (n={n})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symprec", type=float, default=1e-5)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    rp = LMDBReader(P1_CACHE)
    rw = LMDBReader(WQ_CACHE)

    preds = {k: load_predictions(v) for k, v in RUNS.items()}
    ids = sorted(set(preds["WQ"]) & set(preds["P1"]))
    print(f"test structures common to both runs: {len(ids)}")

    # --- build the point group + P18 for every test structure ----------------
    proj, info = {}, {}
    lat_mismatch = 0.0
    label_mismatch = 0.0
    orth_worst = 0.0
    map_worst = 0.0
    for mid in ids:
        gp, gw = rp.get(mid), rw.get(mid)
        lat = np.asarray(gp["lattice"], dtype=np.float64)
        lat_mismatch = max(
            lat_mismatch,
            float(np.abs(lat - np.asarray(gw["lattice"], dtype=np.float64)).max()),
        )
        label_mismatch = max(
            label_mismatch,
            float(np.abs(np.asarray(gp["y_tensor"], dtype=np.float64)
                         - np.asarray(gw["y_tensor"], dtype=np.float64)).max()),
        )
        R, meta = crystal_point_group_rotations(
            lat,
            np.asarray(gp["orbit_rep_frac"], dtype=np.float64),
            np.asarray(gp["orbit_element"], dtype=np.int32),
            symprec=args.symprec,
        )
        orth_worst = max(orth_worst, meta["max_orthogonality_error"])
        map_worst = max(map_worst, meta["structure_map_residual"])
        proj[mid] = build_voigt18_projector(torch.tensor(R, dtype=torch.float64))
        info[mid] = meta

    print(f"P1 vs WQ cache: max |lattice diff|={lat_mismatch:.2e}  "
          f"max |y_tensor diff|={label_mismatch:.2e}")
    print(f"worst Cartesian-rotation orthogonality error: {orth_worst:.2e}")
    print(f"worst structure self-mapping residual:        {map_worst:.2e}")
    idem = max(float((P @ P - P).abs().max()) for P in proj.values())
    print(f"worst P18 idempotence over {len(proj)} structures: {idem:.2e}")

    ng = np.array([info[m]["n_unique_pointgroup_rotations"] for m in ids])
    nsg = np.array([info[m]["n_spacegroup_ops"] for m in ids])
    print(f"|G| point group: median={np.median(ng):.0f} max={ng.max()}  "
          f"(space-group ops median={np.median(nsg):.0f}) -> dedup is doing work")

    # --- groups --------------------------------------------------------------
    T = np.array([preds["WQ"][m][0] for m in ids])
    ln = v18(T)
    inv = np.array([info[m]["contains_inversion"] for m in ids])

    neumann = np.zeros(len(ids))
    for i, mid in enumerate(ids):
        t = torch.tensor(T[i], dtype=torch.float64)
        cart = voigt36_to_cartesian333(t.unsqueeze(0))[0]
        n = float(cart.norm())
        if n < 1e-12:
            neumann[i] = 0.0
            continue
        R = torch.tensor(
            crystal_point_group_rotations(
                np.asarray(rp.get(mid)["lattice"], dtype=np.float64),
                np.asarray(rp.get(mid)["orbit_rep_frac"], dtype=np.float64),
                np.asarray(rp.get(mid)["orbit_element"], dtype=np.int32),
                symprec=args.symprec)[0], dtype=torch.float64)
        neumann[i] = float((project_cartesian_tensor(cart, R, 3) - cart).norm()) / n

    gA = inv & (ln == 0)
    gB = (~inv) & (neumann <= 1e-3)
    gC = neumann > 1e-3
    gC_severe = neumann > 0.1
    print(f"\nGROUP A centrosymmetric & zero label : {int(gA.sum())}")
    print(f"  consistency: contains_inversion={int(inv.sum())}  zero-label={int((ln==0).sum())}"
          f"  both={int(gA.sum())}  inversion-but-nonzero-label={int((inv & (ln>0)).sum())}")
    print(f"GROUP B non-centrosym & label consistent (residual<=1e-3): {int(gB.sum())}")
    print(f"GROUP C label violates its own point group (>1e-3): {int(gC.sum())}"
          f"   severe (>0.1): {int(gC_severe.sum())}")

    # --- project -------------------------------------------------------------
    results = {}
    for tag in RUNS:
        raw = np.array([preds[tag][m][1] for m in ids])
        pj = np.stack([
            (proj[m] @ torch.tensor(preds[tag][m][1].reshape(-1), dtype=torch.float64)
             ).reshape(3, 6).numpy() for m in ids
        ])
        results[tag] = {"raw": raw, "projected": pj}
        for kind, arr in (("raw", raw), ("projected", pj)):
            with open(OUT / f"{tag}_predictions_{kind}.csv", "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["material_id", "y_true_components", "y_pred_components"])
                for i, m in enumerate(ids):
                    w.writerow([m, json.dumps(T[i].reshape(-1).tolist()),
                                json.dumps(arr[i].reshape(-1).tolist())])

    # --- GROUP A hard gate ---------------------------------------------------
    print("\n=== GROUP A gate: centrosymmetric predictions must vanish ===")
    gate_ok = True
    for tag in RUNS:
        rn = v18(results[tag]["raw"][gA])
        pn = v18(results[tag]["projected"][gA])
        n_zero = int((pn < 1e-6).sum())
        print(f"  {tag}: raw ||pred|| max={rn.max():.3e} median={np.median(rn):.3e}"
              f" | projected max={pn.max():.3e} | exact-zero(<1e-6) {n_zero}/{int(gA.sum())}")
        gate_ok &= n_zero == int(gA.sum())

    # --- official metrics ----------------------------------------------------
    print("\n=== OFFICIAL metrics: V18 Fnorm over all %d test samples ===" % len(ids))
    print(f"{'model':<16}{'Fnorm':>9}{'median':>9}   EwT (exclude-zero policy)")
    rows_out = {}
    for name, P in [("zero-predictor", np.zeros_like(T))] + [
        (f"{t} {k}", results[t][k]) for t in RUNS for k in ("raw", "projected")
    ]:
        err = v18(T - P)
        e_ex = ewt(err, ln, "exclude_zero")
        e_in = ewt(err, ln, "include_zero")
        rows_out[name] = (err.mean(), np.median(err), e_ex, e_in)
        print(f"{name:<16}{err.mean():>9.4f}{np.median(err):>9.4f}   "
              f"{fmt_ewt(e_ex, int((ln>0).sum()))}")
    print(f"\n{'model':<16}{'':>18}   EwT (include-zero policy, all %d)" % len(ids))
    for name, (_, _, _, e_in) in rows_out.items():
        print(f"{name:<16}{'':>18}   {fmt_ewt(e_in, len(ids))}")
    print(f"\n  zero-label fraction = {int((ln==0).sum())}/{len(ids)} = "
          f"{100*(ln==0).mean():.2f}%   <- compare with GMTNet EwT@5%=45.7%")
    print("  GMTNet published: Fnorm 0.37  EwT25 49.1%  EwT10 46.3%  EwT5 45.7%")

    # --- grouped -------------------------------------------------------------
    for gname, mask in (("B  non-centrosym, label consistent", gB),
                        ("C  label inconsistent (>1e-3)", gC),
                        ("C' severe (>0.1)", gC_severe)):
        if not mask.any():
            continue
        print(f"\n=== GROUP {gname}   n={int(mask.sum())} ===")
        for tag in RUNS:
            for kind in ("raw", "projected"):
                err = v18(T[mask] - results[tag][kind][mask])
                e = ewt(err, ln[mask], "exclude_zero")
                print(f"  {tag} {kind:<10} Fnorm={err.mean():.4f} median={np.median(err):.4f}"
                      f"   {fmt_ewt(e, int((ln[mask]>0).sum()))}")

    # --- projection magnitude ------------------------------------------------
    print("\n=== projection correction magnitude (all test) ===")
    for tag in RUNS:
        d = v18(results[tag]["projected"] - results[tag]["raw"])
        rn = v18(results[tag]["raw"])
        rel = d / np.maximum(rn, 1e-12)
        print(f"  {tag}: ||Pi(T)-T|| median={np.median(d):.4e} max={d.max():.4e}"
              f" | relative median={np.median(rel):.3f} max={rel.max():.3f}")

    json.dump({"group_sizes": {"A": int(gA.sum()), "B": int(gB.sum()),
                               "C": int(gC.sum()), "C_severe": int(gC_severe.sum())},
               "gate_group_a_exact_zero": bool(gate_ok),
               "worst_p18_idempotence": idem,
               "worst_orthogonality_error": orth_worst},
              open(OUT / "summary.json", "w"), indent=2)
    print(f"\nGROUP A gate: {'PASS' if gate_ok else 'FAIL'}")
    print(f"artifacts -> {OUT}")
    return 0 if gate_ok else 1


if __name__ == "__main__":
    sys.exit(main())
