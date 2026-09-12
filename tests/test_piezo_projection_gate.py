"""P6 regression gate for the global crystal point-group projection.

    E  P18 idempotence over all 3158 structures          < 1e-12
    F  point-group invariance  R^(x)3 (Pi T) = Pi T
    G  227 centrosymmetric test structures: P18 exactly zero, prediction
       exactly zero -- element-wise, no epsilon
    H  JVASP-5197 (SG 19, class 222): projector_rank == 3, not zeroed
    I  batching: mixed-symmetry batch == per-graph single-graph output
    J  WQ and P1 caches hold bit-identical projectors
    K  Voigt18 <-> Cartesian round trip
    L  point-group extraction audit over all 3158 structures

Usage:
    python tests/test_piezo_projection_gate.py
"""
import json
import sys

import numpy as np
import spglib
import torch

sys.path.insert(0, "/public/home/moyk/wyckoff_gnn")

from torch_geometric.data import Batch
from wyckoff_gnn.data.graph_schema import light_dict_to_pyg_data
from wyckoff_gnn.data.lmdb_cache import LMDBReader
from wyckoff_gnn.models.unified_equivariant import UnifiedQuotientEquivariantGNN
from wyckoff_gnn.models.unified_equivariant.crystal_tensor_projection import (
    fractional_to_cartesian_rotations,
    project_cartesian_tensor,
    unique_cartesian_rotations,
)
from wyckoff_gnn.models.unified_equivariant.tensor_adapter import (
    cartesian333_to_voigt36,
    voigt36_to_cartesian333,
)

WQ = "data/processed/gmtnet_piezo_wyckoff_stdframe"
P1 = "data/processed/gmtnet_piezo_p1_stdframe"


def ok(name, val, thresh, mode="lt"):
    good = val < thresh if mode == "lt" else val > thresh
    print(f"  {name:<56} {val:.3e}  [{'PASS' if good else 'FAIL'}]")
    return good


def main():
    passed = True
    rw, rp = LMDBReader(WQ), LMDBReader(P1)
    rows = [json.loads(l) for l in open(f"{WQ}/manifest.jsonl")]
    test_ids = [r["material_id"] for r in rows if r["split"] == "test"]
    print(f"structures={len(rows)}  test={len(test_ids)}")

    # ---- E / J / L -------------------------------------------------------
    print("\n=== E. P18 idempotence + J. WQ/P1 identity + L. extraction audit ===")
    idem = pj_diff = orth = 0.0
    n_inv = n_zeroP = 0
    sg_match = 0
    mismatch = []
    ranks = {}
    for r in rows:
        mid = r["material_id"]
        gw, gp = rw.get(mid), rp.get(mid)
        Pw = np.asarray(gw["tensor_point_group_projector"], dtype=np.float64)
        Pp = np.asarray(gp["tensor_point_group_projector"], dtype=np.float64)
        idem = max(idem, float(np.abs(Pw @ Pw - Pw).max()))
        pj_diff = max(pj_diff, float(np.abs(Pw - Pp).max()))
        inv = bool(gw["contains_inversion"])
        n_inv += inv
        if inv:
            n_zeroP += int(np.all(Pw == 0.0))
        ranks[mid] = int(gw["projector_rank"])
        # audit: does the stored cell still reproduce the recorded space group?
        lat = np.asarray(gp["lattice"], dtype=np.float64)
        ds = spglib.get_symmetry_dataset(
            (lat, np.asarray(gp["orbit_rep_frac"], dtype=np.float64),
             np.asarray(gp["orbit_element"], dtype=np.int32)), symprec=1e-5)
        stored_sg = int(gw["space_group"])
        got = int(ds.number) if ds is not None else -1
        if got == stored_sg:
            sg_match += 1
        else:
            R = unique_cartesian_rotations(fractional_to_cartesian_rotations(
                lat, np.asarray(ds.rotations, dtype=np.float64))) if ds else None
            mismatch.append((mid, stored_sg, got,
                             int(len(ds.rotations)) if ds else 0,
                             int(len(R)) if R is not None else 0,
                             int(gw["point_group_order"]),
                             len(gp["orbit_element"])))
    passed &= ok("E  worst |P.P - P| over 3158", idem, 1e-12)
    passed &= ok("J  worst |P18(WQ) - P18(P1)|", pj_diff, 1e-300)
    print(f"  L  centrosymmetric={n_inv}  of which P18 exactly zero: {n_zeroP}/{n_inv}"
          f"  [{'PASS' if n_zeroP == n_inv else 'FAIL'}]")
    passed &= n_zeroP == n_inv
    print(f"  L  stored cell reproduces recorded SG: {sg_match}/{len(rows)}")
    if mismatch:
        print(f"  L  mismatches ({len(mismatch)}), all listed:")
        print(f"     {'JID':14s}{'storedSG':>9}{'float32SG':>10}{'nops':>6}"
              f"{'nuniq':>7}{'|G|stored':>10}{'natoms':>7}")
        for m in mismatch:
            print(f"     {m[0]:14s}{m[1]:>9}{m[2]:>10}{m[3]:>6}{m[4]:>7}{m[5]:>10}{m[6]:>7}")

    # ---- F / K -----------------------------------------------------------
    print("\n=== F. point-group invariance of the projected tensor + K. round trip ===")
    # The group must be re-derived exactly as the builder did -- from the
    # float64 standardized cell, not from the float32 lattice in the LMDB.
    import pickle

    from wyckoff_gnn.data.adapters.gmtnet_piezo_adapter import (
        _gmtnet_atoms_to_pymatgen_structure,
    )
    from wyckoff_gnn.data.crystal_to_wyckoff import standardize_structure_cell

    pkl = pickle.load(
        open("data/processed/gmtnet_piezo/gmtnet_piezo_filtered.pkl", "rb"))

    def builder_group(mid):
        stru = _gmtnet_atoms_to_pymatgen_structure(pkl[mid]["atoms"], mid)
        std = standardize_structure_cell(stru, symprec=1e-5)
        return torch.tensor(unique_cartesian_rotations(
            fractional_to_cartesian_rotations(
                std["standardized_lattice"],
                np.asarray(std["symmetry_rotations"], dtype=np.float64))),
            dtype=torch.float64)

    worst_inv = worst_rt = 0.0
    for mid in test_ids[:200]:
        R = builder_group(mid)
        P = torch.tensor(np.asarray(rw.get(mid)["tensor_point_group_projector"],
                                   dtype=np.float64))
        v = torch.randn(18, dtype=torch.float64, generator=torch.Generator().manual_seed(3))
        pv = (P @ v).reshape(1, 3, 6)
        cart = voigt36_to_cartesian333(pv)[0]
        n = float(cart.abs().max())
        if n > 1e-30:
            for Ri in R:
                q = project_cartesian_tensor(cart, Ri[None, ...], 3)
                worst_inv = max(worst_inv, float((q - cart).abs().max()) / n)
        back = cartesian333_to_voigt36(voigt36_to_cartesian333(pv))
        worst_rt = max(worst_rt, float((back - pv).abs().max()))
    passed &= ok("F  relative |R^(x)3 Pi T - Pi T| (200 structures)", worst_inv, 1e-10)
    passed &= ok("K  voigt -> cartesian -> voigt", worst_rt, 1e-12)

    if mismatch:
        print("\n=== L. point-group equivalence proof for the SG mismatches ===")
        for m in mismatch:
            mid = m[0]
            R64 = builder_group(mid).numpy()
            gp = rp.get(mid)
            ds = spglib.get_symmetry(
                (np.asarray(gp["lattice"], dtype=np.float64),
                 np.asarray(gp["orbit_rep_frac"], dtype=np.float64),
                 np.asarray(gp["orbit_element"], dtype=np.int32)), symprec=1e-5)
            R32 = unique_cartesian_rotations(fractional_to_cartesian_rotations(
                np.asarray(gp["lattice"], dtype=np.float64),
                np.asarray(ds["rotations"], dtype=np.float64)))
            same = len(R64) == len(R32) and all(
                any(np.abs(a - b).max() < 1e-4 for b in R32) for a in R64)
            print(f"  {mid:14s} |G|float64={len(R64):3d} |G|float32={len(R32):3d}"
                  f"  same rotation set: {same}")

    # ---- G / H / I -------------------------------------------------------
    print("\n=== G/H/I. model forward with projection ===")
    torch.manual_seed(0)
    model = UnifiedQuotientEquivariantGNN(
        hidden_irreps="128x0e + 8x1o + 4x2e", num_layers=2, num_rbf=16,
        rbf_max=8.0, output_type="piezo_rank3",
        global_point_group_projection=True,
    ).float()
    model.eval()

    def graph(mid):
        d = light_dict_to_pyg_data(rw.get(mid))
        d.batch = torch.zeros(d.num_nodes, dtype=torch.long)
        return d

    cs = [m for m in test_ids if bool(rw.get(m)["contains_inversion"])]
    print(f"  centrosymmetric test structures: {len(cs)}")
    worst = 0.0
    n_exact = 0
    for mid in cs:
        with torch.no_grad():
            out = model(graph(mid))["tensor"]
        v = out["voigt"]
        worst = max(worst, float(v.abs().max()))
        n_exact += int(torch.all(v == 0.0))
    print(f"  G  prediction element-wise exactly zero: {n_exact}/{len(cs)}"
          f"  [{'PASS' if n_exact == len(cs) else 'FAIL'}]")
    passed &= n_exact == len(cs)
    passed &= ok("G  max |projected prediction| (must be 0.0)", worst, 1e-300)

    mid5197 = "JVASP-5197"
    if mid5197 in ranks:
        g = rw.get(mid5197)
        with torch.no_grad():
            v = model(graph(mid5197))["tensor"]["voigt"]
        r = int(g["projector_rank"])
        nz = float(v.abs().max())
        print(f"  H  {mid5197}: projector_rank={r} contains_inversion="
              f"{bool(g['contains_inversion'])} |pred|max={nz:.3e}"
              f"  [{'PASS' if r == 3 and nz > 0 else 'FAIL'}]")
        passed &= (r == 3 and nz > 0)
    else:
        print(f"  H  {mid5197} not in test split  [FAIL]")
        passed = False

    # I: mixed-symmetry batch
    by_rank = {}
    for mid in test_ids:
        by_rank.setdefault(ranks[mid], []).append(mid)
    picks = []
    for r in sorted(by_rank):
        picks.extend(by_rank[r][:2])
    picks = picks[:12]
    singles, singles_raw = [], []
    for mid in picks:
        with torch.no_grad():
            t = model(graph(mid))["tensor"]
        singles.append(t["voigt"][0])
        singles_raw.append(t["voigt_raw"][0])
    batch = Batch.from_data_list([light_dict_to_pyg_data(rw.get(m)) for m in picks])
    with torch.no_grad():
        bt = model(batch)["tensor"]
    bout, braw = bt["voigt"], bt["voigt_raw"]
    err = max(float((bout[i] - singles[i]).abs().max()) for i in range(len(picks)))
    err_raw = max(float((braw[i] - singles_raw[i]).abs().max()) for i in range(len(picks)))
    print(f"  I  batch of {len(picks)} graphs, projector ranks "
          f"{sorted({ranks[m] for m in picks})}")
    print(f"  I  pre-projection  max |batched - single| = {err_raw:.3e}"
          "   (float32 reduction-order noise baseline)")
    print(f"  I  post-projection max |batched - single| = {err:.3e}")
    # The projection cannot amplify error beyond the projector norm, so the
    # projected discrepancy must track the raw one. A broadcast bug would show
    # up as a large, structural difference instead.
    passed &= ok("I  projected error vs raw baseline", err, max(err_raw * 10, 1e-9))

    # decisive broadcast check: using graph 0's projector for everyone must
    # give a visibly different answer
    P0 = torch.tensor(
        np.asarray(rw.get(picks[0])["tensor_point_group_projector"],
                   dtype=np.float64), dtype=braw.dtype)
    wrong = torch.einsum("ij,gj->gi", P0, braw.reshape(len(picks), 18))
    delta = float((wrong - bout.reshape(len(picks), 18)).abs().max())
    print(f"  I  broadcast-P0 differs from per-graph result: {delta:.3e}"
          f"  [{'PASS' if delta > 1e-4 else 'FAIL'}]")
    passed &= delta > 1e-4

    print("\n" + ("GATE PASSED" if passed else "GATE FAILED"))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
