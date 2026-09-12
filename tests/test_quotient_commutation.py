"""Quotient-P1 commutation test (NCS final version).

30+ crystals stratified by crystal system; tests 0e / 1o / 2e sectors.
Also diagnoses edge-topology matching between quotient and matched-P1 graphs
(required precondition for the commutation theorem).

Usage:
    python tests/test_quotient_commutation.py
"""
import sys
import json
from collections import Counter

import numpy as np
import torch

sys.path.insert(0, "/public/home/moyk/wyckoff_gnn")

from wyckoff_gnn.data.lmdb_cache import LMDBReader
from wyckoff_gnn.data.graph_schema import light_dict_to_pyg_data
from wyckoff_gnn.models.unified_equivariant import UnifiedQuotientEquivariantGNN

WYCKOFF_DIR = "data/processed/jarvis_lmdb_bandgap"
P1_DIR = "data/processed/jarvis_lmdb_bandgap_p1_std"

CRYSTAL_SYSTEMS = [
    ("triclinic", 1, 2), ("monoclinic", 3, 15), ("orthorhombic", 16, 74),
    ("tetragonal", 75, 142), ("trigonal", 143, 167), ("hexagonal", 168, 194),
    ("cubic", 195, 230),
]


def load_graph(reader, mid):
    g = reader.get(mid)
    return light_dict_to_pyg_data(g)


def edge_topology_match(dq, dp):
    """Gauge-invariant physical edge multiset comparison.

    Compares the multiset of physical edge vectors (target rep, vector, distance)
    between quotient sub-edges and P1 edges.  Both graphs must be built from the
    same conventional cell; the comparison is invariant to atom/orbit ordering,
    fractional-coordinate wrapping, and shift representation.

    Returns (match, n_quotient_edges, n_p1_edges).
    """
    import numpy as np

    def _physical_edges(data):
        lat = data.lattice.double().numpy()
        rep = data.orbit_rep_frac.double().numpy()
        sfrac = data.geo_edge_source_frac.double().numpy()
        sh = data.geo_edge_shift.double().numpy()
        ei = data.geo_edge_index.numpy()
        edges = Counter()
        for e in range(ei.shape[1]):
            t = int(ei[0, e])
            v = (sfrac[e] + sh[e] - rep[t]) @ lat
            d = float(np.linalg.norm(v))
            edges[(t, tuple(np.round(v, 4)), round(d, 4))] += 1
        return edges

    q_sig = _physical_edges(dq)
    p_sig = _physical_edges(dp)
    return q_sig == p_sig, len(q_sig), len(p_sig)


def main():
    torch.manual_seed(0)
    model = UnifiedQuotientEquivariantGNN(
        hidden_irreps="128x0e + 16x1o + 8x2e", num_layers=2, num_rbf=16,
        rbf_max=8.0, block_type="dynamic_tp",
    )
    model.eval()
    scalar_mul = model.scalar_mul
    high_irreps = model.high_irreps

    entries = []
    with open(f"{WYCKOFF_DIR}/manifest.jsonl") as f:
        for line in f:
            entries.append(json.loads(line))

    # Stratified sampling: 5 crystals per crystal system
    picked = []
    for name, lo, hi in CRYSTAL_SYSTEMS:
        cands = [e for e in entries if lo <= e["space_group"] <= hi]
        # prefer compressed crystals (mult>1) so the test is informative
        comp = [e for e in cands if e["num_atoms"] > e["num_orbits"]]
        pool = comp or cands
        for e in pool[:5]:
            picked.append(e["material_id"])
    print(f"采样晶体: {len(picked)}, 晶系覆盖: {[n for n, _, _ in CRYSTAL_SYSTEMS]}")

    wy_reader = LMDBReader(WYCKOFF_DIR)
    p1_reader = LMDBReader(P1_DIR)

    rel_0e, rel_1o, rel_2e = [], [], []
    topo_ok, topo_fail = [], []
    for mid in picked:
        try:
            dq = load_graph(wy_reader, mid)
            dp = load_graph(p1_reader, mid)
        except Exception:
            continue
        K, M = dq.num_nodes, dp.num_nodes
        dq.batch = torch.zeros(K, dtype=torch.long)
        dp.batch = torch.zeros(M, dtype=torch.long)
        if K == M:
            continue  # no compression
        a2o = dq.atom_to_orbit.long()
        img = dq.atom_image_index.long()
        R = dq.orbit_sym_ops_rotations.double()
        assert M == a2o.shape[0] and K == a2o.max().item() + 1

        # edge topology check
        ok, nq, np1 = edge_topology_match(dq, dp)
        (topo_ok if ok else topo_fail).append((mid, dq.space_group.item(), nq, np1))

        # random tied features
        h_q = torch.randn(K, scalar_mul + high_irreps.dim, dtype=torch.float64)
        h_scalar = h_q[:, :scalar_mul]
        h_high = h_q[:, scalar_mul:]
        R_a = R[a2o, img]

        # lift high-l to atoms via D(R_{p,a})
        h_a_high = torch.zeros(M, high_irreps.dim, dtype=torch.float64)
        off = 0
        for mul_i, ir in high_irreps:
            dim = ir.dim
            sec = h_high[:, off:off + mul_i * dim][a2o]
            if ir.l > 0:
                D = ir.D_from_matrix(R_a)
                sec3 = sec.view(M, mul_i, dim)
                rot = torch.einsum("emi,eci->ecm", D, sec3)
                h_a_high[:, off:off + mul_i * dim] = rot.reshape(M, mul_i * dim)
            else:
                h_a_high[:, off:off + mul_i * dim] = sec
            off += mul_i * dim
        h_a = torch.cat([h_scalar[a2o], h_a_high], dim=-1)

        def fwd_with_h(data, h):
            h = h.float()
            e = model.edge_embedding(data.geo_edge_distance.float())
            from wyckoff_gnn.models.unified_equivariant.equivariant_sector import precompute_batch_geometry
            edge_sh, wd = precompute_batch_geometry(data, model.high_irreps, model.lmax)
            agg_norm = model.blocks[0].compute_agg_norm(data.geo_edge_index[0], h.size(0))
            for blk in model.blocks:
                h = blk(h, e, data.geo_edge_index, edge_sh, wd, agg_norm)
            return h

        with torch.no_grad():
            h_p1 = fwd_with_h(dp, h_a).double()
            h_q = fwd_with_h(dq, h_q).double()

        # lift quotient result back to atoms
        q_scalar = h_q[:, :scalar_mul]
        q_high = h_q[:, scalar_mul:]
        lifted_scalar = q_scalar[a2o]
        lifted_high = torch.zeros(M, high_irreps.dim, dtype=torch.float64)
        off = 0
        for mul_i, ir in high_irreps:
            dim = ir.dim
            sec = q_high[:, off:off + mul_i * dim]
            if ir.l > 0:
                D = ir.D_from_matrix(R_a)
                sec3 = sec[a2o].double().view(M, mul_i, dim)
                rot = torch.einsum("emi,eci->ecm", D, sec3)
                lifted_high[:, off:off + mul_i * dim] = rot.reshape(M, mul_i * dim)
            else:
                lifted_high[:, off:off + mul_i * dim] = sec[a2o]
            off += mul_i * dim

        e0 = (h_p1[:, :scalar_mul] - lifted_scalar).norm() / h_p1[:, :scalar_mul].norm()
        rel_0e.append(e0.item())
        # per-sector high-l errors
        off = 0
        for mul_i, ir in high_irreps:
            dim = ir.dim
            sl = slice(off, off + mul_i * dim)
            e_sec = (h_p1[:, sl] - lifted_high[:, sl]).norm() / h_p1[:, sl].norm().clamp(min=1e-8)
            if ir.l == 1:
                rel_1o.append(e_sec.item())
            elif ir.l == 2:
                rel_2e.append(e_sec.item())
            off += mul_i * dim
        print(f"  {mid}: SG={dq.space_group.item()} K={K} M={M} topo={'OK' if ok else 'MISMATCH'} "
              f"0e={e0.item():.2e}")

    def report(name, arr):
        a = np.array(arr)
        print(f"{name}: n={len(a)} mean={a.mean():.3e} median={np.median(a):.3e} "
              f"p95={np.percentile(a,95):.3e} max={a.max():.3e}")

    print("\n=== 结果 ===")
    report("0e", rel_0e)
    report("1o", rel_1o)
    report("2e", rel_2e)
    print(f"\nedge topology: matched={len(topo_ok)}/{len(topo_ok)+len(topo_fail)}")
    for mid, sg, nq, np1 in topo_fail[:10]:
        print(f"  MISMATCH: {mid} SG={sg} quotient_edges={nq} p1_edges={np1}")

    print("\n=== 解释 ===")
    print("0e: 满足 matched topology 的晶体应达机器精度 (float32 ~1e-7)")
    print("1o/2e: target 特征项差异 (representative vs transported image) -> 设计使然的差异")
    print("edge topology mismatch 晶体需从严格 commutation 统计中排除 (定理前提)")


if __name__ == "__main__":
    main()
