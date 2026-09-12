"""Canonical higher-order quotient--P1 commutation audit (Supp. Note 1, Thm 1).

Directly tests  L o Phi_Q = Phi_F o L  on the production canonical
source-image-resolved geometric backbone (``dynamic_tp`` blocks), with the site
stabilizer Reynolds projection applied as the theorem assumes:

    h_q in Fix_{H_q}(V),   P_q^{(l,pi)} = (1/|H_q|) sum_{g in H_q} D^{(l,pi)}(R_g).

Three modes are run for every structure, dtype and layer count:

  A. unprojected     -- negative control (reproduces the old test behaviour)
  B. projected_input -- P_q applied at layer 0 only; per-layer Fix residual logged
  C. projected_each  -- P_q re-applied after every quotient block

Theorem assumptions are audited item by item and recorded per structure rather
than assumed:

  (i)   atom correspondence   quotient atom index <-> P1 node index, verified by
                              fractional coordinate and atomic number
  (ii)  edge bijection        incoming physical-edge multiset at every orbit
                              representative (source orbit + Cartesian
                              displacement + distance) matches the full-atom
                              graph with multiplicity
  (iii) transport consistency the Cartesian rotation carried on a quotient edge
                              equals the rotation the lift applied to the
                              corresponding full-atom source
  (iv)  projector validity    P_q idempotent, H_q-invariant, symmetric; stored
                              Cartesian ops agree with W_frac + lattice
  (v)   gauge consistency     the projected quotient state is H_q-fixed, so the
                              lift is independent of the coset representative

Structure selection is stratified by site stabilizer so that the higher-order
evidence is not silently restricted to generic sites (where every projector is
the identity and mode A == mode C trivially):

  class 1  every H_q = {e}                       (generic sites)
  class 2  some |H_q| > 1 with rank(P_q^{1o}) > 0 (1o survives projection)
  class 3  some |H_q| > 1 with rank(P_q^{2e}) > 0 (2e survives projection)
  class 4  some q with rank(P_q^{1o}) == 0        (1o forbidden by site symmetry)

Usage:
    python tests/test_projected_quotient_commutation.py            # quick, 3 crystals
    python tests/test_projected_quotient_commutation.py --full     # 35+ crystals
    python tests/test_projected_quotient_commutation.py --scan-only

Data paths configurable via env: WYCKOFF_CACHE, P1_CACHE.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from e3nn import o3

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wyckoff_gnn.data.lmdb_cache import LMDBReader
from wyckoff_gnn.data.graph_schema import light_dict_to_pyg_data
from wyckoff_gnn.models.unified_equivariant import UnifiedQuotientEquivariantGNN
from wyckoff_gnn.models.unified_equivariant.crystal_tensor_projection import (
    fractional_to_cartesian_rotations,
)
from wyckoff_gnn.models.unified_equivariant.equivariant_sector import (
    build_wigner_d_cache,
    precompute_batch_geometry,
)

WYCKOFF_DIR = os.environ.get("WYCKOFF_CACHE", "data/processed/jarvis_lmdb_bandgap")
P1_DIR = os.environ.get("P1_CACHE", "data/processed/jarvis_lmdb_bandgap_p1_std")
PROJECTOR_PATH = "data/symmetry/site_irrep_tables/invariant_basis/site_projectors_lmax8.pt"
OUT_DIR = Path("results/commutation")
SCAN_CACHE = OUT_DIR / "stabilizer_scan.json"

HIDDEN = "128x0e + 16x1o + 8x2e"
EDGE_TOL = 1e-3          # Cartesian Angstrom tolerance for edge displacement match
ROT_TOL = 1e-5           # Cartesian rotation-matrix element tolerance
COORD_TOL = 1e-4         # fractional-coordinate tolerance for atom correspondence
EPS = 1e-30
TOL_F64 = 1e-10
TOL_F32 = 1e-5

# ITC lists these 230-group numbers with two origin choices.  The shipped
# projector bank is keyed by (space group, Wyckoff letter) only, so it cannot be
# joined against an spglib-standardized cell here: the same letter denotes a
# different site in the two settings.  See section 9 of
# docs/projected_quotient_commutation_audit.md.
ORIGIN_CHOICE_SG = frozenset({
    48, 50, 59, 68, 70, 85, 86, 88, 125, 126, 129, 130, 133, 134,
    137, 138, 141, 142, 201, 203, 222, 224, 227, 228,
})

CS_SG = [(1, 2, "triclinic"), (3, 15, "monoclinic"), (16, 74, "orthorhombic"),
         (75, 142, "tetragonal"), (143, 167, "trigonal"), (168, 194, "hexagonal"),
         (195, 230, "cubic")]
SYSTEMS = [c[2] for c in CS_SG]


def crystal_system(sg: int) -> str:
    for lo, hi, name in CS_SG:
        if lo <= sg <= hi:
            return name
    return "unknown"


def load_pair(reader_q, reader_p, mid):
    dq = light_dict_to_pyg_data(reader_q.get(mid))
    dp = light_dict_to_pyg_data(reader_p.get(mid))
    dq.batch = torch.zeros(dq.num_nodes, dtype=torch.long)
    dp.batch = torch.zeros(dp.num_nodes, dtype=torch.long)
    return dq, dp


def make_model(num_layers=2, dtype=torch.float32):
    torch.manual_seed(0)
    model = UnifiedQuotientEquivariantGNN(
        hidden_irreps=HIDDEN, num_layers=num_layers, num_rbf=16, rbf_max=8.0,
        block_type="dynamic_tp",
    )
    return model.to(dtype).eval()


# ---------------------------------------------------------------------------
# symmetry / projector construction  -- assumption (iv)
# ---------------------------------------------------------------------------
def stabilizer_cart(dq):
    """(K, S, 3, 3) float64 Cartesian stabilizer rotations plus validity mask."""
    W = dq.orbit_stabilizer_W_frac.double()
    mask = dq.orbit_stabilizer_mask.bool()
    R = fractional_to_cartesian_rotations(
        dq.lattice.double().numpy(), W.reshape(-1, 3, 3).numpy()
    ).reshape(W.shape[0], W.shape[1], 3, 3)
    return torch.tensor(R, dtype=torch.float64), mask


def wigner_d(Rq, ir):
    """D^{(l,pi)}(R) for R of shape (N, 3, 3) -> (N, dim, dim), float64.

    Routed through the production ``build_wigner_d_cache`` so the projector, the
    lift and the message-passing transport share one numerical realisation of
    D.  This matters: e3nn's generic ``D_from_matrix`` reconstructs D from
    extracted Euler angles and is only accurate to ~1e-6 for l=2, whereas the
    analytic fast path the model uses is exact, so mixing the two injects a
    ~1e-7 inconsistency that has nothing to do with the theorem.
    """
    if ir.l == 0:
        return torch.ones(Rq.shape[0], 1, 1, dtype=torch.float64)
    D = build_wigner_d_cache(Rq.double(), o3.Irreps([(1, ir)]))[(ir.l, ir.p)]
    return D.double().reshape(-1, ir.dim, ir.dim)


def reynolds(Rq, ir):
    """P = (1/|H|) sum_g D^{(l,pi)}(R_g), float64."""
    if ir.l == 0:
        return torch.eye(ir.dim, dtype=torch.float64)
    return wigner_d(Rq, ir).mean(dim=0)


def build_projectors(dq, high_irreps):
    """Per-orbit Reynolds projectors (float64) + assumption-(iv) audit."""
    R, mask = stabilizer_cart(dq)
    K = R.shape[0]
    proj = {str(ir): [] for _, ir in high_irreps}
    audit = {
        "idem": 0.0, "sym": 0.0, "invariance": 0.0, "orthogonality": 0.0,
        "stab_order": [], "rank": {str(ir): [] for _, ir in high_irreps},
    }
    for q in range(K):
        Rq = R[q][mask[q]]
        audit["stab_order"].append(int(Rq.shape[0]))
        audit["orthogonality"] = max(
            audit["orthogonality"],
            float((Rq @ Rq.transpose(-1, -2)
                   - torch.eye(3, dtype=torch.float64)).abs().max()))
        for _, ir in high_irreps:
            P = reynolds(Rq, ir)
            proj[str(ir)].append(P)
            audit["idem"] = max(audit["idem"], float((P @ P - P).abs().max()))
            audit["sym"] = max(audit["sym"], float((P - P.T).abs().max()))
            if ir.l > 0:
                D = wigner_d(Rq, ir)
                audit["invariance"] = max(
                    audit["invariance"], float((D @ P - P).abs().max()))
            audit["rank"][str(ir)].append(
                int(torch.linalg.matrix_rank(P, atol=1e-8, rtol=0)))
    return proj, audit


def stored_vs_derived_rotations(dq):
    """Assumption (iv): cached Cartesian image ops == L^T W L^-T."""
    W = dq.orbit_sym_ops_W_frac.double()
    R = torch.tensor(fractional_to_cartesian_rotations(
        dq.lattice.double().numpy(), W.reshape(-1, 3, 3).numpy()
    ), dtype=torch.float64).reshape(W.shape)
    m = dq.orbit_mult_mask.bool()
    return float((R[m] - dq.orbit_sym_ops_rotations.double()[m]).abs().max())


def load_production_projectors(dq, high_irreps):
    """Per-orbit projectors from the production .pt bank (float64, single copy).

    Keys are ``SG{n}_{letter}_l{l}_{e|o}``.  ``o3.Irrep.p`` is the integer +-1,
    so the parity letter has to be spelled out; formatting ``ir.p`` directly
    yields ``l1_-1``, which misses every block and silently turns the comparison
    into "Reynolds vs identity".
    """
    bank = torch.load(PROJECTOR_PATH, map_location="cpu", weights_only=True)
    sg = int(dq.space_group.item())
    letters = dq.orbit_letter_in_sg.tolist()
    proj = {str(ir): [] for _, ir in high_irreps}
    hits = 0
    for q in range(dq.num_nodes):
        letter = chr(ord("a") + int(letters[q]))
        for _, ir in high_irreps:
            key = f"SG{sg}_{letter}_l{ir.l}_{'e' if ir.p == 1 else 'o'}"
            block = bank.get(key)
            if block is None:
                proj[str(ir)].append(torch.eye(ir.dim, dtype=torch.float64))
            else:
                proj[str(ir)].append(block.double().clone())
                hits += 1
    return proj, hits


def apply_projector(h, high_irreps, proj, scalar_mul):
    """Apply per-orbit, per-copy projectors to the high-l block of h (K, D)."""
    out = h.clone()
    off = scalar_mul
    for mul, ir in high_irreps:
        width = mul * ir.dim
        sec = h[:, off:off + width].reshape(-1, mul, ir.dim)
        P = torch.stack(proj[str(ir)]).to(dtype=h.dtype, device=h.device)
        out[:, off:off + width] = torch.einsum(
            "kji,kci->kcj", P, sec).reshape(-1, width)
        off += width
    return out


# ---------------------------------------------------------------------------
# lifting  --  assumption (v)
# ---------------------------------------------------------------------------
def lift(dq, h_q, high_irreps, scalar_mul):
    """Phi: quotient state -> full-atom state via D(R_{q,k}) at each image.

    R_{q,k} is rebuilt from the integer W_frac and the lattice exactly as
    ``compute_source_rotations_with_batch`` does inside the model, rather than
    read from the float32 ``orbit_sym_ops_rotations`` cache field, so the lift
    and the edge transport use bit-identical rotations.
    """
    a2o = dq.atom_to_orbit.long()
    img = dq.atom_image_index.long()
    M = a2o.shape[0]
    W = dq.orbit_sym_ops_W_frac.double()
    R_a = torch.tensor(fractional_to_cartesian_rotations(
        dq.lattice.double().numpy(), W[a2o, img].numpy()), dtype=torch.float64)
    lifted = torch.zeros(M, h_q.shape[1], dtype=h_q.dtype)
    lifted[:, :scalar_mul] = h_q[:, :scalar_mul][a2o]
    off = scalar_mul
    for mul, ir in high_irreps:
        width = mul * ir.dim
        sec = h_q[:, off:off + width][a2o].reshape(M, mul, ir.dim)
        if ir.l > 0:
            D = wigner_d(R_a, ir).to(dtype=h_q.dtype)          # (M, dim, dim)
            sec = torch.einsum("emi,eci->ecm", D, sec)
        lifted[:, off:off + width] = sec.reshape(M, width)
        off += width
    return lifted


def gauge_violation(dq, h_q, high_irreps, scalar_mul):
    """max_q max_{g in H_q} ||D(R_g) h_q - h_q|| / ||h_q||  (0 iff H_q-fixed)."""
    R, mask = stabilizer_cart(dq)
    worst = 0.0
    for q in range(dq.num_nodes):
        Rq = R[q][mask[q]]
        if Rq.shape[0] <= 1:
            continue
        off = scalar_mul
        for mul, ir in high_irreps:
            width = mul * ir.dim
            if ir.l == 0:
                off += width
                continue
            sec = h_q[q, off:off + width].double().reshape(mul, ir.dim)
            nrm = float(sec.norm())
            if nrm > 1e-14:
                D = wigner_d(Rq, ir)                            # (|H|, dim, dim)
                rot = torch.einsum("gmi,ci->gcm", D, sec)
                worst = max(worst, float((rot - sec).norm(dim=(1, 2)).max()) / nrm)
            off += width
    return worst


# ---------------------------------------------------------------------------
# atom correspondence  --  assumption (i)
# ---------------------------------------------------------------------------
def atom_correspondence(dq, dp):
    """Verify quotient atom index a <-> P1 node index a by coordinate + species.

    The quotient builder fills atom_to_orbit / atom_image_index indexed by the
    standardized-cell atom order, and the P1 cache is built from the same
    standardized cell, so the identity map is expected -- but it is checked,
    never assumed.
    """
    a2o = dq.atom_to_orbit.long()
    img = dq.atom_image_index.long()
    W = dq.orbit_sym_ops_W_frac.double()
    w = dq.orbit_sym_ops_w_frac.double()
    rep = dq.orbit_rep_frac.double()
    M = a2o.shape[0]
    pos_q = torch.stack([
        W[a2o[a], img[a]] @ rep[a2o[a]] + w[a2o[a], img[a]] for a in range(M)])
    d = pos_q - dp.orbit_rep_frac.double()
    d = d - d.round()
    coord_res = float(d.abs().max())
    z_q = dq.orbit_element.long()[a2o]
    z_res = int((z_q - dp.orbit_element.long()).abs().max())
    lat_res = float((dq.lattice.double() - dp.lattice.double()).abs().max())
    return coord_res, z_res, lat_res


def representative_atoms(dq):
    """orbit q -> the full-atom index carrying image 0 (identity coset)."""
    a2o = dq.atom_to_orbit.long()
    img = dq.atom_image_index.long()
    return {int(a2o[a]): a for a in range(a2o.shape[0]) if int(img[a]) == 0}


# ---------------------------------------------------------------------------
# edge bijection + transport  --  assumptions (ii), (iii)
# ---------------------------------------------------------------------------
def edge_table(data, node_orbit):
    """target node -> list of (src_orbit, disp_cart(3,), dist, src_image, src_node).

    Displacement follows the production convention used by
    ``precompute_batch_geometry``:  delta = (src_frac - tgt_frac + shift) @ L.
    """
    ei = data.geo_edge_index
    target, source = ei[0].long(), ei[1].long()
    lat = data.lattice.double()
    delta = (data.geo_edge_source_frac.double()
             - data.orbit_rep_frac.double()[target]
             + data.geo_edge_shift.double())
    disp = delta @ lat
    dist = data.geo_edge_distance.double()
    img = data.geo_edge_source_image.long()
    table = collections.defaultdict(list)
    for e in range(ei.shape[1]):
        table[int(target[e])].append(
            (int(node_orbit[int(source[e])]), disp[e], float(dist[e]),
             int(img[e]), int(source[e])))
    return table, float((disp.norm(dim=1) - dist).abs().max())


def match_edges(lq, lp, tol=EDGE_TOL):
    """Optimal bijection between two incoming-edge lists, keyed by source orbit.

    Returns (ok, worst_residual, pairs, detail). ``pairs`` holds matched
    (quotient_record, p1_record) tuples for the transport check.
    """
    if len(lq) != len(lp):
        return False, float("inf"), [], f"count {len(lq)} vs {len(lp)}"
    by_orbit_q = collections.defaultdict(list)
    by_orbit_p = collections.defaultdict(list)
    for r in lq:
        by_orbit_q[r[0]].append(r)
    for r in lp:
        by_orbit_p[r[0]].append(r)
    if set(by_orbit_q) != set(by_orbit_p):
        return False, float("inf"), [], (
            f"source-orbit sets differ {sorted(by_orbit_q)} vs {sorted(by_orbit_p)}")
    worst, pairs = 0.0, []
    for orb in by_orbit_q:
        A, B = by_orbit_q[orb], by_orbit_p[orb]
        if len(A) != len(B):
            return False, float("inf"), [], (
                f"orbit {orb} multiplicity {len(A)} vs {len(B)}")
        cost = np.empty((len(A), len(B)))
        for i, ra in enumerate(A):
            for j, rb in enumerate(B):
                cost[i, j] = max(float((ra[1] - rb[1]).abs().max()),
                                 abs(ra[2] - rb[2]))
        try:
            from scipy.optimize import linear_sum_assignment
            ri, ci = linear_sum_assignment(cost)
        except ImportError:                                   # greedy fallback
            ri, ci, taken = [], [], set()
            for i in range(len(A)):
                j = int(np.argmin([cost[i, k] if k not in taken else np.inf
                                   for k in range(len(B))]))
                taken.add(j)
                ri.append(i)
                ci.append(j)
        for i, j in zip(ri, ci):
            worst = max(worst, float(cost[i, j]))
            pairs.append((A[i], B[j]))
        if worst > tol:
            return False, worst, pairs, f"orbit {orb} residual {worst:.2e} A"
    return True, worst, pairs, ""


def edge_audit(dq, dp, tol=EDGE_TOL):
    """Per-representative edge bijection (ii) and transport consistency (iii).

    Also measures, for every non-representative image k, how well the cached
    full-atom neighbourhood equals R_{q,k} applied to the representative's
    neighbourhood.  That residual is a property of the float32 coordinate
    storage in the cache, not of the theorem, and it is reported separately so
    it is never folded into the representative statistics.
    """
    a2o = dq.atom_to_orbit.long()
    img = dq.atom_image_index.long()
    W = dq.orbit_sym_ops_W_frac.double()
    Rq_ops = torch.tensor(fractional_to_cartesian_rotations(
        dq.lattice.double().numpy(), W.reshape(-1, 3, 3).numpy()
    ), dtype=torch.float64).reshape(W.shape)
    tq, dev_q = edge_table(dq, torch.arange(dq.num_nodes))
    tp, dev_p = edge_table(dp, a2o)
    n_ok, n_fail, failures = 0, 0, []
    worst_disp, worst_rot = 0.0, 0.0
    orbit_resid, orbit_fail = 0.0, 0
    for a in range(dp.num_nodes):
        q, k = int(a2o[a]), int(img[a])
        Rk = Rq_ops[q, k]
        rotated = [(r[0], Rk @ r[1], r[2], r[3], r[4]) for r in tq.get(q, [])]
        ok, resid, pairs, detail = match_edges(rotated, tp.get(a, []), tol)
        if k != 0:
            orbit_resid = max(orbit_resid, resid if ok else float("inf"))
            orbit_fail += 0 if ok else 1
            continue
        if not ok:
            n_fail += 1
            failures.append((q, a, len(tq.get(q, [])), len(tp.get(a, [])), detail))
            continue
        n_ok += 1
        worst_disp = max(worst_disp, resid)
        for rq, rp in pairs:
            # rotation the quotient edge transports the source feature with
            R_edge = Rq_ops[rq[0], rq[3]]
            # rotation the lift already baked into the matched full-atom source
            src_p1 = rp[4]
            R_lift = Rq_ops[int(a2o[src_p1]), int(img[src_p1])]
            worst_rot = max(worst_rot, float((R_edge - R_lift).abs().max()))
    return dict(n_ok=n_ok, n_fail=n_fail, failures=failures,
                worst_disp=worst_disp, worst_rot=worst_rot,
                orbit_resid=orbit_resid, orbit_fail=orbit_fail,
                dist_dev=max(dev_q, dev_p))


_EDGE_AUDIT_CACHE = {}


def edge_audit_cached(dq, dp, mid):
    """Memoised: the edge audit depends only on the graph pair, not on dtype."""
    if mid not in _EDGE_AUDIT_CACHE:
        _EDGE_AUDIT_CACHE[mid] = edge_audit(dq, dp)
    return _EDGE_AUDIT_CACHE[mid]


def agg_norm_audit(model, dq, dp, rep):
    """Compare the 1/sqrt(degree) normaliser per node, not per edge.

    ``compute_agg_norm`` returns an edge-indexed tensor, so it has to be reduced
    to node degrees before the quotient orbit and its representative atom can be
    compared.
    """
    def node_norm(data, n):
        target = data.geo_edge_index[0]
        deg = torch.zeros(n)
        deg.scatter_add_(0, target, torch.ones_like(target, dtype=torch.float))
        return deg, deg.rsqrt().clamp(max=1.0).double()

    dq_deg, nq = node_norm(dq, dq.num_nodes)
    dp_deg, np1 = node_norm(dp, dp.num_nodes)
    worst, deg_mismatch = 0.0, 0
    for q, a0 in rep.items():
        worst = max(worst, float(abs(nq[q] - np1[a0])))
        deg_mismatch += int(dq_deg[q] != dp_deg[a0])
    return worst, deg_mismatch


# ---------------------------------------------------------------------------
# forward
# ---------------------------------------------------------------------------
GEOMETRY_FIELDS = ("orbit_rep_frac", "geo_edge_source_frac", "geo_edge_shift",
                   "geo_edge_distance", "lattice", "orbit_sym_ops_W_frac",
                   "orbit_sym_ops_rotations", "orbit_stabilizer_W_frac",
                   "rep_coords_frac", "multiplicity")


def promote(data, dtype):
    """Cast the geometry fields so the whole forward pass runs at one precision.

    ``precompute_batch_geometry`` derives edge vectors, spherical harmonics and
    the source-image rotations from the cached graph tensors, which are stored
    as float32.  Casting only the resulting features to float64 leaves ~1e-7
    noise in D(R) and Y_l, so a float64 commutation test must promote the inputs
    themselves.  The float32 pass is left untouched and reports production
    precision.
    """
    if dtype == torch.float32:
        return data
    out = data.clone()
    for name in GEOMETRY_FIELDS:
        v = getattr(out, name, None)
        if torch.is_tensor(v) and v.is_floating_point():
            setattr(out, name, v.to(dtype))
    return out


def block_geometry(model, data, dtype):
    e = model.edge_embedding(data.geo_edge_distance.to(dtype))
    edge_sh, wd = precompute_batch_geometry(data, model.high_irreps, model.lmax)
    edge_sh = edge_sh.to(dtype)
    wd = ({k: v.to(dtype) for k, v in wd.items()} if isinstance(wd, dict)
          else wd.to(dtype))
    agg = model.blocks[0].compute_agg_norm(
        data.geo_edge_index[0], data.num_nodes).to(dtype)
    return e, edge_sh, wd, agg


def forward_layers(model, data, h, project=None):
    """Run every dynamic_tp block, returning the state after each one."""
    e, edge_sh, wd, agg = block_geometry(model, data, h.dtype)
    states = [h.clone()]
    cur = h
    for blk in model.blocks:
        cur = blk(cur, e, data.geo_edge_index, edge_sh, wd, agg)
        if project is not None:
            cur = project(cur)
        states.append(cur.clone())
    return states


# ---------------------------------------------------------------------------
# per-structure run
# ---------------------------------------------------------------------------
def sector_blocks(high_irreps, scalar_mul):
    blocks = [("0e", slice(0, scalar_mul))]
    off = scalar_mul
    for mul, ir in high_irreps:
        blocks.append((str(ir), slice(off, off + mul * ir.dim)))
        off += mul * ir.dim
    return blocks


def run_structure(model, dq, dp, mid, dtype, modes):
    rows = []
    hi, sm_ = model.high_irreps, model.scalar_mul
    K, M = dq.num_nodes, dp.num_nodes
    if K == M:
        return rows
    a2o = dq.atom_to_orbit.long()
    sg = int(dq.space_group.item())
    blocks = sector_blocks(hi, sm_)

    proj, pa = build_projectors(dq, hi)
    proj_prod, prod_hits = load_production_projectors(dq, hi)
    coord_res, z_res, lat_res = atom_correspondence(dq, dp)
    rot_store_res = stored_vs_derived_rotations(dq)
    rep = representative_atoms(dq)
    ea = edge_audit_cached(dq, dp, mid)
    agg_diff, agg_deg_mismatch = agg_norm_audit(model, dq, dp, rep)
    ranks = pa["rank"]
    klass = structure_class(pa)

    torch.manual_seed(0)
    z = torch.randn(K, sm_ + hi.dim, dtype=torch.float64)

    for mode in modes:
        h0_64 = z.clone() if mode == "unprojected" else apply_projector(
            z.clone(), hi, proj, sm_)
        h0 = h0_64.to(dtype)
        gauge = gauge_violation(dq, h0_64, hi, sm_)
        prod_res = float("nan")
        if mode != "unprojected":
            hp = apply_projector(z.clone(), hi, proj_prod, sm_)
            prod_res = (float((hp - h0_64).norm())
                        / max(float(h0_64.norm()), EPS))

        step = ((lambda x: apply_projector(x, hi, proj, sm_))
                if mode == "projected_each" else None)
        st_q = forward_layers(model, promote(dq, dtype), h0, project=step)
        st_F = forward_layers(model, promote(dp, dtype), lift(dq, h0, hi, sm_))

        for layer in range(len(st_q)):
            lifted = lift(dq, st_q[layer], hi, sm_)
            hF = st_F[layer]
            denom = max(float(hF.norm()), float(lifted.norm()), EPS)
            overall = float((hF - lifted).norm()) / denom
            # Theorem 1 is a statement about the orbit representative: the
            # quotient node carries that site's state and every other image is
            # obtained by the exact group action.  The all-atom figure below
            # additionally absorbs how faithfully the float32 full-atom cache
            # reproduces its own symmetry, so both are reported.
            rmask = dq.atom_image_index.long() == 0
            rep_denom = max(float(hF[rmask].norm()), float(lifted[rmask].norm()), EPS)
            rep_overall = float((hF[rmask] - lifted[rmask]).norm()) / rep_denom
            sec, sec_rep, active = {}, {}, {}
            for name, sl in blocks:
                a, b = hF[:, sl], lifted[:, sl]
                den = max(float(a.norm()), float(b.norm()), EPS)
                sec[name] = (float((a - b).norm()) / den,
                             float((a - b).norm()), float(a.norm()))
                ar, br = a[rmask], b[rmask]
                denr = max(float(ar.norm()), float(br.norm()), EPS)
                sec_rep[name] = float((ar - br).norm()) / denr
                # A sector the site symmetry forbids is driven to zero by P_q,
                # so its relative error is 0/0.  Flag it instead of reporting a
                # meaningless ratio.
                active[name] = den > 1e-6 * float(hF.norm())
            fix_res = float("nan")
            if layer > 0:
                hq = st_q[layer].double()
                fix_res = (float((apply_projector(hq, hi, proj, sm_) - hq).norm())
                           / max(float(hq.norm()), EPS))
            worst_q, worst_v = -1, 0.0
            for q in range(K):
                m = a2o == q
                v = float((hF[m] - lifted[m]).norm())
                if v > worst_v:
                    worst_v, worst_q = v, q
            rep_err = max((float((hF[a0] - lifted[a0]).norm())
                           for a0 in rep.values()), default=0.0)
            tol = TOL_F64 if dtype == torch.float64 else TOL_F32
            reasons = []
            if rep_overall >= tol:
                reasons.append("rep_rel_error_above_threshold")
            if ea["n_fail"]:
                reasons.append("assumption_ii_edge_bijection")
            if ea["worst_rot"] > ROT_TOL:
                reasons.append("assumption_iii_transport")
            if coord_res > COORD_TOL or z_res:
                reasons.append("assumption_i_atom_correspondence")
            if pa["idem"] > 1e-10:
                reasons.append("assumption_iv_projector")
            if mode != "unprojected" and gauge > 1e-9:
                reasons.append("assumption_v_gauge")
            rows.append(dict(
                material_id=mid, space_group=sg,
                crystal_system=crystal_system(sg), stab_class=klass,
                dtype="float64" if dtype == torch.float64 else "float32",
                mode=mode, layer=layer, K=K, M=M, compression=round(M / K, 4),
                stab_order_max=max(pa["stab_order"]),
                stab_order_min=min(pa["stab_order"]),
                rank_1o_max=max(ranks["1o"]), rank_1o_min=min(ranks["1o"]),
                rank_2e_max=max(ranks["2e"]), rank_2e_min=min(ranks["2e"]),
                proj_idem=pa["idem"], proj_sym=pa["sym"],
                proj_invariance=pa["invariance"],
                stab_orthogonality=pa["orthogonality"],
                rot_store_residual=rot_store_res,
                prod_proj_residual=prod_res, prod_proj_hits=prod_hits,
                atom_coord_residual=coord_res, atom_species_mismatch=z_res,
                lattice_residual=lat_res,
                edge_reps_ok=ea["n_ok"], edge_reps_fail=ea["n_fail"],
                edge_worst_disp=ea["worst_disp"], edge_worst_rot=ea["worst_rot"],
                orbit_image_edge_residual=ea["orbit_resid"],
                orbit_image_edge_fail=ea["orbit_fail"],
                edge_dist_dev=ea["dist_dev"], agg_norm_diff=agg_diff,
                agg_degree_mismatch=agg_deg_mismatch,
                gauge_violation=gauge, fix_residual=fix_res,
                feature_norm=float(hF.norm()),
                rep_rel=rep_overall,
                rep_err_0e=sec_rep["0e"], rep_err_1o=sec_rep["1o"],
                rep_err_2e=sec_rep["2e"],
                err_0e=sec["0e"][0], err_1o=sec["1o"][0], err_2e=sec["2e"][0],
                abs_0e=sec["0e"][1], abs_1o=sec["1o"][1], abs_2e=sec["2e"][1],
                norm_0e=sec["0e"][2], norm_1o=sec["1o"][2], norm_2e=sec["2e"][2],
                active_1o=int(active["1o"]), active_2e=int(active["2e"]),
                overall_rel=overall,
                max_abs_error=float((hF - lifted).abs().max()),
                worst_orbit=worst_q, worst_orbit_err=worst_v,
                rep_atom_err=rep_err,
                passed=int(not reasons), failure_reason=";".join(reasons),
            ))
    return rows


# ---------------------------------------------------------------------------
# stratified structure selection
# ---------------------------------------------------------------------------
def structure_class(pa):
    """Comma-joined stabilizer classes present in this structure."""
    orders = pa["stab_order"]
    r1, r2 = pa["rank"]["1o"], pa["rank"]["2e"]
    cls = set()
    if max(orders) == 1:
        cls.add(1)
    if any(o > 1 and r1[q] > 0 for q, o in enumerate(orders)):
        cls.add(2)
    if any(o > 1 and r2[q] > 0 for q, o in enumerate(orders)):
        cls.add(3)
    if any(r1[q] == 0 for q in range(len(orders))):
        cls.add(4)
    return ",".join(str(c) for c in sorted(cls)) or "0"


def scan_stabilizers(reader_q, high_irreps, per_system=60, refresh=False):
    """Classify compressed structures by site stabilizer; cached to JSON."""
    if SCAN_CACHE.exists() and not refresh:
        return json.loads(SCAN_CACHE.read_text())
    man = [json.loads(l) for l in open(f"{WYCKOFF_DIR}/manifest.jsonl")]
    man = [m for m in man if m["num_atoms"] > m["num_orbits"]]
    by_cs = collections.defaultdict(list)
    for m in man:
        by_cs[crystal_system(int(m["space_group"]))].append(m)
    rng = np.random.default_rng(0)
    out = []
    for cs in SYSTEMS:
        cand = by_cs.get(cs, [])
        if not cand:
            continue
        idx = rng.permutation(len(cand))[:per_system]
        for i in idx:
            m = cand[int(i)]
            try:
                dq = light_dict_to_pyg_data(reader_q.get(m["material_id"]))
            except Exception:
                continue
            _, pa = build_projectors(dq, high_irreps)
            out.append(dict(
                material_id=m["material_id"], space_group=int(m["space_group"]),
                crystal_system=cs, K=int(dq.num_nodes),
                M=int(dq.atom_to_orbit.shape[0]),
                compression=float(m["compression_ratio"]),
                stab_order_max=max(pa["stab_order"]),
                rank_1o_max=max(pa["rank"]["1o"]),
                rank_2e_max=max(pa["rank"]["2e"]),
                rank_1o_min=min(pa["rank"]["1o"]),
                stab_class=structure_class(pa)))
        print(f"  scanned {cs}: {len(out)} total")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SCAN_CACHE.write_text(json.dumps(out, indent=1))
    return out


def select_structures(scan, per_system=5, require_classes=(2, 3)):
    """Pick >= per_system per crystal system, class 2 and 3 guaranteed present."""
    by_cs = collections.defaultdict(list)
    for r in scan:
        if r["M"] > r["K"]:
            by_cs[r["crystal_system"]].append(r)
    picked = []
    for cs in SYSTEMS:
        cand = sorted(by_cs.get(cs, []), key=lambda r: (-r["compression"], -r["M"]))
        chosen = []
        for c in require_classes:
            for r in cand:
                if str(c) in r["stab_class"].split(",") and r not in chosen:
                    chosen.append(r)
                    break
        for r in cand:
            if len(chosen) >= per_system:
                break
            if r not in chosen:
                chosen.append(r)
        picked += chosen
    return picked


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def summarize(rows):
    summary = {}
    for dtype in ("float32", "float64"):
        for mode in ("unprojected", "projected_input", "projected_each"):
            for layer in (1, 2):
                sub = [r for r in rows if r["dtype"] == dtype
                       and r["mode"] == mode and r["layer"] == layer]
                if not sub:
                    continue
                entry = {"n": len(sub)}
                for sec in ("0e", "1o", "2e", "rep_0e", "rep_1o", "rep_2e",
                            "all", "rep"):
                    key = {"all": "overall_rel", "rep": "rep_rel"}.get(
                        sec, f"err_{sec}" if not sec.startswith("rep_")
                        else f"rep_err_{sec[4:]}")
                    act = ("1o" if "1o" in sec else
                           "2e" if "2e" in sec else None)
                    src = ([r for r in sub if r[f"active_{act}"]] if act else sub)
                    v = np.array([r[key] for r in src], dtype=float)
                    v = v[~np.isnan(v)]
                    if len(v):
                        entry[sec] = dict(
                            median=float(np.median(v)),
                            p95=float(np.percentile(v, 95)),
                            max=float(v.max()),
                            n_active=len(src),
                            argmax=src[int(np.argmax(
                                [r[key] for r in src]))]["material_id"])
                tol = TOL_F64 if dtype == "float64" else TOL_F32
                entry["tolerance"] = tol
                entry["n_pass_representative"] = int(
                    sum(r["rep_rel"] < tol for r in sub))
                entry["n_pass_all_atoms"] = int(
                    sum(r["overall_rel"] < tol for r in sub))
                summary[f"{dtype}|{mode}|L{layer}"] = entry
    l0 = [r for r in rows if r["layer"] == 0 and r["dtype"] == "float64"
          and r["mode"] == "projected_each"]
    summary["assumptions"] = dict(
        n_structures=len({r["material_id"] for r in rows}),
        atom_coord_residual_max=max((r["atom_coord_residual"] for r in l0), default=0),
        species_mismatch_total=int(sum(r["atom_species_mismatch"] for r in l0)),
        lattice_residual_max=max((r["lattice_residual"] for r in l0), default=0),
        edge_bijection_rate=float(np.mean([r["edge_reps_fail"] == 0 for r in l0]))
        if l0 else 0.0,
        edge_reps_failed_total=int(sum(r["edge_reps_fail"] for r in l0)),
        edge_worst_disp=max((r["edge_worst_disp"] for r in l0), default=0),
        edge_worst_transport_rot=max((r["edge_worst_rot"] for r in l0), default=0),
        orbit_image_edge_residual_max=max(
            (r["orbit_image_edge_residual"] for r in l0), default=0),
        orbit_image_edge_fail_total=int(
            sum(r["orbit_image_edge_fail"] for r in l0)),
        edge_dist_dev_max=max((r["edge_dist_dev"] for r in l0), default=0),
        agg_norm_diff_max=max((r["agg_norm_diff"] for r in l0), default=0),
        agg_degree_mismatch_total=int(sum(r["agg_degree_mismatch"] for r in l0)),
        projector_idem_max=max((r["proj_idem"] for r in l0), default=0),
        projector_sym_max=max((r["proj_sym"] for r in l0), default=0),
        projector_invariance_max=max((r["proj_invariance"] for r in l0), default=0),
        stab_orthogonality_max=max((r["stab_orthogonality"] for r in l0), default=0),
        rot_store_residual_max=max((r["rot_store_residual"] for r in l0), default=0),
        gauge_violation_max=max((r["gauge_violation"] for r in l0), default=0),
        prod_proj_residual_max=max(
            (r["prod_proj_residual"] for r in l0
             if not np.isnan(r["prod_proj_residual"])), default=float("nan")),
        forbidden_1o_sector_norm_max=max(
            (r["norm_1o"] for r in rows
             if r["mode"] != "unprojected" and r["rank_1o_max"] == 0),
            default=None),
        forbidden_1o_structures=sorted({
            r["material_id"] for r in rows if r["rank_1o_max"] == 0}),
        class_coverage=dict(collections.Counter(r["stab_class"] for r in l0)),
        systems=dict(collections.Counter(r["crystal_system"] for r in l0)),
    )
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--scan-only", action="store_true")
    ap.add_argument("--refresh-scan", action="store_true")
    ap.add_argument("--per-system", type=int, default=5)
    ap.add_argument("--max-structs", type=int, default=0)
    ap.add_argument("--num-layers", type=int, default=2)
    args = ap.parse_args()

    reader_q, reader_p = LMDBReader(WYCKOFF_DIR), LMDBReader(P1_DIR)
    hi = make_model().high_irreps
    print("scanning stabilizers ...")
    scan = scan_stabilizers(reader_q, hi, refresh=args.refresh_scan)
    print(f"scan: {len(scan)} structures, classes "
          f"{dict(collections.Counter(r['stab_class'] for r in scan))}")
    if args.scan_only:
        return 0

    sel = select_structures(scan, per_system=args.per_system if args.full else 1)
    if args.max_structs:
        sel = sel[: args.max_structs]
    print(f"selected {len(sel)} structures:")
    for r in sel:
        print(f"  {r['material_id']:16s} {r['crystal_system']:12s} SG{r['space_group']:<4d}"
              f" K={r['K']:<4d} M={r['M']:<5d} class={r['stab_class']:8s}"
              f" |H|max={r['stab_order_max']:<3d} rank1o={r['rank_1o_max']}"
              f" rank2e={r['rank_2e_max']}")

    all_rows = []
    for r in sel:
        mid = r["material_id"]
        try:
            dq, dp = load_pair(reader_q, reader_p, mid)
        except Exception as exc:
            print(f"  {mid}: load failed {exc}")
            continue
        if dq.num_nodes == dp.num_nodes:
            continue
        for dtype in (torch.float32, torch.float64):
            model = make_model(args.num_layers, dtype)
            rows = run_structure(model, dq, dp, mid, dtype,
                                 ["unprojected", "projected_input", "projected_each"])
            all_rows += rows
            tag = "f32" if dtype == torch.float32 else "f64"
            for mode in ("unprojected", "projected_each"):
                x = [w for w in rows if w["mode"] == mode
                     and w["layer"] == args.num_layers]
                if x:
                    x = x[0]
                    print(f"  {mid} {tag} {mode:16s} rep={x['rep_rel']:.2e} "
                          f"all={x['overall_rel']:.2e} | rep 0e={x['rep_err_0e']:.2e} "
                          f"1o={x['rep_err_1o']:.2e} 2e={x['rep_err_2e']:.2e} | "
                          f"edge={x['edge_reps_ok']}/"
                          f"{x['edge_reps_ok']+x['edge_reps_fail']} "
                          f"imgres={x['orbit_image_edge_residual']:.1e}")
            del model

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if all_rows:
        with open(OUT_DIR / "projected_commutation_details.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()),
                               extrasaction="ignore")
            w.writeheader()
            w.writerows(all_rows)
    summary = summarize(all_rows)
    with open(OUT_DIR / "projected_commutation_summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    print("\n=== summary ===")
    for k, v in summary.items():
        if k == "assumptions":
            continue
        print(f"{k}: n={v['n']} rep median={v['rep']['median']:.3e} "
              f"max={v['rep']['max']:.3e} pass={v['n_pass_representative']}/{v['n']}"
              f"  | all-atom median={v['all']['median']:.3e} "
              f"max={v['all']['max']:.3e} pass={v['n_pass_all_atoms']}/{v['n']}")
    print(json.dumps(summary["assumptions"], indent=1))
    print(f"saved -> {OUT_DIR}")
    return 0


# ---------------------------------------------------------------------------
# pytest hooks
# ---------------------------------------------------------------------------
def _pytest_mids(n=3):
    reader_q = LMDBReader(WYCKOFF_DIR)
    scan = scan_stabilizers(reader_q, make_model().high_irreps)
    sel = select_structures(scan, per_system=1)
    nontrivial = [r for r in sel if r["stab_order_max"] > 1]
    return [r["material_id"] for r in (nontrivial or sel)[:n]]


def _rows(mode, dtype, mids=None, layers=2):
    rq, rp = LMDBReader(WYCKOFF_DIR), LMDBReader(P1_DIR)
    out = []
    for mid in (mids or _pytest_mids()):
        dq, dp = load_pair(rq, rp, mid)
        if dq.num_nodes == dp.num_nodes:
            continue
        out += run_structure(make_model(layers, dtype), dq, dp, mid, dtype, [mode])
    return out


def test_assumption_i_atom_correspondence():
    rq, rp = LMDBReader(WYCKOFF_DIR), LMDBReader(P1_DIR)
    for mid in _pytest_mids():
        dq, dp = load_pair(rq, rp, mid)
        coord, z, lat = atom_correspondence(dq, dp)
        assert coord < COORD_TOL, f"{mid}: coord residual {coord:.2e}"
        assert z == 0, f"{mid}: species mismatch"
        assert lat < 1e-6, f"{mid}: lattice mismatch {lat:.2e}"


def test_assumption_ii_edge_bijection():
    rq, rp = LMDBReader(WYCKOFF_DIR), LMDBReader(P1_DIR)
    for mid in _pytest_mids():
        dq, dp = load_pair(rq, rp, mid)
        ea = edge_audit(dq, dp)
        assert ea["n_fail"] == 0, f"{mid}: {ea['n_fail']} reps, {ea['failures'][:2]}"


def test_assumption_iii_transport_consistency():
    rq, rp = LMDBReader(WYCKOFF_DIR), LMDBReader(P1_DIR)
    for mid in _pytest_mids():
        dq, dp = load_pair(rq, rp, mid)
        ea = edge_audit(dq, dp)
        assert ea["worst_rot"] < ROT_TOL, f"{mid}: rot {ea['worst_rot']:.2e}"


def test_assumption_iv_projector_is_a_projector():
    rq = LMDBReader(WYCKOFF_DIR)
    hi = make_model().high_irreps
    for mid in _pytest_mids():
        dq = light_dict_to_pyg_data(rq.get(mid))
        _, pa = build_projectors(dq, hi)
        assert pa["idem"] < 1e-10
        assert pa["sym"] < 1e-10
        assert pa["invariance"] < 1e-10


def test_assumption_v_projected_state_is_stabilizer_fixed():
    rq = LMDBReader(WYCKOFF_DIR)
    model = make_model()
    hi, sm_ = model.high_irreps, model.scalar_mul
    for mid in _pytest_mids():
        dq = light_dict_to_pyg_data(rq.get(mid))
        proj, _ = build_projectors(dq, hi)
        torch.manual_seed(0)
        z = torch.randn(dq.num_nodes, sm_ + hi.dim, dtype=torch.float64)
        h = apply_projector(z, hi, proj, sm_)
        assert gauge_violation(dq, h, hi, sm_) < 1e-9
        assert gauge_violation(dq, z, hi, sm_) > 1e-3    # control: unprojected is not


def test_aggregation_norm_matches():
    rq, rp = LMDBReader(WYCKOFF_DIR), LMDBReader(P1_DIR)
    model = make_model()
    for mid in _pytest_mids():
        dq, dp = load_pair(rq, rp, mid)
        diff, mismatch = agg_norm_audit(model, dq, dp, representative_atoms(dq))
        assert mismatch == 0, f"{mid}: {mismatch} representatives with wrong degree"
        assert diff < 1e-9


def test_single_layer_commutation_float64():
    r = [x for x in _rows("projected_input", torch.float64) if x["layer"] == 1]
    assert r and max(x["rep_rel"] for x in r) < TOL_F64


def test_multilayer_commutation_float64():
    r = [x for x in _rows("projected_input", torch.float64) if x["layer"] == 2]
    assert r and max(x["rep_rel"] for x in r) < TOL_F64


def test_multilayer_commutation_float32():
    r = [x for x in _rows("projected_input", torch.float32) if x["layer"] == 2]
    assert r and max(x["rep_rel"] for x in r) < TOL_F32


def test_reprojection_residual_equals_fix_space_residual():
    """Mode C's deviation is exactly how far the block output leaves Fix_{H_q}.

    Re-applying P_q inside the quotient branch moves the state by the Fix-space
    residual, which the unprojected full-atom branch cannot follow.  The two
    quantities must agree, which closes the accounting for mode C instead of
    leaving an unexplained ~1e-10 term.  The identity only has content once the
    Fix residual dominates mode B's own accumulation error, which excludes
    generic sites where P_q is the identity and the residual is exactly zero.
    """
    mids = _pytest_mids()
    b = {(x["material_id"], x["layer"]): x
         for x in _rows("projected_input", torch.float64, mids)}
    c = {(x["material_id"], x["layer"]): x
         for x in _rows("projected_each", torch.float64, mids)}
    checked = 0
    for key, cb in b.items():
        fix, base = cb["fix_residual"], cb["rep_rel"]
        if key[1] == 0 or fix < 10 * max(base, 1e-30):
            continue
        rep = c[key]["rep_rel"]
        assert abs(fix - rep) <= 1e-2 * max(fix, rep), (
            f"{key}: fix={fix:.3e} vs mode-C rep={rep:.3e}")
        checked += 1
    assert checked, "no structure with a dominant Fix residual was exercised"


def test_all_atom_residual_is_bounded_by_cache_precision():
    """Non-representative images inherit the float32 coordinate storage error.

    The cached full-atom cell reproduces its own space-group symmetry only to
    single precision, so the whole-orbit comparison carries a floor that the
    representative comparison does not.  Pin it so a real regression in the
    lift or the transport would still show up.
    """
    r = [x for x in _rows("projected_input", torch.float64) if x["layer"] == 2]
    assert max(x["orbit_image_edge_residual"] for x in r) < 1e-4      # Angstrom
    assert max(x["overall_rel"] for x in r) < 1e-7
    assert max(x["rep_rel"] for x in r) < TOL_F64


def test_high_l_sectors_are_nontrivial():
    """Guard: the audit must not pass because 1o/2e project to zero."""
    r = [x for x in _rows("projected_each", torch.float64) if x["layer"] == 2]
    assert max(x["norm_1o"] for x in r) > 1e-3
    assert max(x["norm_2e"] for x in r) > 1e-3
    assert max(x["stab_order_max"] for x in r) > 1
    assert any(x["active_1o"] and x["stab_order_max"] > 1 for x in r)
    assert any(x["active_2e"] and x["stab_order_max"] > 1 for x in r)


def test_unprojected_is_a_valid_negative_control():
    """Without P_q the coset representative is not fixed, so depth >= 2 breaks.

    At depth 1 the source-image transport already reproduces the full-atom state
    exactly whether or not h_q is H_q-fixed, so the control has to be read at
    layer 2, where the gauge ambiguity of the unprojected state shows up.  Only
    structures where the projection is non-vacuous count, measured directly by
    the unprojected state's Fix-space residual; on generic sites P_q is the
    identity and there is nothing to control for.
    """
    mids = _pytest_mids()
    un = {x["material_id"]: x
          for x in _rows("unprojected", torch.float64, mids) if x["layer"] == 2}
    pr = {x["material_id"]: x
          for x in _rows("projected_input", torch.float64, mids) if x["layer"] == 2}
    full_dim = {"1o": 3, "2e": 5}
    checked = 0
    for mid, u in un.items():
        if u["fix_residual"] < 1e-2:
            continue
        p = pr[mid]
        assert u["rep_rel"] > 1e3 * p["rep_rel"], mid
        assert u["overall_rel"] > 1e3 * p["overall_rel"], mid
        for sector, dim in full_dim.items():
            # A sector is only controlled for when P_q actually constrains it:
            # an inversion-only stabilizer leaves 2e untouched (D^2(-I) = +I),
            # and an annihilated sector has a 0/0 ratio.
            if u[f"rank_{sector}_min"] >= dim or not u[f"active_{sector}"]:
                continue
            if not p[f"active_{sector}"]:
                continue
            assert u[f"err_{sector}"] > 1e2 * p[f"err_{sector}"], (
                f"{mid} {sector}: {u[f'err_{sector}']:.2e} vs "
                f"{p[f'err_{sector}']:.2e}")
        checked += 1
    assert checked, "no structure with a non-vacuous projection was exercised"


def test_site_forbidden_sector_is_annihilated():
    """rank(P_q^{1o}) == 0 must drive the whole 1o sector to zero and keep it there.

    This is Neumann's principle at the site level, and it is what makes the
    class-4 structures a meaningful part of the audit rather than a way to pass
    it trivially.
    """
    rq = LMDBReader(WYCKOFF_DIR)
    scan = scan_stabilizers(rq, make_model().high_irreps)
    mids = [r["material_id"] for r in scan if r["rank_1o_max"] == 0][:2]
    if not mids:
        return
    rows = [x for x in _rows("projected_input", torch.float64, mids)
            if x["layer"] == 2]
    assert rows
    for x in rows:
        assert x["norm_1o"] < 1e-6 * x["norm_0e"], (
            f"{x['material_id']}: 1o norm {x['norm_1o']:.3e} not annihilated")
        assert not x["active_1o"]
        assert x["rep_rel"] < TOL_F64


def test_production_projector_bank_matches_reynolds():
    """The shipped bank must equal the Reynolds projector -- outside two-origin SGs.

    The bank is keyed ``SG{n}_{letter}_l{l}_{e|o}``, which does not carry the ITC
    origin choice.  For the 24 space groups that ITC lists with origin choice 1
    and 2 the same Wyckoff letter denotes a different site, so joining the bank
    against an spglib-standardized cell returns another site's projector.  This
    test pins that scope: exact agreement everywhere else, and the mismatch must
    stay confined to ORIGIN_CHOICE_SG.  A regression widens the affected set; a
    fix empties it, and either way the test changes rather than staying silent.

    It also guards the key format -- ``o3.Irrep.p`` is the integer +-1, so
    formatting it directly gives ``l1_-1``, every lookup misses, and the
    comparison degenerates into "Reynolds vs identity".
    """
    rq = LMDBReader(WYCKOFF_DIR)
    hi = make_model().high_irreps
    scan = scan_stabilizers(rq, hi)
    mids = ([r["material_id"] for r in scan
             if r["space_group"] in ORIGIN_CHOICE_SG][:3]
            + [r["material_id"] for r in scan
               if r["space_group"] not in ORIGIN_CHOICE_SG][:5])
    clean, affected = [], []
    for mid in mids:
        dq = light_dict_to_pyg_data(rq.get(mid))
        sg = int(dq.space_group.item())
        reynolds_p, _ = build_projectors(dq, hi)
        bank_p, hits = load_production_projectors(dq, hi)
        assert hits == dq.num_nodes * len(list(hi)), (
            f"{mid}: only {hits} blocks resolved -- key format regression?")
        worst = 0.0
        for _, ir in hi:
            for q in range(dq.num_nodes):
                a, b = reynolds_p[str(ir)][q], bank_p[str(ir)][q]
                worst = max(worst, float((a - b).norm())
                            / max(float(b.norm()), 1.0))
        (affected if sg in ORIGIN_CHOICE_SG else clean).append((mid, sg, worst))
    assert clean, "no unaffected space group exercised"
    for mid, sg, worst in clean:
        assert worst < 1e-5, f"{mid} SG{sg}: bank vs Reynolds {worst:.3e}"
    if affected:
        # documented, still open: see docs/projected_quotient_commutation_audit.md
        assert any(w > 1e-5 for _, _, w in affected), (
            "two-origin space groups now agree with the bank -- if the lookup was "
            "fixed, drop ORIGIN_CHOICE_SG from this test")


def test_reynolds_projector_is_invariant_under_real_site_symmetry():
    """Whatever the bank says, the projector the audit uses must be correct here.

    This is the property that actually matters: P must be invariant under the
    stabilizer of the site as it appears in *this* crystal.  It holds for the
    Reynolds projector on every space group, including the two-origin ones.
    """
    rq = LMDBReader(WYCKOFF_DIR)
    hi = make_model().high_irreps
    scan = scan_stabilizers(rq, hi)
    mids = ([r["material_id"] for r in scan
             if r["space_group"] in ORIGIN_CHOICE_SG][:2]
            + [r["material_id"] for r in scan
               if r["space_group"] not in ORIGIN_CHOICE_SG][:2])
    worst, n = 0.0, 0
    for mid in mids:
        dq = light_dict_to_pyg_data(rq.get(mid))
        R, mask = stabilizer_cart(dq)
        proj, _ = build_projectors(dq, hi)
        for q in range(dq.num_nodes):
            Rq = R[q][mask[q]]
            if Rq.shape[0] <= 1:
                continue
            for _, ir in hi:
                P = proj[str(ir)][q]
                if float(P.norm()) < 1e-12:
                    continue
                D = wigner_d(Rq, ir)
                worst = max(worst, float((D @ P - P).abs().max()))
                n += 1
    assert n
    assert worst < 1e-9, f"Reynolds projector not site-invariant: {worst:.3e}"


if __name__ == "__main__":
    sys.exit(main())
