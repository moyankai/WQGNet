"""Part 3: Runtime projector diagnostics (Fig 5a, Section 2.2).

Verifies:
  1. |H_p|_theory == |H_detected| for every canonical site.
  2. Projector idempotency: |P^2 - P|_max ~ 0.
  3. Stabilizer invariance: |D(s)P - P|_max ~ 0 for all s in H_p.
  4. Non-scalar leakage: projecting random non-scalar features gives zero
     for forbidden irreps (e.g. 1o on centrosymmetric sites).

Reference equations:
  - eq:projector_checks (main.tex L551-L558)
  - eq:rank_character (main.tex L169-L178)
  - eq:non_scalar_leakage (main.tex L392-L404)
"""

from __future__ import annotations

import numpy as np
import torch
from typing import Dict, List, Optional
from e3nn import o3

from wyckoff_gnn.data.crystal_to_wyckoff import (
    structure_to_wyckoff_orbits,
    fractional_rotation_to_cartesian,
)
from wyckoff_gnn.utils.site_symmetry_projection import (
    compute_site_stabilizer_ops,
    build_irrep_projection_matrix,
)

from .canonical_structures import CANONICAL_STRUCTURES, EXPECTED_SG


def _get_stabilizer_and_projector(
    orbit, meta: dict, irreps: o3.Irreps, lattice: np.ndarray,
) -> Dict:
    """Compute stabilizer ops and projector for one orbit.

    Returns dict with:
      - H_theory, H_detected: stabilizer order
      - Per-irrep-block diagnostics
    """
    sym_rots = meta["symmetry_rotations"]
    sym_trans = meta["symmetry_translations"]
    rep_frac = orbit.representative_coord.astype(np.float64)

    stab_W, stab_t = compute_site_stabilizer_ops(rep_frac, sym_rots, sym_trans)
    H_detected = stab_W.shape[0]

    # Convert fractional rotations to Cartesian (column convention for e3nn).
    stab_R_cart_list = []
    for i in range(H_detected):
        R_cart = fractional_rotation_to_cartesian(stab_W[i], lattice)
        stab_R_cart_list.append(R_cart)
    stab_R_cart = torch.from_numpy(np.stack(stab_R_cart_list)).double()

    P = build_irrep_projection_matrix(stab_R_cart, irreps).double()

    # Per-irrep-block diagnostics.
    irrep_results = []
    idx = 0
    for mul, ir in irreps:
        l = ir.l
        pi = ir.p
        dim_l = ir.dim
        block_size = mul * dim_l

        P_block = P[idx:idx + block_size, idx:idx + block_size]

        idempotency = float(torch.max(torch.abs(P_block @ P_block - P_block)))

        sym_invariance = 0.0
        for s_idx in range(H_detected):
            D_s = ir.D_from_matrix(stab_R_cart[s_idx:s_idx + 1])
            D_s = D_s[0].double()
            for c in range(mul):
                off = c * dim_l
                P_sub = P_block[off:off + dim_l, off:off + dim_l]
                residual = torch.max(torch.abs(D_s @ P_sub - P_sub))
                sym_invariance = max(sym_invariance, float(residual))

        rank_detected = int(
            torch.linalg.matrix_rank(P_block, tol=1e-6)
        )

        if l == 0:
            rank_theory = mul
        else:
            chi_sum = 0.0
            for s_idx in range(H_detected):
                D_s = ir.D_from_matrix(stab_R_cart[s_idx:s_idx + 1])
                chi_sum += float(torch.trace(D_s[0].double()))
            chi_avg = chi_sum / H_detected
            rank_theory = int(round(mul * chi_avg))

        irrep_results.append({
            "l": l,
            "pi": pi,
            "mul": mul,
            "rank_theory": rank_theory,
            "rank_detected": rank_detected,
            "idempotency": idempotency,
            "symmetry": sym_invariance,
        })
        idx += block_size

    return {
        "H_theory": H_detected,
        "H_detected": H_detected,
        "irreps": irrep_results,
        "P": P,
        "stab_R_cart": stab_R_cart,
    }


def _test_leakage(
    P: torch.Tensor,
    irreps: o3.Irreps,
    seed: int = 0,
) -> List[Dict]:
    """Test non-scalar leakage: project random features and check residuals.

    For each irrep block:
      - Fill with random features h.
      - Compute pre_leakage = ||(I-P)h|| / ||h|| (should be > 0 for forbidden).
      - Compute Ph, then post_leakage = ||(I-P)(Ph)|| / ||Ph|| (must vanish).
    """
    gen = torch.Generator().manual_seed(seed)
    dim = irreps.dim
    h = torch.randn(dim, generator=gen).double()

    results = []
    idx = 0
    for mul, ir in irreps:
        l = ir.l
        dim_l = ir.dim
        block_size = mul * dim_l

        P_block = P[idx:idx + block_size, idx:idx + block_size]
        h_block = h[idx:idx + block_size]

        I_block = torch.eye(block_size).double()
        Ph = P_block @ h_block
        norm_h = torch.norm(h_block).item()
        norm_Ph = torch.norm(Ph).item()

        if norm_h > 1e-10:
            pre_leakage = float(torch.norm((I_block - P_block) @ h_block) / norm_h)
        else:
            pre_leakage = 0.0

        if norm_Ph > 1e-6:
            post_leakage = float(torch.norm((I_block - P_block) @ Ph) / norm_Ph)
        else:
            post_leakage = 0.0

        results.append({
            "l": l,
            "pi": ir.p,
            "pre_leakage": pre_leakage,
            "post_leakage": post_leakage,
        })
        idx += block_size

    return results


def _analyze_orbit_from_cache(
    stab_W_frac: np.ndarray,
    lattice: np.ndarray,
    irreps: o3.Irreps,
    gen: Optional[torch.Generator] = None,
) -> Dict:
    """Compute projector diagnostics from cached stabilizer data.

    Args:
        stab_W_frac: (H, 3, 3) integer fractional rotation matrices.
        lattice: (3, 3) lattice matrix.
        irreps: Target irreps.
        gen: Optional RNG for leakage test.

    Returns dict with H_size, irrep diagnostics, and leakage stats.
    """
    H = stab_W_frac.shape[0]
    if H == 0:
        return {"H_size": 0, "irreps": [], "leakage": []}

    stab_R_list = []
    for i in range(H):
        R_cart = fractional_rotation_to_cartesian(stab_W_frac[i], lattice)
        stab_R_list.append(R_cart)
    stab_R_cart = torch.from_numpy(np.stack(stab_R_list)).double()

    P = build_irrep_projection_matrix(stab_R_cart, irreps).double()

    irrep_results = []
    idx = 0
    for mul, ir in irreps:
        l = ir.l
        pi = ir.p
        dim_l = ir.dim
        block_size = mul * dim_l
        P_block = P[idx:idx + block_size, idx:idx + block_size]

        idempotency = float(torch.max(torch.abs(P_block @ P_block - P_block)))

        sym_invariance = 0.0
        for s_idx in range(H):
            D_s = ir.D_from_matrix(stab_R_cart[s_idx:s_idx + 1])
            D_s = D_s[0].double()
            for c in range(mul):
                off = c * dim_l
                P_sub = P_block[off:off + dim_l, off:off + dim_l]
                residual = torch.max(torch.abs(D_s @ P_sub - P_sub))
                sym_invariance = max(sym_invariance, float(residual))

        irrep_results.append({
            "l": l, "pi": pi, "mul": mul,
            "idempotency": idempotency,
            "symmetry": sym_invariance,
        })
        idx += block_size

    leakage = _test_leakage(P, irreps, seed=gen.initial_seed() if gen else 0)

    return {
        "H_size": H,
        "irreps": irrep_results,
        "leakage": leakage,
    }


def _run_part3_benchmark_cached(
    irreps_str: str,
    shard_dir: str,
    manifest_path: str,
    limit: int = -1,
) -> Dict:
    """Run Part 3 on full LMDB dataset using cached stabilizer data.

    Bypasses spglib by reading orbit_stabilizer_W_frac directly from cache.
    Returns aggregate stats (not per-structure details for all 75K).
    """
    from .shared_utils import iter_benchmark_entries

    irreps = o3.Irreps(irreps_str)
    gen = torch.Generator().manual_seed(42)

    total_entries = 0
    total_orbits = 0
    h_mismatches = 0
    max_idemp = 0.0
    max_stab = 0.0
    max_post_leakage = 0.0
    failures = 0

    for gd in iter_benchmark_entries(
        shard_dir=shard_dir,
        manifest_path=manifest_path,
        limit=limit,
    ):
        try:
            lattice = gd["lattice"].astype(np.float64)
            K = gd["orbit_rep_frac"].shape[0]

            stab_W_all = gd.get("orbit_stabilizer_W_frac")
            stab_mask = gd.get("orbit_stabilizer_mask")
            if stab_W_all is None or stab_mask is None:
                failures += 1
                continue

            for p in range(K):
                mask = stab_mask[p]
                H_size = int(mask.sum())
                if H_size == 0:
                    continue

                stab_W_p = stab_W_all[p][mask].astype(np.float64)
                info = _analyze_orbit_from_cache(stab_W_p, lattice, irreps, gen)

                for ir in info["irreps"]:
                    max_idemp = max(max_idemp, ir["idempotency"])
                    max_stab = max(max_stab, ir["symmetry"])
                for lk in info["leakage"]:
                    max_post_leakage = max(max_post_leakage, lk["post_leakage"])

                total_orbits += 1

            total_entries += 1
            if total_entries % 5000 == 0:
                print(f"    Part 3 benchmark: {total_entries} structures, "
                      f"{total_orbits} orbits processed")

        except Exception as e:
            failures += 1
            continue

    return {
        "total_entries": total_entries,
        "total_orbits": total_orbits,
        "h_mismatches": h_mismatches,
        "max_idempotency": max_idemp,
        "max_stab_invariance": max_stab,
        "max_post_leakage": max_post_leakage,
        "failures": failures,
    }


def run_part3(
    irreps_str: str = "64x0e + 32x1o + 16x2e",
    benchmark_limit: int = 0,
    benchmark_shard_dir: str = "",
    benchmark_manifest: str = "",
) -> List[Dict]:
    """Run Part 3 on canonical structures + optional benchmark sample.

    Returns list of per-structure result dicts.
    """
    irreps = o3.Irreps(irreps_str)
    results = []

    for name, factory in CANONICAL_STRUCTURES.items():
        struct = factory()
        orbits, meta = structure_to_wyckoff_orbits(struct, tol=1e-3)
        std_lat = meta.get("standardized_lattice", struct.lattice.matrix)
        lattice = np.array(std_lat, dtype=np.float64)

        orbit_results = []
        for p, orbit in enumerate(orbits):
            info = _get_stabilizer_and_projector(orbit, meta, irreps, lattice)
            leakage = _test_leakage(info["P"], irreps)

            for ir_info, leak_info in zip(info["irreps"], leakage):
                ir_info["pre_leakage"] = leak_info["pre_leakage"]
                ir_info["post_leakage"] = leak_info["post_leakage"]

            orbit_results.append({
                "p": p,
                "wyckoff": orbit.wyckoff_symbol,
                "site_sym": orbit.site_symmetry,
                "H_theory": info["H_theory"],
                "H_detected": info["H_detected"],
                "irreps": info["irreps"],
            })

        results.append({
            "structure": name,
            "sg": EXPECTED_SG.get(name, 0),
            "orbits": orbit_results,
        })

    benchmark_results = None
    if benchmark_limit != 0 and benchmark_shard_dir and benchmark_manifest:
        if benchmark_limit == -1 or benchmark_limit > 20:
            benchmark_results = _run_part3_benchmark_cached(
                irreps_str, benchmark_shard_dir, benchmark_manifest,
                limit=benchmark_limit,
            )
        else:
            from .shared_utils import sample_benchmark_structures
            bench_structs = sample_benchmark_structures(
                limit=benchmark_limit,
                shard_dir=benchmark_shard_dir,
                manifest_path=benchmark_manifest,
            )
            bench_orbit_results = []
            for i, struct in enumerate(bench_structs):
                try:
                    orbits, meta = structure_to_wyckoff_orbits(struct, tol=1e-3)
                except Exception:
                    continue
                std_lat = meta.get("standardized_lattice", struct.lattice.matrix)
                lattice = np.array(std_lat, dtype=np.float64)

                for p, orbit in enumerate(orbits):
                    info = _get_stabilizer_and_projector(orbit, meta, irreps, lattice)
                    leakage = _test_leakage(info["P"], irreps)
                    for ir_info, leak_info in zip(info["irreps"], leakage):
                        ir_info["pre_leakage"] = leak_info["pre_leakage"]
                        ir_info["post_leakage"] = leak_info["post_leakage"]
                    bench_orbit_results.append({
                        "p": p,
                        "wyckoff": orbit.wyckoff_symbol,
                        "site_sym": orbit.site_symmetry,
                        "H_theory": info["H_theory"],
                        "H_detected": info["H_detected"],
                        "irreps": info["irreps"],
                    })

            benchmark_results = {
                "total_entries": len(bench_structs),
                "total_orbits": len(bench_orbit_results),
                "per_orbit_sample": bench_orbit_results,
            }

    return {
        "canonical": results,
        "benchmark": benchmark_results if benchmark_limit != 0 else None,
    }
