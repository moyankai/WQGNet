"""Part 1: Periodic neighbour recovery (Table 1 row 1).

Verifies that the quotient graph's sub-edge displacement vectors exactly
match the matched P1 full-atom graph's edge displacement vectors for each
representative orbit/atom pair.

Reference equations:
  - eq:subedge_vector (main.tex L506-L512)
  - eq:quotient_neighbour_multiset (main.tex L516-L522)

Procedure:
  For each canonical structure:
    1. Build quotient graph (WyckoffGraphBuilder, cutoff=5.0).
    2. Build matched P1 graph (_p1_fallback + WyckoffGraphBuilder, cutoff=5.0).
       **Critical**: P1 uses the SAME standardized conventional cell as quotient.
    3. For each representative orbit p:
       - Collect all Cartesian displacements r_alpha from quotient side.
       - Collect all Cartesian displacements from matched P1 side for the
         corresponding standardized atom (image_index=0).
       - Sort both multisets and compare.
    4. Report max vector error, duplicate count, missing count.

Expected: float64 numeric noise (~1e-13 to 1e-15).
"""

from __future__ import annotations

import numpy as np
import torch
from typing import Dict, List, Tuple

from wyckoff_gnn.data.crystal_to_wyckoff import (
    structure_to_wyckoff_orbits,
    _p1_fallback,
)
from wyckoff_gnn.data.wyckoff_graph import WyckoffGraphBuilder

from .canonical_structures import CANONICAL_STRUCTURES, EXPECTED_SG


def _round_sort_key(vec: np.ndarray, decimals: int = 6) -> tuple:
    """Lexicographic sort key for a 3-vector."""
    norm = np.linalg.norm(vec)
    return (round(norm, decimals),) + tuple(round(v, decimals) for v in vec)


def _collect_quotient_neighbors(
    data, orbit_idx: int, lattice: np.ndarray
) -> List[np.ndarray]:
    """Collect all Cartesian displacement vectors incident on orbit p.

    Convention: geo_edge_index[0] = target (receiver), [1] = source.
    vec = A @ (source_frac + shift - target_rep_frac)
    """
    edge_index = data.geo_edge_index
    source_frac = data.geo_edge_source_frac
    shift = data.geo_edge_shift
    rep_frac = data.orbit_rep_frac

    target_mask = edge_index[0] == orbit_idx
    if not target_mask.any():
        return []

    src_f = source_frac[target_mask].numpy()  # (E_p, 3)
    sh = shift[target_mask].numpy()  # (E_p, 3)
    tgt_f = rep_frac[orbit_idx].numpy()  # (3,)

    vec_frac = src_f + sh - tgt_f[None, :]  # (E_p, 3)
    vec_cart = vec_frac @ lattice  # (E_p, 3)

    return [vec_cart[i] for i in range(len(vec_cart))]


def _collect_p1_neighbors(
    data, atom_idx: int, lattice: np.ndarray
) -> List[np.ndarray]:
    """Collect all displacement vectors where atom_idx is the target.

    P1 graph convention (same as quotient):
      geo_edge_index[0] = target (receiver), [1] = source.
      vec = A @ (source_frac + shift - target_rep_frac)

    For matched P1, each atom is its own orbit, so atom_idx == orbit_idx.
    """
    edge_index = data.geo_edge_index
    source_frac = data.geo_edge_source_frac
    shift = data.geo_edge_shift
    rep_frac = data.orbit_rep_frac

    target_mask = edge_index[0] == atom_idx
    if not target_mask.any():
        return []

    src_f = source_frac[target_mask].numpy()  # (E_p, 3)
    sh = shift[target_mask].numpy()  # (E_p, 3)
    tgt_f = rep_frac[atom_idx].numpy()  # (3,)

    vec_frac = src_f + sh - tgt_f[None, :]  # (E_p, 3)
    vec_cart = vec_frac @ lattice  # (E_p, 3)

    return [vec_cart[i] for i in range(len(vec_cart))]


def _find_atom0_for_orbit(
    meta: dict, orbit_idx: int
) -> int:
    """Find the atom index of image a=0 for orbit p."""
    atom_to_orbit = meta["atom_to_orbit"]
    atom_image_index = meta["atom_image_index"]
    mask = (atom_to_orbit == orbit_idx) & (atom_image_index == 0)
    indices = np.where(mask)[0]
    if len(indices) == 0:
        return -1
    return int(indices[0])


def _compare_multisets(
    vecs_q: List[np.ndarray],
    vecs_f: List[np.ndarray],
    tol: float = 1e-6,
) -> Dict:
    """Compare two neighbor multisets. Returns error metrics."""
    n_q = len(vecs_q)
    n_f = len(vecs_f)

    if n_q == 0 and n_f == 0:
        return {
            "n_quotient": 0, "n_fullatom": 0,
            "max_vec_err": 0.0, "duplicates": 0, "missing": 0,
            "boundary_events": 0,
        }

    sorted_q = sorted(vecs_q, key=_round_sort_key)
    sorted_f = sorted(vecs_f, key=_round_sort_key)

    max_err = 0.0
    matched_f = set()
    missing = 0
    duplicates = 0

    for vq in sorted_q:
        best_err = float("inf")
        best_j = -1
        for j, vf in enumerate(sorted_f):
            if j in matched_f:
                continue
            err = np.linalg.norm(vq - vf)
            if err < best_err:
                best_err = err
                best_j = j
        if best_j >= 0 and best_err < tol:
            matched_f.add(best_j)
            max_err = max(max_err, best_err)
        else:
            missing += 1

    duplicates = max(0, n_f - len(matched_f))

    boundary_events = sum(
        1 for v in vecs_q if abs(np.linalg.norm(v) - 5.0) < 1e-3
    )

    return {
        "n_quotient": n_q,
        "n_fullatom": n_f,
        "max_vec_err": float(max_err),
        "duplicates": duplicates,
        "missing": missing,
        "boundary_events": boundary_events,
    }


def run_part1(
    cutoff: float = 5.0,
    dtype: torch.dtype = torch.float64,
) -> List[Dict]:
    """Run Part 1 on all canonical structures.

    Returns list of per-structure result dicts.
    """
    torch.set_default_dtype(dtype)
    results = []

    for name, factory in CANONICAL_STRUCTURES.items():
        struct = factory()

        # Build quotient graph
        q_orbits, q_meta = structure_to_wyckoff_orbits(struct, tol=1e-3)
        std_lat = q_meta.get("standardized_lattice", struct.lattice.matrix)

        q_builder = WyckoffGraphBuilder(
            cutoff_radius=cutoff, subedge_aggregation="sum"
        )
        q_data = q_builder.build(
            q_orbits, std_lat,
            atom_to_orbit=q_meta.get("atom_to_orbit"),
            atom_image_index=q_meta.get("atom_image_index"),
        )

        # Build matched P1 graph (same standardized cell!)
        std_positions = q_meta["standardized_positions"]
        std_numbers = q_meta["standardized_numbers"]
        p1_orbits, p1_meta = _p1_fallback(
            struct, std_lat, std_positions, std_numbers
        )

        p1_builder = WyckoffGraphBuilder(
            cutoff_radius=cutoff, subedge_aggregation="sum"
        )
        p1_data = p1_builder.build(
            p1_orbits, std_lat,
            atom_to_orbit=p1_meta.get("atom_to_orbit"),
            atom_image_index=p1_meta.get("atom_image_index"),
        )

        lattice = np.array(std_lat, dtype=np.float64)

        site_results = []
        max_err_all = 0.0

        for p in range(len(q_orbits)):
            atom0 = _find_atom0_for_orbit(q_meta, p)
            if atom0 < 0:
                continue

            vecs_q = _collect_quotient_neighbors(q_data, p, lattice)
            vecs_p1 = _collect_p1_neighbors(p1_data, atom0, lattice)

            site_info = _compare_multisets(vecs_q, vecs_p1)
            site_info["p"] = p
            site_info["wyckoff"] = q_orbits[p].wyckoff_symbol
            site_results.append(site_info)

            max_err_all = max(max_err_all, site_info["max_vec_err"])

        struct_result = {
            "structure": name,
            "sg": EXPECTED_SG.get(name, 0),
            "n_orbits": len(q_orbits),
            "n_p1_atoms": len(p1_orbits),
            "sites": site_results,
            "max_vec_err_all_sites": float(max_err_all),
        }
        results.append(struct_result)

    return results
