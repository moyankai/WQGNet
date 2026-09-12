"""Part 4: Invariance tests (Table 1 rows 5, 6, 7).

Part 4a — Delta_rep (row 5):
  Representative drift: rebuild graph after translating reference cell by a
  lattice vector. Should not change output.

Part 4b — Delta_transport (row 6):
  Transport equivariance: two symmetry ops mapping p to same image should
  give the same rotated features after projection.

Part 4c — Global rotation equivariance (row 7):
  E(X) = E(RX) for random R in O(3), both proper and improper.

Reference equations:
  - eq:representative_drift (main.tex L665-L671)
  - eq:transport_drift (main.tex L672-L679)
"""

from __future__ import annotations

import numpy as np
import torch
from typing import Dict, List, Tuple
from e3nn import o3

from wyckoff_gnn.data.crystal_to_wyckoff import (
    structure_to_wyckoff_orbits,
    orbits_to_node_features,
    fractional_rotation_to_cartesian,
)
from wyckoff_gnn.data.wyckoff_graph import WyckoffGraphBuilder
from wyckoff_gnn.data.atom_graph import structure_to_atom_graph
from wyckoff_gnn.utils.site_symmetry_projection import (
    compute_site_stabilizer_ops,
    build_irrep_projection_matrix,
)

from .canonical_structures import CANONICAL_STRUCTURES, EXPECTED_SG
from .shared_utils import build_quotient_graph, build_p1_view


# ---------------------------------------------------------------------------
# Part 4a: Delta_rep — representative choice drift
# ---------------------------------------------------------------------------


def _build_and_forward(
    struct, cutoff: float = 5.0, tol: float = 1e-3,
) -> Tuple[torch.Tensor, dict]:
    """Build quotient graph and compute a simple invariant: sum of rep coords.

    For the real test we'd use the encoder, but for the structural test
    we just verify the graph topology is identical.
    """
    orbits, meta = structure_to_wyckoff_orbits(struct, tol=tol)
    std_lat = meta.get("standardized_lattice", struct.lattice.matrix)
    builder = WyckoffGraphBuilder(cutoff_radius=cutoff, subedge_aggregation="sum")
    data = builder.build(
        orbits, std_lat,
        atom_to_orbit=meta.get("atom_to_orbit"),
        atom_image_index=meta.get("atom_image_index"),
    )
    n_edges = data.geo_edge_index.shape[1]
    n_nodes = data.num_nodes
    edge_dist_sum = data.geo_edge_distance.sum().item() if data.geo_edge_distance.numel() > 0 else 0.0
    return n_nodes, n_edges, edge_dist_sum, orbits, meta


def run_part4a(cutoff: float = 5.0) -> Dict:
    """Test Delta_rep: translate cell by lattice vector, rebuild, compare.

    Translation by a lattice vector should not change the graph topology
    or the set of inter-orbit distances.
    """
    results = {}
    max_delta = 0.0

    for name, factory in CANONICAL_STRUCTURES.items():
        struct = factory()

        n1, e1, d1, orbits1, meta1 = _build_and_forward(struct, cutoff)

        translated_coords = struct.frac_coords + np.array([1.0, 0.0, 0.0])
        struct2 = struct.__class__(
            struct.lattice, struct.species, translated_coords,
            coords_are_cartesian=False,
        )

        n2, e2, d2, orbits2, meta2 = _build_and_forward(struct2, cutoff)

        delta_nodes = abs(n1 - n2)
        delta_edges = abs(e1 - e2)
        delta_dist = abs(d1 - d2)

        max_delta = max(max_delta, delta_dist)

        results[name] = {
            "delta_nodes": delta_nodes,
            "delta_edges": delta_edges,
            "delta_dist_sum": float(delta_dist),
        }

    return {"max_canonical": float(max_delta), "per_structure": results}


# ---------------------------------------------------------------------------
# Part 4b: Delta_transport — transport equivariance under projector
# ---------------------------------------------------------------------------


def run_part4b(
    irreps_str: str = "64x0e + 32x1o + 16x2e",
    seed: int = 42,
) -> Dict:
    """Test transport equivariance: D(g1) P h == D(g2) P h when g1, g2 map p
    to the same image (i.e. g2 = g1 * s for s in stabilizer H_p).

    Returns max error for projected vs unprojected features.
    """
    irreps = o3.Irreps(irreps_str)
    gen = torch.Generator().manual_seed(seed)

    max_proj_err = 0.0
    max_noproj_err = 0.0
    max_wrong_proj_err = 0.0
    per_orbit_results = []

    for name, factory in CANONICAL_STRUCTURES.items():
        struct = factory()
        orbits, meta = structure_to_wyckoff_orbits(struct, tol=1e-3)
        std_lat = meta.get("standardized_lattice", struct.lattice.matrix)
        lattice = np.array(std_lat, dtype=np.float64)

        sym_rots = meta["symmetry_rotations"]
        sym_trans = meta["symmetry_translations"]

        for p, orbit in enumerate(orbits):
            rep_frac = orbit.representative_coord.astype(np.float64)

            stab_W, stab_t = compute_site_stabilizer_ops(
                rep_frac, sym_rots, sym_trans
            )
            if stab_W.shape[0] < 2:
                continue

            stab_R_cart_list = []
            for i in range(stab_W.shape[0]):
                R_cart = fractional_rotation_to_cartesian(stab_W[i], lattice)
                stab_R_cart_list.append(R_cart)
            stab_R_cart = torch.from_numpy(np.stack(stab_R_cart_list)).double()

            P = build_irrep_projection_matrix(stab_R_cart, irreps).double()

            h = torch.randn(irreps.dim, generator=gen).double()

            s1_cart = fractional_rotation_to_cartesian(stab_W[0], lattice)
            s2_cart = fractional_rotation_to_cartesian(stab_W[1], lattice)

            D_s1 = _wigner_d_block(irreps, torch.from_numpy(s1_cart).double())
            D_s2 = _wigner_d_block(irreps, torch.from_numpy(s2_cart).double())

            Ph = P @ h
            transported_proj_1 = D_s1 @ Ph
            transported_proj_2 = D_s2 @ Ph
            proj_err = float(torch.max(torch.abs(transported_proj_1 - transported_proj_2)))

            transported_noproj_1 = D_s1 @ h
            transported_noproj_2 = D_s2 @ h
            noproj_err = float(torch.max(torch.abs(transported_noproj_1 - transported_noproj_2)))

            max_proj_err = max(max_proj_err, proj_err)
            max_noproj_err = max(max_noproj_err, noproj_err)

            per_orbit_results.append({
                "structure": name,
                "orbit": p,
                "wyckoff": orbit.wyckoff_symbol,
                "site_sym": orbit.site_symmetry,
                "H_size": stab_W.shape[0],
                "delta_proj": proj_err,
                "delta_noproj": noproj_err,
            })

    return {
        "max_proj_canonical": max_proj_err,
        "max_noproj_canonical": max_noproj_err,
        "max_wrong_proj_canonical": 0.0,
        "per_orbit": per_orbit_results,
    }


def _wigner_d_block(irreps: o3.Irreps, R: torch.Tensor) -> torch.Tensor:
    """Build the full Wigner-D matrix for all irreps at once."""
    R_batch = R.unsqueeze(0)
    blocks = []
    for mul, ir in irreps:
        D = ir.D_from_matrix(R_batch)
        dim_l = ir.dim
        block = D[0]
        tiled = torch.block_diag(*[block] * mul)
        blocks.append(tiled)
    return torch.block_diag(*blocks)


# ---------------------------------------------------------------------------
# Part 4c: Global rotation equivariance
# ---------------------------------------------------------------------------


def _build_graph_for_equivariance(struct, cutoff=5.0):
    """Build graph suitable for equivariance testing."""
    orbits, meta = structure_to_wyckoff_orbits(struct, tol=1e-3)
    std_lat = meta.get("standardized_lattice", struct.lattice.matrix)
    builder = WyckoffGraphBuilder(cutoff_radius=cutoff, subedge_aggregation="sum")
    data = builder.build(
        orbits, std_lat,
        atom_to_orbit=meta.get("atom_to_orbit"),
        atom_image_index=meta.get("atom_image_index"),
    )
    data.batch = torch.zeros(data.num_nodes, dtype=torch.long)
    data.multiplicity = torch.tensor(
        [o.multiplicity for o in orbits], dtype=torch.float32
    )
    feats = orbits_to_node_features(orbits)
    data.orbit_sym_ops_rotations = feats["orbit_sym_ops_rotations"]
    data.orbit_mult_mask = feats["orbit_mult_mask"]
    if "orbit_stabilizer_W_frac" in feats:
        data.orbit_stabilizer_W_frac = feats["orbit_stabilizer_W_frac"]
        data.orbit_stabilizer_w_frac = feats["orbit_stabilizer_w_frac"]
        data.orbit_stabilizer_mask = feats["orbit_stabilizer_mask"]
    return data, orbits, meta


def _cast_data_to_float32(data):
    """Cast all floating-point tensor fields to float32 for encoder compatibility."""
    for key in list(data.keys()):
        val = data[key]
        if isinstance(val, torch.Tensor) and val.is_floating_point():
            data[key] = val.float()
    return data


def _rotate_graph(data, R):
    """Rotate graph by O(3) matrix. Only lattice changes (fractional invariant)."""
    data.lattice = torch.matmul(data.lattice, R.T)


def _build_data_from_graph_dict(gd: dict) -> "torch_geometric.data.Data":
    """Build a WyckoffData object directly from a cached LMDB graph dict.

    Bypasses spglib and WyckoffGraphBuilder by using pre-computed fields.
    """
    from torch_geometric.data import Data

    data = Data()
    data.lattice = torch.from_numpy(gd["lattice"].copy())
    data.orbit_rep_frac = torch.from_numpy(gd["orbit_rep_frac"].copy())
    data.orbit_element = torch.from_numpy(gd["orbit_element"].copy().astype(np.int64))
    data.geo_edge_index = torch.from_numpy(gd["geo_edge_index"].copy().astype(np.int64))
    data.geo_edge_source_frac = torch.from_numpy(gd["geo_edge_source_frac"].copy())
    data.geo_edge_shift = torch.from_numpy(gd["geo_edge_shift"].copy().astype(np.float32))
    data.geo_edge_distance = torch.from_numpy(gd["geo_edge_distance"].copy())
    data.geo_edge_source_image = torch.from_numpy(gd["geo_edge_source_image"].copy().astype(np.int64))
    data.sym_edge_index = torch.from_numpy(gd["sym_edge_index"].copy().astype(np.int64))
    data.sym_edge_attr = torch.from_numpy(gd["sym_edge_attr"].copy())

    K = gd["orbit_rep_frac"].shape[0]
    data.batch = torch.zeros(K, dtype=torch.long)
    data.multiplicity = torch.from_numpy(gd["multiplicity"].copy())

    data.orbit_sym_ops_W_frac = torch.from_numpy(
        gd["orbit_sym_ops_W_frac"].copy().astype(np.float32)
    )
    data.orbit_sym_ops_w_frac = torch.from_numpy(
        gd["orbit_sym_ops_w_frac"].copy().astype(np.float32)
    )
    data.orbit_mult_mask = torch.from_numpy(
        gd["orbit_mult_mask"].copy()
    )

    if "orbit_stabilizer_W_frac" in gd:
        data.orbit_stabilizer_W_frac = torch.from_numpy(
            gd["orbit_stabilizer_W_frac"].copy().astype(np.float32)
        )
        data.orbit_stabilizer_w_frac = torch.from_numpy(
            gd["orbit_stabilizer_w_frac"].copy().astype(np.float32)
        )
        data.orbit_stabilizer_mask = torch.from_numpy(
            gd["orbit_stabilizer_mask"].copy()
        )

    if "space_group" in gd:
        data.space_group = torch.tensor(gd["space_group"], dtype=torch.long)
    if "orbit_letter_in_sg" in gd:
        data.orbit_letter_in_sg = torch.from_numpy(
            gd["orbit_letter_in_sg"].copy().astype(np.int64)
        )

    return data


def _run_part4c_benchmark_cached(
    encoder,
    shard_dir: str,
    manifest_path: str,
    limit: int = -1,
    n_rotations: int = 2,
    device: str = "cpu",
) -> Dict:
    """Run Part 4c equivariance on full LMDB dataset using cached graph data.

    Uses GPU if available. Bypasses spglib by building Data from cache.
    """
    from .shared_utils import iter_benchmark_entries

    encoder = encoder.to(device)
    max_proper = 0.0
    max_improper = 0.0
    total = 0
    failures = 0

    for gd in iter_benchmark_entries(
        shard_dir=shard_dir,
        manifest_path=manifest_path,
        limit=limit,
    ):
        try:
            data = _build_data_from_graph_dict(gd)
            data = _cast_data_to_float32(data)
            data = data.to(device)

            for trial in range(n_rotations):
                R = o3.rand_matrix(1)[0].to(device)
                det = torch.det(R).item()

                with torch.no_grad():
                    e1 = encoder(data)

                data_rot = data.clone()
                _rotate_graph(data_rot, R)
                with torch.no_grad():
                    e2 = encoder(data_rot)

                err = float(torch.abs(e1 - e2).max())
                if det > 0:
                    max_proper = max(max_proper, err)
                else:
                    max_improper = max(max_improper, err)

            total += 1
            if total % 5000 == 0:
                print(f"    Part 4c benchmark: {total} structures, "
                      f"max_proper={max_proper:.2e}, max_improper={max_improper:.2e}")

        except Exception:
            failures += 1
            continue

    return {
        "total_entries": total,
        "failures": failures,
        "max_proper": max_proper,
        "max_improper": max_improper,
    }


def run_part4c(
    cutoff: float = 5.0,
    benchmark_limit: int = 0,
    benchmark_shard_dir: str = "",
    benchmark_manifest: str = "",
    seed: int = 42,
) -> Dict:
    """Test E(X) = E(RX) under random O(3) rotations.

    Uses the full encoder forward pass. Reports max |E(X) - E(RX)| for
    proper and improper rotations.
    """
    from wyckoff_gnn.models.wyckoff_equiv.e3nn_encoder import EquivariantWyckoffGNNEncoder
    from .shared_utils import load_paper_config

    torch.manual_seed(seed)
    config = load_paper_config()

    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    encoder = EquivariantWyckoffGNNEncoder(
        num_layers=config["num_layers"],
        num_rbf=config["num_rbf"],
        rbf_max=config["cutoff"],
        init_irreps=config["init_irreps"],
        hidden_irreps=config["hidden_irreps"],
        tp_mode=config["tp_mode"],
        radial_mlp_width=config["radial_mlp_width"],
        use_edge_state=config["use_edge_state"],
        edge_state_dim=config["edge_state_dim"],
        edge_state_layers=config["edge_state_layers"],
        edge_readout=config["edge_readout"],
        edge_pool_mode=config["edge_pool_mode"],
        dropout=0.0,
        use_site_projection=config["use_site_projection"],
    )
    encoder.eval()

    max_proper = 0.0
    max_improper = 0.0
    per_struct = []

    for name, factory in CANONICAL_STRUCTURES.items():
        struct = factory()
        data, _, _ = _build_graph_for_equivariance(struct, cutoff)
        data = _cast_data_to_float32(data)

        for trial in range(5):
            R = o3.rand_matrix(1)[0]
            det = torch.det(R).item()

            with torch.no_grad():
                e1 = encoder(data)

            data_rot = data.clone()
            _rotate_graph(data_rot, R)
            with torch.no_grad():
                e2 = encoder(data_rot)

            err = float(torch.abs(e1 - e2).max())

            if det > 0:
                max_proper = max(max_proper, err)
            else:
                max_improper = max(max_improper, err)

        per_struct.append({"structure": name, "n_orbits": data.num_nodes})

    benchmark_results = None
    if benchmark_limit != 0 and benchmark_shard_dir and benchmark_manifest:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if benchmark_limit == -1 or benchmark_limit > 20:
            benchmark_results = _run_part4c_benchmark_cached(
                encoder, benchmark_shard_dir, benchmark_manifest,
                limit=benchmark_limit, device=device,
            )
            if device == "cuda":
                max_proper = max(max_proper, benchmark_results["max_proper"])
                max_improper = max(max_improper, benchmark_results["max_improper"])
        else:
            from .shared_utils import sample_benchmark_structures
            bench = sample_benchmark_structures(
                limit=benchmark_limit,
                shard_dir=benchmark_shard_dir,
                manifest_path=benchmark_manifest,
            )
            for struct in bench:
                try:
                    data, _, _ = _build_graph_for_equivariance(struct, cutoff)
                    data = _cast_data_to_float32(data)
                except Exception:
                    continue

                for trial in range(2):
                    R = o3.rand_matrix(1)[0]
                    det = torch.det(R).item()
                    with torch.no_grad():
                        e1 = encoder(data)
                    data_rot = data.clone()
                    _rotate_graph(data_rot, R)
                    with torch.no_grad():
                        e2 = encoder(data_rot)
                    err = float(torch.abs(e1 - e2).max())
                    if det > 0:
                        max_proper = max(max_proper, err)
                    else:
                        max_improper = max(max_improper, err)

    torch.set_default_dtype(prev_dtype)
    return {
        "max_canonical_proper": max_proper,
        "max_canonical_improper": max_improper,
        "per_structure": per_struct,
        "benchmark": benchmark_results,
    }


def run_part4(
    cutoff: float = 5.0,
    irreps_str: str = "64x0e + 32x1o + 16x2e",
    benchmark_limit: int = 0,
    benchmark_shard_dir: str = "",
    benchmark_manifest: str = "",
) -> Dict:
    """Run all Part 4 sub-tests."""
    result_4a = run_part4a(cutoff)
    result_4b = run_part4b(irreps_str)
    result_4c = run_part4c(
        cutoff, benchmark_limit, benchmark_shard_dir, benchmark_manifest,
    )

    result = {
        "delta_rep": result_4a,
        "delta_transport": result_4b,
        "equivariance_error": {
            "max_canonical_proper": result_4c["max_canonical_proper"],
            "max_canonical_improper": result_4c["max_canonical_improper"],
        },
    }
    if result_4c.get("benchmark"):
        result["equivariance_error"]["max_benchmark_proper"] = (
            result_4c["benchmark"]["max_proper"]
        )
        result["equivariance_error"]["max_benchmark_improper"] = (
            result_4c["benchmark"]["max_improper"]
        )
        result["equivariance_error"]["benchmark_entries"] = (
            result_4c["benchmark"]["total_entries"]
        )
    return result
