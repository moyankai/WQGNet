"""Occupied Wyckoff Graph builder with thick geometric edges at sub-edge level.

Constructs a PyG Data object where:
- Nodes = occupied Wyckoff orbit instances.
- Geometric edges = **sub-edge level**: for each ordered orbit pair (p, q),
  one sub-edge per equivalent atom of q (plus PBC images) that falls within
  the cutoff radius from the representative atom of p.  Each sub-edge stores
  the source fractional coordinate, integer PBC shift, and precomputed
  distance — the model uses these to reconstruct differentiable edge vectors.

Message passing convention (documented once, relied upon everywhere):
  - **target orbit p**: receives the message (message receiver).
  - **source orbit q**: provides the message (message sender).
  - **Edge vector**::
        vec_{p←q,k,L} = x_{q,k} + L - x_p
    i.e. vector FROM target representative x_p TO source image x_{q,k}+L.
  - ``geo_edge_index[0]`` = target (dst), ``geo_edge_index[1]`` = source.
  - In the e3nn TP layer:  h_q (D-rotated) ⊗ SH(vec/r) → message to p,
    aggregated via scatter at target p.

Why sub-edges (not thick-edge pooling before the MLP)?
  - Each equivalent atom q_k sits at a different position and thus has a
    different geometric relationship (distance, direction) with p.  Pooling
    before the TP loses this information.
  - Sub-edges preserve per-atom geometry; the model aggregates them via
    scatter_sum (or weighted variants), which recovers the correct
    equivariant message.

Configurable sub-edge aggregation:
  - ``"sum"`` (default): weight=1.0 on each sub-edge; scatter_sum naturally
    sums over all equivalent images.  This best approximates "all-atom" MP.
  - ``"mean"``: weight = 1/k_pq where k_pq is the number of sub-edges
    between (p, q).  Normalises each orbit pair's contribution.
  - ``"normalized"``: weight = 1/sqrt(k_pq).  Intermediate scaling.

MLFF / symmetry-breaking boundary:
  - This Wyckoff-compressed graph is designed for high-symmetry equilibrium
    crystals.  For MD, phonons, defects, or surface reconstructions where
    atoms within one Wyckoff orbit are no longer symmetry-equivalent, use
    a P1 fallback (each atom = separate orbit, multiplicity=1) or a hybrid
    E = E_sym + E_res model.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch_geometric.data import Data

from wyckoff_gnn.data.crystal_to_wyckoff import WyckoffOrbit, orbits_to_node_features
from wyckoff_gnn.utils.wyckoff_utils import site_symmetry_hierarchy


class WyckoffGraphBuilder:
    """Builds an Occupied Wyckoff Graph from a list of WyckoffOrbit instances.

    The graph has:
    - One node per occupied Wyckoff orbit.
    - Geometric sub-edges with PBC shifts (thick edges at sub-edge resolution).
    - Optional symmetry hierarchy edges.

    Args:
        cutoff_radius: Maximum inter-orbit distance for geometric edges (Å).
        num_rbf: Number of RBF kernels (for legacy edge_attr).
        rbf_min: Minimum RBF distance.
        rbf_max: Maximum RBF distance.
        subedge_aggregation: ``"sum"`` (default), ``"mean"``, or ``"normalized"``.
        pbc_tol: Tolerance for self-edge exclusion.
    """

    def __init__(
        self,
        cutoff_radius: float = 8.0,
        num_rbf: int = 32,
        rbf_min: float = 0.0,
        rbf_max: float = 8.0,
        subedge_aggregation: str = "sum",
        pbc_tol: float = 1e-4,
        global_max_mult: int = 0,
        max_pbc_images: Optional[int] = None,
        angle_pair_top_k: int = 0,
    ):
        self.cutoff_radius = cutoff_radius
        self.num_rbf = num_rbf
        self.rbf_min = rbf_min
        self.rbf_max = rbf_max
        self.subedge_aggregation = subedge_aggregation
        self.pbc_tol = pbc_tol
        self.global_max_mult = global_max_mult
        self.max_pbc_images = max_pbc_images
        # Top-K nearest kj-neighbours to pair with each ij edge for angle MP.
        # 0 = disable angle preprocessing entirely.
        self.angle_pair_top_k = angle_pair_top_k

        self.rbf_centers = torch.linspace(rbf_min, rbf_max, num_rbf)
        gamma_denom = rbf_max - rbf_min
        self.rbf_gamma = (num_rbf - 1) / gamma_denom if gamma_denom > 0 else 1.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self,
        orbits: List[WyckoffOrbit],
        lattice: np.ndarray,
        atom_to_orbit: Optional[np.ndarray] = None,
        atom_image_index: Optional[np.ndarray] = None,
    ) -> Data:
        """Build the complete Occupied Wyckoff Graph.

        Args:
            orbits: List of occupied Wyckoff orbit instances.
            lattice: (3, 3) lattice matrix in Å (row vectors).

        Returns:
            PyG Data object with node features, geometric sub-edges, and
            optional symmetry edges.
        """
        n_nodes = len(orbits)

        # Node features (pad sym_ops to global_max_mult for batch collation).
        node_feats = orbits_to_node_features(
            orbits, target_max_mult=self.global_max_mult
        )
        # Save raw fractional W/w BEFORE Cartesian conversion.
        node_feats["orbit_sym_ops_W_frac"] = node_feats["orbit_sym_ops_rotations"].clone()
        node_feats["orbit_sym_ops_w_frac"] = node_feats["orbit_sym_ops_translations"].clone()
        # Convert to Cartesian R_e3nn for the main sym_ops field (backward compat).
        node_feats = _convert_sym_ops_to_cartesian(node_feats, lattice)

        # Geometric sub-edges.
        geo = self._build_geometric_edges(orbits, lattice)

        # Angle pair index for three-body equivariant MP.
        angle_pair_index, angle_pair_target = self._build_angle_pairs(
            geo["edge_index"], geo["distance"]
        )

        # Orbit multiplicities.
        multiplicity = torch.tensor(
            [o.multiplicity for o in orbits], dtype=torch.float32
        )

        # Symmetry edges.
        sym_edge_index, sym_edge_attr = self._build_symmetry_edges(orbits)

        # Space group.
        sg_number = orbits[0].space_group if orbits else 1

        data = Data(
            # Node features
            orbit_element=node_feats["orbit_element"],
            orbit_rep_frac=node_feats["orbit_rep_frac"],
            orbit_letter_in_sg=node_feats["orbit_letter_in_sg"],
            orbit_site_sym=node_feats["orbit_site_sym"],
            letter_in_sg_token=node_feats["letter_in_sg_token"],
            # Backward-compatible aliases
            rep_coords_frac=node_feats["orbit_rep_frac"],
            atomic_numbers=node_feats["orbit_element"],
            # Geometric sub-edges
            geo_edge_index=geo["edge_index"],
            geo_edge_source_image=geo["source_image"],
            geo_edge_source_frac=geo["source_frac"],
            geo_edge_shift=geo["shift"],
            geo_edge_distance=geo["distance"],
            geo_edge_weight=geo["weight"],
            geo_edge_attr=geo["edge_attr"],
            # Angle pair index for three-body equivariant MP (empty if disabled).
            # angle_pair_index[0] = "kj" edge index (the pivot / conditioning edge)
            # angle_pair_index[1] = "ij" edge index (the edge being updated)
            # Both edges share the same target orbit i (== j).
            angle_pair_index=angle_pair_index,
            angle_pair_target=angle_pair_target,
            # Orbit multiplicities for readout weighting
            multiplicity=multiplicity,
            # Atom-level mapping: each original atom → (orbit, image_index)
            atom_to_orbit=(
                torch.from_numpy(atom_to_orbit.astype(np.int64)).long()
                if atom_to_orbit is not None else None
            ),
            atom_image_index=(
                torch.from_numpy(atom_image_index.astype(np.int64)).long()
                if atom_image_index is not None else None
            ),
            # Symmetry edges
            sym_edge_index=sym_edge_index,
            sym_edge_attr=sym_edge_attr,
            # Per-orbit symmetry operations:
            # - orbit_sym_ops_rotations: Cartesian R_e3nn (cached image
            #   generators, m_p per orbit — NOT the stabilizer).
            # - orbit_sym_ops_W_frac: raw fractional W of image generators.
            # - orbit_sym_ops_w_frac: raw fractional w of image generators.
            # - orbit_stabilizer_W_frac: full site stabilizer H_p (|H_p| per
            #   orbit, padded to max_H in the batch). This is what the
            #   Reynolds projector averages over.
            orbit_sym_ops_rotations=node_feats.get("orbit_sym_ops_rotations"),
            orbit_sym_ops_W_frac=node_feats.get("orbit_sym_ops_W_frac"),
            orbit_sym_ops_w_frac=node_feats.get("orbit_sym_ops_w_frac"),
            orbit_mult_mask=node_feats.get("orbit_mult_mask"),
            orbit_stabilizer_W_frac=node_feats.get("orbit_stabilizer_W_frac"),
            orbit_stabilizer_w_frac=node_feats.get("orbit_stabilizer_w_frac"),
            orbit_stabilizer_mask=node_feats.get("orbit_stabilizer_mask"),
            # Free-parameter DOF for generalized force.
            orbit_dof=node_feats.get("orbit_dof"),
            orbit_param_basis_frac=node_feats.get("orbit_param_basis_frac"),
            # Global
            space_group=torch.tensor([sg_number]),
            lattice=torch.from_numpy(np.ascontiguousarray(lattice)).float(),
            # Metadata
            num_nodes=n_nodes,
            num_orbits=n_nodes,
        )
        return data

    # ------------------------------------------------------------------
    # Geometric sub-edges
    # ------------------------------------------------------------------

    def _build_angle_pairs(
        self,
        geo_edge_index: torch.Tensor,
        geo_edge_distance: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build angle pair index for three-body equivariant MP.

        For each target orbit i, gather its incoming edges N_i and form ordered
        pairs (kj, ij) with kj != ij, where "kj" is treated as the conditioning
        pivot and "ij" is the edge being updated.  To keep P bounded we retain
        only the ``top_k`` nearest incoming edges as candidate pivots for each
        target; each of those pivots is paired with every other incoming edge.
        Returned tensors:
          - angle_pair_index: (2, P) int64 with rows [kj_edge_idx, ij_edge_idx]
          - angle_pair_target: (P,) int64 giving the shared target orbit id.
        If ``self.angle_pair_top_k <= 0`` or the graph has no edges, both
        tensors are empty.
        """
        top_k = int(self.angle_pair_top_k)
        E = int(geo_edge_index.size(1)) if geo_edge_index.numel() else 0
        empty_pair = torch.zeros(2, 0, dtype=torch.long)
        empty_tgt = torch.zeros(0, dtype=torch.long)
        if top_k <= 0 or E <= 1:
            return empty_pair, empty_tgt

        targets = geo_edge_index[0].long()
        distances = geo_edge_distance.float()
        n_orbits = int(targets.max().item()) + 1 if E > 0 else 0

        # Group edge indices by target orbit.
        pairs_kj: List[torch.Tensor] = []
        pairs_ij: List[torch.Tensor] = []
        pairs_tgt: List[torch.Tensor] = []
        for i in range(n_orbits):
            mask = targets == i
            edge_ids = torch.nonzero(mask, as_tuple=False).view(-1)
            n_i = int(edge_ids.numel())
            if n_i < 2:
                continue  # need at least two neighbours for a triangle
            # Sort by distance and pick top-K nearest as pivot candidates.
            d_i = distances[edge_ids]
            order = torch.argsort(d_i)
            edge_ids_sorted = edge_ids[order]
            k_eff = min(top_k, n_i)
            kj_candidates = edge_ids_sorted[:k_eff]
            # For each pivot kj, pair with every ij in N_i \ {kj}.
            # Use broadcasting: (k_eff, 1) and (1, n_i) then mask kj==ij.
            kj_mat = kj_candidates.view(-1, 1).expand(k_eff, n_i)
            ij_mat = edge_ids_sorted.view(1, -1).expand(k_eff, n_i)
            keep = kj_mat != ij_mat
            pairs_kj.append(kj_mat[keep])
            pairs_ij.append(ij_mat[keep])
            pairs_tgt.append(torch.full(
                (int(keep.sum().item()),), i, dtype=torch.long,
            ))

        if not pairs_kj:
            return empty_pair, empty_tgt
        kj_all = torch.cat(pairs_kj, dim=0)
        ij_all = torch.cat(pairs_ij, dim=0)
        tgt_all = torch.cat(pairs_tgt, dim=0)
        angle_pair_index = torch.stack([kj_all, ij_all], dim=0).contiguous()
        return angle_pair_index, tgt_all

    def _build_geometric_edges(
        self,
        orbits: List[WyckoffOrbit],
        lattice: np.ndarray,
    ) -> Dict[str, torch.Tensor]:
        """Build sub-edge-level geometric edges with PBC shifts (vectorized).

        For each ordered pair of orbits (p, q):
        1. Fix p's representative atom x_p.
        2. Enumerate q's equivalent atoms x_{q,k} for k in [0, m_q).
        3. Enumerate lattice translations L = [i,j,k] within cutoff range.
        4. If d = ||(x_{q,k} + L - x_p) @ lattice|| < R_c + tol, add sub-edge.

        Self-edges (p=q, k yields rep atom, L=[0,0,0]) with d < tol are excluded.

        Returns dict:
            edge_index:   [2, E_sub]  (target=p at row 0, source=q at row 1)
            source_image: [E_sub]     k in [0, m_q)
            source_frac:  [E_sub, 3]  x_{q,k} fractional coords
            shift:        [E_sub, 3]  integer PBC shift L (as float32)
            distance:     [E_sub]     precomputed Euclidean distance
            weight:       [E_sub]     per-sub-edge aggregation weight
            edge_attr:    [E_sub, num_rbf+3]  legacy RBF+unit_dir
        """
        n = len(orbits)
        lat_t = torch.from_numpy(np.array(lattice, copy=True)).float()

        empty = {
            "edge_index": torch.zeros(2, 0, dtype=torch.long),
            "source_image": torch.zeros(0, dtype=torch.long),
            "source_frac": torch.zeros(0, 3),
            "shift": torch.zeros(0, 3),
            "distance": torch.zeros(0),
            "weight": torch.zeros(0),
            "edge_attr": torch.zeros(0, self.num_rbf + 3),
        }
        # Only truly empty graphs (no orbits) have no edges.
        # Single-orbit graphs with multiplicity > 1 MUST have self-subedges:
        # different equivalent atoms of the SAME orbit are at different
        # positions and produce valid geometric (p==q, k≠k') edges.
        # These are essential for high-symmetry crystals where only one
        # Wyckoff orbit is occupied (e.g. diamond Si: 8a, Fd-3m).
        if n == 0:
            return empty

        # PBC image enumeration range (strict bound).
        #
        # The old heuristic ``ceil(cutoff / min_edge)`` uses the shortest lattice
        # vector length, which is NOT a sufficient bound for skewed lattices:
        # a combination shift (e.g. [1,2,0]) can place a periodic image within
        # the cutoff even when every single-axis image is outside it.
        #
        # Correct bound: for each lattice direction i, the face height
        #     h_i = V / |a_j x a_k|
        # is the perpendicular distance between adjacent lattice planes normal
        # to direction i.  Any image with |n_i| > cutoff / h_i cannot contribute
        # an edge within the cutoff, so
        #     nmax_i = ceil(cutoff / h_i) + 1
        # is a sufficient (and tight) per-direction enumeration range.
        #
        # Lattice convention: rows of ``lattice`` are a1, a2, a3 (row-vector
        # convention, offsets_cart = offsets_frac @ lat_t).
        a1, a2, a3 = lat_t[0], lat_t[1], lat_t[2]
        V = torch.abs(torch.dot(a1, torch.cross(a2, a3)))
        h1 = V / torch.norm(torch.cross(a2, a3))
        h2 = V / torch.norm(torch.cross(a3, a1))
        h3 = V / torch.norm(torch.cross(a1, a2))
        heights = [float(h1), float(h2), float(h3)]
        nmax = [
            max(1, int(math.ceil(self.cutoff_radius / h)) + 1)
            for h in heights
        ]

        # Allow truncation only when explicitly requested.
        if self.max_pbc_images is not None and max(nmax) > self.max_pbc_images:
            import warnings
            warnings.warn(
                f"PBC image range {nmax} exceeds max_pbc_images="
                f"{self.max_pbc_images}; truncating to {self.max_pbc_images}. "
                f"Edges may be MISSING for lattices where a face height "
                f"({heights}) ≪ cutoff ({self.cutoff_radius} Å). "
                f"Set max_pbc_images=None to disable this truncation."
            )
            nmax = [min(n, self.max_pbc_images) for n in nmax]

        offsets_frac = torch.tensor(
            [[i, j, k]
             for i in range(-nmax[0], nmax[0] + 1)
             for j in range(-nmax[1], nmax[1] + 1)
             for k in range(-nmax[2], nmax[2] + 1)],
            dtype=torch.float32,
        )  # (N_img, 3)
        offsets_cart = torch.matmul(offsets_frac, lat_t)  # (N_img, 3)
        N_img = offsets_frac.shape[0]

        # Precompute representative Cartesian coords.
        rep_fracs = torch.stack(
            [torch.from_numpy(o.representative_coord).float() for o in orbits]
        )  # (n, 3)
        rep_carts = torch.matmul(rep_fracs, lat_t)  # (n, 3)

        # Build flat array of all (q, k) source positions.
        # all_src_frac[j] = fractional coord of image k of orbit q
        # src_orbit_idx[j] = q,  src_image_idx[j] = k
        all_src_frac_list = []
        src_orbit_idx_list = []
        src_image_idx_list = []
        for q, orb in enumerate(orbits):
            pos = torch.from_numpy(orb.all_positions).float()  # (m_q, 3)
            m_q = pos.shape[0]
            all_src_frac_list.append(pos)
            src_orbit_idx_list.append(torch.full((m_q,), q, dtype=torch.long))
            src_image_idx_list.append(torch.arange(m_q, dtype=torch.long))

        all_src_frac = torch.cat(all_src_frac_list, dim=0)     # (S, 3) where S = sum(m_q)
        src_orbit_idx = torch.cat(src_orbit_idx_list)           # (S,)
        src_image_idx = torch.cat(src_image_idx_list)           # (S,)
        all_src_cart = torch.matmul(all_src_frac, lat_t)        # (S, 3)
        S = all_src_frac.shape[0]

        # Vectorized distance computation:
        # diff[p, j, L] = all_src_cart[j] + offsets_cart[L] - rep_carts[p]
        # Shape: (n, S, N_img, 3)
        # For memory efficiency, compute per target orbit p.
        cutoff_sq = self.cutoff_radius ** 2
        tol = self.pbc_tol

        edge_target_list = []
        edge_source_list = []
        img_idx_out = []
        src_frac_out = []
        shift_out = []
        dist_out = []
        pair_id_out = []

        # src_cart_plus_L: (S, N_img, 3) — precompute once
        src_cart_plus_L = all_src_cart.unsqueeze(1) + offsets_cart.unsqueeze(0)  # (S, N_img, 3)

        for p in range(n):
            # diff: (S, N_img, 3)
            diff = src_cart_plus_L - rep_carts[p].unsqueeze(0).unsqueeze(0)
            d_sq = (diff * diff).sum(dim=-1)  # (S, N_img)

            # Mask: within cutoff and not self (d > tol)
            mask = (d_sq > tol) & (d_sq < cutoff_sq + tol)

            if not mask.any():
                continue

            # Get indices where mask is True
            s_idx, l_idx = torch.where(mask)  # each (E_p,)
            E_p = s_idx.shape[0]

            edge_target_list.append(torch.full((E_p,), p, dtype=torch.long))
            edge_source_list.append(src_orbit_idx[s_idx])
            img_idx_out.append(src_image_idx[s_idx])
            src_frac_out.append(all_src_frac[s_idx])
            shift_out.append(offsets_frac[l_idx])
            dist_out.append(torch.sqrt(d_sq[s_idx, l_idx]))

            # pair_id: unique per (p, q) pair
            pair_id_out.append(p * n + src_orbit_idx[s_idx])

        if not edge_target_list:
            return empty

        edge_target_t = torch.cat(edge_target_list)
        edge_source_t = torch.cat(edge_source_list)
        source_image = torch.cat(img_idx_out)
        source_frac = torch.cat(src_frac_out, dim=0)
        shift = torch.cat(shift_out, dim=0)
        distance = torch.cat(dist_out)
        pair_id_t = torch.cat(pair_id_out)

        E = distance.shape[0]
        edge_index = torch.stack([edge_target_t, edge_source_t], dim=0)

        # Sub-edge weights based on aggregation mode.
        unique_pairs, counts = torch.unique(pair_id_t, return_counts=True)
        k_per_pair = torch.zeros(n * n, dtype=torch.float32)
        k_per_pair[unique_pairs] = counts.float()
        k_pq_per_edge = k_per_pair[pair_id_t]

        if self.subedge_aggregation == "sum":
            weight = torch.ones(E, dtype=torch.float32)
        elif self.subedge_aggregation == "mean":
            weight = 1.0 / k_pq_per_edge.clamp(min=1)
        elif self.subedge_aggregation == "normalized":
            weight = 1.0 / k_pq_per_edge.sqrt().clamp(min=1)
        else:
            weight = torch.ones(E, dtype=torch.float32)

        # Legacy edge_attr (RBF + unit direction).
        edge_attr = torch.zeros(E, self.num_rbf + 3)
        rbf = self._rbf_expand(distance)
        edge_attr[:, : self.num_rbf] = rbf
        rep_frac_target = rep_fracs[edge_target_t.long()]
        vec_frac = source_frac + shift - rep_frac_target
        vec_cart = torch.matmul(vec_frac, lat_t)
        unit_dir = vec_cart / (distance.unsqueeze(1) + 1e-8)
        edge_attr[:, self.num_rbf:] = unit_dir

        return {
            "edge_index": edge_index,
            "source_image": source_image,
            "source_frac": source_frac,
            "shift": shift,
            "distance": distance,
            "weight": weight,
            "edge_attr": edge_attr,
        }

    # ------------------------------------------------------------------
    # Symmetry hierarchy edges
    # ------------------------------------------------------------------

    def _build_symmetry_edges(
        self,
        orbits: List[WyckoffOrbit],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build directed symmetry-hierarchy prior edges (heuristic, NOT strict algebraic).

        Edge i → j exists when H_i ⊃ H_j (i's site-symmetry group is a proper
        supergroup of j's).  The hierarchy is based on the 32 crystallographic
        point group Hasse diagram — this is a **heuristic prior** derived from
        spglib's string labels.  It is NOT a rigorous subgroup proof and should
        be treated as an ablatable inductive bias, not a hard constraint.

        Edge features (symmetry_hierarchy_prior):
          +1.0 → high-symmetry → low-symmetry (constraint / prior)
          -1.0 → low-symmetry → high-symmetry (feedback)

        No edges between equal or incomparable site-symmetry groups.
        """
        n = len(orbits)
        if n <= 1:
            return torch.zeros(2, 0, dtype=torch.long), torch.zeros(0, 1)

        edge_list: List[List[int]] = []
        attr_list: List[float] = []

        for i in range(n):
            ss_i = orbits[i].site_symmetry
            for j in range(n):
                if i == j:
                    continue
                ss_j = orbits[j].site_symmetry
                relation = site_symmetry_hierarchy(ss_i, ss_j)
                if relation != 0:
                    edge_list.append([i, j])
                    attr_list.append(float(relation))

        if len(edge_list) == 0:
            return torch.zeros(2, 0, dtype=torch.long), torch.zeros(0, 1)

        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(attr_list, dtype=torch.float32).unsqueeze(-1)
        return edge_index, edge_attr

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _rbf_expand(self, distances: torch.Tensor) -> torch.Tensor:
        """Expand distances in Gaussian RBF basis.

        φ_k(d) = exp(-γ · (d - μ_k)²)
        """
        centers = self.rbf_centers.to(distances.device)
        diff = distances.unsqueeze(1) - centers.unsqueeze(0)
        return torch.exp(-self.rbf_gamma * diff.pow(2))


def _convert_sym_ops_to_cartesian(
    node_feats: dict, lattice: np.ndarray
) -> dict:
    """Convert orbit sym_ops from fractional W to Cartesian R_e3nn.

    spglib W operates on column vectors in fractional space:  r' = W @ r + w.
    e3nn's D_from_matrix expects a column-vector Cartesian rotation matrix.

    Derivation (see docs/coordinate_convention.md):
        x_cart = x_frac @ A                          (row convention)
        R_e3nn = A.T @ W @ A^{-T}                     (column convention)

    For cubic lattice (A = a·I): R_e3nn = W.

    This is done at graph-build time so the stored rotations transform
    correctly under global O(3) rotations: R_e3nn → Q @ R_e3nn @ Q^{-1}.

    Args:
        node_feats: dict from ``orbits_to_node_features``.
        lattice: (3, 3) lattice matrix (row vectors).

    Returns:
        Updated node_feats dict with Cartesian (column) R_e3nn.
    """
    A = torch.from_numpy(np.ascontiguousarray(lattice)).float()
    A_T = A.T
    A_inv_T = torch.inverse(A).T  # = A^{-T}

    W = node_feats["orbit_sym_ops_rotations"]  # (K, M, 3, 3) fractional
    K, M = W.shape[:2]
    W_flat = W.reshape(K * M, 3, 3).to(A.dtype)
    A_T_exp = A_T.unsqueeze(0).expand(K * M, 3, 3)
    A_inv_T_exp = A_inv_T.unsqueeze(0).expand(K * M, 3, 3)
    # R_e3nn = A.T @ W @ A^{-T}
    R_flat = torch.bmm(torch.bmm(A_T_exp, W_flat), A_inv_T_exp)
    node_feats["orbit_sym_ops_rotations"] = R_flat.reshape(K, M, 3, 3)
    return node_feats


__all__ = ["WyckoffGraphBuilder", "_convert_sym_ops_to_cartesian"]
