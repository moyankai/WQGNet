"""Equivariant WyckoffGNN encoder (Stage B).

Stacks the equivariant layers produced by :mod:`wyckoff_gnn.models.e3nn_layers`
into a full encoder that produces graph-level invariant predictions.

Data flow
---------
1. NodeEncoder:  chemical(32x0e) + radial(16x0e) + vector(16x1o) + proto(16x0e)
                  -> init irreps "64x0e + 16x1o"
2. ExpandLayer:   init x edge SH -> first CG MP -> node irreps "64x0e+32x1o+16x2e"
3. WyckoffLayer x L-1:  dual-channel equivariant MP (geo + sym)
4. Readout:       0e-channel pooling -> scalar-energy prediction

Edge direction convention:
  - geo_edge_index[0] = target orbit p (message receiver)
  - geo_edge_index[1] = source orbit q (message sender)
  - vec = (source_frac + shift - target_rep_frac) @ lattice

The encoder operates on PyG ``Data`` objects that carry:
  - ``orbit_element``, ``orbit_rep_frac``, ``lattice`` (node features)
  - ``geo_edge_index``, ``geo_edge_source_frac``, ``geo_edge_shift``,
    ``geo_edge_source_image``, ``geo_edge_weight`` (geometric sub-edges)
  - ``sym_edge_index``, ``sym_edge_attr`` (symmetry edges)
  - ``batch`` (graph assignment for pooling)
  - ``multiplicity`` (orbit multiplicities for weighted readout)
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
from torch_scatter import scatter

from e3nn import o3

from wyckoff_gnn.models.wyckoff_equiv.e3nn_layers import (
    ElementOnlyNodeEncoder,
    EquivariantWyckoffLayer,
    GeometricLiftingLayer,
    EquivariantReadout,
    EquivariantGeometricMP_v2,
    _rotate_node_features,
    _IRREPS_NODE,
    _IRREPS_SH,
    _IRREPS_MESSAGE,
    _DEFAULT_HIDDEN_IRREPS,
    compute_irreps,
)
from wyckoff_gnn.utils.radial import RadialFeat

from wyckoff_gnn.models.wyckoff_equiv.edge_state import (
    EdgeStateInit,
    EdgeStateUpdate,
    InvariantMessageSummary,
    EdgePool,
)


def _build_source_D_cache(source_rotations, irreps):
    """Pre-compute Wigner-D matrices for all unique (l, p) in *irreps*.

    Returns a dict ``{(l, p): D_l}`` where ``D_l`` has shape ``(E, 2l+1, 2l+1)``.
    Scalar (l=0) entries are omitted (identity is implicit).

    This avoids calling ``D_from_matrix`` once per layer — the source
    rotations don't change across layers, so the D matrices can be shared.
    """
    E = source_rotations.shape[0]
    if E == 0:
        return {}
    cache = {}
    for _, ir in irreps:
        if ir.l == 0:
            continue
        key = (ir.l, ir.p)
        if key in cache:
            continue
        D_l = ir.D_from_matrix(source_rotations)
        if D_l.dim() == 2:
            D_l = D_l.unsqueeze(0).expand(E, -1, -1)
        elif D_l.size(0) != E:
            D_l = D_l.expand(E, -1, -1)
        cache[key] = D_l
    return cache


class EquivariantWyckoffGNNEncoder(nn.Module):
    """Full equivariant Wyckoff GNN encoder (Stage B).

    Args:
        num_layers: Number of equivariant MP layers (>= 1).
        num_rbf: Number of RBF kernels for distance encoding.
        rbf_max: Maximum distance for RBF expansion (Å).
        pool: Graph-level pooling method (mean/sum/max/attention).
        dropout: Dropout rate.
        init_irreps: Pure-scalar irreps for the element-only node embedding
            (e.g. "128x0e"). Decoupled from hidden_irreps.
        readout_mode: "intensive" (default) or "extensive" for
            multiplicity-weighted readout.
    """

    def __init__(
        self,
        num_layers: int = 4,
        num_rbf: int = 32,
        rbf_max: float = 8.0,
        pool: str = "mean",
        dropout: float = 0.1,
        init_irreps: str = "64x0e",
        readout_mode: str = "intensive",
        geometry_mode: str = "precomputed",
        use_site_projection: bool = False,
        hidden_irreps: str = _DEFAULT_HIDDEN_IRREPS,
        radial_gate_mode: str = "per_type",
        use_edge_state: bool = False,
        edge_state_dim: int = 64,
        edge_state_layers: int = 2,
        edge_readout: bool = False,
        edge_pool_mode: str = "normalized",
        tp_mode: str = "dynamic_v2",
        radial_mlp_width: int = 128,
        n_bessel: int = 16,
        # --- Phase 2: precomputed site-irrep projection (Phase 1 tables) ---
        use_site_irrep_projection: bool = False,
        site_projector_path: str = "",
        site_irrep_table_path: str = "",
        site_irrep_lmax: int = 4,
        apply_site_projection: str = "each_layer",
        # --- Phase 3: formal edge readout ---
        edge_readout_mode: str = "legacy",
        edge_invariant_features: Optional[List[str]] = None,
        edge_pool_reduce: str = "sum",
        orbit_pair_hidden_dim: int = 64,
        orbit_pair_output_dim: int = 64,
        # --- Phase 4: NequIP-style per-element scale/shift ---
        per_type_shift: bool = False,
        per_type_scale: bool = False,
        max_atomic_number: int = 118,
        use_atom_props: bool = False,
    ):
        super().__init__()
        assert num_layers >= 1, "Need at least 1 equivariant layer."
        self.num_layers = num_layers
        self.init_irreps = init_irreps
        self.readout_mode = readout_mode
        self.geometry_mode = geometry_mode
        self.use_site_projection = use_site_projection
        self.hidden_irreps = hidden_irreps
        self.radial_gate_mode = radial_gate_mode
        # Phase 2: precomputed site-irrep projection settings
        self.use_site_irrep_projection = use_site_irrep_projection
        self.apply_site_projection = apply_site_projection
        self.site_projector_path = site_projector_path
        self.site_irrep_table_path = site_irrep_table_path
        self.site_irrep_lmax = site_irrep_lmax
        self.use_edge_state = use_edge_state
        self.edge_state_dim = edge_state_dim if use_edge_state else 0
        self.edge_readout = edge_readout
        self.tp_mode = tp_mode

        # Compute all dependent irreps from the hidden string.
        ir = compute_irreps(hidden_irreps)
        self._irreps_node = ir["node"]
        self._irreps_node_init = o3.Irreps(init_irreps)
        self._irreps_message = ir["message"]
        self._irreps_sh = ir["sh"]

        # Radial features: Bessel basis + polynomial cutoff (forward-weight init)
        self.n_bessel = n_bessel
        self.rbf = RadialFeat(r_max=rbf_max, n_bessel=n_bessel, n_out=num_rbf,
                              mlp_hidden=[64, 64])
        self.num_rbf = num_rbf  # compatibility with existing code

        # Element-only node encoder: Z -> pure-scalar init_irreps.
        self.node_encoder = ElementOnlyNodeEncoder(
            init_irreps=self._irreps_node_init,
            use_atom_props=use_atom_props,
        )

        # Edge state dimension passed to layers.
        _es_dim = self.edge_state_dim

        # Geometric lifting layer: init scalars -> hidden irreps via edge SH.
        self.expand = GeometricLiftingLayer(
            irreps_in=self._irreps_node_init,
            irreps_out=ir["node"],
            irreps_sh=ir["sh"],
            num_rbf=num_rbf,
            radial_gate_width=radial_mlp_width,
            dropout=dropout,
        )

        # MP layers.
        self.layers = nn.ModuleList([
            EquivariantWyckoffLayer(
                irreps_node=ir["node"],
                irreps_message=ir["message"],
                irreps_sh=ir["sh"],
                num_rbf=num_rbf,
                dropout=dropout,
                radial_gate_mode=radial_gate_mode,
                edge_state_dim=_es_dim,
            )
            for _ in range(num_layers - 1)
        ])

        # Replace geo_mp with dynamic_v2 if requested.
        if tp_mode == "dynamic_v2":
            for layer in self.layers:
                layer.geo_mp = EquivariantGeometricMP_v2(
                    irreps_node=ir["node"],
                    irreps_message=ir["message"],
                    irreps_sh=ir["sh"],
                    num_rbf=num_rbf,
                    edge_state_dim=_es_dim,
                    radial_mlp_width=radial_mlp_width,
                )

        # Edge state modules.
        self.edge_readout_mode = edge_readout_mode
        if use_edge_state:
            edge_init_input_dim = num_rbf + 1
            self.edge_state_init = EdgeStateInit(edge_init_input_dim, edge_state_dim)

            self.msg_summary = InvariantMessageSummary(ir["message"])
            msg_inv_dim = self.msg_summary.output_dim

            self.edge_state_updates = nn.ModuleList([
                EdgeStateUpdate(edge_state_dim, msg_inv_dim, num_rbf)
                for _ in range(num_layers - 1)
            ])
        else:
            self.edge_state_init = None
            self.msg_summary = None
            self.edge_state_updates = None

        # Edge readout dispatch. This is independent of use_edge_state: features
        # like rbf_hist / edge_count / norm / tp_l work without edge_state.
        _effective_es_dim = edge_state_dim if use_edge_state else 0
        if edge_readout_mode == "legacy":
            if use_edge_state and edge_readout:
                self.edge_pool = EdgePool(edge_state_dim, pool_mode=edge_pool_mode)
            else:
                self.edge_pool = None
            self.edge_invariant_readout = None
        elif edge_readout_mode == "none":
            self.edge_pool = None
            self.edge_invariant_readout = None
        else:
            from wyckoff_gnn.models.wyckoff_equiv.readouts import EdgeInvariantReadout
            feats = edge_invariant_features or ["0e", "rbf_hist", "edge_count"]
            # If user asked for 0e but disabled use_edge_state, drop 0e —
            # we have no edge_state to source it from.
            if not use_edge_state and "0e" in feats:
                feats = [f for f in feats if f != "0e"]
            # message_irreps is needed whenever any feature consumes the
            # per-edge message tensor. That includes "norm", "tp0" (legacy
            # alias for tp_l1), and any "tp_l<N>" for N >= 1. Note this is
            # independent of use_edge_state: messages are produced by every
            # MP layer regardless of edge_state.
            _needs_msg_irreps = any(
                f == "norm" or f == "tp0" or f.startswith("tp_l")
                for f in feats
            )
            _msg_irreps_for_readout = ir["message"] if _needs_msg_irreps else None
            self.edge_invariant_readout = EdgeInvariantReadout(
                edge_state_dim=_effective_es_dim,
                num_rbf=num_rbf,
                mode=edge_readout_mode,
                features=feats,
                pool_reduce=edge_pool_reduce,
                pair_hidden_dim=orbit_pair_hidden_dim,
                pair_output_dim=orbit_pair_output_dim,
                message_irreps=_msg_irreps_for_readout,
            )
            self.edge_pool = None
            # Remember the effective inputs so callers (and integration tests)
            # can introspect what was actually wired up.
            self._edge_invariant_features_effective = list(feats)
            self._edge_readout_message_irreps = _msg_irreps_for_readout

        # Phase 2: precomputed site-symmetry irrep projection.
        self.site_projector = None
        if use_site_irrep_projection:
            if not site_projector_path:
                raise ValueError(
                    "use_site_irrep_projection=True but site_projector_path is empty. "
                    "Point it to a Phase 1 site_projectors_lmaxN.pt file."
                )
            from wyckoff_gnn.models.wyckoff_equiv.site_projection import SiteIrrepProjector
            self.site_projector = SiteIrrepProjector(
                irreps=ir["node"],
                projector_path=site_projector_path,
                table_path=site_irrep_table_path or None,
                lmax=site_irrep_lmax,
            )

        # Pre-readout projection: extract invariant scalars from full irreps.
        # o3.Linear only connects 0e→0e paths; l>0 channels are dropped here.
        # They contribute only through message passing, not directly to readout.
        full_dim = ir["node"].dim
        self._readout_irreps_in = o3.Irreps(f"{full_dim}x0e")
        self.pre_readout = o3.Linear(
            irreps_in=ir["node"],
            irreps_out=self._readout_irreps_in,
        )

        # Readout.
        # Compute edge_pool_dim consumed by EquivariantReadout.
        if self.edge_invariant_readout is not None:
            _edge_pool_dim = self.edge_invariant_readout.output_dim
        elif use_edge_state and edge_readout_mode == "legacy" and edge_readout:
            _edge_pool_dim = edge_state_dim
        else:
            _edge_pool_dim = 0
        self.readout = EquivariantReadout(
            irreps_node=self._readout_irreps_in,
            pool=pool,
            readout_mode=readout_mode,
            dropout=dropout,
            edge_pool_dim=_edge_pool_dim,
            per_type_shift=per_type_shift,
            per_type_scale=per_type_scale,
            max_atomic_number=max_atomic_number,
        )

    # ------------------------------------------------------------------
    # Edge feature computation
    # ------------------------------------------------------------------

    def _compute_edge_features(
        self,
        data,
        lattice_per_node: torch.Tensor,
        rep_frac: Optional[torch.Tensor] = None,
    ):
        """Compute edge features.  Uses dynamic rep_frac if provided."""
        if rep_frac is not None and self.geometry_mode == "dynamic_symmetric":
            return self._compute_edge_features_dynamic(
                data, lattice_per_node, rep_frac
            )
        return self._compute_edge_features_precomputed(data, lattice_per_node)

    def _compute_edge_features_precomputed(
        self,
        data,
        lattice_per_node: torch.Tensor,
    ):
        """Compute per-sub-edge SH and RBF features from stored fields.

        Uses the stored ``geo_edge_source_frac`` and ``geo_edge_shift`` to
        reconstruct the minimum-image-corrected displacement vector purely
        through differentiable operations (matrix multiply + addition).
        No ``torch.round`` or argmin-based minimum-image search.

        Formula:
            vec_frac = source_frac + shift - target_rep_frac
            vec_cart = vec_frac @ lattice
            d = ||vec_cart||
            rhat = vec_cart / d
            edge_sh = spherical_harmonics(rhat)
            edge_rbf = GaussianRBF(d)

        Args:
            data: PyG Data/Batch.
            lattice_per_node: (K_total, 3, 3) per-node lattice matrices.

        Returns:
            edge_sh: (E_geo, 9) spherical-harmonic features.
            edge_rbf: (E_geo, num_rbf) RBF distance features.
            source_rotations: (E_geo, 3, 3) rotation matrices, or None.
        """
        edge_index = data.geo_edge_index  # (2, E): row0=target, row1=source
        if edge_index.numel() == 0:
            dim_sh = self._irreps_sh.dim
            return (
                torch.zeros(0, dim_sh, device=data.orbit_rep_frac.device),
                torch.zeros(0, self.num_rbf, device=data.orbit_rep_frac.device),
                None,
            )

        target_orbit, source_orbit = edge_index[0], edge_index[1]

        # Lattice per edge.
        lat_per_edge = lattice_per_node[target_orbit]  # (E, 3, 3)

        # Target representative fractional coordinates.
        target_rep_frac = data.orbit_rep_frac[target_orbit]  # (E, 3)

        # Source equivalent-atom fractional coords (from stored sub-edge data).
        source_frac = data.geo_edge_source_frac  # (E, 3)

        # PBC shift (integer, stored as float32).
        shift = data.geo_edge_shift  # (E, 3)

        # Minimum-image-corrected fractional displacement.
        vec_frac = source_frac + shift - target_rep_frac  # (E, 3)

        # Cartesian displacement (differentiable).
        vec_cart = torch.bmm(vec_frac.unsqueeze(1), lat_per_edge).squeeze(1)  # (E, 3)

        d = torch.norm(vec_cart, dim=-1)  # (E,)
        rhat = vec_cart / (d.unsqueeze(-1).clamp(min=1e-8))
        edge_sh = o3.spherical_harmonics(
            self._irreps_sh, rhat, normalize=True, normalization="component"
        )
        edge_rbf = self.rbf(d)

        # Source image rotations (fractional → Cartesian converted).
        source_rotations = self._compute_source_rotations(data, lattice_per_node)

        return edge_sh, edge_rbf, source_rotations

    # ------------------------------------------------------------------
    # Dynamic edge features: differentiable source coordinates
    # ------------------------------------------------------------------

    def _compute_edge_features_dynamic(
        self,
        data,
        lattice_per_node: torch.Tensor,
        rep_frac: torch.Tensor,
    ):
        """Compute edge features with DIFFERENTIABLE source image coordinates.

        In this mode, source equivalent-atom coordinates are computed
        dynamically from the SOURCE orbit's representative coordinate
        and the affine W,w operation::

            r_{q,k} = r_q @ W_{q,k}.T + w_{q,k}
            v_frac = r_{q,k} + L - r_p
            v_cart = v_frac @ A

        This ensures that both ∂v/∂r_p (target) and ∂v/∂r_q (source)
        are in the autograd graph, enabling correct force computation.

        Requires ``orbit_sym_ops_W_frac`` and ``orbit_sym_ops_w_frac``
        on the Data object.
        """
        edge_index = data.geo_edge_index
        if edge_index.numel() == 0:
            dim_sh = self._irreps_sh.dim
            dev = rep_frac.device
            return (
                torch.zeros(0, dim_sh, device=dev),
                torch.zeros(0, self.num_rbf, device=dev),
                None,
            )

        target_orbit, source_orbit = edge_index[0], edge_index[1]
        source_image = data.geo_edge_source_image  # (E,)
        shift = data.geo_edge_shift  # (E, 3)

        # Target representative (differentiable).
        r_p = rep_frac[target_orbit]  # (E, 3)

        # Source representative (differentiable).
        r_q = rep_frac[source_orbit]  # (E, 3)

        # Source equivalent-atom: r_{q,k} = r_q @ W.T + w  (row convention).
        if hasattr(data, "orbit_sym_ops_W_frac") and data.orbit_sym_ops_W_frac is not None:
            W = data.orbit_sym_ops_W_frac[source_orbit, source_image]  # (E, 3, 3) frac
            w = data.orbit_sym_ops_w_frac[source_orbit, source_image]  # (E, 3) frac
        else:
            # Fallback: use frozen source_frac (no source gradient).
            W = None
            w = None

        if W is not None and w is not None:
            r_qk = torch.bmm(r_q.unsqueeze(1), W.transpose(1, 2)).squeeze(1) + w
        else:
            r_qk = data.geo_edge_source_frac  # frozen fallback

        # Edge vector: v = r_qk + L - r_p.
        v_frac = r_qk + shift - r_p  # (E, 3)

        # Cartesian displacement.
        lat_per_edge = lattice_per_node[target_orbit]
        v_cart = torch.bmm(v_frac.unsqueeze(1), lat_per_edge).squeeze(1)

        d = torch.norm(v_cart, dim=-1)
        rhat = v_cart / (d.unsqueeze(-1).clamp(min=1e-8))
        edge_sh = o3.spherical_harmonics(
            self._irreps_sh, rhat, normalize=True, normalization="component"
        )
        edge_rbf = self.rbf(d)

        source_rotations = self._compute_source_rotations(data, lattice_per_node)
        return edge_sh, edge_rbf, source_rotations

    # ------------------------------------------------------------------
    # Source image rotation matrices
    # ------------------------------------------------------------------

    def _compute_source_rotations(
        self, data, lattice_per_node: torch.Tensor, use_dynamic: bool = True
    ) -> Optional[torch.Tensor]:
        """Build per-sub-edge **Cartesian** rotation matrices.

        Uses dynamic computation from stored fractional W when
        ``use_dynamic=True``:

            R_e3nn(A, W) = A.T @ W @ A^{-T}

        This ensures that when the lattice transforms A → A @ Q.T,
        the source rotations automatically conjugate Q·R·Q^T,
        keeping the model consistent under global rotation.

        Falls back to precomputed ``orbit_sym_ops_rotations`` if
        ``orbit_sym_ops_W_frac`` is unavailable.

        Returns (E_geo, 3, 3) Cartesian rotation matrices, or None.
        """
        if use_dynamic and hasattr(data, "orbit_sym_ops_W_frac") and data.orbit_sym_ops_W_frac is not None:
            W_frac = data.orbit_sym_ops_W_frac  # (K, max_mult, 3, 3) fractional
        elif hasattr(data, "orbit_sym_ops_rotations") and data.orbit_sym_ops_rotations is not None:
            sym_rot = data.orbit_sym_ops_rotations
            source_orbit = data.geo_edge_index[1]
            source_image = data.geo_edge_source_image
            return sym_rot[source_orbit, source_image]
        else:
            return None

        source_orbit = data.geo_edge_index[1]
        source_image = data.geo_edge_source_image
        A = lattice_per_node[source_orbit]  # (E, 3, 3)
        dev = A.device

        # Explicitly move W_frac to the same device as lattice.
        W = W_frac[source_orbit, source_image].to(device=dev, dtype=A.dtype)

        # R_e3nn = A.T @ W @ A^{-T}
        A_T = A.transpose(1, 2)
        A_inv_T = torch.inverse(A).transpose(1, 2)
        R_e3nn = torch.bmm(torch.bmm(A_T, W), A_inv_T)

        return R_e3nn

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward_features(
        self,
        data,
        pos_cart: Optional[torch.Tensor] = None,
    ):
        """Run the encoder up to the pre-readout projection and return
        per-orbit equivariant features.

        Unlike ``forward``, this method does **not** apply the pre-readout
        collapse to scalars and does not run the final graph-level readout.
        Downstream heads (e.g. :class:`TensorPropertyHead`) can therefore
        consume node-level equivariant features directly.

        Returns:
            dict with keys:
              * ``h``: (num_orbits, irreps_node.dim) equivariant node
                features in the layout given by ``self._irreps_node``.
              * ``irreps``: the ``o3.Irreps`` describing ``h``.
              * ``batch``: (num_orbits,) graph index per orbit.
              * ``edge_state``: final edge_state tensor (or None).
              * ``last_messages``: last per-edge message tensor (or None).
              * ``edge_rbf``: (E, num_rbf) invariant radial features.
              * ``geo_edge_index``: (2, E) geometric edge index.
              * ``geo_edge_weight``: (E,) or None.
              * ``num_graphs``: batch size.
        """
        batch = data.batch
        if batch is None:
            batch = torch.zeros(
                data.num_nodes, dtype=torch.long,
                device=data.orbit_element.device,
            )

        num_graphs = int(batch.max().item()) + 1

        lattice = data.lattice
        if lattice.dim() == 2:
            lattice = lattice.reshape(num_graphs, 3, 3)
        lattice_per_node = lattice[batch]

        if pos_cart is None:
            rep_frac_for_edges = data.orbit_rep_frac
            pos_cart = torch.bmm(
                data.orbit_rep_frac.unsqueeze(1), lattice_per_node
            ).squeeze(1)
        else:
            rep_frac_for_edges = torch.bmm(
                pos_cart.unsqueeze(1),
                torch.inverse(lattice_per_node)
            ).squeeze(1)

        h = self.node_encoder(atomic_numbers=data.orbit_element)

        edge_sh, edge_rbf, source_rotations = self._compute_edge_features(
            data, lattice_per_node, rep_frac=rep_frac_for_edges
        )
        geo_edge_weight = getattr(data, "geo_edge_weight", None)

        sym_edge_attr = data.sym_edge_attr
        if sym_edge_attr.dim() == 2 and sym_edge_attr.size(-1) > 1:
            sym_edge_attr = sym_edge_attr[:, :1]

        edge_state = None
        if self.use_edge_state and edge_rbf.size(0) > 0:
            ew = geo_edge_weight if geo_edge_weight is not None else torch.ones(
                edge_rbf.size(0), device=edge_rbf.device)
            edge_init_features = torch.cat([edge_rbf, ew.unsqueeze(-1)], dim=-1)
            edge_state = self.edge_state_init(edge_init_features)

        h = self.expand(
            h,
            data.geo_edge_index,
            edge_sh,
            edge_rbf,
            source_rotations=source_rotations,
            geo_edge_weight=geo_edge_weight,
        )
        _apply_after_expand = (
            self.use_site_projection or (
                self.use_site_irrep_projection and
                self.apply_site_projection in ("encoder_only", "each_layer", "readout_only")
            )
        )
        if self.use_site_irrep_projection and self.apply_site_projection == "readout_only":
            _apply_after_expand = False

        # Build compact projector blocks ONCE per forward (not per layer).
        _proj_blocks = None
        if _apply_after_expand:
            if self.use_site_projection and self.site_projector is None:
                _proj_blocks = self._build_projection_blocks(data, h.dtype)
                h = self._apply_projection_blocks(h, _proj_blocks)
            else:
                h = self._apply_site_projection(h, data)

        # Build source Wigner-D cache ONCE per forward.
        _src_D_cache = None
        if source_rotations is not None:
            _src_D_cache = _build_source_D_cache(source_rotations, self._irreps_node)

        last_messages: Optional[torch.Tensor] = None
        # Do we need to capture per-edge messages for the invariant readout?
        _need_messages_for_readout = (
            self.edge_invariant_readout is not None
            and getattr(self, "_edge_readout_message_irps", None) is not None
        )
        for i, layer in enumerate(self.layers):
            if self.use_edge_state and edge_state is not None:
                layer_result = layer(
                    h,
                    data.geo_edge_index,
                    edge_sh,
                    edge_rbf,
                    data.sym_edge_index,
                    sym_edge_attr,
                    source_rotations=source_rotations,
                    geo_edge_weight=geo_edge_weight,
                    edge_state=edge_state,
                    return_messages=True,
                    source_D_cache=_src_D_cache,
                )
                h, messages = layer_result
                last_messages = messages

                msg_inv = self.msg_summary(messages)
                edge_state = self.edge_state_updates[i](
                    edge_state, msg_inv, edge_rbf
                )
            elif _need_messages_for_readout:
                layer_result = layer(
                    h,
                    data.geo_edge_index,
                    edge_sh,
                    edge_rbf,
                    data.sym_edge_index,
                    sym_edge_attr,
                    source_rotations=source_rotations,
                    geo_edge_weight=geo_edge_weight,
                    return_messages=True,
                    source_D_cache=_src_D_cache,
                )
                h, messages = layer_result
                last_messages = messages
            else:
                h = layer(
                    h,
                    data.geo_edge_index,
                    edge_sh,
                    edge_rbf,
                    data.sym_edge_index,
                    sym_edge_attr,
                    source_rotations=source_rotations,
                    geo_edge_weight=geo_edge_weight,
                    source_D_cache=_src_D_cache,
                )

            _apply_after_mp = (
                self.use_site_projection or (
                    self.use_site_irrep_projection and
                    self.apply_site_projection == "each_layer"
                )
            )
            if _apply_after_mp:
                if _proj_blocks is not None:
                    h = self._apply_projection_blocks(h, _proj_blocks)
                else:
                    h = self._apply_site_projection(h, data)

        # Site projection right before readout, if configured. We apply it
        # here (still on the equivariant h, before pre_readout) so that
        # forward_features consumers see the same site-projected features
        # that forward() would carry into the scalar readout.
        if self.use_site_irrep_projection and self.apply_site_projection == "readout_only":
            h = self._apply_site_projection(h, data)

        return {
            "h": h,
            "irreps": self._irreps_node,
            "batch": batch,
            "edge_state": edge_state,
            "last_messages": last_messages,
            "edge_rbf": edge_rbf,
            "geo_edge_index": data.geo_edge_index,
            "geo_edge_weight": geo_edge_weight,
            "num_graphs": num_graphs,
        }

    def forward(
        self,
        data,
        pos_cart: Optional[torch.Tensor] = None,
    ):
        """Encode a batch of Wyckoff graphs.

        Args:
            data: PyG Data/Batch.
            pos_cart: Optional (K, 3) precomputed Cartesian positions.

        Returns:
            (num_graphs,) scalar predictions.
        """
        feats = self.forward_features(
            data, pos_cart=pos_cart
        )
        h = feats["h"]
        batch = feats["batch"]
        edge_state = feats["edge_state"]
        last_messages = feats["last_messages"]
        edge_rbf = feats["edge_rbf"]
        geo_edge_weight = feats["geo_edge_weight"]
        num_graphs = feats["num_graphs"]

        # --- Pre-readout projection (l>0 → scalars) ---
        h = self.pre_readout(h)

        # --- Readout ---
        node_weights = getattr(data, "multiplicity", None)

        edge_graph_repr = None
        if (
            self.use_edge_state
            or self.edge_invariant_readout is not None
        ) and data.geo_edge_index.numel() > 0:
            target_orbit = data.geo_edge_index[0]
            batch_edge = batch[target_orbit]
            ew = geo_edge_weight if geo_edge_weight is not None else torch.ones(
                target_orbit.size(0), device=h.device)

            # Path 1: Phase 3 EdgeInvariantReadout
            if self.edge_invariant_readout is not None:
                edge_graph_repr = self.edge_invariant_readout(
                    edge_state=edge_state,
                    edge_rbf=edge_rbf,
                    edge_weight=ew,
                    batch_edge=batch_edge,
                    num_graphs=num_graphs,
                    geo_edge_index=data.geo_edge_index,
                    num_nodes=h.shape[0],
                    message_features=last_messages,
                )
            # Path 2: legacy EdgePool
            elif self.edge_pool is not None and self.edge_readout and edge_state is not None:
                edge_graph_repr = self.edge_pool(
                    edge_state, ew, batch_edge, num_graphs
                )

        return self.readout(h, batch, node_weights=node_weights,
                           edge_graph_repr=edge_graph_repr,
                           atomic_numbers=data.orbit_element)

    def _apply_site_projection(self, h: torch.Tensor, data) -> torch.Tensor:
        """Apply per-orbit site-symmetry projection to enforce stabilizer invariance.

        Dispatch order (first available wins):
          1. Phase 2 precomputed projector (self.site_projector) — lookup by
             (space_group, orbit_letter_in_sg) from Phase 1 .pt table.
          2. Precomputed data.orbit_stabilizer_projections — legacy path.
          3. On-the-fly stabilizer identification + Wigner-D projection.
          4. No-op (if use_site_projection=False and no other path available).
        """
        # Phase 2 path (recommended)
        if self.site_projector is not None:
            return self._apply_site_irrep_projector(h, data)

        # Legacy on-the-fly / precomputed paths
        projections = getattr(data, "orbit_stabilizer_projections", None)
        if projections is not None:
            return self._apply_precomputed_projection(h, projections)

        if not self.use_site_projection:
            return h

        return self._compute_and_apply_projection(h, data)

    def _apply_site_irrep_projector(self, h: torch.Tensor, data) -> torch.Tensor:
        """Apply the Phase 2 SiteIrrepProjector using batch metadata."""
        sg = data.space_group
        letter = data.orbit_letter_in_sg
        batch = data.batch if hasattr(data, "batch") else None
        if batch is None:
            batch = torch.zeros(h.shape[0], dtype=torch.long, device=h.device)
        return self.site_projector(h, sg, letter, batch)

    def _apply_site_projection_ORIGINAL(self, h: torch.Tensor, data) -> torch.Tensor:
        """Deprecated entry point (kept for import compatibility)."""
        return self._apply_site_projection(h, data)

    def _apply_precomputed_projection(self, h, projections):
        """Apply precomputed (K, dim, dim) projection matrices."""
        P = projections.to(device=h.device, dtype=h.dtype)
        return torch.bmm(h.unsqueeze(1), P.transpose(1, 2)).squeeze(1)

    def _compute_and_apply_projection(self, h, data):
        """Compute stabilizer projections on-the-fly and apply them.

        Uses ``data.orbit_stabilizer_W_frac`` / ``orbit_stabilizer_mask`` which
        hold the FULL site stabilizer H_p per orbit (populated by
        :func:`structure_to_wyckoff_orbits`). For each orbit p, the projector
        is the Reynolds average
            P_p = (1/|H_p|) sum_{g in H_p} D(R_g)
        computed per (l, pi) irrep block. Scalar (l=0) blocks receive the
        identity (which is what D acts as for l=0).

        Falls back to the legacy filter over ``orbit_sym_ops_W_frac`` (image
        generators) if the stabilizer field is absent — this keeps old
        preprocessed shards importable but always returns identity on that
        legacy path (the filter is provably degenerate).

        The projector is built without grad; multiplication onto ``h`` keeps
        gradients flowing.
        """
        blocks = self._build_projection_blocks(data, h.dtype)
        return self._apply_projection_blocks(h, blocks)

    def _build_projection_blocks(self, data, dtype=torch.float32):
        """Build compact per-irrep projector blocks (built once per forward).

        Returns a dict mapping ``(l, p) -> P_block`` where ``P_block`` has
        shape ``(K, dim_l, dim_l)``.  Scalar (l=0) blocks are omitted — they
        are identity by definition and the apply function skips them.

        This replaces the old approach of building a full ``(K, D, D)`` matrix
        (e.g. 560x560 for "512x0e+16x1o") with only the small blocks that
        actually need projection (e.g. 3x3 for 1o, 5x5 for 2e).
        """
        from e3nn import o3

        device = data.orbit_element.device
        irreps = self._irreps_node
        K = data.orbit_element.shape[0]

        with torch.no_grad():
            stab_W = getattr(data, "orbit_stabilizer_W_frac", None)
            stab_mask = getattr(data, "orbit_stabilizer_mask", None)
            if stab_W is None or stab_mask is None:
                stab_W = data.orbit_sym_ops_W_frac
                w_frac = data.orbit_sym_ops_w_frac
                rep_frac = data.orbit_rep_frac
                mult_mask = data.orbit_mult_mask
                M = stab_W.shape[1]
                rep_expanded = rep_frac.unsqueeze(1).unsqueeze(-1).expand(K, M, 3, 1)
                W_float = stab_W.float()
                mapped = (W_float @ rep_expanded).squeeze(-1) + w_frac
                diff = (mapped - rep_frac.unsqueeze(1)) % 1.0
                diff = torch.min(diff, 1.0 - diff)
                stab_mask = (diff.sum(dim=-1) < 0.01) & mult_mask
            else:
                W_float = stab_W.float()
            stab_counts = stab_mask.sum(dim=1)

            batch_idx = data.batch if hasattr(data, "batch") else torch.zeros(
                K, dtype=torch.long, device=device,
            )
            lattice = data.lattice
            if lattice.dim() == 2:
                num_graphs = int(batch_idx.max().item()) + 1 if K > 0 else 1
                lattice = lattice.reshape(num_graphs, 3, 3)
            lat_per_node = lattice[batch_idx]

            nontrivial = (stab_counts > 1).nonzero(as_tuple=True)[0]
            if nontrivial.numel() == 0:
                return {}

            all_R_stab = []
            stab_slices = []
            cur = 0
            for p_idx in nontrivial.tolist():
                n_s = int(stab_counts[p_idx].item())
                W_s = W_float[p_idx][stab_mask[p_idx]]
                A = lat_per_node[p_idx]
                A_T = A.T
                A_inv_T = torch.inverse(A).T
                R_s = torch.bmm(
                    torch.bmm(A_T.unsqueeze(0).expand(n_s, 3, 3), W_s),
                    A_inv_T.unsqueeze(0).expand(n_s, 3, 3),
                )
                all_R_stab.append(R_s)
                stab_slices.append((cur, cur + n_s, p_idx))
                cur += n_s

            all_R = torch.cat(all_R_stab, dim=0)

            blocks = {}
            for _, ir in irreps:
                if ir.l == 0:
                    continue
                key = (ir.l, ir.p)
                if key in blocks:
                    continue
                D_l = ir.D_from_matrix(all_R.to(device=device, dtype=torch.float32))
                if D_l.dim() == 2:
                    D_l = D_l.unsqueeze(0)
                D_avg_all = torch.zeros(K, D_l.shape[-2], D_l.shape[-1],
                                        device=device, dtype=dtype)
                for start, end, p_idx in stab_slices:
                    D_avg_all[p_idx] = D_l[start:end].mean(dim=0).to(dtype)
                trivial = (stab_counts <= 1).nonzero(as_tuple=True)[0]
                if trivial.numel() > 0:
                    D_avg_all[trivial] = torch.eye(
                        D_l.shape[-1], device=device, dtype=dtype,
                    )
                blocks[key] = D_avg_all

        return blocks

    def _apply_projection_blocks(self, h, blocks):
        """Apply cached compact projector blocks to node features.

        Scalar (l=0) channels pass through unchanged.  For each l>0 irrep
        block, applies the ``(K, dim_l, dim_l)`` projector via einsum.
        """
        if not blocks:
            return h

        irreps = self._irreps_node
        out_parts = []
        idx = 0
        for mul, ir in irreps:
            dim_l = ir.dim
            block_size = mul * dim_l
            h_block = h[:, idx:idx + block_size]

            if ir.l == 0 or (ir.l, ir.p) not in blocks:
                out_parts.append(h_block)
            else:
                P = blocks[(ir.l, ir.p)]
                K = h.shape[0]
                h_reshaped = h_block.reshape(K, mul, dim_l)
                projected = torch.einsum("kij,kmj->kmi", P, h_reshaped)
                out_parts.append(projected.reshape(K, block_size))
            idx += block_size

        return torch.cat(out_parts, dim=-1)


__all__ = ["EquivariantWyckoffGNNEncoder"]
