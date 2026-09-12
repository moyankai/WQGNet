"""WyckoffGNN model for crystal property prediction (Stage B).

O(3)-equivariant architecture with e3nn Clebsch-Gordan tensor products,
gated non-linearities.

Force modes / Symmetry-projected gradients
---------------------------------------------

**WyckoffGNN cannot predict physical atomic forces by design.**
It operates on the symmetry-constrained Wyckoff manifold where
representative coordinates are constrained by space-group symmetry.
Fixed Wyckoff sites (dof=0) have no Cartesian degree of freedom.

- ``"none"``: No gradient (default for scalar property prediction).
- ``"cartesian"``: P1 / mult=1 only.  True Cartesian force
  F = -dE/dx_atom via autograd.  Only valid when each atom is its
  own orbit (no symmetry constraint on coordinates).
- ``"symmetric_expand"``: Symmetry-projected gradient (NOT a force).
  G_rep = -(1/m_p)*dE/dx_rep, expanded to all atoms via
  G_{p,a}=R_{p,a}@G_rep.  Returns (M_total,3) per-atom values of
  the constrained gradient.  For dof=0 sites this is a projection,
  not a physical force.
- ``"wyckoff_generalized"``: Generalized force Q_u = -dE/du on the
  Wyckoff free-parameter manifold.  Only meaningful for dof>0 orbits.

.. warning::
   The outputs of ``symmetric_expand`` and ``wyckoff_generalized``
   are NOT physical atomic forces.  Physical force prediction
   requires P1 fallback (``cartesian``) or a residual atom graph
   that lifts the symmetry constraint.  See theory.md §11.3, §11.6.
"""

from __future__ import annotations

import warnings
from typing import Optional, Tuple

import torch
import torch.nn as nn

from wyckoff_gnn.models.wyckoff_equiv.e3nn_encoder import EquivariantWyckoffGNNEncoder
from wyckoff_gnn.utils.symmetry_expand import (
    expand_orbit_vectors_to_atoms,
    force_rep_from_symmetric_energy_gradient,
)


class WyckoffGNN(nn.Module):
    """O(3)-equivariant WyckoffGNN for crystal property prediction.

    Uses e3nn-based equivariant message passing with Clebsch-Gordan tensor
    products and gated non-linearities. Initial node features encode only
    element identity (Z) as pure scalars; geometry is introduced by the
    geometric lifting layer via real edge spherical harmonics.

    Forward interface: ``model(data) -> (batch_size,)`` scalar predictions.

    Args:
        num_layers: Number of equivariant MP layers (>= 1).
        num_rbf: Number of RBF kernels for distance encoding.
        rbf_max: Maximum distance for RBF expansion (Å).
        pool: Graph-level pooling (mean/sum/max/attention).
        readout_mode: ``"intensive"`` (per-atom, default) or ``"extensive"``
            (total energy via multiplicity-weighted sum).
        gradient_mode: ``"none"``, ``"cartesian"``, or ``"symmetric_expand"``.
            See module docstring for semantics.
        dropout: Dropout rate.
        init_irreps: Pure-scalar irreps for the element-only node embedding
            (default ``"128x0e"``). Decoupled from hidden_irreps.
    """

    def __init__(
        self,
        num_layers: int = 4,
        num_rbf: int = 32,
        rbf_max: float = 8.0,
        pool: str = "mean",
        readout_mode: str = "intensive",
        gradient_mode: str = "none",
        dropout: float = 0.1,
        init_irreps: str = "64x0e",
        use_site_projection: bool = False,
        hidden_irreps: str = "64x0e+16x1o+8x2e",
        radial_gate_mode: str = "per_type",
        use_edge_state: bool = False,
        edge_state_dim: int = 64,
        edge_state_layers: int = 2,
        edge_readout: bool = False,
        edge_pool_mode: str = "normalized",
        tp_mode: str = "dynamic_v2",
        radial_mlp_width: int = 128,
        n_bessel: int = 16,
        # Phase 2: precomputed site-irrep projection
        use_site_irrep_projection: bool = False,
        site_projector_path: str = "",
        site_irrep_table_path: str = "",
        site_irrep_lmax: int = 4,
        apply_site_projection: str = "each_layer",
        # Phase 3: formal edge readout
        edge_readout_mode: str = "legacy",
        edge_invariant_features: Optional[list] = None,
        edge_pool_reduce: str = "sum",
        orbit_pair_hidden_dim: int = 64,
        orbit_pair_output_dim: int = 64,
        # Phase 4: NequIP-style learnable per-element scale/shift (readout stage)
        per_type_shift_learnable: bool = False,
        per_type_scale_learnable: bool = False,
        max_atomic_number: int = 118,
        use_atom_props: bool = False,   # periodic-table physical properties in embedding
        # Multi-property support
        property_type: str = "graph_scalar_intensive",
        property_name: Optional[str] = None,
        target_irreps: Optional[str] = None,
    ):
        super().__init__()
        self.init_irreps = init_irreps
        self.readout_mode = readout_mode
        self.gradient_mode = gradient_mode
        self.use_site_projection = use_site_projection
        self.hidden_irreps = hidden_irreps
        self.radial_gate_mode = radial_gate_mode
        self.use_edge_state = use_edge_state
        self.tp_mode = tp_mode

        # --- Resolve PropertySpec from registry ---
        from wyckoff_gnn.properties.registry import (
            get_property_spec, PhysicalType, Status, PROPERTY_REGISTRY,
        )
        # If user gave property_name, use it directly. Otherwise map old
        # property_type to a default name (backward compat).
        if property_name is not None:
            self._property_spec = get_property_spec(property_name)
        else:
            # Legacy: map old property_type string to closest registry entry
            _TYPE_TO_DEFAULT_NAME = {
                "graph_scalar_intensive": "formation_energy_peratom",
                "graph_scalar_extensive": "optb88vdw_total_energy",
                "graph_vector": None,
                "graph_tensor": None,
                "atom_scalar": None,
                "atom_vector": None,
                "atom_tensor": None,
                "hamiltonian": "hamiltonian",
            }
            default_name = _TYPE_TO_DEFAULT_NAME.get(property_type)
            if default_name and default_name in PROPERTY_REGISTRY:
                self._property_spec = get_property_spec(default_name)
            else:
                # Fallback: create a minimal spec for backward compat
                from wyckoff_gnn.properties.registry import PropertySpec
                self._property_spec = PropertySpec(
                    name=property_type,
                    source="config",
                    level="graph" if "graph" in property_type else ("atom" if "atom" in property_type else "hamiltonian"),
                    physical_type=property_type,
                    target_shape=(1,),
                    target_irreps=target_irreps,
                    required_label_components=[],
                    output_head="TensorPropertyHead" if target_irreps else "ScalarReadout",
                    pooling="mean",
                    status=Status.SMOKE_ONLY,
                )

        # Check status: unsupported raises immediately
        if self._property_spec.status == Status.UNSUPPORTED:
            raise ValueError(
                f"Property '{self._property_spec.name}' is marked UNSUPPORTED. "
                f"Reason: {self._property_spec.notes}"
            )

        # Expose for external inspection
        self.property_name = self._property_spec.name
        self.property_type = self._property_spec.physical_type
        self._target_irreps = self._property_spec.target_irreps or target_irreps

        # Pool from spec (authoritative)
        spec_pool = self._property_spec.pooling
        if spec_pool != "none" and spec_pool in ("mean", "sum"):
            pool = spec_pool

        # Per-type energy statistics (for formation energy prediction)
        max_z = 118
        self.register_buffer("per_type_scale", torch.ones(max_z + 1))
        self.register_buffer("per_type_shift", torch.zeros(max_z + 1))

        # Equivariant encoder.
        self.encoder = EquivariantWyckoffGNNEncoder(
            num_layers=num_layers,
            num_rbf=num_rbf,
            rbf_max=rbf_max,
            pool=pool,
            dropout=dropout,
            init_irreps=init_irreps,
            readout_mode=readout_mode,
            use_site_projection=use_site_projection,
            hidden_irreps=hidden_irreps,
            radial_gate_mode=radial_gate_mode,
            use_edge_state=use_edge_state,
            edge_state_dim=edge_state_dim,
            edge_state_layers=edge_state_layers,
            edge_readout=edge_readout,
            use_atom_props=use_atom_props,
            edge_pool_mode=edge_pool_mode,
            tp_mode=tp_mode,
            radial_mlp_width=radial_mlp_width,
            n_bessel=n_bessel,
            # Phase 2 pass-through
            use_site_irrep_projection=use_site_irrep_projection,
            site_projector_path=site_projector_path,
            site_irrep_table_path=site_irrep_table_path,
            site_irrep_lmax=site_irrep_lmax,
            apply_site_projection=apply_site_projection,
            # Phase 3 pass-through
            edge_readout_mode=edge_readout_mode,
            edge_invariant_features=edge_invariant_features,
            edge_pool_reduce=edge_pool_reduce,
            orbit_pair_hidden_dim=orbit_pair_hidden_dim,
            orbit_pair_output_dim=orbit_pair_output_dim,
            # Phase 4 pass-through
            per_type_shift=per_type_shift_learnable,
            per_type_scale=per_type_scale_learnable,
            max_atomic_number=max_atomic_number,
            # Phase 5 removed (angular edge was impractical — OOM on 80GB GPU)
        )

        # --- Property head (for non-scalar property types) ---
        self.property_head = None
        if self._property_spec.output_head != "ScalarReadout":
            self._build_property_head(hidden_irreps, pool)

    def _build_property_head(self, hidden_irreps, pool):
        """Build the output head as specified by self._property_spec."""
        from wyckoff_gnn.properties.registry import PhysicalType
        from wyckoff_gnn.models.wyckoff_equiv.e3nn_layers import compute_irreps

        ir = compute_irreps(hidden_irreps)
        irreps_node = ir["node"]
        spec = self._property_spec

        if spec.output_head == "TensorPropertyHead":
            from wyckoff_gnn.models.wyckoff_equiv.readouts import TensorPropertyHead
            t_irreps = spec.target_irreps or self._target_irreps
            if not t_irreps:
                raise ValueError(
                    f"Property '{spec.name}' requires target_irreps but none specified "
                    f"in registry or config."
                )
            use_pool = spec.pooling if spec.pooling != "none" else "none"
            self.property_head = TensorPropertyHead(
                irreps_node_in=irreps_node,
                target_irreps=t_irreps,
                pool=use_pool,
            )

        elif spec.output_head == "DiagonalCanonicalHead":
            from wyckoff_gnn.models.wyckoff_equiv.readouts import TensorPropertyHead
            n_components = len(spec.required_label_components)
            self.property_head = TensorPropertyHead(
                irreps_node_in=irreps_node,
                target_irreps=f"{n_components}x0e",
                pool=spec.pooling,
            )

        elif spec.output_head == "WyckoffHamiltonianHead":
            from wyckoff_gnn.models.wyckoff_equiv.readouts import WyckoffHamiltonianHead
            node_dim = self.encoder._readout_irreps_in.dim
            edge_dim = self.encoder.num_rbf
            self.property_head = WyckoffHamiltonianHead(
                node_feature_dim=node_dim,
                edge_feature_dim=edge_dim,
                basis="spd",
            )

        elif spec.output_head in ("ElasticTensorHead", "PiezoTensorHead"):
            raise NotImplementedError(
                f"Output head '{spec.output_head}' for property '{spec.name}' "
                f"is not yet implemented. Status: {spec.status}."
            )

        else:
            raise ValueError(
                f"Unknown output_head '{spec.output_head}' in PropertySpec '{spec.name}'"
            )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, data) -> torch.Tensor:
        """Predict property from Wyckoff graph.

        Output shape depends on the PropertySpec:
          - ScalarReadout: (num_graphs,)
          - TensorPropertyHead: (num_graphs, dim) or (num_orbits, dim)
          - DiagonalCanonicalHead: (num_graphs, n_components)
          - WyckoffHamiltonianHead: Dict["onsite": ..., "offsite": ...]
        """
        from wyckoff_gnn.properties.registry import PhysicalType

        spec = self._property_spec

        # Scalar path: use the encoder's built-in readout (backward-compat).
        if spec.output_head == "ScalarReadout":
            return self.encoder(data)

        # Non-scalar: get equivariant per-orbit features.
        feats = self.encoder.forward_features(data)
        h = feats["h"]
        batch = feats["batch"]

        # Hamiltonian: needs scalar projection + edge features.
        if spec.output_head == "WyckoffHamiltonianHead":
            h_scalars = self.encoder.pre_readout(h)
            edge_features = feats["edge_rbf"]
            return self.property_head(
                h_scalars, edge_features, data.geo_edge_index
            )

        # Graph-level tensor/vector or DiagonalCanonical: pool over orbits.
        if spec.level == "graph":
            node_weights = getattr(data, "multiplicity", None)
            return self.property_head(h, batch, node_weights=node_weights)

        # Atom/orbit-level: per-node output (pool="none").
        return self.property_head(h, batch)

    def set_per_type_params(self, scale, shift):
        """Set frozen per-type scale and shift from training statistics."""
        self.per_type_scale[:len(scale)] = scale
        self.per_type_shift[:len(shift)] = shift

    # ------------------------------------------------------------------
    # Inference & force prediction
    # ------------------------------------------------------------------

    def predict(self, data) -> torch.Tensor:
        """Inference mode prediction."""
        self.eval()
        with torch.no_grad():
            return self.forward(data)

    def predict_energy_and_gradient(
        self,
        data,
        want_grad: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Predict energy and (optionally) forces via autograd.

        Force semantics depend on ``self.gradient_mode``:

        - ``"none"``: Returns ``None`` for forces.
        - ``"cartesian"``: F = -dE/d(pos_cart).  Works when each orbit
          has mult=1 (P1 / atomic graph).  Returns per-orbit forces
          (K_total, 3).
        - ``"symmetric_expand"``: For strictly symmetric structures.
          Computes F_rep = -(1/m_p) * dE/dx_rep, then expands to all
          atoms via F_{p,a} = R_{p,a} @ F_rep.  Returns per-atom forces
          (M_total, 3).  Requires ``atom_to_orbit`` and
          ``atom_image_index`` on the Data object.

        Args:
            data: PyG Data/Batch.
            want_grad: If True, compute gradient.

        Returns:
            energy: (batch_size,) scalar predictions.
            forces: Per-orbit or per-atom forces, depending on gradient_mode.
        """
        if not want_grad or self.gradient_mode == "none":
            energy = self.encoder(data)
            return energy, None

        # --- Enable dynamic_symmetric geometry for correct gradient ---
        prev_geom = self.encoder.geometry_mode
        self.encoder.geometry_mode = "dynamic_symmetric"

        # --- Prepare autograd-traceable positions ---
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

        frac = data.orbit_rep_frac.detach().requires_grad_(True)
        pos_cart = torch.bmm(
            frac.unsqueeze(1), lattice_per_node
        ).squeeze(1)

        # --- Forward ---
        energy = self.encoder(
            data, pos_cart=pos_cart,
        )

        # --- Compute gradient w.r.t. orbit representative positions ---
        grad = torch.autograd.grad(
            outputs=energy.sum(),
            inputs=pos_cart,
            create_graph=self.training,
            retain_graph=True,
        )[0]  # (K, 3)

        # --- Gradient mode dispatch ---
        if self.gradient_mode == "cartesian":
            grad_output = -grad  # (K, 3) per-orbit Cartesian
        elif self.gradient_mode == "wyckoff_generalized":
            grad_output = self._compute_generalized_gradient(grad, data)
        elif self.gradient_mode == "symmetric_expand":
            grad_output = self._symmetric_expand_gradient(grad, data)
        else:
            raise ValueError(
                f"Unknown gradient_mode='{self.gradient_mode}'. "
                f"Use 'none', 'cartesian', 'wyckoff_generalized', "
                f"or 'symmetric_expand'."
            )

        # --- Restore geometry mode ---
        self.encoder.geometry_mode = prev_geom

        return energy, grad_output

    def _compute_generalized_gradient(
        self, grad_rep: torch.Tensor, data
    ) -> torch.Tensor:
        """Compute generalized forces on Wyckoff free-parameter manifold.

        For each orbit p with free parameters u_p (dim = dof_p)::

            dE/du_{p,alpha} = grad_cart_p · (b_{p,alpha} @ A)

        where b_{p,alpha} = ∂r_p/∂u_{p,alpha} is the tangent basis
        in fractional space, and A is the lattice.

        For fixed Wyckoff sites (dof=0), the generalized force is empty.

        Returns:
            (K, max_dof) generalized forces, with zeros for dof=0 orbits.
        """
        K = grad_rep.shape[0]
        max_dof = int(data.orbit_dof.max().item()) if hasattr(data, "orbit_dof") else 0
        if max_dof == 0:
            return torch.zeros(K, 0, device=grad_rep.device)

        basis = data.orbit_param_basis_frac  # (K, max_dof, 3)
        if basis is None:
            return torch.zeros(K, 0, device=grad_rep.device)

        # Lattice for Cartesian projection.
        lattice = data.lattice
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(K, dtype=torch.long, device=grad_rep.device)
        if lattice.dim() == 2:
            lattice = lattice.unsqueeze(0)
        elif lattice.dim() == 3:
            pass
        else:
            num_graphs = int(batch.max().item()) + 1
            lattice = lattice.reshape(num_graphs, 3, 3)
        A = lattice[batch]  # (K, 3, 3)

        Q = torch.zeros(K, max_dof, device=grad_rep.device)
        dof = data.orbit_dof  # (K,)

        for p in range(K):
            d = int(dof[p].item())
            if d == 0:
                continue
            Ap = A[p]  # (3, 3)
            for alpha in range(d):
                b_frac = basis[p, alpha]  # (3,)
                b_cart = torch.matmul(b_frac, Ap)  # (3,) row convention
                Q[p, alpha] = -torch.dot(grad_rep[p], b_cart)

        return Q  # (K, max_dof)

    def _symmetric_expand_gradient(
        self, grad_rep: torch.Tensor, data
    ) -> torch.Tensor:
        """Expand orbit-representative gradient to all-atom forces.

        Uses the symmetry relation::

            F_p^rep = - grad_rep / m_p
            F_{p,a} = R_{p,a} @ F_p^rep

        where R_{p,a} is the **Cartesian** rotation matrix derived from
        the stored fractional W via R_cart = A @ W @ A^{-1}.

        Requires ``atom_to_orbit``, ``atom_image_index``,
        ``orbit_sym_ops_rotations``, ``orbit_mult_mask``,
        ``orbit_multiplicity``, and ``lattice`` on the Data object.
        """
        if not hasattr(data, "atom_to_orbit") or data.atom_to_orbit is None:
            raise RuntimeError(
                "symmetric_expand requires atom_to_orbit on the Data. "
                "Re-run graph building with atom_to_orbit from meta dict."
            )
        if not hasattr(data, "atom_image_index") or data.atom_image_index is None:
            raise RuntimeError(
                "symmetric_expand requires atom_image_index. "
                "Re-run graph building with atom_image_index from meta dict."
            )

        mult = data.multiplicity  # (K,)
        has_sym = (
            hasattr(data, "orbit_sym_ops_rotations")
            and data.orbit_sym_ops_rotations is not None
        )
        if not has_sym:
            raise RuntimeError(
                "symmetric_expand requires orbit_sym_ops_rotations."
            )

        # orbit_sym_ops_rotations is already Cartesian R_cart
        # (converted from fractional W at graph-build time).
        R_cart = data.orbit_sym_ops_rotations  # (K, M, 3, 3) Cartesian

        # F_rep = -grad / m_p
        F_rep = force_rep_from_symmetric_energy_gradient(grad_rep, mult)

        # Expand to atoms: F_{p,a} = R_cart_{p,a} @ F_rep
        F_atom = expand_orbit_vectors_to_atoms(
            orbit_vectors=F_rep,
            orbit_sym_ops_rotations=R_cart,
            orbit_mult_mask=data.orbit_mult_mask,
            atom_to_orbit=data.atom_to_orbit,
            atom_image_index=data.atom_image_index,
            vector_type="polar",
        )
        return F_atom


__all__ = ["WyckoffGNN"]
