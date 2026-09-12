"""UnifiedQuotientEquivariantGNN: a single backbone with hidden_irreps as the sole config.

``hidden_irreps="128x0e"`` and ``hidden_irreps="128x0e + 8x1o + 4x2e"`` use
the same class, the same ``self.blocks``, and the same forward loop.  The
high-l channels are different irrep sectors of one hidden representation,
not a separate branch.

When ``high_irreps`` is empty the model is numerically identical to
WyckoffCoGN0e — same parameter count (697,601), same forward outputs,
same gradients, same optimizer steps.

With ``output_type="symmetric_rank2"``, an additional tensor readout head
produces a per-graph symmetric 3x3 tensor (1x0e + 1x2e).  The scalar path
is completely unchanged when ``output_type="scalar"`` (default).
"""

from __future__ import annotations

from typing import Dict, Optional, Union

import torch
import torch.nn as nn
from e3nn import o3

from .scalar_sector import InputEmbedding, EdgeEmbedding
from .dynamic_tp_block import UnifiedDynamicTPBlock, split_scalar_and_high
from .tensor_adapter import voigt36_to_cartesian333
from .equivariant_sector import (
    UnifiedQuotientEquivariantBlock,
    precompute_batch_geometry,
)


def _parse_hidden_irreps(hidden_irreps_str: str):
    """Parse hidden_irreps into (irreps, scalar_mul, high_irreps, high_dim, lmax).

    ``simplify()`` merges repeated entries such as "2x1o + 3x1o" so that a
    sector cannot appear twice in the state layout.
    """
    irreps = o3.Irreps(hidden_irreps_str).simplify()

    scalar_mul, high_irreps = split_scalar_and_high(irreps)
    if scalar_mul == 0:
        raise ValueError(
            f"hidden_irreps must contain at least one 0e term, got '{hidden_irreps_str}'"
        )

    lmax = max((ir.l for _, ir in irreps), default=0)

    return irreps, scalar_mul, high_irreps, high_irreps.dim, lmax


class UnifiedQuotientEquivariantGNN(nn.Module):
    """Unified crystallographic quotient-equivariant backbone.

    The single configuration ``hidden_irreps`` determines the full hidden
    representation.  When it contains only ``Nx0e`` the model reduces to
    the coGN-compatible scalar model; when it includes higher-order
    irreps the same blocks jointly update all sectors.

    Args:
        hidden_irreps: Full irreps string, e.g. ``"128x0e"`` or
            ``"128x0e + 8x1o + 4x2e"``.
        num_layers: Number of message-passing layers (default 5).
        num_rbf: Gaussian RBF bins (default 32).
        rbf_max: Maximum RBF distance in Angstrom (default 8.0).
        output_type: ``"scalar"`` (default) or ``"symmetric_rank2"``.
            In scalar mode, forward returns a 1-D tensor of per-graph
            predictions.  In symmetric_rank2 mode, forward returns a dict
            with ``"scalar"`` and ``"tensor"`` keys.
    """

    def __init__(
        self,
        hidden_irreps: str = "128x0e",
        num_layers: int = 5,
        num_rbf: int = 32,
        rbf_max: float = 8.0,
        output_type: str = "scalar",
        use_source_image_transport: bool = True,
        point_group_symmetry: bool = False,
        global_point_group_projection: bool = False,
        block_type: str = "scalar",
        use_scalar_to_high_l: bool = True,
        use_high_l_propagation: bool = True,
        use_site_irrep_projection: bool = False,
        site_projector_path: str = "",
        site_irrep_table_path: str = "",
        site_irrep_lmax: int = 4,
        apply_site_projection: str = "each_layer",
        unknown_projector_policy: str = "identity",
        missing_irrep_policy: str = "identity",
        high_l_feedback_type: str = "norm2",
        high_l_aggregation: str = "mean",
        pool_mode: str = "mean",
        **kwargs,
    ):
        super().__init__()

        self.irreps, self.scalar_mul, self.high_irreps, self.high_dim, self.lmax = (
            _parse_hidden_irreps(hidden_irreps)
        )
        self.num_layers = num_layers
        self.output_type = output_type
        self.use_source_image_transport = use_source_image_transport
        self.point_group_symmetry = point_group_symmetry
        self.global_point_group_projection = global_point_group_projection
        self.block_type = block_type
        self.pool_mode = pool_mode

        self.input_embedding = InputEmbedding(self.scalar_mul, num_rbf)
        self.edge_embedding = EdgeEmbedding(num_rbf, rbf_max, self.scalar_mul)

        self.site_projector = None
        self.apply_site_projection = apply_site_projection

        if block_type == "dynamic_tp":
            self.blocks = nn.ModuleList([
                UnifiedDynamicTPBlock(
                    hidden_irreps=str(self.irreps),
                    layer_idx=i,
                )
                for i in range(num_layers)
            ])
            self.use_angular_readout = False

            if use_site_irrep_projection and self.high_dim > 0:
                if apply_site_projection not in ("each_layer", "final", "none"):
                    raise ValueError(
                        f"apply_site_projection must be 'each_layer', 'final' or "
                        f"'none'; got {apply_site_projection!r}"
                    )
                if not site_projector_path:
                    raise ValueError(
                        "use_site_irrep_projection=True but site_projector_path is "
                        "empty. Point it at a Phase 1 site_projectors_lmaxN.pt file."
                    )
                from wyckoff_gnn.models.wyckoff_equiv.site_projection import (
                    SiteIrrepProjector,
                )
                # Projected over the full unified state; 0e blocks are identity,
                # so the scalar sector passes through unchanged.
                self.site_projector = SiteIrrepProjector(
                    irreps=self.irreps,
                    projector_path=site_projector_path,
                    table_path=site_irrep_table_path or None,
                    lmax=site_irrep_lmax,
                    unknown_projector_policy=unknown_projector_policy,
                    missing_irrep_policy=missing_irrep_policy,
                )
        elif block_type in ("legacy", "scalar"):
            block_kwargs = dict(
                hidden_irreps=hidden_irreps,
                use_scalar_to_high_l=use_scalar_to_high_l,
                use_high_l_propagation=use_high_l_propagation,
                use_site_irrep_projection=use_site_irrep_projection,
                high_l_feedback_type=high_l_feedback_type,
                high_l_aggregation=high_l_aggregation,
            )
            self.blocks = nn.ModuleList([
                UnifiedQuotientEquivariantBlock(layer_idx=i, **block_kwargs)
                for i in range(num_layers)
            ])
        else:
            raise ValueError(
                f"Unknown block_type={block_type!r}. Valid: 'dynamic_tp', 'legacy', 'scalar'"
            )

        self.readout = nn.Linear(self.scalar_mul, 1, bias=True)

        if output_type == "symmetric_rank2":
            from .tensor_readout import SymmetricRank2Readout
            self.tensor_readout = SymmetricRank2Readout(
                scalar_mul=self.scalar_mul,
                high_irreps=str(self.high_irreps),
            )
            if point_group_symmetry:
                from .point_group_symmetry import PointGroupSymmetryEnforcement
                self.symmetry_enforcement = PointGroupSymmetryEnforcement(enabled=True)
        elif output_type == "piezo_rank3":
            from .tensor_readout import PiezoRank3Readout
            self.tensor_readout = PiezoRank3Readout(
                scalar_mul=self.scalar_mul,
                high_irreps=str(self.high_irreps),
            )
            if point_group_symmetry:
                raise ValueError(
                    "point_group_symmetry is implemented for rank-2 tensors only "
                    "and cannot be combined with output_type='piezo_rank3'."
                )
        elif output_type != "scalar":
            raise ValueError(
                f"Unknown output_type={output_type!r}. "
                f"Valid: 'scalar', 'symmetric_rank2', 'piezo_rank3'"
            )

        if global_point_group_projection and output_type not in (
            "piezo_rank3", "symmetric_rank2"
        ):
            raise ValueError(
                "global_point_group_projection requires a tensor output_type "
                f"('symmetric_rank2' or 'piezo_rank3'), got {output_type!r}."
            )

    def apply_reference_init(self):
        """Apply Keras-compatible initialization (matches WyckoffCoGN0e).

        - nn.Linear → GlorotUniform (Xavier), bias → zeros
        - nn.Embedding → uniform(-0.05, 0.05)
        - ZeroInitHighLToScalar.mlp[-1] → preserved zero-init (not overwritten)
        - Angular modules in dynamic_tp blocks → preserved (zero or small init)

        This ensures that when hidden_irreps="128x0e", the Unified model
        starts from the same initial conditions as WyckoffCoGN0e.
        """
        import math
        from wyckoff_gnn.models.unified_equivariant.equivariant_sector import (
            ZeroInitHighLToScalar,
        )
        from wyckoff_gnn.models.unified_equivariant.dynamic_tp_block import (
            UnifiedDynamicTPBlock,
        )

        protected_ids = set()

        for module in self.modules():
            if isinstance(module, ZeroInitHighLToScalar):
                protected_ids.add(id(module.mlp[-1]))

        for module in self.modules():
            if isinstance(module, UnifiedDynamicTPBlock):
                for name in module._get_angular_module_names():
                    angular_mod = getattr(module, name, None)
                    if angular_mod is not None:
                        for sub in angular_mod.modules():
                            if isinstance(sub, nn.Linear):
                                protected_ids.add(id(sub))

        if hasattr(self, "angular_readout"):
            protected_ids.add(id(self.angular_readout[-1]))

        for module in self.modules():
            if id(module) in protected_ids:
                continue
            if isinstance(module, nn.Linear):
                fan_in = module.in_features
                fan_out = module.out_features
                limit = math.sqrt(6.0 / (fan_in + fan_out))
                nn.init.uniform_(module.weight, -limit, limit)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.uniform_(module.weight, -0.05, 0.05)

    def forward_features(self, data) -> Dict[str, torch.Tensor]:
        """Run the backbone and return node-level features.

        This is the shared encoding path used by both scalar and tensor
        readout heads.  No task-specific computation happens here.

        Returns:
            dict with keys:
                scalar_node: (N, scalar_mul)
                high_l_node: (N, high_dim) or None
                multiplicity: (N,)
                batch: (N,)
                num_graphs: int
        """
        z = data.orbit_element
        edge_index = data.geo_edge_index
        distances = data.geo_edge_distance
        multiplicity = data.multiplicity
        batch = data.batch
        num_graphs = getattr(data, 'num_graphs', None) or (batch.max().item() + 1)

        h = self.input_embedding(z)
        e = self.edge_embedding(distances)

        if self.block_type == "dynamic_tp":
            return self._forward_features_dynamic(h, e, data, batch, multiplicity, num_graphs)

        high_l = None
        edge_sh = None
        wigner_d_cache = None

        if self.high_dim > 0:
            high_l = h.new_zeros(h.size(0), self.high_dim)
            edge_sh, wigner_d_cache = precompute_batch_geometry(
                data, self.high_irreps, self.lmax
            )
            if not self.use_source_image_transport:
                wigner_d_cache = {
                    k: torch.eye(v.size(-1), dtype=v.dtype, device=v.device)
                    .expand_as(v).contiguous()
                    for k, v in wigner_d_cache.items()
                }

        for block in self.blocks:
            h, high_l = block(h, high_l, e, edge_index, edge_sh, wigner_d_cache)

        return {
            "scalar_node": h,
            "high_l_node": high_l,
            "multiplicity": multiplicity,
            "batch": batch,
            "num_graphs": num_graphs,
        }

    def _apply_site_projection(self, h: torch.Tensor, data) -> torch.Tensor:
        """Project the unified state onto each orbit's site-symmetry invariants."""
        return self.site_projector(
            h,
            data.space_group,
            data.orbit_letter_in_sg,
            data.batch,
            stab_hash=getattr(data, "orbit_stab_hash", None),
            projector_key_idx=getattr(data, "site_projector_key_idx", None),
        )

    def _forward_features_dynamic(self, h, e, data, batch, multiplicity, num_graphs):
        """Forward for dynamic_tp blocks: unified state tensor."""
        edge_index = data.geo_edge_index

        if self.high_dim > 0:
            high_l = h.new_zeros(h.size(0), self.high_dim)
            h = torch.cat([h, high_l], dim=-1)
            edge_sh, wigner_d_cache = precompute_batch_geometry(
                data, self.high_irreps, self.lmax
            )
            if not self.use_source_image_transport:
                wigner_d_cache = {
                    k: torch.eye(v.size(-1), dtype=v.dtype, device=v.device)
                    .expand_as(v).contiguous()
                    for k, v in wigner_d_cache.items()
                }
            agg_norm = self.blocks[0].compute_agg_norm(edge_index[0], h.size(0))
        else:
            edge_sh = None
            wigner_d_cache = None
            agg_norm = None

        project_each_layer = (
            self.site_projector is not None
            and self.apply_site_projection == "each_layer"
        )

        for block in self.blocks:
            h = block(h, e, edge_index, edge_sh, wigner_d_cache, agg_norm)
            if project_each_layer:
                h = self._apply_site_projection(h, data)

        if self.site_projector is not None and self.apply_site_projection == "final":
            h = self._apply_site_projection(h, data)

        scalar_node = h[:, :self.scalar_mul]
        high_l_node = h[:, self.scalar_mul:] if self.high_dim > 0 else None

        return {
            "scalar_node": scalar_node,
            "high_l_node": high_l_node,
            "multiplicity": multiplicity,
            "batch": batch,
            "num_graphs": num_graphs,
        }

    def forward(
        self, data
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        features = self.forward_features(data)

        if self.output_type == "scalar":
            g = self._scalar_readout(
                features["scalar_node"],
                features["multiplicity"],
                features["batch"],
                features["num_graphs"],
                pool_mode=self.pool_mode,
            )
            scalar_pred = self.readout(g).squeeze(-1)

            if (
                self.high_dim > 0
                and getattr(self, "use_angular_readout", False)
                and hasattr(self, "angular_readout")
            ):
                high_l = features["high_l_node"]
                multiplicity = features["multiplicity"]
                batch = features["batch"]
                num_graphs = features["num_graphs"]

                inv_parts = []
                offset = 0
                for mul, ir in self.high_irreps:
                    if ir.l == 0:
                        offset += mul * ir.dim
                        continue
                    h_sec = high_l[:, offset:offset + mul * ir.dim]
                    h_3d = h_sec.reshape(-1, mul, ir.dim)
                    inv = h_3d.pow(2).sum(dim=-1) / ir.dim
                    inv_parts.append(inv)
                    offset += mul * ir.dim

                if inv_parts:
                    inv_all = torch.cat(inv_parts, dim=-1)
                    g_inv = self._scalar_readout(
                        inv_all, multiplicity, batch, num_graphs,
                        pool_mode=self.pool_mode,
                    )
                    angular_correction = self.angular_readout(g_inv).squeeze(-1)
                    scalar_pred = scalar_pred + angular_correction

            return scalar_pred

        # tensor output: symmetric_rank2 or piezo_rank3
        g_scalar = self._scalar_readout(
            features["scalar_node"],
            features["multiplicity"],
            features["batch"],
            features["num_graphs"],
            pool_mode=self.pool_mode,
        )
        scalar_pred = self.readout(g_scalar).squeeze(-1)

        tensor_pred = self.tensor_readout(
            features["scalar_node"],
            features["high_l_node"],
            features["multiplicity"],
            features["batch"],
            features["num_graphs"],
        )

        if self.point_group_symmetry and hasattr(self, "symmetry_enforcement"):
            lattice = data.lattice  # (G, 3, 3)
            space_group = data.space_group  # (G,)
            tensor_pred["cartesian"] = self.symmetry_enforcement(
                tensor_pred["cartesian"],
                lattice,
                space_group,
            )

        if self.global_point_group_projection:
            # Neumann-symmetry projection of the global Cartesian tensor
            # output. Each graph carries its own projector, applied with a
            # batched matmul so no graph ever borrows another's group.
            P = data.tensor_point_group_projector
            if self.output_type == "piezo_rank3":
                raw = tensor_pred["voigt"]
                G, dim = raw.shape[0], 18
            else:
                raw = tensor_pred["cartesian"]
                G, dim = raw.shape[0], 9
            P = P.reshape(G, dim, dim).to(dtype=raw.dtype, device=raw.device)
            projected = torch.einsum("gij,gj->gi", P, raw.reshape(G, dim))
            if self.output_type == "piezo_rank3":
                projected = projected.reshape(G, 3, 6)
                tensor_pred["voigt_raw"] = raw
                tensor_pred["cartesian_raw"] = tensor_pred["cartesian"]
                tensor_pred["voigt"] = projected
                tensor_pred["cartesian"] = voigt36_to_cartesian333(projected)
            else:
                tensor_pred["cartesian_raw"] = raw
                tensor_pred["cartesian"] = projected.reshape(G, 3, 3)

        return {
            "scalar": scalar_pred,
            "tensor": tensor_pred,
        }

    @staticmethod
    def _scalar_readout(
        h: torch.Tensor,
        multiplicity: torch.Tensor,
        batch: torch.Tensor,
        num_graphs: int,
        pool_mode: str = "mean",
    ) -> torch.Tensor:
        mult = multiplicity.to(dtype=h.dtype, device=h.device).unsqueeze(-1)
        weighted = h * mult
        pooled_sum = h.new_zeros(num_graphs, h.size(1))
        pooled_sum.index_add_(0, batch, weighted)
        if pool_mode == "sum":
            # Extensive property (e.g. total energy): sum over atoms, no
            # division by multiplicity.
            return pooled_sum
        norm_sum = h.new_zeros(num_graphs, 1)
        norm_sum.index_add_(0, batch, mult)
        g = pooled_sum / norm_sum.clamp(min=1e-8)
        return g
