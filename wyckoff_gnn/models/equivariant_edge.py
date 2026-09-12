"""WyckoffGNN: Edge-dominant equivariant GNN with site-symmetry by construction.

Key design principles:
    1. Node features ALWAYS live in the site-symmetry invariant subspace.
       Enforced by a precomputed projection P_i applied after every scatter-to-node.
       Cost: one batched matmul per layer (negligible vs TP).

    2. Only 1 TP per InteractionBlock: edge × D(R_k)·h_source → edge_update.
       Node update is a simple Linear + scatter + project (no extra TP).

    3. Edge features are NOT symmetry-constrained (pairwise interactions).
       User-configurable arbitrary o3.Irreps.

    4. Angular information enters via CG (source rotation D(R_k) encodes the
       geometry of equivalent-atom positions relative to the representative).

Data flow per layer:
    h_src_real = D(R_k) · h_q           [coset Wigner rotation, uses cached D]
    edge_msg = TP(edge, h_src_real) * gate(RBF)
    e_edge' = Norm(e_edge + Linear(edge_msg))
    node_agg = scatter_sum(Linear(e_edge'), target)
    h_node' = P_i @ Norm(h_node + node_agg)   [projection = 1 bmm, near-zero cost]
"""

from __future__ import annotations

import torch
import torch.nn as nn
from e3nn import o3
from torch_scatter import scatter

from wyckoff_gnn.utils.radial import RadialFeat

# GPU Wigner-D patch: move so3_generators to device of input.
try:
    import e3nn.o3._wigner as _w
    _orig_so3_gen = _w.so3_generators
    def _gpu_wigner_D(l, alpha, beta, gamma):
        dev = alpha.device
        Xa, Xb, Xc = _orig_so3_gen(l)
        Xa, Xb, Xc = Xa.to(dev), Xb.to(dev), Xc.to(dev)
        a = alpha.reshape(-1, 1, 1)
        b = beta.reshape(-1, 1, 1)
        c = gamma.reshape(-1, 1, 1)
        return (torch.matrix_exp(a * Xb) @
                torch.matrix_exp(b * Xa) @
                torch.matrix_exp(c * Xb))
    _w.wigner_D = _gpu_wigner_D
except Exception:
    pass

_IRREPS_SH = o3.Irreps("1x0e + 1x1o + 1x2e")


# ---------------------------------------------------------------------------
# Core utilities
# ---------------------------------------------------------------------------

class IrrepNorm(nn.Module):
    """Per-irrep normalisation: LayerNorm for l=0, RMS for l>0."""

    def __init__(self, irreps: o3.Irreps, eps: float = 1e-5):
        super().__init__()
        self.irreps = irreps
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(len(list(irreps))))
        self.scalar_norms = nn.ModuleDict()
        for i, (mul, ir) in enumerate(irreps):
            if ir.l == 0:
                self.scalar_norms[str(i)] = nn.LayerNorm(mul * ir.dim, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        blocks = []
        offset = 0
        for i, (mul, ir) in enumerate(self.irreps):
            dim = mul * ir.dim
            block = x[:, offset:offset + dim]
            if ir.l == 0:
                block = self.scalar_norms[str(i)](block)
            else:
                block = block.reshape(-1, mul, ir.dim)
                rms = (block.pow(2).mean(dim=-1, keepdim=True) + self.eps).sqrt()
                block = (block / rms.clamp(min=self.eps)).reshape(-1, mul * ir.dim)
            blocks.append(block * self.scale[i])
            offset += dim
        return torch.cat(blocks, dim=-1)


class ElementEncoder(nn.Module):
    """Z -> pure-scalar embedding, optionally with period-table physical properties."""
    def __init__(self, dim: int, max_z: int = 118, use_atom_props: bool = False):
        super().__init__()
        self.use_atom_props = use_atom_props
        self.embedding = nn.Embedding(max_z + 1, dim)
        if use_atom_props:
            from wyckoff_gnn.models.e3nn_layers import _N_ATOM_PROPERTIES
            self.prop_proj = nn.Linear(dim + _N_ATOM_PROPERTIES, dim)
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.embedding(z.clamp(0, self.embedding.num_embeddings - 1))
        if self.use_atom_props:
            from wyckoff_gnn.models.e3nn_layers import _gather_atom_props
            props = _gather_atom_props(z, h.device)
            h = self.prop_proj(torch.cat([h, props], dim=-1))
        return h


def precompute_wigner_D(R: torch.Tensor, irreps: o3.Irreps):
    """Compute Wigner-D once. Returns {l: (E, 2l+1, 2l+1)}."""
    D = {}
    for _, ir in irreps:
        if ir.l > 0 and ir.l not in D:
            D[ir.l] = ir.D_from_matrix(R)
    return D


def rotate_irreps(h, R, irreps, D_cache=None):
    """Apply D(R) to h. Uses precomputed D_cache if available."""
    E = h.size(0)
    if E == 0:
        return h
    if D_cache is None:
        D_cache = precompute_wigner_D(R, irreps)
    parts = []
    offset = 0
    for mul, ir in irreps:
        dim = mul * ir.dim
        block = h[:, offset:offset + dim]
        if ir.l == 0:
            parts.append(block)
        else:
            D_l = D_cache[ir.l]
            block_3d = block.reshape(E, mul, ir.dim)
            rotated = torch.einsum("eij,emj->emi", D_l, block_3d)
            parts.append(rotated.reshape(E, mul * ir.dim))
        offset += dim
    return torch.cat(parts, dim=-1)


# ---------------------------------------------------------------------------
# Geometric Lift: scalar nodes → equivariant via edge SH
# ---------------------------------------------------------------------------

class GeometricLift(nn.Module):
    """Lift pure-scalar element features to node_irreps using edge CG."""

    def __init__(self, scalar_dim: int, irreps_out: o3.Irreps, num_rbf: int):
        super().__init__()
        self.irreps_out = irreps_out
        self._blocks = []
        n_coeff = 0
        for mul, ir in irreps_out:
            can_init = (ir.l, ir.p) in {(0, 1), (1, -1), (2, 1)} and ir.l <= 2
            self._blocks.append((mul, ir.l, ir.dim, can_init))
            if can_init:
                n_coeff += mul
        self.lin = nn.Linear(scalar_dim, n_coeff)
        self.gate = nn.Sequential(
            nn.Linear(num_rbf, 64), nn.SiLU(), nn.Linear(64, n_coeff), nn.Sigmoid()
        )
        self.norm = IrrepNorm(irreps_out)

    def forward(self, h_scalar, geo_edge_index, edge_sh, edge_rbf, K):
        target, src = geo_edge_index[0], geo_edge_index[1]
        coeff = self.lin(h_scalar[src]) * self.gate(edge_rbf)
        sh_slices = {0: (0, 1), 1: (1, 4), 2: (4, 9)}
        parts = []
        c_off = 0
        for mul, l, dim_l, can_init in self._blocks:
            if can_init:
                a = coeff[:, c_off:c_off + mul]; c_off += mul
                Y = edge_sh[:, sh_slices[l][0]:sh_slices[l][1]]
                msg = (a.unsqueeze(-1) * Y[:, None, :]).reshape(-1, mul * dim_l)
            else:
                msg = torch.zeros(geo_edge_index.size(1), mul * dim_l,
                                  device=h_scalar.device, dtype=h_scalar.dtype)
            parts.append(scatter(msg, target, dim=0, dim_size=K, reduce="sum"))
        return self.norm(torch.cat(parts, dim=-1))


# ---------------------------------------------------------------------------
# InteractionBlock: 1 TP + 1 bmm projection (fast)
# ---------------------------------------------------------------------------

class InteractionBlock(nn.Module):
    """One layer: edge update (1 TP) + node update (linear + scatter + project).

    Edge update:
        edge_msg = TP(edge, D(R_k)·h_source) * gate(RBF)
        e' = Norm(e + Linear(edge_msg))

    Node update:
        node_msg = Linear(e') * gate(RBF)
        h' = P_i @ Norm(h + scatter(node_msg))

    Only 1 TP per layer. Site projection = 1 bmm (precomputed P_i).
    """

    def __init__(self, irreps_node, irreps_edge, num_rbf, radial_hidden=64):
        super().__init__()
        self.irreps_node = o3.Irreps(irreps_node)
        self.irreps_edge = o3.Irreps(irreps_edge)

        # Edge update: TP(edge, rotated_source_node) → edge update.
        # Use uvu mode: weight shared across multiplicities (1 per (u,v) pair per path).
        # uvu constraint: mul_in1 == mul_out for each instruction.
        # This is ~100x faster than uvw.
        instructions = []
        irreps_edge_list = list(self.irreps_edge)
        irreps_node_list = list(self.irreps_node)
        for i, (mul_e, ir_e) in enumerate(irreps_edge_list):
            for j, (mul_n, ir_n) in enumerate(irreps_node_list):
                for ir_out in ir_e * ir_n:
                    for k, (mul_o, ir_o) in enumerate(irreps_edge_list):
                        if ir_o == ir_out and mul_o == mul_e:  # uvu: same mul
                            instructions.append((i, j, k, "uvu", True))
                            break
        self.edge_tp = o3.TensorProduct(
            self.irreps_edge, self.irreps_node, self.irreps_edge,
            instructions, shared_weights=True, internal_weights=True,
        )
        self.edge_gate = nn.Sequential(
            nn.Linear(num_rbf, radial_hidden), nn.SiLU(),
            nn.Linear(radial_hidden, len(list(self.irreps_edge))), nn.Sigmoid(),
        )
        egate_map = []
        for i, (mul, ir) in enumerate(self.irreps_edge):
            egate_map.extend([i] * (mul * ir.dim))
        self.register_buffer("egate_map", torch.tensor(egate_map, dtype=torch.long))
        self.edge_proj = o3.Linear(self.irreps_edge, self.irreps_edge)
        self.edge_norm = IrrepNorm(self.irreps_edge)

        # Node update: Linear(edge → node) + scatter + project.
        self.node_lin = o3.Linear(self.irreps_edge, self.irreps_node)
        self.node_gate = nn.Sequential(
            nn.Linear(num_rbf, radial_hidden), nn.SiLU(),
            nn.Linear(radial_hidden, len(list(self.irreps_node))), nn.Sigmoid(),
        )
        ngate_map = []
        for i, (mul, ir) in enumerate(self.irreps_node):
            ngate_map.extend([i] * (mul * ir.dim))
        self.register_buffer("ngate_map", torch.tensor(ngate_map, dtype=torch.long))
        self.node_norm = IrrepNorm(self.irreps_node)

    def forward(self, h_node, e_edge, geo_edge_index, edge_rbf,
                source_rotations, D_cache, projections):
        target, src = geo_edge_index[0], geo_edge_index[1]
        K = h_node.size(0)

        # Edge update: 1 TP.
        h_src_real = rotate_irreps(h_node[src], source_rotations, self.irreps_node, D_cache)
        edge_msg = self.edge_tp(e_edge, h_src_real)
        edge_msg = edge_msg * self.edge_gate(edge_rbf)[:, self.egate_map]
        e_edge = self.edge_norm(e_edge + self.edge_proj(edge_msg))

        # Node update: Linear + gate + scatter + residual + project.
        node_msg = self.node_lin(e_edge) * self.node_gate(edge_rbf)[:, self.ngate_map]
        agg = scatter(node_msg, target, dim=0, dim_size=K, reduce="sum")
        h_node = self.node_norm(h_node + agg)

        # Site projection: 1 bmm (near-zero cost, precomputed P_i).
        if projections is not None:
            h_node = torch.bmm(
                projections, h_node.unsqueeze(-1)
            ).squeeze(-1)

        return h_node, e_edge


# ---------------------------------------------------------------------------
# EquivDecoder: edge-dominant O(3)-invariant readout
# ---------------------------------------------------------------------------

class EquivDecoder(nn.Module):
    """Edge-dominant readout: TP(edge, D(R_k)·h_source) → invariants → MLP.

    Design principles:
        1. All information lives in real atom pairs (edges), not representative
           nodes. Nodes are Inv(H_i)-projected (limited capacity), edges carry
           the full O(3) representation.
        2. Compute per-edge invariants via a single cross TP with output = "Kx0e".
           The TP mixes edge irreps (chemistry-in-context) with node irreps
           (full atom features recovered by D(R_k)) and collapses to O(3) scalars.
        3. Pool per-edge invariants to graph level using the true atom-pair
           weight w_e = geo_edge_weight * multiplicity[target]. This gives the
           mean over all real atom pairs (intensive) — correct for e_form,
           bandgap, etc.
        4. Deep MLP with SiLU + Dropout + residual on the pooled scalar.

    No self-mixing (TP(h,h) or TP(e,e)); every interaction is edge×node,
    consistent with InteractionBlock.
    """

    def __init__(
        self,
        irreps_node: o3.Irreps,
        irreps_edge: o3.Irreps,
        hidden: int = 128,
        n_invariants: int = 64,
        n_mlp_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.irreps_node = o3.Irreps(irreps_node)
        self.irreps_edge = o3.Irreps(irreps_edge)
        self._irreps_inv = o3.Irreps(f"{n_invariants}x0e")

        # Cross TP: edge × node → n_invariants scalars. Fully-connected so every
        # (l_e, l_n → 0) path gets its own learnable weight matrix (mul_e × mul_n
        # → n_invariants), producing K learned "Gram-like" scalars per edge.
        self.edge_tp = o3.FullyConnectedTensorProduct(
            self.irreps_edge, self.irreps_node, self._irreps_inv,
            internal_weights=True, shared_weights=True,
        )

        # Deep MLP head with SiLU + Dropout + residual.
        self.n_invariants = n_invariants
        layers = []
        in_dim = n_invariants
        for _ in range(n_mlp_layers - 1):
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.SiLU())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden
        self.mlp = nn.Sequential(*layers)
        self.residual = nn.Linear(n_invariants, hidden) if n_invariants != hidden else nn.Identity()
        self.out = nn.Linear(hidden, 1)

    def forward(
        self,
        h_node: torch.Tensor,
        e_edge: torch.Tensor,
        geo_edge_index: torch.Tensor,
        source_rotations: torch.Tensor,
        D_cache,
        batch: torch.Tensor,
        multiplicity: torch.Tensor,
        geo_edge_weight,
        num_graphs: int,
    ) -> torch.Tensor:
        target, src = geo_edge_index[0], geo_edge_index[1]
        E = geo_edge_index.size(1)

        if E == 0:
            # No edges: fall back to zero-inv → MLP produces a bias-only output.
            zero_inv = torch.zeros(num_graphs, self.n_invariants,
                                   device=h_node.device, dtype=h_node.dtype)
            h_mlp = self.mlp(zero_inv) + self.residual(zero_inv)
            return self.out(h_mlp).squeeze(-1)

        # Recover full atom features on the source side (D(R_k) · h_orbit).
        h_src_real = rotate_irreps(h_node[src], source_rotations, self.irreps_node, D_cache)

        # Per-edge O(3) invariants (E, n_invariants).
        edge_inv = self.edge_tp(e_edge, h_src_real)

        # Weight = per-edge PBC/multiplicity weight × orbit size of target.
        # This turns the sum over representative-target edges into a weighted
        # mean over all real (atom, atom) pairs in the crystal.
        w = torch.ones(E, device=edge_inv.device, dtype=edge_inv.dtype)
        if geo_edge_weight is not None:
            w = w * geo_edge_weight
        if multiplicity is not None:
            w = w * multiplicity[target]

        batch_edge = batch[target]
        w_sum = scatter(w, batch_edge, dim=0, dim_size=num_graphs, reduce="sum")
        w_sum = w_sum.clamp(min=1e-8).unsqueeze(-1)
        g = scatter(edge_inv * w.unsqueeze(-1), batch_edge,
                    dim=0, dim_size=num_graphs, reduce="sum") / w_sum

        # Deep MLP + residual + head.
        h_mlp = self.mlp(g) + self.residual(g)
        return self.out(h_mlp).squeeze(-1)


# ---------------------------------------------------------------------------
# Edge initialization
# ---------------------------------------------------------------------------

class EdgeInit(nn.Module):
    """Initialize equivariant edges from node pair + geometry."""

    def __init__(self, irreps_node, irreps_edge, num_rbf, radial_hidden=64):
        super().__init__()
        self.irreps_node = o3.Irreps(irreps_node)
        self.irreps_edge = o3.Irreps(irreps_edge)
        self.tp = o3.FullyConnectedTensorProduct(
            self.irreps_node, self.irreps_node, self.irreps_edge,
            internal_weights=True, shared_weights=True,
        )
        n_gate = len(list(self.irreps_edge))
        self.gate = nn.Sequential(
            nn.Linear(num_rbf, radial_hidden), nn.SiLU(),
            nn.Linear(radial_hidden, n_gate), nn.Sigmoid(),
        )
        gate_map = []
        for i, (mul, ir) in enumerate(self.irreps_edge):
            gate_map.extend([i] * (mul * ir.dim))
        self.register_buffer("gate_map", torch.tensor(gate_map, dtype=torch.long))

    def forward(self, h_node, geo_edge_index, edge_rbf, source_rotations, D_cache):
        target, src = geo_edge_index[0], geo_edge_index[1]
        h_src = rotate_irreps(h_node[src], source_rotations, self.irreps_node, D_cache)
        h_tgt = h_node[target]
        e = self.tp(h_src, h_tgt) * self.gate(edge_rbf)[:, self.gate_map]
        return e


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------
