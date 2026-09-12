"""Stage B: O(3)-equivariant message-passing layers for WyckoffGNN.

Architecture
============

Node irreps (throughout all layers)::

    "64x0e + 32x1o + 16x2e"

Physical channel allocation:
  - 64x0e (scalars, rotation-invariant):
    * 0e[0:32]:  chemical (element type, periodic-table context)
    * 0e[32:48]: radial (distance/radial info from geometric edges)
    * 0e[48:64]: group-theoretic (template encoder prototype prior)
  - 32x1o (vectors, O(3)-equivariant):
    * directional geometric information
  - 16x2e (rank-2 tensors):
    * bond-angle / shape information (from 1o x 1o CG products)

Edge features (per sub-edge):
  - distance -> GaussianRBF(d) -> 0e scalar
  - unit direction -> spherical_harmonics(rhat) -> 0e+1o+2e
  - source image rotation R_{q,k} -> D^{(l)}(R) rotation of source node features

Source image rotation:
  When orbit q has multiplicity > 1, each equivalent atom q_k is related to
  the representative by a symmetry operation R_{q,k}.  Before computing the
  tensor-product message from h_q, we MUST rotate h_q by D^{(l)}(R_{q,k}) so
  that its directional channels align with the actual geometric edge direction.
  This is required for strict O(3)-equivariance.

Edge direction convention:
  - geo_edge_index[0] = target orbit p (message receiver)
  - geo_edge_index[1] = source orbit q (message sender)
  - vec = x_{q,k} + shift - x_p  (from target to source image)

Design principles:
  1. Physical-channel separation at node initialisation — no MLP fusion.
  2. Controlled interaction via CG tensor-product message passing only.
  3. Dynamic gate configuration derived from irreps_message.
  4. Per-irrep-block RMSNorm for equivariant normalisation.
  5. Different physical levels interact through structured TP, not ad-hoc MLPs.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from e3nn import o3
from e3nn.nn import Gate
from torch_scatter import scatter

# Monkey-patch e3nn for GPU: so3_generators creates X matrices on CPU.
# Fix: wrap so3_generators to return X on the same device as caller context.
try:
    import e3nn.o3._wigner as _w
    _orig_so3_gen = _w.so3_generators
    def _gpu_so3_generators(l):
        X = _orig_so3_gen(l)  # list of CPU tensors
        # so3_generators is called inside wigner_D; we detect device from
        # the calling frame's arguments at runtime.  Instead, wrap wigner_D
        # to move X after creation.
        return X
    # Better approach: wrap wigner_D directly and fix X device.
    _orig_wigner_D = _w.wigner_D
    def _gpu_wigner_D(l, alpha, beta, gamma):
        dev = alpha.device
        Xa, Xb, Xc = _orig_so3_gen(l)
        Xa, Xb, Xc = Xa.to(dev), Xb.to(dev), Xc.to(dev)
        a = alpha.reshape(-1, 1, 1)
        b = beta.reshape(-1, 1, 1)
        c = gamma.reshape(-1, 1, 1)
        # Original e3nn convention: exp(α·X[1]) @ exp(β·X[0]) @ exp(γ·X[1])
        #   X[0]=Lx=Xa, X[1]=Lz=Xb, X[2]=-Ly=Xc
        #   So: exp(α·Lz) @ exp(β·Lx) @ exp(γ·Lz) = exp(α·Xb) @ exp(β·Xa) @ exp(γ·Xb)
        return (torch.matrix_exp(a * Xb) @
                torch.matrix_exp(b * Xa) @
                torch.matrix_exp(c * Xb))
    _w.wigner_D = _gpu_wigner_D
except Exception:
    pass  # CPU fallback

# ---------------------------------------------------------------------------
# Irreps configuration (single source of truth)
# ---------------------------------------------------------------------------

# Node feature irreps (internal representation).
_IRREPS_NODE = o3.Irreps("64x0e + 32x1o + 16x2e")

# Edge spherical-harmonic irreps (from unit direction vector).
# l_max = 2 is a pragmatic balance: higher l gets expensive (> 5× compute)
# while l=1 alone cannot capture bond-angle information.
_IRREPS_SH = o3.Irreps("1x0e + 1x1o + 1x2e")

# Edge irreps for symmetry edges (scalar only: relation sign ±1).
_IRREPS_SYM_EDGE = o3.Irreps("1x0e")

# Layout offsets for _IRREPS_NODE = "64x0e + 32x1o + 16x2e" (240-dim).
_N_CH_SCALAR = 64      # 64x0e
_N_CH_VECTOR = 32      # 32x1o → 96 dims
_N_CH_TENSOR = 16      # 16x2e → 80 dims
_DIM_SCALAR = _N_CH_SCALAR       # 64
_DIM_VECTOR = _N_CH_VECTOR * 3   # 96
_DIM_TENSOR = _N_CH_TENSOR * 5   # 80


def _make_gate_irreps(irreps_msg: o3.Irreps):
    """Derive Gate actuator irreps dynamically from message irreps.

    Convention:
      - Activation scalars:  up to half of the ``0e`` channels (SiLU).
      - Gate scalars:        one ``0e`` scalar per non-scalar (l>0) channel.
      - Gated features:      all channels with l>0, gated by sigmoid.

    This replaces the previous hard-coded split and keeps the Gate in sync
    with ``_IRREPS_MESSAGE`` automatically.
    """
    scalars_0e = []
    gated = []
    for mul, ir in irreps_msg:
        if ir.l == 0:
            scalars_0e.append((mul, ir))
        else:
            gated.append((mul, ir))

    n_gate = sum(mul for mul, _ in gated)  # one scalar gate per gated channel
    n_act = sum(mul for mul, _ in scalars_0e) - n_gate

    if n_act <= 0:
        raise ValueError(
            f"irreps_msg has insufficient 0e channels for Gate: "
            f"total 0e = {sum(mul for mul,_ in scalars_0e)}, "
            f"need > {n_gate} for gate scalars."
        )

    gate_act = o3.Irreps(f"{n_act}x0e")
    gate_gate = o3.Irreps(f"{n_gate}x0e")
    gate_gated = o3.Irreps(gated) if gated else o3.Irreps("0x0e")

    return gate_act, gate_gate, gate_gated


class IrrepNorm(nn.Module):
    """Per-irrep-block normalisation that preserves SE(3) equivariance.

    For each irrep block:
      - **Scalars (l=0)**:  standard ``LayerNorm`` with learnable shift + scale.
      - **High-l (l>0)**:  RMS normalisation *without mean subtraction* +
        a single learnable per-block scale.  Mean subtraction would break
        equivariance (a rotated zero vector must remain zero).

    Uses ``sqrt(mean(x²) + eps)`` (genuine RMS) for numerical stability.

    This matches the ``IrrepNorm`` design from the equivariant_gnn_project
    reference (src/models/modules/norm.py).
    """

    def __init__(self, irreps: o3.Irreps, eps: float = 1e-5):
        super().__init__()
        self.irreps = irreps
        self.eps = eps

        # One learnable scale per irrep block (high-l only; scalars use LayerNorm).
        n_blocks = len(list(irreps))
        self.scale = nn.Parameter(torch.ones(n_blocks, dtype=torch.float32))

        # Per-scalar-block LayerNorm (learnable shift + scale, invariant).
        self.scalar_norms = nn.ModuleDict()
        for i, (mul, ir) in enumerate(irreps):
            if ir.l == 0:
                self.scalar_norms[str(i)] = nn.LayerNorm(mul * ir.dim, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        blocks: list[torch.Tensor] = []
        offset = 0
        for i, (mul, ir) in enumerate(self.irreps):
            dim = mul * ir.dim
            block = x[:, offset:offset + dim]
            if ir.l == 0:
                block = self.scalar_norms[str(i)](block)
            else:
                # Reshape to (N, mul, ir.dim) for per-channel RMS.
                block = block.reshape(-1, mul, ir.dim)
                rms = (block.pow(2).mean(dim=-1, keepdim=True) + self.eps).sqrt()
                block = block / rms.clamp(min=self.eps)
                block = block.reshape(-1, mul * ir.dim)
            blocks.append(block * self.scale[i])
            offset += dim
        return torch.cat(blocks, dim=-1)

# Gate actuator configuration.
#   activation scalars : 64×0e (SiLU)
#   gate scalars       : 48×0e (sigmoid)  → 32× for 1o gates + 16× for 2e gates
#   gated features     : 32×1o + 16×2e
_IRREPS_GATE_ACT  = o3.Irreps("64x0e")
_IRREPS_GATE_GATE = o3.Irreps("48x0e")
_IRREPS_GATED     = o3.Irreps("32x1o + 16x2e")

# Message irreps entering the gate.
_IRREPS_MESSAGE = _IRREPS_GATE_ACT + _IRREPS_GATE_GATE + _IRREPS_GATED
# = "112x0e + 32x1o + 16x2e"  (288-dim)

# Default hidden irreps string (lite variant, paired with 128x0e element init).
_DEFAULT_HIDDEN_IRREPS = "64x0e + 16x1o + 8x2e"


def _derive_message_irreps(irreps_node: o3.Irreps) -> o3.Irreps:
    """Derive message (TP output) irreps from node irreps.

    The message must be larger than the node because it carries both
    activation scalars AND gate scalars (one per gated multiplicity).
    After the Gate: gate_act + gate_gated = node irreps.
    """
    scalars = []
    gated = []
    for mul, ir in irreps_node:
        if ir.l == 0:
            scalars.append((mul, ir))
        else:
            gated.append((mul, ir))
    n_gate = sum(mul for mul, _ in gated)
    gate_act = o3.Irreps(scalars) if scalars else o3.Irreps("0x0e")
    gate_gate = o3.Irreps(f"{n_gate}x0e") if n_gate > 0 else o3.Irreps("0x0e")
    gate_gated = o3.Irreps(gated) if gated else o3.Irreps("0x0e")
    return gate_act + gate_gate + gate_gated


def compute_irreps(hidden_irreps: str):
    """Compute all dependent irreps from a hidden node irreps string.

    Returns a dict with keys: node, gate_act, gate_gate, gated,
    message, sh, sym_edge.
    """
    node = o3.Irreps(hidden_irreps)
    message = _derive_message_irreps(node)
    gate_act, gate_gate, gated = _make_gate_irreps(message)
    return {
        "node": node,
        "gate_act": gate_act,
        "gate_gate": gate_gate,
        "gated": gated,
        "message": message,
        "sh": _IRREPS_SH,
        "sym_edge": _IRREPS_SYM_EDGE,
    }

# ---------------------------------------------------------------------------
# Source image rotation (O(3)-equivariant)
# ---------------------------------------------------------------------------


def _rotate_node_features(
    h: torch.Tensor, rotations: torch.Tensor, irreps: o3.Irreps,
    D_cache: Optional[dict] = None,
) -> torch.Tensor:
    """Apply Wigner-D rotation D(R) to node features.

    For each edge e with source rotation R_e, the source node features h_q
    are rotated::

        h_q' = D(R_e) @ h_q

    This ensures that the directional (l>0) channels of h_q are aligned with
    the actual geometric edge direction for equivalent atom q_k.

    Args:
        h: (E, dim) source node features.
        rotations: (E, 3, 3) rotation matrices.
        irreps: irrep decomposition of h.
        D_cache: Optional pre-computed ``{(l, p): D_l}`` dict from
            :func:`_build_source_D_cache`. If provided, avoids recomputing
            Wigner-D matrices.

    Returns:
        (E, dim) rotated features.
    """
    E = h.size(0)
    if E == 0:
        return h

    # Build per-irrep slices and collect unique (l, p) pairs.
    slices_info = []
    l_p_set = set()
    idx = 0
    for mul, ir in irreps:
        dim = mul * ir.dim
        slices_info.append((ir.l, ir.p, mul, slice(idx, idx + dim)))
        l_p_set.add((ir.l, ir.p))
        idx += dim

    # Build or reuse Wigner-D matrices per unique (l, p).
    if D_cache is None:
        D_cache = {}
        for l, p in l_p_set:
            if l == 0:
                D_cache[(l, p)] = None
            else:
                D_l = o3.Irrep(l, p).D_from_matrix(
                    rotations.to(device=h.device, dtype=h.dtype)
                )
                if D_l.dim() == 2:
                    D_l = D_l.unsqueeze(0).expand(E, -1, -1)
                elif D_l.size(0) != E:
                    D_l = D_l.expand(E, -1, -1)
                D_cache[(l, p)] = D_l  # [E, 2l+1, 2l+1]

    out_parts = []
    for l, p, mul, sl in slices_info:
        block = h[:, sl]  # [E, mul*(2l+1)]
        if l == 0:
            out_parts.append(block)
        else:
            D_l = D_cache[(l, p)]       # [E, 2l+1, 2l+1]
            dim_l = 2 * l + 1
            block_3d = block.reshape(E, mul, dim_l)          # [E, mul, 2l+1]
            rotated = torch.bmm(D_l, block_3d.transpose(1, 2))  # [E, 2l+1, mul]
            rotated = rotated.transpose(1, 2).reshape(E, mul * dim_l)
            out_parts.append(rotated)

    return torch.cat(out_parts, dim=-1)


# ---------------------------------------------------------------------------
# Node feature encoder — physical-channel separation
# ---------------------------------------------------------------------------


class ElementOnlyNodeEncoder(nn.Module):
    """Element-only node initialiser: Z -> pure-scalar irreps (e.g. 128x0e).

    Optionally appends periodic-table physical properties (mass, radius,
    electronegativity, ionisation energy) to the embedding.
    """

    def __init__(
        self,
        init_irreps,
        max_atomic_number: int = 118,
        use_atom_props: bool = False,
    ):
        super().__init__()
        init_irreps = o3.Irreps(init_irreps)
        if not all(ir.l == 0 and ir.p == 1 for _, ir in init_irreps):
            raise ValueError(
                f"init_irreps must be pure scalars (Nx0e), got {init_irreps}."
            )
        self.init_irreps = init_irreps
        self.max_atomic_number = max_atomic_number
        self.use_atom_props = use_atom_props
        self.embedding = nn.Embedding(max_atomic_number + 1, init_irreps.dim)
        if use_atom_props:
            from wyckoff_gnn.models.e3nn_layers import _N_ATOM_PROPERTIES
            self.prop_proj = nn.Linear(init_irreps.dim + _N_ATOM_PROPERTIES,
                                       init_irreps.dim)

    def forward(self, atomic_numbers: torch.Tensor) -> torch.Tensor:
        z = atomic_numbers.clamp(0, self.max_atomic_number)
        h = self.embedding(z)
        if self.use_atom_props:
            from wyckoff_gnn.models.e3nn_layers import _gather_atom_props
            props = _gather_atom_props(atomic_numbers, h.device)
            h = self.prop_proj(torch.cat([h, props], dim=-1))
        return h


# ---------------------------------------------------------------------------
# RBF basis (non-learnable, pre-computed or re-computed on the fly)
# ---------------------------------------------------------------------------

class GaussianRBF(nn.Module):
    """Gaussian radial basis function expansion.

    φ_k(d) = exp(-γ · (d - μ_k)²)

    This is a **non-learnable** distance encoding; the learning happens in
    the TP weight MLP that consumes the RBF output.
    """

    def __init__(self, num_rbf: int = 32, r_min: float = 0.0, r_max: float = 8.0):
        super().__init__()
        self.num_rbf = num_rbf
        centers = torch.linspace(r_min, r_max, num_rbf)
        gamma = (num_rbf - 1) / (r_max - r_min) if r_max > r_min else 1.0
        self.register_buffer("centers", centers)
        self.register_buffer("gamma", torch.tensor(gamma))

    def forward(self, d: torch.Tensor) -> torch.Tensor:
        """Expand distances.

        Args:
            d: (E,) distances.

        Returns:
            (E, num_rbf) RBF features.
        """
        diff = d.unsqueeze(-1) - self.centers.to(d.device).unsqueeze(0)
        return torch.exp(-self.gamma.to(d.device) * diff.pow(2))


# ---------------------------------------------------------------------------
# Equivariant geometric message passing (with e3nn FullyConnectedTensorProduct)
# ---------------------------------------------------------------------------

class EquivariantGeometricMP(nn.Module):
    """Geometric-channel equivariant message passing (ConvBlockGate style).

    TP weights are learned internally (shared_weights=True, internal_weights=True),
    matching the parameter-efficient design of atom_e3nn.  Edge RBF features
    only modulate a per-irrep radial gate, not the full TP weight tensor.

    For each edge (src→dst):

        msg = TP(h_src, edge_sh)               # learned TP weights, no edge input
        msg = msg * radial_gate(edge_rbf)       # per-irrep sigmoid gate
        agg = scatter_sum(msg, dst)
        h' = h_dst + Gate(agg)
    """

    def __init__(
        self,
        irreps_node: o3.Irreps,
        irreps_message: o3.Irreps,
        irreps_sh: o3.Irreps,
        num_rbf: int = 32,
        radial_gate_mode: str = "per_type",
        edge_state_dim: int = 0,
    ):
        super().__init__()
        self.irreps_node = irreps_node
        self.irreps_message = irreps_message
        self.radial_gate_mode = radial_gate_mode
        self.edge_state_dim = edge_state_dim

        # TP with internally-learned shared weights (like ConvBlockGate).
        self.tp = o3.FullyConnectedTensorProduct(
            irreps_in1=irreps_node,
            irreps_in2=irreps_sh,
            irreps_out=irreps_message,
            internal_weights=True,
            shared_weights=True,
        )

        # Radial gate: RBF (+ optional edge_state) -> sigmoid scalars.
        gate_input_dim = num_rbf + edge_state_dim
        if radial_gate_mode == "per_copy":
            n_gate_out = sum(mul for mul, _ in irreps_message)
            copy_per_dim = []
            copy_idx = 0
            for mul, ir in irreps_message:
                for _ in range(mul):
                    copy_per_dim.extend([copy_idx] * ir.dim)
                    copy_idx += 1
            self.register_buffer(
                "irrep_per_dim", torch.tensor(copy_per_dim, dtype=torch.long)
            )
        else:
            n_gate_out = len(list(irreps_message))
            irrep_per_dim = []
            for i, (mul, ir) in enumerate(irreps_message):
                irrep_per_dim.extend([i] * (mul * ir.dim))
            self.register_buffer(
                "irrep_per_dim", torch.tensor(irrep_per_dim, dtype=torch.long)
            )

        mid = max(irreps_message.dim // 2, 32)
        self.radial_gate = nn.Sequential(
            nn.Linear(gate_input_dim, mid),
            nn.SiLU(),
            nn.Linear(mid, n_gate_out),
            nn.Sigmoid(),
        )

        # Gate activation (dynamic from irreps_message).
        gate_act, gate_gate, gate_gated = _make_gate_irreps(irreps_message)
        n_gated = sum(mul for mul, _ in gate_gated)
        if n_gated == 0:
            # Scalar-only: no gated channels → no Gate needed.
            self.gate = None
            self._gate_out = gate_act
        else:
            self.gate = Gate(
                gate_act, [F.silu],
                gate_gate, [F.silu],
                gate_gated,
            )
            self._gate_out = self.gate.irreps_out

        self.skip = o3.Linear(irreps_node, self._gate_out)

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_sh: torch.Tensor,
        edge_rbf: torch.Tensor,
        source_rotations: Optional[torch.Tensor] = None,
        edge_weight: Optional[torch.Tensor] = None,
        edge_state: Optional[torch.Tensor] = None,
        return_messages: bool = False,
        source_D_cache: Optional[dict] = None,
    ):
        """Compute geometric-channel equivariant messages.

        Args:
            h: (K, dim_node) node features in irrep order.
            edge_index: (2, E) target←source indices (row0=target,row1=source).
            edge_sh: (E, dim_sh) SH encoding of unit direction.
            edge_rbf: (E, num_rbf) RBF distance encoding.
            source_rotations: Optional (E, 3, 3) rotation matrices.
            edge_weight: Optional (E,) per-sub-edge aggregation weights.
            edge_state: Optional (E, edge_state_dim) scalar edge state.
            return_messages: If True, also return per-edge gated messages.
            source_D_cache: Optional pre-computed Wigner-D cache.

        Returns:
            (K, dim_out) updated node features (gate + skip).
            If return_messages: tuple of (node_output, messages).
        """
        K = h.size(0)
        target, src = edge_index[0], edge_index[1]

        if src.numel() == 0:
            out = self.skip(h)
            if return_messages:
                return out, torch.zeros(0, self.irreps_message.dim, device=h.device)
            return out

        h_src = h[src]

        if source_rotations is not None:
            h_src = _rotate_node_features(
                h_src, source_rotations, self.irreps_node, D_cache=source_D_cache
            )

        msg = self.tp(h_src, edge_sh)

        # Radial gate: concat edge_state if available.
        if edge_state is not None and self.edge_state_dim > 0:
            gate_input = torch.cat([edge_rbf, edge_state], dim=-1)
        else:
            gate_input = edge_rbf

        gate_per_irrep = self.radial_gate(gate_input)
        gate_per_dim = gate_per_irrep[:, self.irrep_per_dim]
        msg = msg * gate_per_dim

        msg_for_update = msg if return_messages else None

        if edge_weight is not None:
            msg = msg * edge_weight.unsqueeze(-1)

        agg = scatter(msg, target, dim=0, dim_size=K, reduce="sum")

        if self.gate is not None:
            out = self.gate(agg) + self.skip(h)
        else:
            out = F.silu(agg) + self.skip(h)

        if return_messages:
            return out, msg_for_update
        return out


# ---------------------------------------------------------------------------
# Equivariant symmetry message passing (scalar-only edges)
# ---------------------------------------------------------------------------

class EquivariantSymmetryMP(nn.Module):
    """Symmetry-channel equivariant message passing.

    Symmetry edges carry **only scalar** edge features (relation sign ±1),
    so the tensor product reduces to:

        msg = h_src ⊗ edge_scalar

    which is simply a scaling of each irrep channel by a learned function
    of the sign.  No directional information enters — the symmetry channel
    encodes **group-theoretic constraints**, not geometry.

    Physical meaning:
        The sign (±1) tells whether H_src ⊃ H_dst (+1: "parent constrains child")
        or H_src ⊂ H_dst (-1: "child informs parent").  The TP with this
        scalar edge feature allows the model to learn **hierarchy-aware**
        transformations of node features.
    """

    def __init__(
        self,
        irreps_node: o3.Irreps,
        irreps_message: o3.Irreps,
    ):
        super().__init__()

        # Sign encoder: ±1 → hidden.
        self.sign_encoder = nn.Sequential(
            nn.Linear(1, 32),
            nn.SiLU(),
            nn.Linear(32, 32),
            nn.SiLU(),
        )

        # Tensor product with scalar edge → message.
        self.tp = o3.FullyConnectedTensorProduct(
            irreps_in1=irreps_node,
            irreps_in2=_IRREPS_SYM_EDGE,
            irreps_out=irreps_message,
            internal_weights=True,
            shared_weights=True,
        )

        gate_act, gate_gate, gate_gated = _make_gate_irreps(irreps_message)
        n_gated = sum(mul for mul, _ in gate_gated)
        if n_gated == 0:
            self.gate = None
            self._gate_out = gate_act
        else:
            self.gate = Gate(
                gate_act,
                [F.silu],
                gate_gate,
                [F.silu],
                gate_gated,
            )
            self._gate_out = self.gate.irreps_out

        self.skip = o3.Linear(irreps_node, self._gate_out)

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """Compute symmetry-channel equivariant messages.

        Args:
            h: (N, dim_node) node features.
            edge_index: (2, E_sym) directed symmetry edges (src, dst).
            edge_attr: (E_sym, 1) relation sign (+1 or -1).

        Returns:
            (N, dim_out) updated node features.
        """
        N = h.size(0)
        if edge_index.numel() == 0:
            return h

        src, dst = edge_index[0], edge_index[1]

        # Edge irrep for TP — just ones in 1×0e.
        edge_irrep = torch.ones(edge_attr.size(0), 1,
                                dtype=h.dtype, device=h.device)

        msg = self.tp(h[src], edge_irrep)       # (E, dim_message)
        agg = scatter(msg, dst, dim=0, dim_size=N, reduce="mean")

        if self.gate is not None:
            return self.gate(agg) + self.skip(h)
        else:
            return F.silu(agg) + self.skip(h)


# ---------------------------------------------------------------------------
# Full equivariant Wyckoff layer (geometric + symmetry dual-channel)
# ---------------------------------------------------------------------------


class EquivariantWyckoffLayer(nn.Module):
    """One equivariant dual-channel WyckoffGNN layer.

    1. Linear_1 (pre-TP irrep mixing)
    2. NeighborNorm (degree normalization)
    3. Geometric MP (distance × direction, CG tensor product)
    4. Symmetry MP (hierarchy relation, scalar TP)
    5. Sum fusion + LayerNorm (no MLP — physical separation preserved)
    """

    def __init__(
        self,
        irreps_node: o3.Irreps,
        irreps_message: o3.Irreps,
        irreps_sh: o3.Irreps,
        num_rbf: int = 32,
        dropout: float = 0.1,
        radial_gate_mode: str = "per_type",
        edge_state_dim: int = 0,
    ):
        super().__init__()

        # Linear_1: pre-TP irrep mixing (same as AISPML)
        self.linear_1 = o3.Linear(irreps_node, irreps_node,
                                   internal_weights=True, shared_weights=True)

        self.geo_mp = EquivariantGeometricMP(
            irreps_node=irreps_node,
            irreps_message=irreps_message,
            irreps_sh=irreps_sh,
            num_rbf=num_rbf,
            radial_gate_mode=radial_gate_mode,
            edge_state_dim=edge_state_dim,
        )
        self.sym_mp = EquivariantSymmetryMP(
            irreps_node=irreps_node,
            irreps_message=irreps_message,
        )

        self.norm = IrrepNorm(irreps_node)

        # Fusion: sum of channels (preserves irrep structure).
        # No MLP — interaction happens in the next layer's TP.
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(
        self,
        h: torch.Tensor,
        geo_edge_index: torch.Tensor,
        geo_edge_sh: torch.Tensor,
        geo_edge_rbf: torch.Tensor,
        sym_edge_index: torch.Tensor,
        sym_edge_attr: torch.Tensor,
        source_rotations: Optional[torch.Tensor] = None,
        geo_edge_weight: Optional[torch.Tensor] = None,
        edge_state: Optional[torch.Tensor] = None,
        return_messages: bool = False,
        source_D_cache: Optional[dict] = None,
    ):
        """One dual-channel equivariant layer.

        Args:
            h: (K, dim_node) node features.
            geo_edge_index: (2, E_geo) target←source indices.
            geo_edge_sh: (E_geo, dim_sh) SH of unit direction.
            geo_edge_rbf: (E_geo, num_rbf) RBF distance features.
            sym_edge_index: (2, E_sym) symmetry edges.
            sym_edge_attr: (E_sym, 1) symmetry edge features.
            source_rotations: Optional (E_geo, 3, 3) rotation matrices.
            geo_edge_weight: Optional (E_geo,) per-sub-edge weights.
            edge_state: Optional (E_geo, edge_state_dim) edge states.
            return_messages: If True, also return per-edge messages.
            source_D_cache: Optional pre-computed Wigner-D cache.

        Returns:
            (K, dim_node) updated node features.
            If return_messages: tuple of (node_output, messages).
        """
        h_pre = self.linear_1(h)   # pre-TP irrep mixing
        geo_result = self.geo_mp(
            h_pre, geo_edge_index, geo_edge_sh, geo_edge_rbf,
            source_rotations=source_rotations,
            edge_weight=geo_edge_weight,
            edge_state=edge_state,
            return_messages=return_messages,
            source_D_cache=source_D_cache,
        )

        if return_messages:
            m_geo, messages = geo_result
        else:
            m_geo = geo_result
            messages = None

        if sym_edge_index.numel() > 0:
            m_sym = self.sym_mp(h, sym_edge_index, sym_edge_attr)
        else:
            m_sym = torch.zeros_like(m_geo)

        h_new = m_geo + m_sym
        h_new = self.norm(h_new)
        h_new = self.dropout(h_new)

        if return_messages:
            return h_new, messages
        return h_new


# ---------------------------------------------------------------------------
# Geometric lifting layer (first layer only) — init scalars → hidden irreps
# ---------------------------------------------------------------------------

class GeometricLiftingLayer(nn.Module):
    """Minimal scalar-input CG lift: element scalars × RBF gate × SH direction.

    Input:  ``irreps_in``   (pure scalars, e.g. ``64x0e``)
    Output: ``irreps_out``  (e.g. ``64x0e + 16x1o + 8x2e``)

    From a pure-scalar node feature, the only coordinate-free way to create
    ``l>0`` channels is ``scalar_coeff ⊗ Y_l(rhat)``. Because the input is a
    scalar, this reduces to the **minimal** CG: a per-channel scalar coefficient
    multiplied by the real spherical harmonic ``Y_l(rhat)`` and gated by a
    scalar function of the edge distance ``RBF(d)``. No full dynamic tensor
    product, no dense per-edge weight matrix.

    For each geometric sub-edge ``α: j → i`` with unit direction ``r̂_α``:

        a_l = A_l · h_j              (invariant scalar coefficients, l=0,1,2)
        g_l = gate_l(RBF(d_α))       (per-edge, per-channel scalar gate)
        m_α^{0e} = g_0 ⊙ a_0                    (Y_0 ≡ const under `component`)
        m_α^{1o} = (g_1 ⊙ a_1) ⊗ Y_1(r̂_α)      → 1o channels
        m_α^{2e} = (g_2 ⊙ a_2) ⊗ Y_2(r̂_α)      → 2e channels
        h_i = Σ_α m_α   (+ scalar skip on the 0e block)

    Equivariance: ``a_l`` and ``g_l`` depend only on invariant scalars and the
    distance; ``Y_l`` carries the irrep-``l`` transform; the source image is
    already baked into ``r̂_α``. The residual is scalar-only (0e→0e). Site
    symmetry is handled by the site projector, so this layer uses only real
    geometric edges — no symmetry-MP channel.

    Scalar-only output (``c1 = c2 = 0``) is supported: only the ``m_0`` path
    runs.
    """

    def __init__(
        self,
        irreps_in: o3.Irreps,
        irreps_out: o3.Irreps,
        irreps_sh: o3.Irreps,
        num_rbf: int = 32,
        radial_gate_width: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps(irreps_out)

        # Per-l output multiplicities (l in {0,1,2}); other l are unsupported.
        mul = {0: 0, 1: 0, 2: 0}
        for m, ir in self.irreps_out:
            if ir.l not in mul:
                raise ValueError(
                    f"GeometricLiftingLayer supports l<=2 outputs, got {ir}."
                )
            mul[ir.l] += m
        self.c0, self.c1, self.c2 = mul[0], mul[1], mul[2]
        self.n_coeff = self.c0 + self.c1 + self.c2

        # Spherical-harmonic column layout (must be 1x0e + 1x1o + 1x2e).
        self.irreps_sh = o3.Irreps(irreps_sh)

        # Source scalar coefficients: D0×0e → (c0+c1+c2)×0e.
        self.lin_coeff = o3.Linear(
            self.irreps_in, o3.Irreps(f"{self.n_coeff}x0e"),
        )

        # Per-edge, per-channel radial gate from RBF(d).
        self.radial_gate = nn.Sequential(
            nn.Linear(num_rbf, radial_gate_width),
            nn.SiLU(),
            nn.Linear(radial_gate_width, self.n_coeff),
        )
        with torch.no_grad():
            self.radial_gate[-1].weight.mul_(1.0 / (radial_gate_width ** 0.5))

        # Scalar-only skip: 0e→0e residual.
        self.skip_scalar = o3.Linear(
            self.irreps_in, o3.Irreps(f"{self.c0}x0e"),
        )

        self.norm = IrrepNorm(self.irreps_out)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(
        self,
        h: torch.Tensor,
        geo_edge_index: torch.Tensor,
        geo_edge_sh: torch.Tensor,
        geo_edge_rbf: torch.Tensor,
        source_rotations: Optional[torch.Tensor] = None,
        geo_edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        K = h.size(0)
        target, src = geo_edge_index[0], geo_edge_index[1]

        skip0 = self.skip_scalar(h)  # (K, c0)

        def _empty_out():
            parts = [skip0]
            if self.c1 > 0:
                parts.append(h.new_zeros(K, self.c1 * 3))
            if self.c2 > 0:
                parts.append(h.new_zeros(K, self.c2 * 5))
            return self.dropout(self.norm(torch.cat(parts, dim=-1)))

        if src.numel() == 0:
            return _empty_out()

        # Source scalar coefficients (invariant); split per l.
        coeff = self.lin_coeff(h)  # (K, n_coeff)
        a0, a1, a2 = torch.split(coeff, [self.c0, self.c1, self.c2], dim=-1)

        # Per-edge radial gates (invariant); split per l.
        g = self.radial_gate(geo_edge_rbf)  # (E, n_coeff)
        g0, g1, g2 = torch.split(g, [self.c0, self.c1, self.c2], dim=-1)

        # SH direction: cols [0]=Y0, [1:4]=Y1, [4:9]=Y2.
        ew = geo_edge_weight.unsqueeze(-1) if geo_edge_weight is not None else None

        # l=0 message: Y0 is constant under `component` normalisation.
        m0 = g0 * a0[src]  # (E, c0)
        if ew is not None:
            m0 = m0 * ew
        agg0 = scatter(m0, target, dim=0, dim_size=K, reduce="sum") + skip0

        parts = [agg0]

        if self.c1 > 0:
            Y1 = geo_edge_sh[:, 1:4]                    # (E, 3)
            s1 = (g1 * a1[src]).unsqueeze(-1)           # (E, c1, 1)
            m1 = (s1 * Y1[:, None, :]).reshape(-1, self.c1 * 3)
            if ew is not None:
                m1 = m1 * ew
            parts.append(scatter(m1, target, dim=0, dim_size=K, reduce="sum"))

        if self.c2 > 0:
            Y2 = geo_edge_sh[:, 4:9]                    # (E, 5)
            s2 = (g2 * a2[src]).unsqueeze(-1)           # (E, c2, 1)
            m2 = (s2 * Y2[:, None, :]).reshape(-1, self.c2 * 5)
            if ew is not None:
                m2 = m2 * ew
            parts.append(scatter(m2, target, dim=0, dim_size=K, reduce="sum"))

        h_new = torch.cat(parts, dim=-1)
        h_new = self.norm(h_new)
        return self.dropout(h_new)


# ---------------------------------------------------------------------------
# Readout: SE(3)-invariant graph-level pooling + prediction head
# ---------------------------------------------------------------------------


class EquivariantReadout(nn.Module):
    """Pool node irreps into an SE(3)-invariant graph-level representation.

    Strategy:
        - Take only the ``0e`` scalar channels (guaranteed SE(3)-invariant).
        - Pool to graph-level (mean/sum/attention).
        - Optionally concatenate edge pool representation.
        - Predict via a small MLP.

    Optionally supports a NequIP-style per-element scale/shift stage that
    learns a chemistry baseline: ``y_per_orbit = shift[Z] + scale[Z] * MLP(h)``
    applied at the orbit level *before* pooling. For intensive properties
    (mean-pool) the graph value becomes a multiplicity-weighted mean of
    per-orbit predictions, so element-averaged behaviour is baked in early
    and the MLP only has to model residuals.
    """

    def __init__(
        self,
        irreps_node: o3.Irreps,
        hidden_dim: int = 128,
        pool: str = "mean",
        readout_mode: str = "intensive_scalar",
        dropout: float = 0.1,
        edge_pool_dim: int = 0,
        per_type_shift: bool = False,
        per_type_scale: bool = False,
        max_atomic_number: int = 118,
    ):
        super().__init__()
        self.pool = pool
        self.readout_mode = readout_mode
        self.edge_pool_dim = edge_pool_dim
        if readout_mode not in (
            "extensive_scalar", "intensive_scalar",
            "nonadditive_graph_scalar", "graph_vector", "atom_vector",
        ):
            _map = {"intensive": "intensive_scalar", "extensive": "extensive_scalar"}
            self.readout_mode = _map.get(readout_mode, readout_mode)

        scalar_dim = 0
        for mul, ir in irreps_node:
            if ir == o3.Irrep(0, 1):
                scalar_dim += mul
        assert scalar_dim > 0, "irreps_node must have 0e channels for readout"

        self.scalar_dim = scalar_dim

        if pool == "attention":
            self.attn = nn.Sequential(
                nn.Linear(scalar_dim, scalar_dim // 2),
                nn.SiLU(),
                nn.Linear(scalar_dim // 2, 1),
            )

        head_in = scalar_dim + edge_pool_dim
        self.head = nn.Sequential(
            nn.Linear(head_in, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        # --- Per-element scale/shift (NequIP-style) ---
        self.per_type_shift = per_type_shift
        self.per_type_scale = per_type_scale
        self.max_atomic_number = max_atomic_number
        if self.per_type_shift:
            self.type_shift = nn.Parameter(
                torch.zeros(max_atomic_number + 1)
            )
        else:
            self.register_parameter("type_shift", None)
        if self.per_type_scale:
            self.type_scale = nn.Parameter(
                torch.ones(max_atomic_number + 1)
            )
        else:
            self.register_parameter("type_scale", None)

        # Separate MLP head for per-type branch: consumes scalar_dim only.
        # (edge pool information enters as an additive residual after pooling.)
        if per_type_shift or per_type_scale:
            self.per_type_head = nn.Sequential(
                nn.Linear(scalar_dim, hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1),
            )
        else:
            self.per_type_head = None

        # Edge-pool residual channel used only when per_type path is on and
        # edge_pool_dim > 0. Small init so training starts from a pure
        # per-type baseline.
        if (per_type_shift or per_type_scale) and edge_pool_dim > 0:
            self.edge_residual = nn.Linear(edge_pool_dim, 1, bias=False)
            nn.init.zeros_(self.edge_residual.weight)
        else:
            self.edge_residual = None

    def _apply_per_type(self, y_per_orbit, atomic_numbers):
        """y ← shift[Z] + scale[Z] * y (both learnable if enabled)."""
        if self.type_scale is not None:
            z_clamped = atomic_numbers.clamp(0, self.max_atomic_number).long()
            y_per_orbit = self.type_scale[z_clamped] * y_per_orbit
        if self.type_shift is not None:
            z_clamped = atomic_numbers.clamp(0, self.max_atomic_number).long()
            y_per_orbit = y_per_orbit + self.type_shift[z_clamped]
        return y_per_orbit

    def forward(
        self,
        h: torch.Tensor,
        batch: torch.Tensor,
        node_weights: Optional[torch.Tensor] = None,
        edge_graph_repr: Optional[torch.Tensor] = None,
        atomic_numbers: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Pool to graph-level and predict based on readout_mode.

        Args:
            h: (N_total, dim_irreps) node features.
            batch: (N_total,) graph assignment indices.
            node_weights: Optional (N_total,) per-node weights.
            edge_graph_repr: Optional (num_graphs, edge_pool_dim) edge pool.
            atomic_numbers: (N_total,) orbit element numbers, required only
                when per_type_shift or per_type_scale is enabled.

        Returns:
            (num_graphs,) scalar predictions.
        """
        h_scalar = h[:, :self.scalar_dim]

        _use_per_type = (self.type_shift is not None) or (self.type_scale is not None)
        if _use_per_type:
            assert atomic_numbers is not None, (
                "per_type_shift/scale enabled but atomic_numbers not passed."
            )

        if self.readout_mode == "extensive_scalar" and node_weights is not None and self.edge_pool_dim == 0:
            head_fn = self.per_type_head if _use_per_type else self.head
            y_per_orbit = head_fn(h_scalar).squeeze(-1)
            if _use_per_type:
                y_per_orbit = self._apply_per_type(y_per_orbit, atomic_numbers)
            y_total = scatter(
                y_per_orbit * node_weights, batch, dim=0, reduce="sum"
            )
            return y_total

        # --- Intensive / graph readout paths ---
        # If per-type is enabled we compute head at orbit level, mix with
        # chemistry baseline, then pool weighted mean. edge_graph_repr (if
        # any) is added AFTER pooling as a graph-level correction MLP.
        if _use_per_type:
            y_per_orbit = self.per_type_head(h_scalar).squeeze(-1)
            y_per_orbit = self._apply_per_type(y_per_orbit, atomic_numbers)
            w = node_weights if node_weights is not None else torch.ones_like(y_per_orbit)
            if self.pool == "sum":
                y_graph = scatter(y_per_orbit * w, batch, dim=0, reduce="sum")
            else:
                num = scatter(y_per_orbit * w, batch, dim=0, reduce="sum")
                den = scatter(w, batch, dim=0, reduce="sum").clamp(min=1e-6)
                y_graph = num / den
            if edge_graph_repr is not None and self.edge_residual is not None:
                y_graph = y_graph + self.edge_residual(edge_graph_repr).squeeze(-1)
            return y_graph

        w = node_weights.unsqueeze(-1) if node_weights is not None else None
        if w is not None:
            h_weighted = h_scalar * w
            if self.pool == "attention":
                attn_w = torch_geometric_softmax(self.attn(h_scalar), batch)
                h_graph = scatter(h_weighted * attn_w, batch, dim=0, reduce="sum")
                norm = scatter(w * attn_w, batch, dim=0, reduce="sum").clamp(min=1e-6)
                h_graph = h_graph / norm
            elif self.pool == "mean":
                h_graph = scatter(h_weighted, batch, dim=0, reduce="sum")
                norm = scatter(w, batch, dim=0, reduce="sum").clamp(min=1e-6)
                h_graph = h_graph / norm
            elif self.pool == "sum":
                h_graph = scatter(h_weighted, batch, dim=0, reduce="sum")
            else:
                h_graph = scatter(h_scalar, batch, dim=0, reduce="max")
        elif self.pool == "attention":
            attn_w = torch_geometric_softmax(self.attn(h_scalar), batch)
            h_graph = scatter(h_scalar * attn_w, batch, dim=0, reduce="sum")
        elif self.pool == "sum":
            h_graph = scatter(h_scalar, batch, dim=0, reduce="sum")
        elif self.pool == "max":
            h_graph = scatter(h_scalar, batch, dim=0, reduce="max")
        else:
            h_graph = scatter(h_scalar, batch, dim=0, reduce="mean")

        if edge_graph_repr is not None and self.edge_pool_dim > 0:
            h_graph = torch.cat([h_graph, edge_graph_repr], dim=-1)

        return self.head(h_graph).squeeze(-1)


def torch_geometric_softmax(
    src: torch.Tensor, index: torch.Tensor
) -> torch.Tensor:
    """Softmax over nodes within each graph (numerically stable)."""
    from torch_geometric.utils import softmax
    return softmax(src, index)


# ---------------------------------------------------------------------------
# Per-path dynamic TP (EdgeState-v2)
# ---------------------------------------------------------------------------


class EquivariantGeometricMP_v2(nn.Module):
    """Geometric MP with per-path dynamic TP weights (NequIP-style).

    Key differences from v1 (shared TP + gate):
    - Filtered CG instructions: only paths producing target output irreps
    - Per-edge weights from MLP(concat(RBF, edge_state))
    - linear_2 bottleneck: TP mid-irreps → output irreps
    - Pre-TP normalization for stable aggregation
    """

    def __init__(
        self,
        irreps_node: o3.Irreps,
        irreps_message: o3.Irreps,
        irreps_sh: o3.Irreps,
        num_rbf: int = 32,
        edge_state_dim: int = 128,
        radial_mlp_width: int = 128,
    ):
        super().__init__()
        self.irreps_node = irreps_node
        self.irreps_message = irreps_message
        self.edge_state_dim = edge_state_dim

        # Build filtered TP instructions (only paths → output irreps in node)
        irreps_mid = []
        instructions = []
        for i, (mul, ir_in) in enumerate(irreps_node):
            for j, (_, ir_edge) in enumerate(irreps_sh):
                for ir_out in ir_in * ir_edge:
                    if ir_out in irreps_node:
                        k = len(irreps_mid)
                        irreps_mid.append((mul, ir_out))
                        instructions.append((i, j, k, "uvu", True))

        irreps_mid = o3.Irreps(irreps_mid)
        irreps_mid, p, _ = irreps_mid.sort()
        instructions = [
            (i1, i2, p[i_out], mode, train)
            for i1, i2, i_out, mode, train in instructions
        ]
        self._irreps_mid = irreps_mid

        self.tp = o3.TensorProduct(
            irreps_node,
            irreps_sh,
            irreps_mid,
            instructions,
            shared_weights=False,
            internal_weights=False,
        )

        # linear_2: compress mid → message irreps (bottleneck)
        self.linear_2 = o3.Linear(
            irreps_mid.simplify(),
            irreps_message,
            internal_weights=True,
            shared_weights=True,
        )

        # Weight MLP with proper initialization
        weight_input_dim = num_rbf + edge_state_dim
        self.weight_mlp = nn.Sequential(
            nn.Linear(weight_input_dim, radial_mlp_width),
            nn.SiLU(),
            nn.Linear(radial_mlp_width, self.tp.weight_numel),
        )
        # Scale final layer for stable TP weight magnitudes
        with torch.no_grad():
            self.weight_mlp[-1].weight.mul_(1.0 / (radial_mlp_width ** 0.5))

        # Pre-TP normalization: learnable per-irrep scale
        self.pre_norm_scale = nn.Parameter(torch.ones(1))

        # Gate after linear_2
        gate_act, gate_gate, gate_gated = _make_gate_irreps(irreps_message)
        n_gated = sum(mul for mul, _ in gate_gated)
        if n_gated == 0:
            self.gate = None
            self._gate_out = gate_act
        else:
            self.gate = Gate(
                gate_act, [F.silu],
                gate_gate, [F.silu],
                gate_gated,
            )
            self._gate_out = self.gate.irreps_out

        self.skip = o3.Linear(irreps_node, self._gate_out)

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_sh: torch.Tensor,
        edge_rbf: torch.Tensor,
        source_rotations: Optional[torch.Tensor] = None,
        edge_weight: Optional[torch.Tensor] = None,
        edge_state: Optional[torch.Tensor] = None,
        return_messages: bool = False,
        source_D_cache: Optional[dict] = None,
    ):
        K = h.size(0)
        target, src = edge_index[0], edge_index[1]

        if src.numel() == 0:
            out = self.skip(h)
            if return_messages:
                return out, torch.zeros(0, self.irreps_message.dim, device=h.device)
            return out

        h_src = h[src]

        if source_rotations is not None:
            h_src = _rotate_node_features(
                h_src, source_rotations, self.irreps_node, D_cache=source_D_cache
            )

        # Pre-TP normalization (approximate avg_num_neighbors)
        h_src = h_src * self.pre_norm_scale

        # Per-path weights from edge_state + RBF
        if edge_state is not None and self.edge_state_dim > 0:
            weight_input = torch.cat([edge_rbf, edge_state], dim=-1)
        else:
            weight_input = edge_rbf

        tp_weights = self.weight_mlp(weight_input)

        # Filtered TP with per-edge external weights
        msg_mid = self.tp(h_src, edge_sh, tp_weights)

        # linear_2: compress mid → message (bottleneck)
        msg = self.linear_2(msg_mid)

        msg_for_update = msg if return_messages else None

        if edge_weight is not None:
            msg = msg * edge_weight.unsqueeze(-1)

        agg = scatter(msg, target, dim=0, dim_size=K, reduce="sum")

        if self.gate is not None:
            out = self.gate(agg) + self.skip(h)
        else:
            out = F.silu(agg) + self.skip(h)

        if return_messages:
            return out, msg_for_update
        return out


__all__ = [
    "ElementOnlyNodeEncoder",
    "GaussianRBF",
    "EquivariantGeometricMP",
    "EquivariantGeometricMP_v2",
    "EquivariantSymmetryMP",
    "EquivariantWyckoffLayer",
    "GeometricLiftingLayer",
    "EquivariantReadout",
    "_make_gate_irreps",
    "compute_irreps",
    "IrrepNorm",
    "_rotate_node_features",
    "_IRREPS_NODE",
    "_IRREPS_SH",
    "_IRREPS_MESSAGE",
    "_DEFAULT_HIDDEN_IRREPS",
]
