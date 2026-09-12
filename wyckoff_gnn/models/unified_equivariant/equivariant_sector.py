"""Unified per-layer equivariant high-l sector.

Implements the correct equivariant high-l channels:
- LightweightScalarToHighL: scalar × Y_l(edge_direction) → high-l
- CachedSourceImageTransport: apply cached Wigner-D to source features
- SameLHighLPropagation: lightweight same-l channel mixing
- CopyWiseIrrepInvariants: per-copy ||h_c^(l)||² invariants
- ZeroInitHighLToScalar: true zero-init high-l → scalar feedback
- UnifiedQuotientEquivariantBlock: joint scalar + high-l per-layer block

All geometry (SH, Wigner-D, edge vectors) is expected to be pre-cached
on the data object by the preprocessing/collate step.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from e3nn import o3


# ---------------------------------------------------------------------------
# Utility functions for geometry computation (used in preprocessing/caching)
# ---------------------------------------------------------------------------


def compute_source_rotations_with_batch(
    orbit_sym_ops_W_frac: torch.Tensor,
    geo_edge_index: torch.Tensor,
    geo_edge_source_image: torch.Tensor,
    lattice: torch.Tensor,
    batch: torch.Tensor,
    num_graphs: int,
) -> torch.Tensor:
    """Compute Cartesian rotation matrices with batch info.

    R_e3nn = A^T @ W_frac @ A^{-T}

    Args:
        orbit_sym_ops_W_frac: (N, max_mult, 3, 3)
        geo_edge_index: (2, E)
        geo_edge_source_image: (E,)
        lattice: (num_graphs * 3, 3) or (num_graphs, 3, 3)
        batch: (N,) node → graph assignment
        num_graphs: number of graphs

    Returns:
        R_e3nn: (E, 3, 3)
    """
    source_orbit = geo_edge_index[1]
    W_frac = orbit_sym_ops_W_frac[source_orbit, geo_edge_source_image]  # (E, 3, 3)

    lattice_3d = lattice.reshape(num_graphs, 3, 3)  # (G, 3, 3)

    A_inv_T = torch.linalg.inv(lattice_3d).transpose(1, 2)  # (G, 3, 3)
    A_T = lattice_3d.transpose(1, 2)  # (G, 3, 3)

    A_T_per_edge = A_T[batch[source_orbit]]  # (E, 3, 3)
    A_inv_T_per_edge = A_inv_T[batch[source_orbit]]  # (E, 3, 3)

    R_e3nn = torch.bmm(torch.bmm(A_T_per_edge, W_frac), A_inv_T_per_edge)
    return R_e3nn


def _wigner_d1(R: torch.Tensor) -> torch.Tensor:
    """Analytical D^1(R) = R for e3nn's real SH basis (m=-1,0,1) = (x,y,z)."""
    return R


def _wigner_d2(R: torch.Tensor) -> torch.Tensor:
    """Analytical D^2(R) for e3nn's real SH basis.

    e3nn l=2 basis (m=-2,-1,0,1,2):
        g_{-2} = sqrt(15) * xz,  g_{-1} = sqrt(15) * xy
        g_0    = sqrt(5/4) * (2y^2 - x^2 - z^2)
        g_1    = sqrt(15) * yz,  g_2    = sqrt(15/4) * (z^2 - x^2)

    Directly computes the 5x5 Wigner-D matrix from R elements
    via traceless symmetric tensor transformation + change of basis.
    """
    R00, R01, R02 = R[..., 0, 0], R[..., 0, 1], R[..., 0, 2]
    R10, R11, R12 = R[..., 1, 0], R[..., 1, 1], R[..., 1, 2]
    R20, R21, R22 = R[..., 2, 0], R[..., 2, 1], R[..., 2, 2]

    # Only compute the S-basis rows needed:
    # M[3,:] (S_xz), M[2,:] (S_xy), M[1,:] (S_yy), M[4,:] (S_yz), M[0,:2] (S_xx)

    # M[3,:] = S_xz transformation
    M30 = R00 * R20 - R02 * R22
    M31 = R01 * R21 - R02 * R22
    M32 = R00 * R21 + R01 * R20
    M33 = R00 * R22 + R02 * R20
    M34 = R01 * R22 + R02 * R21

    # M[2,:] = S_xy transformation
    M20 = R00 * R10 - R02 * R12
    M21 = R01 * R11 - R02 * R12
    M22 = R00 * R11 + R01 * R10
    M23 = R00 * R12 + R02 * R10
    M24 = R01 * R12 + R02 * R11

    # M[1,:] = S_yy transformation
    M10 = R10 * R10 - R12 * R12
    M11 = R11 * R11 - R12 * R12
    M12 = 2 * R10 * R11
    M13 = 2 * R10 * R12
    M14 = 2 * R11 * R12

    # M[4,:] = S_yz transformation
    M40 = R10 * R20 - R12 * R22
    M41 = R11 * R21 - R12 * R22
    M42 = R10 * R21 + R11 * R20
    M43 = R10 * R22 + R12 * R20
    M44 = R11 * R22 + R12 * R21

    # M[0,0] and M[0,1] = S_xx transformation (only cols 0,1 needed)
    M00 = R00 * R00 - R02 * R02
    M01 = R01 * R01 - R02 * R02

    # Direct D^2 = P_inv @ M @ P (expanded)
    inv_sqrt3 = 0.5773502691896258  # 1/sqrt(3)
    D = R.new_empty(R.shape[:-2] + (5, 5))

    # Row 0 (g_{-2}): depends on M[3,:]
    D[..., 0, 0] = M33
    D[..., 0, 1] = M32
    D[..., 0, 2] = -inv_sqrt3 * M30 + 2 * inv_sqrt3 * M31
    D[..., 0, 3] = M34
    D[..., 0, 4] = -M30

    # Row 1 (g_{-1}): depends on M[2,:]
    D[..., 1, 0] = M23
    D[..., 1, 1] = M22
    D[..., 1, 2] = -inv_sqrt3 * M20 + 2 * inv_sqrt3 * M21
    D[..., 1, 3] = M24
    D[..., 1, 4] = -M20

    # Row 2 (g_0): depends on M[1,:]
    sqrt3_2 = 0.8660254037844386  # sqrt(3)/2
    D[..., 2, 0] = sqrt3_2 * M13
    D[..., 2, 1] = sqrt3_2 * M12
    D[..., 2, 2] = -0.5 * M10 + M11
    D[..., 2, 3] = sqrt3_2 * M14
    D[..., 2, 4] = -sqrt3_2 * M10

    # Row 3 (g_1): depends on M[4,:]
    D[..., 3, 0] = M43
    D[..., 3, 1] = M42
    D[..., 3, 2] = -inv_sqrt3 * M40 + 2 * inv_sqrt3 * M41
    D[..., 3, 3] = M44
    D[..., 3, 4] = -M40

    # Row 4 (g_2): depends on M[0,:] and M[1,:]
    D[..., 4, 0] = -2 * R00 * R02 - R10 * R12
    D[..., 4, 1] = -2 * R00 * R01 - R10 * R11
    D[..., 4, 2] = 0.5773502691896258 * M00 - 1.1547005383792515 * M01 + 0.2886751345948129 * M10 - 0.5773502691896258 * M11
    D[..., 4, 3] = -2 * R01 * R02 - R11 * R12
    D[..., 4, 4] = M00 + 0.5 * M10

    return D


def build_wigner_d_cache(
    R_e3nn: torch.Tensor,
    hidden_irreps: o3.Irreps,
) -> Dict[Tuple[int, int], torch.Tensor]:
    """Pre-compute Wigner-D matrices for all (l, p) in hidden_irreps.

    e3nn defines D^{(l,p)}(R) = D^l(det(R) * R) * p^k with k = (1-det(R))/2, so
    the analytical fast paths are only parity-independent for two irreps:
      (1,-1): D = (det R * R) * (-1)^k = R for both proper and improper R.
      (2, 1): D^2 is quadratic in R, so D^2(-R) = D^2(R), and p^k = 1.
    Every other (l,p) — 1e, 2o, 3e, 3o, ... — picks up a det-dependent sign and
    must go through e3nn, otherwise improper source images transport wrongly.

    Args:
        R_e3nn: (E, 3, 3) rotation matrices (may be improper)
        hidden_irreps: node irreps

    Returns:
        D_cache: dict mapping (l, p) -> D of shape (E, 2l+1, 2l+1)
    """
    FAST_PATHS = {(1, -1): _wigner_d1, (2, 1): _wigner_d2}

    D_cache = {}
    for l, p in {(ir.l, ir.p) for _, ir in hidden_irreps}:
        if l == 0:
            continue
        fast = FAST_PATHS.get((l, p))
        D_cache[(l, p)] = (
            fast(R_e3nn) if fast is not None
            else o3.Irrep(l, p).D_from_matrix(R_e3nn)
        )

    return D_cache


def spherical_harmonics_layout(lmax: int) -> Dict[Tuple[int, int], slice]:
    """Map (l, parity) -> slice into the SH vector produced by compute_edge_sh.

    The SH basis carries natural parity, p = (-1)^l, so only those keys exist.
    """
    layout = {}
    offset = 0
    for mul, ir in o3.Irreps.spherical_harmonics(lmax):
        layout[(ir.l, ir.p)] = slice(offset, offset + ir.dim)
        offset += mul * ir.dim
    return layout


def compute_edge_sh(
    edge_vec: torch.Tensor,
    lmax: int,
) -> torch.Tensor:
    """Compute spherical harmonics for edge directions.

    Args:
        edge_vec: (E, 3) Cartesian edge vectors
        lmax: max angular momentum (any value >= 0)

    Returns:
        edge_sh: (E, (lmax+1)^2) spherical harmonics with natural parity
    """
    if lmax < 0:
        raise ValueError(f"lmax must be >= 0, got {lmax}")

    edge_rhat = edge_vec / edge_vec.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    return o3.spherical_harmonics(
        o3.Irreps.spherical_harmonics(lmax), edge_rhat,
        normalize=True, normalization="component",
    )


def precompute_batch_geometry(
    data,
    hidden_irreps: o3.Irreps,
    lmax: int,
) -> Tuple[torch.Tensor, Dict[Tuple[int, int], torch.Tensor]]:
    """Compute all geometry needed by unified blocks from batch data.

    Computes edge vectors, spherical harmonics, source rotations, and
    Wigner-D matrices. Called once per forward pass in model.forward().

    Args:
        data: PyG Batch with fields:
            orbit_rep_frac, geo_edge_index, geo_edge_source_frac,
            geo_edge_shift, lattice, batch, num_graphs,
            orbit_sym_ops_W_frac, geo_edge_source_image
        hidden_irreps: node irreps for Wigner-D computation
        lmax: max angular momentum for SH computation

    Returns:
        edge_sh: (E, dim_sh)
        wigner_d_cache: dict (l, p) → (E, 2l+1, 2l+1)
    """
    num_graphs = getattr(data, 'num_graphs', None) or (data.batch.max().item() + 1)
    batch = data.batch
    edge_index = data.geo_edge_index
    target, source = edge_index[0], edge_index[1]

    # Lattice per graph: (G, 3, 3)
    lattice_3d = data.lattice.reshape(num_graphs, 3, 3)

    # Edge vectors in Cartesian coordinates
    # frac: r_cart = frac @ A  (A = lattice with row vectors)
    target_frac = data.orbit_rep_frac[target]  # (E, 3)
    source_frac = data.geo_edge_source_frac  # (E, 3)
    shift = data.geo_edge_shift  # (E, 3)
    delta_frac = source_frac - target_frac + shift  # (E, 3)
    lattice_e = lattice_3d[batch[target]]  # (E, 3, 3)
    edge_vec = (delta_frac.unsqueeze(-1) * lattice_e).sum(1)

    # Spherical harmonics
    edge_sh = compute_edge_sh(edge_vec, lmax)

    # Source rotations for Wigner-D transport
    R_e3nn = compute_source_rotations_with_batch(
        data.orbit_sym_ops_W_frac,
        edge_index,
        data.geo_edge_source_image,
        data.lattice,
        batch,
        num_graphs,
    )

    # Wigner-D cache
    wigner_d_cache = build_wigner_d_cache(R_e3nn, hidden_irreps)

    return edge_sh, wigner_d_cache


# ---------------------------------------------------------------------------
# LightweightScalarToHighL: scalar × Y_l → high-l
# ---------------------------------------------------------------------------


class LightweightScalarToHighL(nn.Module):
    """Generate high-l messages via scalar coefficients × spherical harmonics.

    For each edge, compute:
        coeff = MLP([edge_feat, scalar_target, scalar_source]) → (n_coeffs,)
        msg = coeff × Y_l(rhat_edge)

    This is the legal equivariant lifting: 0e ⊗ Y_l → l.

    For 8x1o + 4x2e: n_coeffs = 8 + 4 = 12.
    """

    def __init__(
        self,
        hidden_scalar: int = 128,
        hidden_irreps: str = "8x1o + 4x2e",
        lmax: int = 2,
    ):
        super().__init__()
        self.hidden_irreps = o3.Irreps(hidden_irreps)
        self.lmax = lmax

        # Count number of high-l copies (each needs one scalar coefficient)
        self.n_coeffs = sum(mul for mul, ir in self.hidden_irreps if ir.l > 0)

        # MLP: [edge, s_target, s_source] → n_coeffs
        input_dim = hidden_scalar * 3  # edge + target + source
        self.coeff_mlp = nn.Sequential(
            nn.Linear(input_dim, 32, bias=True),
            nn.SiLU(),
            nn.Linear(32, self.n_coeffs, bias=True),
        )

        # Build SH irreps string
        if lmax == 1:
            self.irreps_sh = o3.Irreps("1x0e + 1x1o")
        else:
            self.irreps_sh = o3.Irreps("1x0e + 1x1o + 1x2e")

        # Pre-compute l → start index in edge_sh tensor
        # edge_sh layout: [l=0 (1) | l=1 (3) | l=2 (5)]
        self._sh_offset = {}
        offset = 0
        for mul, ir in self.irreps_sh:
            self._sh_offset[ir.l] = offset
            offset += mul * ir.dim

    def forward(
        self,
        scalar_edge: torch.Tensor,
        scalar_node_target: torch.Tensor,
        scalar_node_source: torch.Tensor,
        edge_sh: torch.Tensor,
    ) -> torch.Tensor:
        """Generate high-l messages.

        Args:
            scalar_edge: (E, hidden_scalar)
            scalar_node_target: (E, hidden_scalar)
            scalar_node_source: (E, hidden_scalar)
            edge_sh: (E, dim_sh) spherical harmonics

        Returns:
            msg: (E, dim_high_l) generated high-l messages
        """
        # Compute scalar coefficients
        ctx = torch.cat([scalar_edge, scalar_node_target, scalar_node_source], dim=-1)
        coeffs = self.coeff_mlp(ctx)  # (E, n_coeffs)

        # Multiply coefficients by SH, using explicit l-based offsets
        msg_parts = []
        coeff_idx = 0

        for mul, ir in self.hidden_irreps:
            if ir.l == 0:
                continue
            dim = ir.dim  # 2l+1
            sh_start = self._sh_offset[ir.l]
            block_parts = []
            for c in range(mul):
                c_val = coeffs[:, coeff_idx:coeff_idx + 1]
                sh_block = edge_sh[:, sh_start:sh_start + dim]
                block_parts.append(c_val * sh_block)
                coeff_idx += 1
            msg_parts.append(torch.cat(block_parts, dim=-1))

        return torch.cat(msg_parts, dim=-1)


# ---------------------------------------------------------------------------
# CachedSourceImageTransport
# ---------------------------------------------------------------------------


class CachedSourceImageTransport(nn.Module):
    """Apply cached Wigner-D matrices to transport source high-l features.

    For each sub-edge, rotate source features to align with edge frame:
        h_rotated = D^l(R_image) @ h_source
    """

    def __init__(self, hidden_irreps: str = "8x1o + 4x2e"):
        super().__init__()
        self.hidden_irreps = o3.Irreps(hidden_irreps)

    def forward(
        self,
        high_l_source: torch.Tensor,
        wigner_d_cache: Dict[Tuple[int, int], torch.Tensor],
    ) -> torch.Tensor:
        """Apply Wigner-D transport to source features.

        Args:
            high_l_source: (E, dim_high_l) source node high-l features
            wigner_d_cache: dict mapping (l, p) → D_l of shape (E, 2l+1, 2l+1)

        Returns:
            transported: (E, dim_high_l) rotated features
        """
        parts = []
        idx = 0

        for mul, ir in self.hidden_irreps:
            dim = mul * ir.dim
            block = high_l_source[:, idx:idx + dim]

            if ir.l == 0:
                parts.append(block)
            else:
                D_l = wigner_d_cache[(ir.l, ir.p)]  # (E, 2l+1, 2l+1)
                # Reshape to (E, mul, 2l+1)
                block_3d = block.reshape(-1, mul, ir.dim)
                # Rotate: D @ h^T → (E, 2l+1, mul)
                rotated = torch.bmm(D_l, block_3d.transpose(1, 2))
                # Transpose back: (E, mul, 2l+1) → flatten
                rotated = rotated.transpose(1, 2).reshape(-1, dim)
                parts.append(rotated)

            idx += dim

        return torch.cat(parts, dim=-1)


# ---------------------------------------------------------------------------
# SameLHighLPropagation
# ---------------------------------------------------------------------------


class SameLHighLPropagation(nn.Module):
    """Lightweight same-l channel mixing with scalar coefficients.

    For each transported high-l feature, apply per-copy scalar gating:
        h_prop = coeff × h_transported

    No full tensor product — just scalar multiplication per copy.
    """

    def __init__(
        self,
        hidden_scalar: int = 128,
        hidden_irreps: str = "8x1o + 4x2e",
    ):
        super().__init__()
        self.hidden_irreps = o3.Irreps(hidden_irreps)

        # Count number of high-l copies
        self.n_coeffs = sum(mul for mul, ir in self.hidden_irreps if ir.l > 0)

        # MLP: [edge, s_target, s_source] → n_coeffs
        input_dim = hidden_scalar * 3
        self.coeff_mlp = nn.Sequential(
            nn.Linear(input_dim, 32, bias=True),
            nn.SiLU(),
            nn.Linear(32, self.n_coeffs, bias=True),
        )

    def forward(
        self,
        transported: torch.Tensor,
        scalar_edge: torch.Tensor,
        scalar_node_target: torch.Tensor,
        scalar_node_source: torch.Tensor,
    ) -> torch.Tensor:
        """Apply same-l propagation with scalar coefficients.

        Args:
            transported: (E, dim_high_l) Wigner-D transported features
            scalar_edge: (E, hidden_scalar)
            scalar_node_target: (E, hidden_scalar)
            scalar_node_source: (E, hidden_scalar)

        Returns:
            propagated: (E, dim_high_l) propagated features
        """
        ctx = torch.cat([scalar_edge, scalar_node_target, scalar_node_source], dim=-1)
        coeffs = self.coeff_mlp(ctx)  # (E, n_coeffs)

        # Apply per-copy scalar coefficients
        parts = []
        idx = 0
        coeff_idx = 0

        for mul, ir in self.hidden_irreps:
            if ir.l == 0:
                continue
            dim = mul * ir.dim
            block = transported[:, idx:idx + dim]  # (E, mul * (2l+1))
            block_3d = block.reshape(-1, mul, ir.dim)

            # Per-copy gating: (E, mul, 1) × (E, mul, 2l+1)
            copy_coeffs = coeffs[:, coeff_idx:coeff_idx + mul].unsqueeze(-1)  # (E, mul, 1)
            gated = copy_coeffs * block_3d  # (E, mul, 2l+1)

            parts.append(gated.reshape(-1, dim))
            idx += dim
            coeff_idx += mul

        return torch.cat(parts, dim=-1)


# ---------------------------------------------------------------------------
# CopyWiseIrrepInvariants
# ---------------------------------------------------------------------------


class CopyWiseIrrepInvariants(nn.Module):
    """Extract per-copy rotation invariants ||h_c^(l)||².

    For 8x1o + 4x2e, produces 12 independent invariants:
        [||h_1^(1o)||², ..., ||h_8^(1o)||², ||h_1^(2e)||², ..., ||h_4^(2e)||²]
    """

    def __init__(self, hidden_irreps: str = "8x1o + 4x2e"):
        super().__init__()
        self.hidden_irreps = o3.Irreps(hidden_irreps)
        self.n_invariants = sum(mul for mul, ir in self.hidden_irreps if ir.l > 0)

    def forward(self, high_l_node: torch.Tensor) -> torch.Tensor:
        """Compute per-copy invariants.

        Args:
            high_l_node: (N, dim_high_l)

        Returns:
            invariants: (N, n_invariants) per-copy squared norms
        """
        invariants = []
        idx = 0

        for mul, ir in self.hidden_irreps:
            if ir.l == 0:
                continue
            dim = mul * ir.dim
            block = high_l_node[:, idx:idx + dim]
            # Reshape to (N, mul, 2l+1)
            block_3d = block.reshape(-1, mul, ir.dim)
            # Per-copy norm²: (N, mul)
            norm_sq = block_3d.pow(2).sum(dim=-1)
            # Append each copy separately
            for c in range(mul):
                invariants.append(norm_sq[:, c])
            idx += dim

        return torch.stack(invariants, dim=-1)  # (N, n_invariants)


# ---------------------------------------------------------------------------
# ZeroInitHighLToScalar
# ---------------------------------------------------------------------------


class ZeroInitHighLToScalar(nn.Module):
    """Map high-l invariants to scalar correction with true zero-init.

    The final linear layer is zero-initialized so that loading a scalar
    checkpoint + enabling high-l produces identical initial output.

    A learnable ``inv_temperature`` bounds MLP inputs via ``tanh`` to prevent
    the quadratic feedback loop (||h||² grows unbounded → huge correction →
    scalar explosion).  At init, norms are small so tanh is linear and the
    zero-init output is unchanged; as norms grow, tanh saturates at 1,
    capping the correction magnitude.
    """

    def __init__(
        self,
        n_invariants: int = 12,
        hidden_scalar: int = 128,
    ):
        super().__init__()
        self.inv_temperature = nn.Parameter(torch.tensor(0.1))
        self.mlp = nn.Sequential(
            nn.Linear(n_invariants, 32, bias=True),
            nn.SiLU(),
            nn.Linear(32, hidden_scalar, bias=True),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, invariants: torch.Tensor) -> torch.Tensor:
        """Compute scalar correction.

        Args:
            invariants: (N, n_invariants)

        Returns:
            correction: (N, hidden_scalar)
        """
        bounded = torch.tanh(invariants * self.inv_temperature)
        return self.mlp(bounded)


# ---------------------------------------------------------------------------
# UnifiedQuotientEquivariantBlock
# ---------------------------------------------------------------------------


class UnifiedQuotientEquivariantBlock(nn.Module):
    """One unified per-layer block acting on the full hidden representation.

    The hidden representation is H = scalar_mul x 0e [+ high_irreps].
    When high_irreps is empty, the block naturally reduces to the
    coGN-compatible scalar update with no high-l overhead.

    Ablation switches (all default to enabled for full model):
        use_scalar_to_high_l: toggle scalar→high-l generation
        use_high_l_propagation: toggle same-l propagation
        use_site_irrep_projection: toggle site stabilizer projection
        high_l_feedback_type: "norm2" | "none"
        high_l_aggregation: "mean" | "sum" | "normalized"
    """

    def __init__(
        self,
        hidden_irreps: str = "128x0e",
        use_scalar_to_high_l: bool = True,
        use_high_l_propagation: bool = True,
        use_site_irrep_projection: bool = False,
        high_l_feedback_type: str = "norm2",
        high_l_aggregation: str = "mean",
        layer_idx: int = 0,
        **kwargs,
    ):
        super().__init__()
        from wyckoff_gnn.models.scalar.scalar_block import ScalarProcessingBlock

        self.irreps = o3.Irreps(hidden_irreps)

        scalar_mul = 0
        high_parts = []
        for mul, ir in self.irreps:
            if ir.l == 0 and ir.p == 1:
                scalar_mul += mul
            else:
                high_parts.append(f"{mul}x{ir}")
        self.scalar_mul = scalar_mul
        self.high_irreps = o3.Irreps(" + ".join(high_parts)) if high_parts else o3.Irreps("")
        self.high_dim = self.high_irreps.dim
        self.lmax = max((ir.l for _, ir in self.irreps), default=0)

        self.use_scalar_to_high_l = use_scalar_to_high_l and self.high_dim > 0
        self.use_high_l_propagation = use_high_l_propagation and self.high_dim > 0
        self.use_site_irrep_projection = use_site_irrep_projection and self.high_dim > 0
        self.high_l_feedback_type = high_l_feedback_type if self.high_dim > 0 else "none"
        self.high_l_aggregation = high_l_aggregation
        self.layer_idx = layer_idx

        self.scalar_update = ScalarProcessingBlock(scalar_mul)

        if self.high_dim > 0:
            high_irreps_str = str(self.high_irreps)
            if self.use_scalar_to_high_l:
                self.scalar_to_high_l = LightweightScalarToHighL(
                    scalar_mul, high_irreps_str, self.lmax
                )
            self.transport = CachedSourceImageTransport(high_irreps_str)
            if self.use_high_l_propagation:
                self.high_l_propagation = SameLHighLPropagation(scalar_mul, high_irreps_str)
            self.invariants = CopyWiseIrrepInvariants(high_irreps_str)
            if self.high_l_feedback_type != "none":
                self.high_l_to_scalar = ZeroInitHighLToScalar(
                    self.invariants.n_invariants, scalar_mul
                )
            if self.use_site_irrep_projection:
                self._site_projector = None

    def set_site_projector(self, projector_fn):
        """Inject site projector function: (high_l, node_indices) -> projected_high_l"""
        self._site_projector = projector_fn

    def forward(
        self,
        scalar_node: torch.Tensor,
        high_l_node: Optional[torch.Tensor],
        scalar_edge: torch.Tensor,
        edge_index: torch.Tensor,
        edge_sh: Optional[torch.Tensor] = None,
        wigner_d_cache: Optional[Dict[Tuple[int, int], torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        target, source = edge_index[0], edge_index[1]

        scalar_next = self.scalar_update(scalar_edge, scalar_node, edge_index)

        if self.high_dim == 0:
            return scalar_next, None

        high_l_agg_input = torch.zeros_like(high_l_node)

        if self.use_scalar_to_high_l:
            gen_msg = self.scalar_to_high_l(
                scalar_edge,
                scalar_node[target],
                scalar_node[source],
                edge_sh,
            )
            high_l_agg_input_gen = torch.zeros_like(high_l_node)
            high_l_agg_input_gen.index_add_(0, target, gen_msg)
            high_l_agg_input = high_l_agg_input + high_l_agg_input_gen

        transported = self.transport(high_l_node[source], wigner_d_cache)

        if self.use_high_l_propagation:
            prop_msg = self.high_l_propagation(
                transported,
                scalar_edge,
                scalar_node[target],
                scalar_node[source],
            )
        else:
            prop_msg = transported

        high_l_delta = prop_msg
        high_l_agg = torch.zeros_like(high_l_node)
        high_l_agg.index_add_(0, target, high_l_delta)

        if self.high_l_aggregation == "mean":
            count = torch.zeros(high_l_node.size(0), 1, device=high_l_node.device)
            count.index_add_(0, target, torch.ones(high_l_delta.size(0), 1, device=high_l_delta.device))
            high_l_agg = high_l_agg / count.clamp(min=1)
        elif self.high_l_aggregation == "normalized":
            count = torch.zeros(high_l_node.size(0), 1, device=high_l_node.device)
            count.index_add_(0, target, torch.ones(high_l_delta.size(0), 1, device=high_l_delta.device))
            high_l_agg = high_l_agg / count.sqrt().clamp(min=1)

        high_l_next = high_l_node + high_l_agg + high_l_agg_input

        if self.use_site_irrep_projection and self._site_projector is not None:
            high_l_next = self._site_projector(high_l_next)

        if self.high_l_feedback_type != "none" and hasattr(self, "high_l_to_scalar"):
            inv = self.invariants(high_l_next)
            correction = self.high_l_to_scalar(inv)
            scalar_next = scalar_next + correction

        return scalar_next, high_l_next
