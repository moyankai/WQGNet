"""UnifiedDynamicTPBlock: DTP1-compatible lightweight arbitrary-irrep block.

Strict degeneracy: when hidden_irreps="Nx0e", this block is numerically
identical to ScalarProcessingBlock.

For natural-parity high-l sectors (1o, 2e, 3o, 4e, ...), each sector gets:
  A) Lifting:        source_proj(h_0e[source]) * g_lift * Y_l
  B) Propagation:    Wigner-D transport -> copy_mixing -> g_prop
  C) Feedback:       (transported * Y_l).sum(-1) * g_fb -> feedback_linear (zero-init)

Gate MLP reads only edge features e (no target/source context).
No tanh, no LayerNorm, no factor_rank, no context projections.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from e3nn import o3

from wyckoff_gnn.models.scalar.scalar_block import ScalarProcessingBlock


@dataclass
class SectorMeta:
    """Metadata for one natural-parity high-l sector."""
    index: int
    mul: int
    ir: o3.Irrep
    state_slice: slice
    sh_slice: slice
    gate_slices: dict = field(default_factory=dict)

    @property
    def ir_dim(self) -> int:
        return self.ir.dim

    @property
    def flat_dim(self) -> int:
        return self.mul * self.ir.dim

    @property
    def key(self) -> str:
        sign = "p" if self.ir.p == 1 else "m"
        return f"sector_{self.index}_l{self.ir.l}_{sign}"

    @property
    def natural_parity(self) -> bool:
        return self.ir.p == (-1) ** self.ir.l


def split_scalar_and_high(irreps: o3.Irreps) -> Tuple[int, o3.Irreps]:
    """Split into (scalar 0e multiplicity, l>0 irreps)."""
    scalar_mul = 0
    high_parts = []
    for mul, ir in irreps:
        if ir.l == 0 and ir.p == 1:
            if high_parts:
                raise ValueError(f"hidden_irreps must list 0e first, got '{irreps}'")
            scalar_mul += mul
        else:
            high_parts.append(f"{mul}x{ir}")
    high = o3.Irreps("+".join(high_parts)) if high_parts else o3.Irreps("")
    return scalar_mul, high


def build_sector_metadata(high_irreps: o3.Irreps, lmax: int) -> List[SectorMeta]:
    """Build per-sector metadata; validate natural parity."""
    from wyckoff_gnn.models.unified_equivariant.equivariant_sector import (
        spherical_harmonics_layout,
    )
    sh_layout = spherical_harmonics_layout(lmax)

    sectors = []
    state_offset = 0
    for mul, ir in high_irreps:
        if not (ir.p == (-1) ** ir.l):
            raise ValueError(
                f"Current scalar-seeded DynamicTP supports natural-parity high-l sectors only. "
                f"Got {ir} (l={ir.l}, p={ir.p}), expected p={(-1)**ir.l}."
            )
        sectors.append(SectorMeta(
            index=len(sectors),
            mul=mul,
            ir=ir,
            state_slice=slice(state_offset, state_offset + mul * ir.dim),
            sh_slice=sh_layout[(ir.l, (-1) ** ir.l)],
        ))
        state_offset += mul * ir.dim
    return sectors


def assign_gate_slices(sectors: List[SectorMeta]) -> int:
    """Each sector gets 3*mul gates: lift (mul) + prop (mul) + fb (mul)."""
    gate_idx = 0
    for sec in sectors:
        sec.gate_slices["lift"] = slice(gate_idx, gate_idx + sec.mul)
        gate_idx += sec.mul
    for sec in sectors:
        sec.gate_slices["prop"] = slice(gate_idx, gate_idx + sec.mul)
        gate_idx += sec.mul
    for sec in sectors:
        sec.gate_slices["fb"] = slice(gate_idx, gate_idx + sec.mul)
        gate_idx += sec.mul
    return gate_idx


class UnifiedDynamicTPBlock(nn.Module):
    """DTP1-compatible lightweight arbitrary-irrep block.

    Args:
        hidden_irreps: Full irreps string; must lead with 0e.
        layer_idx: Block index.
    """

    def __init__(
        self,
        hidden_irreps: str = "128x0e",
        layer_idx: int = 0,
        **kwargs,
    ):
        super().__init__()
        self.irreps = o3.Irreps(hidden_irreps).simplify()
        self.layer_idx = layer_idx

        self.scalar_mul, self.high_irreps = split_scalar_and_high(self.irreps)
        self.high_dim = self.high_irreps.dim
        self.lmax = max((ir.l for _, ir in self.irreps), default=0)

        self.scalar_update = ScalarProcessingBlock(self.scalar_mul)
        self.sectors: List[SectorMeta] = []

        if self.high_dim > 0:
            self.sectors = build_sector_metadata(self.high_irreps, self.lmax)
            self._build_dtp_modules()

    def _build_dtp_modules(self):
        """Build DTP1-compatible lightweight modules."""
        sectors = self.sectors
        total_gate_dim = assign_gate_slices(sectors)

        # Gate MLP: reads only edge features
        self.radial_gate_mlp = nn.Sequential(
            nn.Linear(self.scalar_mul, 32, bias=True),
            nn.SiLU(),
            nn.Linear(32, total_gate_dim, bias=True),
        )

        # Per-sector modules
        for sec in sectors:
            # Lifting: scalar -> mul coefficients
            source_proj = nn.Linear(self.scalar_mul, sec.mul, bias=False)
            setattr(self, f"source_proj_{sec.key}", source_proj)

            # Copy mixing: o3.Linear on mul x ir
            sec_irreps = o3.Irreps(f"{sec.mul}x{sec.ir}")
            copy_mixing = o3.Linear(sec_irreps, sec_irreps)
            setattr(self, f"copy_mixing_{sec.key}", copy_mixing)

            # Feedback: mul -> scalar_mul, zero-init
            feedback_linear = nn.Linear(sec.mul, self.scalar_mul, bias=False)
            nn.init.zeros_(feedback_linear.weight)
            setattr(self, f"feedback_linear_{sec.key}", feedback_linear)

        self._init_angular_weights()

    def _init_angular_weights(self):
        """Small non-zero init for angular modules."""
        for sec in self.sectors:
            proj = getattr(self, f"source_proj_{sec.key}")
            nn.init.uniform_(proj.weight, -0.05, 0.05)
            mixing = getattr(self, f"copy_mixing_{sec.key}")
            for p in mixing.parameters():
                if p.numel() > 0:
                    nn.init.uniform_(p, -0.05, 0.05)

        for p in self.radial_gate_mlp.parameters():
            if p.numel() > 0:
                nn.init.uniform_(p, -0.05, 0.05)

    def forward(
        self,
        h: torch.Tensor,
        e: torch.Tensor,
        edge_index: torch.Tensor,
        edge_sh: Optional[torch.Tensor] = None,
        wigner_d_cache: Optional[Dict[Tuple[int, int], torch.Tensor]] = None,
        agg_norm: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        target, source = edge_index[0], edge_index[1]
        h_0e = h[:, :self.scalar_mul]
        h_0e_new = self.scalar_update(e, h_0e, edge_index)

        if self.high_dim == 0:
            return h_0e_new

        return self._forward_high_l(
            h, h_0e, h_0e_new, e, edge_index, target, source,
            edge_sh, wigner_d_cache, agg_norm,
        )

    def _forward_high_l(
        self,
        h: torch.Tensor,
        h_0e: torch.Tensor,
        h_0e_new: torch.Tensor,
        e: torch.Tensor,
        edge_index: torch.Tensor,
        target: torch.Tensor,
        source: torch.Tensor,
        edge_sh: torch.Tensor,
        wigner_d_cache: Optional[Dict],
        agg_norm: Optional[torch.Tensor],
    ) -> torch.Tensor:
        E = target.size(0)
        N = h.size(0)

        if agg_norm is None:
            agg_norm = self.compute_agg_norm(target, N)

        h_high = h[:, self.scalar_mul:]
        h_0e_src = h_0e[source]
        h_high_src = h_high[source]

        # Gates from edge features only
        gates = self.radial_gate_mlp(e)

        lift_msgs = []
        prop_msgs = []
        fb_msgs = []
        src_rotated_list = []

        for sec in self.sectors:
            Y_l = edge_sh[:, sec.sh_slice]

            # Lifting: source_proj(h_0e[source]) * g_lift * Y_l
            proj = getattr(self, f"source_proj_{sec.key}")
            coeff = proj(h_0e_src)  # E, mul
            g_lift = gates[:, sec.gate_slices["lift"]]  # E, mul
            lift = (coeff * g_lift).unsqueeze(-1) * Y_l.unsqueeze(1)  # E, mul, 2l+1
            lift_msgs.append(lift.reshape(E, sec.flat_dim))

            # Transport: Wigner-D
            src = h_high_src[:, sec.state_slice]
            if wigner_d_cache is not None:
                src = self._apply_wigner_d(src, sec, wigner_d_cache)
            src_rotated_list.append(src)
            src_3d = src.view(E, sec.mul, sec.ir_dim)

            # Propagation: copy_mixing -> g_prop
            mixing = getattr(self, f"copy_mixing_{sec.key}")
            mixed = mixing(src)  # E, mul*ir_dim
            mixed_3d = mixed.view(E, sec.mul, sec.ir_dim)
            g_prop = gates[:, sec.gate_slices["prop"]].unsqueeze(-1)  # E, mul, 1
            prop = mixed_3d * g_prop
            prop_msgs.append(prop.reshape(E, sec.flat_dim))

            # Feedback: (src_rotated * Y_l).sum(-1) * g_fb -> feedback_linear
            q = (src_3d * Y_l.unsqueeze(1)).sum(dim=-1)  # E, mul
            g_fb = gates[:, sec.gate_slices["fb"]]  # E, mul
            q = q * g_fb
            fb_linear = getattr(self, f"feedback_linear_{sec.key}")
            fb = fb_linear(q)  # E, scalar_mul
            fb_msgs.append(fb)

        # Aggregate all messages with single scatter
        all_high_msgs = []
        for i, sec in enumerate(self.sectors):
            msg = lift_msgs[i] + prop_msgs[i]
            all_high_msgs.append(msg)

        high_concat = torch.cat(all_high_msgs, dim=-1)  # E, high_dim
        fb_concat = sum(fb_msgs)  # E, scalar_mul (sum contributions)

        # Apply bias-free feedback_linear after scatter (optimization)
        # But for compatibility, we scatter fb_concat directly
        agg_high = self._scatter(high_concat, target, N, agg_norm)
        agg_fb = self._scatter(fb_concat, target, N, agg_norm)

        # Residual update
        h_high_new = []
        offset = 0
        for sec in self.sectors:
            h_sector = h_high[:, sec.state_slice] + agg_high[:, offset:offset + sec.flat_dim]
            h_high_new.append(h_sector)
            offset += sec.flat_dim

        h_0e_new = h_0e_new + agg_fb

        return torch.cat([h_0e_new] + h_high_new, dim=-1)

    @staticmethod
    def compute_agg_norm(target: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """Per-edge 1/sqrt(degree)."""
        degree = torch.zeros(num_nodes, device=target.device)
        degree.scatter_add_(0, target, torch.ones_like(target, dtype=torch.float))
        return degree.rsqrt().clamp(max=1.0)[target].unsqueeze(-1)

    @staticmethod
    def _scatter(
        msg: torch.Tensor, target: torch.Tensor, num_nodes: int,
        agg_norm: torch.Tensor,
    ) -> torch.Tensor:
        out = torch.zeros(num_nodes, msg.size(-1), device=msg.device, dtype=msg.dtype)
        out.index_add_(0, target, msg * agg_norm)
        return out

    @staticmethod
    def _apply_wigner_d(
        h_sector: torch.Tensor,
        sector: SectorMeta,
        wigner_d_cache: Dict[Tuple[int, int], torch.Tensor],
    ) -> torch.Tensor:
        """Transport: h <- D_(l,p) h."""
        E = h_sector.size(0)
        D = wigner_d_cache[(sector.ir.l, sector.ir.p)]
        block_3d = h_sector.reshape(E, sector.mul, sector.ir_dim)
        rotated = torch.einsum("emi,eci->ecm", D, block_3d)
        return rotated.reshape(E, sector.flat_dim)

    def _get_angular_module_names(self) -> set:
        """Angular submodule names for param group separation."""
        if not self.sectors:
            return set()
        names = {"radial_gate_mlp"}
        for sec in self.sectors:
            names.add(f"source_proj_{sec.key}")
            names.add(f"copy_mixing_{sec.key}")
            names.add(f"feedback_linear_{sec.key}")
        return names

    def angular_parameters(self) -> list:
        """Angular parameters for slow-LR param group."""
        if not self.sectors:
            return []
        params = []
        for name in self._get_angular_module_names():
            params.extend(getattr(self, name).parameters())
        return params
