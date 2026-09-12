"""Unified quotient-equivariant model package."""

from .model import UnifiedQuotientEquivariantGNN
from .scalar_sector import InputEmbedding, EdgeEmbedding
from .equivariant_sector import (
    UnifiedQuotientEquivariantBlock,
    LightweightScalarToHighL,
    CachedSourceImageTransport,
    SameLHighLPropagation,
    CopyWiseIrrepInvariants,
    ZeroInitHighLToScalar,
    compute_source_rotations_with_batch,
    build_wigner_d_cache,
    compute_edge_sh,
    precompute_batch_geometry,
)
from .dynamic_tp_block import UnifiedDynamicTPBlock
from .tensor_adapter import SymmetricRank2Adapter
from .tensor_readout import SymmetricRank2Readout

__all__ = [
    "UnifiedQuotientEquivariantGNN",
    "InputEmbedding",
    "EdgeEmbedding",
    "UnifiedQuotientEquivariantBlock",
    "UnifiedDynamicTPBlock",
    "LightweightScalarToHighL",
    "CachedSourceImageTransport",
    "SameLHighLPropagation",
    "CopyWiseIrrepInvariants",
    "ZeroInitHighLToScalar",
    "compute_source_rotations_with_batch",
    "build_wigner_d_cache",
    "compute_edge_sh",
    "precompute_batch_geometry",
    "SymmetricRank2Adapter",
    "SymmetricRank2Readout",
]
