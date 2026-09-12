"""WyckoffGNN equivariant architecture — production model.

This subpackage contains the ``wyckoff_gnn`` model that produced the best
JARVIS bandgap ``test_mae=0.158`` with ``512x0e+16x1o`` hidden irreps.
"""
from wyckoff_gnn.models.wyckoff_equiv.model import WyckoffGNN
from wyckoff_gnn.models.wyckoff_equiv.e3nn_encoder import EquivariantWyckoffGNNEncoder
from wyckoff_gnn.models.wyckoff_equiv.e3nn_layers import (
    ElementOnlyNodeEncoder,
    GaussianRBF,
    EquivariantWyckoffLayer,
    GeometricLiftingLayer,
    EquivariantReadout,
    IrrepNorm,
)

__all__ = [
    "WyckoffGNN",
    "EquivariantWyckoffGNNEncoder",
    "ElementOnlyNodeEncoder",
    "GaussianRBF",
    "EquivariantWyckoffLayer",
    "GeometricLiftingLayer",
    "EquivariantReadout",
    "IrrepNorm",
]
