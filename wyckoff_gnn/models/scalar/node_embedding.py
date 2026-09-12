"""Shared node-embedding blocks (element embedding + kgcnn features).

This model tests whether the Wyckoff quotient graph retains all predictive
information needed for scalar property prediction when using a pure scalar
(0e) message-passing operator.

Architecture
------------
- Node input: orbit_element (Z) → Embedding(119, 128) + atom_props(4) + ox_states(14) = 146
  → Linear(146, 128) no activation
- Edge input: geo_edge_distance → GaussBasisExpansion(32, 0, 8) → Linear(32, 128) no activation
- 5 × ScalarProcessingBlock:
    edge_mlp: [e, h[target], h[src]] (384) → Linear+SiLU → 128 → (4× Linear+SiLU → 128)
    node_mlp: agg(128) → Linear+SiLU → 128
    h = h + node_mlp(agg)   # residual
    e unchanged              # reuse initial edge state
- Readout: multiplicity-weighted mean → Linear(128, 1)

Key design decisions
--------------------
- Graph: Wyckoff quotient (same as all other WyckoffGNN models)
- Edge features: geo_edge_distance → RBF (true sub-edge distance, NOT recomputed)
- Multiplicity: Used in readout (weighted mean), NOT in message passing
- No tensor products, no Wigner-D, no spherical harmonics
- edge_index: Uses data.geo_edge_index (target=row0, source=row1)
- Forward signature: forward(self, data) — same as other WyckoffGNN models

Parameter budget (matches coGN exactly)
---------------------------------------
- Embedding: 15,232
- Input node proj: 18,816
- Input edge proj: 4,224
- 5 blocks × 131,840: 659,200
- Readout: 129
- Total: 697,601
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from wyckoff_gnn.models.scalar.scalar_block import GaussBasisExpansion, ScalarProcessingBlock


def _glorot_uniform_(tensor: torch.Tensor, fan_in: int, fan_out: int):
    """Initialize tensor like Keras GlorotUniform."""
    limit = math.sqrt(6.0 / (fan_in + fan_out))
    nn.init.uniform_(tensor, -limit, limit)


def init_like_keras(module: nn.Module):
    """Initialize a PyTorch module to match Keras 2.12 defaults."""
    if isinstance(module, nn.Linear):
        _glorot_uniform_(module.weight, module.in_features, module.out_features)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.uniform_(module.weight, -0.05, 0.05)


def _load_kgcnn_features():
    """Load kgcnn pre-normalized atom properties and oxidation states.

    Returns:
        atom_props: (119, 4) tensor, dummy row at index 0
        ox_states: (119, 14) tensor, dummy row at index 0
    """
    import os
    import numpy as np

    _REF_DIR = os.path.join(
        os.path.dirname(__file__), "..", "..", "..",
        "artifacts", "reference"
    )
    path = os.path.join(_REF_DIR, "kgcnn_atom_features_v3_0_1.npz")

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"kgcnn_atom_features_v3_0_1.npz not found at {path}. "
            "This file is required for WyckoffCoGN0e node features."
        )

    data = np.load(path)
    atom_props = data["atom_props"]
    ox_states = data["oxidation_states"]

    dummy_props = np.zeros((1, 4), dtype=np.float32)
    dummy_ox = np.zeros((1, 14), dtype=np.float32)
    atom_props_119 = np.concatenate([dummy_props, atom_props], axis=0).astype(np.float32)
    ox_states_119 = np.concatenate([dummy_ox, ox_states], axis=0).astype(np.float32)

    return (
        torch.tensor(atom_props_119, dtype=torch.float32),
        torch.tensor(ox_states_119, dtype=torch.float32),
    )


class AtomEmbedding(nn.Module):
    """Node embedding: concat(Embedding(128), 4 props, 14 ox) → 146-dim.

    Uses kgcnn pre-normalized atom properties (z-score with pandas ddof=1).
    Table has 119 rows: dummy at 0, H at 1, ..., Og at 118.
    """

    def __init__(self, node_size: int = 128):
        super().__init__()
        self.embedding = nn.Embedding(119, node_size)
        atom_props, ox_states = _load_kgcnn_features()
        self.register_buffer("atom_props", atom_props, persistent=True)
        self.register_buffer("oxidation_states", ox_states, persistent=True)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        zc = z.clamp(0, 118).long()
        h = self.embedding(zc)
        props = self.atom_props[zc]
        ox = self.oxidation_states[zc]
        return torch.cat([h, props, ox], dim=-1)
