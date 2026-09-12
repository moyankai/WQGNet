"""Core equivariant building blocks used by the WyckoffEdgeTP model."""

import torch
import torch.nn as nn
from e3nn import o3

# ---------------------------------------------------------------------------
# Periodic-table static physical properties (scalar, rotation-invariant).
# All arrays are indexed by atomic number Z from 0 to 118 (index 0 = padding).
# Length is 119 for all four tables.
#
# Sources:
#   mass                : pymatgen.Element.atomic_mass  (NIST 2018 standard atomic weights)
#   covalent radius (Å) : Cordero et al. 2008, DOI 10.1039/B801115J  (same as coGN/kgcnn/MEGNet)
#   electronegativity   : pymatgen.Element.X            (Pauling scale)
#   ionization_energy   : pymatgen.Element.ionization_energy (first IE, eV)
#
# Missing values (noble gases have no Pauling EN; superheavy elements have no
# measured radius/IE) are filled with the column mean over Z=1..118.
# _PROPS_MEAN/_PROPS_STD are the z-score stats of these tables (Z=1..118).
# ---------------------------------------------------------------------------

_ATOMIC_MASS = torch.tensor([
    0.0000, 1.0079, 4.0026, 6.9410, 9.0122, 10.8110, 12.0107, 14.0067, 15.9994, 18.9984, 20.1797, 22.9898,
    24.3050, 26.9815, 28.0855, 30.9738, 32.0650, 35.4530, 39.9480, 39.0983, 40.0780, 44.9559, 47.8670, 50.9415,
    51.9961, 54.9380, 55.8450, 58.9332, 58.6934, 63.5460, 65.4090, 69.7230, 72.6400, 74.9216, 78.9600, 79.9040,
    83.7980, 85.4678, 87.6200, 88.9059, 91.2240, 92.9064, 95.9400, 98.0000, 101.0700, 102.9055, 106.4200, 107.8682,
    112.4110, 114.8180, 118.7100, 121.7600, 127.6000, 126.9045, 131.2930, 132.9055, 137.3270, 138.9055, 140.1160, 140.9077,
    144.2420, 145.0000, 150.3600, 151.9640, 157.2500, 158.9254, 162.5000, 164.9303, 167.2590, 168.9342, 173.0400, 174.9670,
    178.4900, 180.9479, 183.8400, 186.2070, 190.2300, 192.2170, 195.0840, 196.9666, 200.5900, 204.3833, 207.2000, 208.9804,
    210.0000, 210.0000, 220.0000, 223.0000, 226.0000, 227.0000, 232.0381, 231.0359, 238.0289, 237.0000, 244.0000, 243.0000,
    247.0000, 247.0000, 251.0000, 252.0000, 257.0000, 258.0000, 259.0000, 262.0000, 267.0000, 268.0000, 269.0000, 270.0000,
    270.0000, 278.0000, 281.0000, 282.0000, 285.0000, 286.0000, 289.0000, 290.0000, 293.0000, 294.0000, 294.0000,
], dtype=torch.float32)   # length = 119

_ATOMIC_RADIUS = torch.tensor([
    0.0000, 0.3100, 0.2800, 1.2800, 0.9600, 0.8400, 0.7600, 0.7100, 0.6600, 0.5700, 0.5800, 1.6600,
    1.4100, 1.2100, 1.1100, 1.0700, 1.0500, 1.0200, 1.0600, 2.0300, 1.7600, 1.7000, 1.6000, 1.5300,
    1.3900, 1.3900, 1.3200, 1.2600, 1.2400, 1.3200, 1.2200, 1.2200, 1.2000, 1.1900, 1.2000, 1.2000,
    1.1600, 2.2000, 1.9500, 1.9000, 1.7500, 1.6400, 1.5400, 1.4700, 1.4600, 1.4200, 1.3900, 1.4500,
    1.4400, 1.4200, 1.3900, 1.3900, 1.3800, 1.3900, 1.4000, 2.4400, 2.1500, 2.0700, 2.0400, 2.0300,
    2.0100, 1.9900, 1.9800, 1.9800, 1.9600, 1.9400, 1.9200, 1.9200, 1.8900, 1.9000, 1.8700, 1.8700,
    1.7500, 1.7000, 1.6200, 1.5100, 1.4400, 1.4100, 1.3600, 1.3600, 1.3200, 1.4500, 1.4600, 1.4800,
    1.4000, 1.5000, 1.5000, 2.6000, 2.2100, 2.1500, 2.0600, 2.0000, 1.9600, 1.9000, 1.8700, 1.8000,
    1.6900, 1.5199, 1.5199, 1.5199, 1.5199, 1.5199, 1.5199, 1.5199, 1.5199, 1.5199, 1.5199, 1.5199,
    1.5199, 1.5199, 1.5199, 1.5199, 1.5199, 1.5199, 1.5199, 1.5199, 1.5199, 1.5199, 1.5199,
], dtype=torch.float32)   # length = 119

_ELECTRONEGATIVITY = torch.tensor([
    0.0000, 2.2000, 1.7130, 0.9800, 1.5700, 2.0400, 2.5500, 3.0400, 3.4400, 3.9800, 1.7130, 0.9300,
    1.3100, 1.6100, 1.9000, 2.1900, 2.5800, 3.1600, 1.7130, 0.8200, 1.0000, 1.3600, 1.5400, 1.6300,
    1.6600, 1.5500, 1.8300, 1.8800, 1.9100, 1.9000, 1.6500, 1.8100, 2.0100, 2.1800, 2.5500, 2.9600,
    3.0000, 0.8200, 0.9500, 1.2200, 1.3300, 1.6000, 2.1600, 1.9000, 2.2000, 2.2800, 2.2000, 1.9300,
    1.6900, 1.7800, 1.9600, 2.0500, 2.1000, 2.6600, 2.6000, 0.7900, 0.8900, 1.1000, 1.1200, 1.1300,
    1.1400, 1.1300, 1.1700, 1.2000, 1.2000, 1.1000, 1.2200, 1.2300, 1.2400, 1.2500, 1.1000, 1.2700,
    1.3000, 1.5000, 2.3600, 1.9000, 2.2000, 2.2000, 2.2800, 2.5400, 2.0000, 1.6200, 2.3300, 2.0200,
    2.0000, 2.2000, 2.2000, 0.7000, 0.9000, 1.1000, 1.3000, 1.5000, 1.3800, 1.3600, 1.2800, 1.3000,
    1.3000, 1.3000, 1.3000, 1.3000, 1.3000, 1.3000, 1.3000, 1.3000, 1.7130, 1.7130, 1.7130, 1.7130,
    1.7130, 1.7130, 1.7130, 1.7130, 1.7130, 1.7130, 1.7130, 1.7130, 1.7130, 1.7130, 1.7130,
], dtype=torch.float32)   # length = 119

_IONIZATION_ENERGY = torch.tensor([
    0.0000, 13.5984, 24.5874, 5.3917, 9.3227, 8.2980, 11.2603, 14.5341, 13.6181, 17.4228, 21.5645, 5.1391,
    7.6462, 5.9858, 8.1517, 10.4867, 10.3600, 12.9676, 15.7596, 4.3407, 6.1132, 6.5615, 6.8281, 6.7462,
    6.7665, 7.4340, 7.9025, 7.8810, 7.6399, 7.7264, 9.3942, 5.9993, 7.8994, 9.7886, 9.7524, 11.8138,
    13.9996, 4.1771, 5.6949, 6.2173, 6.6341, 6.7589, 7.0924, 7.1194, 7.3605, 7.4589, 8.3368, 7.5762,
    8.9938, 5.7864, 7.3439, 8.6084, 9.0098, 10.4513, 12.1298, 3.8939, 5.2117, 5.5769, 5.5386, 5.4702,
    5.5250, 5.5819, 5.6437, 5.6704, 6.1498, 5.8638, 5.9391, 6.0215, 6.1077, 6.1843, 6.2542, 5.4259,
    6.8251, 7.5496, 7.8640, 7.8335, 8.4382, 8.9670, 8.9588, 9.2256, 10.4375, 6.1083, 7.4167, 7.2855,
    8.4181, 9.3175, 10.7485, 4.0727, 5.2784, 5.3802, 6.3067, 5.8900, 6.1940, 6.2655, 6.0258, 5.9738,
    5.9914, 6.1979, 6.2817, 6.3676, 6.5000, 6.5800, 6.6262, 4.9600, 6.0200, 6.8000, 7.8000, 7.7000,
    7.6000, 50.0000, 65.0000, 8.8245, 8.8245, 8.8245, 8.8245, 8.8245, 8.8245, 8.8245, 8.8245,
], dtype=torch.float32)   # length = 119

_N_ATOM_PROPERTIES = 4   # mass, radius, electronegativity, ionization_energy

# z-score normalisation stats, computed over Z=1..118 of the tables above.
_PROPS_MEAN = torch.tensor([146.4628, 1.5199, 1.713, 8.8245], dtype=torch.float32)
_PROPS_STD  = torch.tensor([89.2593, 0.3976, 0.579, 7.1755], dtype=torch.float32)


def _gather_atom_props(atomic_numbers: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Gather z-score normalised physical properties (mass/radius/EN/IE).

    Returns:
        (N, 4) tensor: [mass, radius, electronegativity, ionization_energy],
        each column z-score normalised with _PROPS_MEAN / _PROPS_STD.
    """
    z = atomic_numbers.clamp(0, 118).long()
    mass_t   = _ATOMIC_MASS.to(device)[z]
    radius_t = _ATOMIC_RADIUS.to(device)[z]
    en_t     = _ELECTRONEGATIVITY.to(device)[z]
    ie_t     = _IONIZATION_ENERGY.to(device)[z]
    raw = torch.stack([mass_t, radius_t, en_t, ie_t], dim=-1)
    mean = _PROPS_MEAN.to(device).unsqueeze(0)
    std  = _PROPS_STD.to(device).unsqueeze(0)
    return (raw - mean) / std.clamp(min=1e-6)


class IrrepNorm(nn.Module):
    """Per-irrep-block normalisation that preserves SE(3) equivariance.

    - Scalars (l=0): LayerNorm with learnable shift + scale.
    - High-l (l>0): RMS normalisation (no mean subtraction) + learnable scale.
    """

    def __init__(self, irreps: o3.Irreps, eps: float = 1e-5):
        super().__init__()
        self.irreps = irreps
        self.eps = eps

        n_blocks = len(list(irreps))
        self.scale = nn.Parameter(torch.ones(n_blocks, dtype=torch.float32))

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
                block = block.reshape(-1, mul, ir.dim)
                rms = (block.pow(2).mean(dim=-1, keepdim=True) + self.eps).sqrt()
                block = block / rms.clamp(min=self.eps)
                block = block.reshape(-1, mul * ir.dim)
            blocks.append(block * self.scale[i])
            offset += dim
        return torch.cat(blocks, dim=-1)


class ElementOnlyNodeEncoder(nn.Module):
    """Element-only node initialiser: Z -> pure-scalar irreps.

    Optionally appends periodic-table physical properties (mass, radius,
    electronegativity, ionisation energy) to the embedding.

    The output irreps must be pure scalars (Nx0e).
    """

    def __init__(self, init_irreps, max_atomic_number: int = 118,
                 use_atom_props: bool = False):
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
            self.prop_proj = nn.Linear(init_irreps.dim + _N_ATOM_PROPERTIES,
                                       init_irreps.dim)

    def forward(self, atomic_numbers: torch.Tensor) -> torch.Tensor:
        z = atomic_numbers.clamp(0, self.max_atomic_number)
        h = self.embedding(z)
        if self.use_atom_props:
            props = _gather_atom_props(atomic_numbers, h.device)
            h = self.prop_proj(torch.cat([h, props], dim=-1))
        return h


__all__ = ["IrrepNorm", "ElementOnlyNodeEncoder"]
