"""Five canonical crystal structures for synthetic correctness tests.

All structures are exact (no noise), built from published lattice parameters,
and chosen to exercise different symmetry regimes:
  - P1 (trivial symmetry)
  - High-symmetry symmorphic (cubic perovskite, NaCl)
  - Non-symmorphic (diamond Si, rutile TiO2)
"""

from __future__ import annotations

from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure


def make_p1_si2o2() -> Structure:
    """P1 distorted Si2O2 (SG 1). Trivial symmetry, all mult=1."""
    lat = Lattice.from_parameters(4.0, 4.2, 3.9, 89.5, 90.5, 89.0)
    return Structure(
        lat,
        ["Si", "Si", "O", "O"],
        [
            [0.01, 0.02, 0.03],
            [0.48, 0.51, 0.49],
            [0.25, 0.25, 0.25],
            [0.75, 0.75, 0.75],
        ],
    )


def make_perovskite_cubic() -> Structure:
    """Ideal cubic perovskite CaTiO3 (Pm-3m, SG 221).

    5 atoms -> 3 orbits:
      Ca 1a (0,0,0)       mult=1, |H|=48
      Ti 1b (1/2,1/2,1/2) mult=1, |H|=48
      O  3c (0,1/2,1/2)   mult=3, |H|=8  (mmm site symmetry)
    """
    lat = Lattice.cubic(3.9)
    return Structure(
        lat,
        ["Ca", "Ti", "O", "O", "O"],
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5],
        ],
    )


def make_nacl() -> Structure:
    """NaCl (Fm-3m, SG 225). Multiplicity > 1."""
    lat = Lattice.cubic(5.64)
    return Structure(
        lat,
        ["Na", "Cl"],
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )


def make_diamond_si() -> Structure:
    """Diamond Si (Fd-3m, SG 227). Non-symmorphic with glide planes.

    Wyckoff 8a: (0,0,0) and (1/4,1/4,1/4) + FCC centering.
    8 atoms in conventional cell, 1 orbit, mult=8.
    """
    lat = Lattice.cubic(5.43)
    return Structure(
        lat,
        ["Si"] * 8,
        [
            [0.0, 0.0, 0.0],
            [0.25, 0.25, 0.25],
            [0.5, 0.5, 0.0],
            [0.75, 0.75, 0.25],
            [0.5, 0.0, 0.5],
            [0.75, 0.25, 0.75],
            [0.0, 0.5, 0.5],
            [0.25, 0.75, 0.75],
        ],
    )


def make_rutile_tio2() -> Structure:
    """Rutile TiO2 (P4_2/mnm, SG 136). Non-symmorphic screw axis.

    Ti at 2a (0,0,0), O at 4f (x,x,0) with x~0.305.
    6 atoms in conventional cell.
    """
    lat = Lattice.from_parameters(4.594, 4.594, 2.959, 90, 90, 90)
    x_o = 0.3053
    return Structure(
        lat,
        ["Ti", "Ti", "O", "O", "O", "O"],
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            [x_o, x_o, 0.0],
            [1 - x_o, 1 - x_o, 0.0],
            [0.5 + x_o, 0.5 - x_o, 0.5],
            [0.5 - x_o, 0.5 + x_o, 0.5],
        ],
    )


CANONICAL_STRUCTURES = {
    "P1_Si2O2": make_p1_si2o2,
    "CaTiO3": make_perovskite_cubic,
    "NaCl": make_nacl,
    "Si_diamond": make_diamond_si,
    "TiO2_rutile": make_rutile_tio2,
}

EXPECTED_SG = {
    "P1_Si2O2": 1,
    "CaTiO3": 221,
    "NaCl": 225,
    "Si_diamond": 227,
    "TiO2_rutile": 136,
}
