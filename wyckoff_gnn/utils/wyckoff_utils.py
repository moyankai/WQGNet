"""Wyckoff position encoding utilities.

Maps Wyckoff letters, multiplicity, site-symmetry groups, and DOF patterns
to numerical features for the WyckoffGNN node type descriptor.
"""

from functools import lru_cache
from typing import Dict

import torch

# All possible site-symmetry groups (crystallographic point groups compatible
# with space group symmetry), ordered roughly by decreasing symmetry.
SITE_SYMMETRY_GROUPS = [
    "m-3m", "m-3", "-43m", "432", "23",                              # cubic
    "6/mmm", "6/m", "6mm", "622", "6", "-6m2", "-6",                  # hexagonal
    "4/mmm", "4/m", "4mm", "422", "4", "-42m", "-4",                  # tetragonal
    "-3m", "3m", "32", "-3", "3",                                      # trigonal
    "mmm", "mm2", "222",                                                # orthorhombic
    "2/m", "m", "2",                                                    # monoclinic
    "-1", "1",                                                          # triclinic
]

# Map site-symmetry group name to index for embedding
SITE_SYM_TO_IDX = {sg: i for i, sg in enumerate(SITE_SYMMETRY_GROUPS)}

# Number of site-symmetry groups used for embeddings (single source of truth).
NUM_SITE_SYM_GROUPS = len(SITE_SYMMETRY_GROUPS)

# Order |H| of each crystallographic point group. Used by
# ``site_symmetry_hierarchy`` consumers (e.g. symmetry edge features) to
# build dimensionless quantities such as log(|H_p| / |H_q|).
POINT_GROUP_ORDER = {
    "m-3m": 48, "m-3": 24, "-43m": 24, "432": 24, "23": 12,
    "6/mmm": 24, "6/m": 12, "6mm": 12, "622": 12, "6": 6,
    "-6m2": 12, "-6": 6,
    "4/mmm": 16, "4/m": 8, "4mm": 8, "422": 8, "4": 4,
    "-42m": 8, "-4": 4,
    "-3m": 12, "3m": 6, "32": 6, "-3": 6, "3": 3,
    "mmm": 8, "mm2": 4, "222": 4,
    "2/m": 4, "m": 2, "2": 2,
    "-1": 2, "1": 1,
}

# DOF patterns describe which coordinates are free parameters in a given
# Wyckoff position. Each pattern lists which of (x, y, z) are independent.
# Compiled from ITC-A + empirical coverage of mp_20 dataset (54 patterns).
DOF_PATTERNS = [
    # === DOF=3: General position ===
    "(x,y,z)",         # general position, 3 free params
    # === DOF=2: One constraint or one fixed ===
    # z fixed
    "(x,y,0)",         # mirror z=0
    "(x,y,1/2)",       # mirror z=1/2
    # x fixed
    "(0,y,z)",         # mirror x=0
    "(1/2,y,z)",       # mirror x=1/2
    # Diagonal constraints (y tied to x)
    "(x,x,z)",         # y=x (diagonal mirror)
    "(x,-x,z)",        # y=-x (antisymmetric diagonal)
    "(x,2x,z)",        # y=2x (hexagonal .m.)
    # y fixed
    "(x,0,z)",         # y=0
    "(x,1/2,z)",       # y=1/2
    # z tied to x or y
    "(x,y,x)",         # z=x
    "(x,y,-x)",        # z=-x
    "(x,y,y)",         # z=y
    "(x,y,-y)",        # z=-y
    # === DOF=1: Two constraints or two fixed ===
    # Two coordinates fixed, one free (line positions)
    "(x,0,0)",         # y=0, z=0
    "(0,y,0)",         # x=0, z=0
    "(0,0,z)",         # x=0, y=0
    "(x,1/2,0)",       # y=1/2, z=0
    "(x,0,1/2)",       # y=0, z=1/2
    "(x,1/2,1/2)",     # y=1/2, z=1/2
    "(0,y,1/2)",       # x=0, z=1/2
    "(1/2,y,0)",       # x=1/2, z=0
    "(1/2,y,1/2)",     # x=1/2, z=1/2
    "(0,1/2,z)",       # x=0, y=1/2
    "(1/2,0,z)",       # x=1/2, y=0
    "(1/2,1/2,z)",     # x=1/2, y=1/2
    # Diagonal + fixed (DOF=1)
    "(x,x,0)",         # y=x, z=0
    "(x,-x,0)",        # y=-x, z=0
    "(x,x,1/2)",       # y=x, z=1/2
    "(x,-x,1/2)",      # y=-x, z=1/2
    "(x,2x,0)",        # y=2x, z=0 (hexagonal)
    "(x,2x,1/2)",      # y=2x, z=1/2 (hexagonal)
    # Coupled: two of {x,y,z} tied + one fixed or all tied
    "(x,x,x)",         # body diagonal y=x, z=x
    "(x,-x,x)",        # y=-x, z=x
    "(x,-x,-x)",       # y=-x, z=-x
    "(x,x,-x)",        # y=x, z=-x
    "(x,0,x)",         # y=0, z=x
    "(x,0,-x)",        # y=0, z=-x
    "(x,1/2,x)",       # y=1/2, z=x
    "(x,1/2,-x)",      # y=1/2, z=-x
    "(0,y,y)",         # x=0, z=y
    "(0,y,-y)",        # x=0, z=-y
    "(1/2,y,y)",       # x=1/2, z=y
    "(1/2,y,-y)",      # x=1/2, z=-y
    "(x,2x,x)",        # y=2x, z=x (hexagonal + cubic)
    "(x,2x,-x)",       # y=2x, z=-x
    # === DOF=0: All fixed (special points) ===
    "(0,0,0)",
    "(0,0,1/2)",
    "(0,1/2,0)",
    "(1/2,0,0)",
    "(1/2,1/2,0)",
    "(1/2,1/2,1/2)",
    "(0,1/2,1/2)",
    "(1/2,0,1/2)",
]

DOF_TO_IDX = {p: i for i, p in enumerate(DOF_PATTERNS)}

# Aliases for oriented site-symmetry symbols produced by spglib.
# Keys are the dot-stripped symbol, values are canonical Hermann-Mauguin
# short notation present in SITE_SYMMETRY_GROUPS.
_SITE_SYM_ALIASES = {
    "m2m": "mm2",
    "2mm": "mm2",
    "mm":  "mm2",   # partial oriented form (.mm or mm.) -> C2v
    "-4m2": "-42m",
    "-62m": "-6m2",
}


def encode_dof_pattern(dof_str: str) -> int:
    """Return the categorical index of a DOF pattern string."""
    return DOF_TO_IDX.get(dof_str, 0)


def _normalize_site_sym(ss_str: str) -> str:
    """Normalize spglib's oriented site-symmetry symbol to a canonical form.

    Strips whitespace and all directional dots (e.g. ``..2/m`` -> ``2/m``,
    ``m.mm`` -> ``mmm``) and applies a small alias table for permutations
    of equivalent Hermann-Mauguin symbols.
    """
    s = ss_str.strip().replace(" ", "").replace(".", "")
    return _SITE_SYM_ALIASES.get(s, s)


def encode_site_symmetry(ss_str: str) -> int:
    """Return the categorical index of a site-symmetry group string.

    Handles spglib's oriented notation. Unknown symbols fall back to ``"1"``
    (the lowest symmetry) instead of doing fuzzy substring matching, which
    could otherwise alias ``"3"`` to ``"-3m"`` and corrupt the hierarchy.
    """
    s = _normalize_site_sym(ss_str)
    if s in SITE_SYM_TO_IDX:
        return SITE_SYM_TO_IDX[s]
    return SITE_SYM_TO_IDX["1"]


def site_symmetry_hierarchy(ss_a: str, ss_b: str) -> int:
    """Compare two site-symmetry groups.

    Returns:
        ``1``  if H_a is a proper supergroup of H_b (H_a ⊃ H_b),
        ``-1`` if H_b is a proper supergroup of H_a,
        ``0``  if equal or incomparable.
    """
    a = _normalize_site_sym(ss_a)
    b = _normalize_site_sym(ss_b)
    if a == b:
        return 0
    if _is_supergroup(a, b):
        return 1
    if _is_supergroup(b, a):
        return -1
    return 0


# Direct (not transitive) supergroup-to-subgroup edges in the crystallographic
# point-group lattice. Used by _is_supergroup with transitive closure.
_HIERARCHY = {
    # Cubic — Oh (order 48)
    # mmm removed: mmm ⊂ m-3 ⊂ m-3m (non-maximal, reachable via m-3)
    "m-3m": {"m-3", "4/mmm", "-3m", "432", "-43m"},
    "m-3":  {"23", "mmm", "-3"},
    "-43m": {"23", "-42m", "3m"},
    "432":  {"23", "422", "32"},
    "23":   {"222", "3"},
    # Hexagonal — D6h (order 24)
    "6/mmm": {"6/m", "6mm", "622", "-6m2", "-3m", "mmm"},
    "6/m":  {"6", "-6", "2/m", "-3"},
    "6mm":  {"6", "3m", "mm2"},
    "622":  {"6", "32", "222"},
    "-6m2": {"-6", "mm2", "3m"},
    "6":    {"3", "2"},
    "-6":   {"3", "m"},
    # Tetragonal — D4h (order 16)
    "4/mmm": {"4/m", "4mm", "422", "-42m", "mmm"},
    "4/m":  {"4", "-4", "2/m"},
    "4mm":  {"4", "mm2"},
    "422":  {"4", "222"},
    "-42m": {"-4", "mm2", "222"},
    "4":    {"2"},
    "-4":   {"2"},
    # Trigonal — D3d (order 12)
    # 2/m added: C2h is a maximal subgroup of D3d (index 3, Bilbao-verified)
    "-3m":  {"-3", "3m", "32", "2/m"},
    "3m":   {"3", "m"},
    "32":   {"3", "2"},
    "-3":   {"3", "-1"},
    "3":    {"1"},
    # Orthorhombic — D2h (order 8)
    "mmm":  {"mm2", "222", "2/m"},
    "mm2":  {"m", "2"},
    "222":  {"2"},
    # Monoclinic — C2h (order 4)
    "2/m":  {"2", "m", "-1"},
    "m":    {"1"},
    "2":    {"1"},
    # Triclinic — Ci (order 2)
    "-1":   {"1"},
}


@lru_cache(maxsize=None)
def _is_supergroup(high_sym: str, low_sym: str) -> bool:
    """Return True iff ``high_sym`` is a proper supergroup of ``low_sym``
    in the crystallographic point-group hierarchy (transitive closure).
    """
    if high_sym == low_sym:
        return False
    direct = _HIERARCHY.get(high_sym, set())
    if low_sym in direct:
        return True
    for intermediate in direct:
        if _is_supergroup(intermediate, low_sym):
            return True
    return False


def get_degree_of_freedom(dof_str: str) -> int:
    """Count the number of free parameters in a DOF pattern.
    Examples: (x,y,z)=3, (x,x,0)=1, (0,0,0)=0.
    """
    # Count distinct free variables (x, y, z)
    free_vars = set()
    for c in dof_str:
        if c in ("x", "y", "z"):
            free_vars.add(c)
    return len(free_vars)


def get_wyckoff_letter_index(letter: str) -> int:
    """Map Wyckoff letter to 0-indexed integer.

    ITC-A convention: a=0, b=1, ..., z=25, A=26.
    SG #47 (Pmmm) has 27 WP and uses uppercase A as the 27th letter.
    """
    if len(letter) == 1 and letter.isalpha():
        if letter.islower():
            return ord(letter) - ord("a")  # a→0, ..., z→25
        else:
            return 26  # A→26 (27th position)
    return 0


def get_point_group_order(ss_str: str) -> int:
    """Return the order |H| of a (possibly oriented) site-symmetry group.

    Unknown symbols fall back to 1 (identity), matching ``encode_site_symmetry``.
    """
    s = _normalize_site_sym(ss_str)
    return POINT_GROUP_ORDER.get(s, 1)


# ---------------------------------------------------------------------------
# Space-group-level metadata derived from ITC-A.
# Used by build_wyckoff_templates.py to populate template node features.
# ---------------------------------------------------------------------------

# Crystal system per SG number: 0=triclinic, 1=monoclinic, 2=orthorhombic,
# 3=tetragonal, 4=trigonal, 5=hexagonal, 6=cubic.
_SG_CS_RANGES = [
    (1,   2,   0),  # triclinic
    (3,   15,  1),  # monoclinic
    (16,  74,  2),  # orthorhombic
    (75,  142, 3),  # tetragonal
    (143, 167, 4),  # trigonal
    (168, 194, 5),  # hexagonal
    (195, 230, 6),  # cubic
]
SG_CRYSTAL_SYSTEM: Dict[int, int] = {}
for _lo, _hi, _cs in _SG_CS_RANGES:
    for _sg in range(_lo, _hi + 1):
        SG_CRYSTAL_SYSTEM[_sg] = _cs

NUM_CRYSTAL_SYSTEMS: int = 7

# ITC-A standard: geometric point group (crystal class) of each SG.
_SG_PG_RANGES = [
    (1,   1,   "1"),    (2,   2,   "-1"),
    (3,   5,   "2"),    (6,   9,   "m"),     (10,  15,  "2/m"),
    (16,  24,  "222"),  (25,  46,  "mm2"),   (47,  74,  "mmm"),
    (75,  80,  "4"),    (81,  82,  "-4"),    (83,  88,  "4/m"),
    (89,  98,  "422"),  (99,  110, "4mm"),   (111, 122, "-42m"),
    (123, 142, "4/mmm"),
    (143, 146, "3"),    (147, 148, "-3"),
    (149, 155, "32"),   (156, 161, "3m"),    (162, 167, "-3m"),
    (168, 173, "6"),    (174, 174, "-6"),    (175, 176, "6/m"),
    (177, 182, "622"),  (183, 186, "6mm"),   (187, 190, "-6m2"),
    (191, 194, "6/mmm"),
    (195, 199, "23"),   (200, 206, "m-3"),   (207, 214, "432"),
    (215, 220, "-43m"), (221, 230, "m-3m"),
]
SG_POINT_GROUP: Dict[int, str] = {}
for _lo, _hi, _pg in _SG_PG_RANGES:
    for _sg in range(_lo, _hi + 1):
        SG_POINT_GROUP[_sg] = _pg

# Bravais centering letter → embedding index.
# A/B/C are all "base-centered" (same class).
CENTERING_TO_IDX: Dict[str, int] = {
    "P": 0,  # primitive
    "I": 1,  # body-centered
    "F": 2,  # face-centered
    "A": 3, "B": 3, "C": 3,  # base-centered
    "R": 4,  # rhombohedral
}
NUM_CENTERING_TYPES: int = 5  # P / I / F / base / R


def get_sg_crystal_system(sg: int) -> int:
    """Return crystal system index (0..6) for a space group number."""
    return SG_CRYSTAL_SYSTEM.get(sg, 0)


def get_sg_point_group_idx(sg: int) -> int:
    """Return SITE_SYM_TO_IDX of the SG's geometric point group (0..31)."""
    pg = SG_POINT_GROUP.get(sg, "1")
    return SITE_SYM_TO_IDX.get(pg, SITE_SYM_TO_IDX["1"])
