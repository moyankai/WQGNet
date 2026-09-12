"""Symmetry operations and site-symmetry group utilities.

Handles space group operations, site-symmetry group determination,
and symmetry-constrained coordinate generation.
"""

from typing import List, Tuple, Optional
import numpy as np
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.core.structure import Structure
import spglib

_SG_TO_HALL: Optional[dict] = None
_SG_OPS_CACHE: dict = {}


def _build_sg_to_hall() -> dict:
    """Build a map from international SG number (1..230) to the default Hall
    number (the first Hall entry encountered for that SG).

    spglib's `get_symmetry_from_database(hall_number)` takes a Hall number
    (1..530), NOT an international SG number. Passing the SG number silently
    returns the operations of a different setting (e.g. Hall 221 -> Pnn2,
    while the user wanted SG 221 = Pm-3m with Hall 517).
    """
    mapping = {}
    for h in range(1, 531):
        try:
            st = spglib.get_spacegroup_type(h)
        except Exception:
            continue
        if st is None:
            continue
        # spglib 2.x may return an object with .number or a dict
        try:
            n = st.number  # attribute API
        except AttributeError:
            n = st["number"]
        if n not in mapping:
            mapping[n] = h
    return mapping


def sg_number_to_hall(sg_number: int) -> int:
    """Return the default Hall number for the given international SG number."""
    global _SG_TO_HALL
    if _SG_TO_HALL is None:
        _SG_TO_HALL = _build_sg_to_hall()
    if sg_number not in _SG_TO_HALL:
        raise ValueError(f"Unknown space group number: {sg_number}")
    return _SG_TO_HALL[sg_number]


def get_space_group_operations(sg_number: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (rotations, translations) for a given international SG number.

    Args:
        sg_number: International space group number (1-230).

    Returns:
        rotations:    (n_ops, 3, 3) integer rotation matrices (float dtype).
        translations: (n_ops, 3) fractional translations.

    Uses spglib's Hall-number database with the default Hall choice for each
    international SG. Result is cached per SG number.
    """
    if sg_number in _SG_OPS_CACHE:
        return _SG_OPS_CACHE[sg_number]
    hall = sg_number_to_hall(sg_number)
    dataset = spglib.get_symmetry_from_database(hall)
    if dataset is None:
        raise RuntimeError(
            f"spglib returned no dataset for SG {sg_number} (Hall {hall})"
        )
    rotations = np.array(dataset["rotations"], dtype=np.float32)
    translations = np.array(dataset["translations"], dtype=np.float32)
    _SG_OPS_CACHE[sg_number] = (rotations, translations)
    return rotations, translations


def apply_symmetry_operations(
    coords: np.ndarray, rotations: np.ndarray, translations: np.ndarray
) -> np.ndarray:
    """Apply space group operations to a set of coordinates.

    Args:
        coords: (N, 3) or (3,) fractional coordinates.
        rotations: (n_ops, 3, 3) rotation matrices.
        translations: (n_ops, 3) translations.

    Returns:
        (n_ops, N, 3) generated coordinates.
    """
    if coords.ndim == 1:
        coords = coords[None, :]  # (1, 3)
    # coords: (N, 3), rotations: (n_ops, 3, 3), translations: (n_ops, 3)
    # result: (n_ops, N, 3)
    result = np.einsum("nij,kj->nki", rotations, coords) + translations[:, None, :]
    # Modulo 1 to wrap into unit cell
    result = result % 1.0
    return result


def get_unique_positions(coords: np.ndarray, tol: float = 1e-3) -> np.ndarray:
    """Remove duplicate positions (mod 1) from a set of fractional coordinates.

    Args:
        coords: (M, N, 3) or (M, 3) array of fractional coordinates.
        tol: tolerance for position comparison.

    Returns:
        Unique positions array.
    """
    if coords.ndim == 2:
        coords = coords.reshape(-1, 3)
    else:
        coords = coords.reshape(-1, coords.shape[-1])

    unique = []
    for pos in coords:
        pos_mod = pos % 1.0
        if not any(np.allclose(pos_mod, u, atol=tol) for u in unique):
            unique.append(pos_mod)
    return np.array(unique)


def get_site_symmetry_group(
    structure: Structure, wyckoff_letter: str
) -> Optional[str]:
    """Get the site-symmetry group for a given Wyckoff position in a structure.

    Uses pymatgen's SpacegroupAnalyzer to determine the Wyckoff positions and
    extract the site-symmetry group.

    Args:
        structure: pymatgen Structure object.
        wyckoff_letter: Wyckoff letter (e.g., 'a', 'b', 'c').

    Returns:
        Site-symmetry group string (e.g., 'mm2', '4/m') or None.
    """
    try:
        sga = SpacegroupAnalyzer(structure)
        sym_struct = sga.get_symmetrized_structure()
        wyckoff_positions = sym_struct.wyckoff_positions

        for wp in wyckoff_positions:
            if wp.letter == wyckoff_letter:
                return wp.site_symmetry
        return None
    except Exception:
        return None


def get_wyckoff_sites_from_structure(
    structure: Structure,
) -> List[dict]:
    """Extract occupied Wyckoff sites from a pymatgen Structure.

    Args:
        structure: pymatgen Structure.

    Returns:
        List of dicts, each containing:
          - wyckoff_letter: str
          - multiplicity: int
          - site_symmetry: str
          - element: str
          - occupancy: float
          - representative_coord: (3,) np.ndarray (fractional)
          - all_positions: (multiplicity, 3) np.ndarray (fractional)
    """
    sga = SpacegroupAnalyzer(structure)
    sym_struct = sga.get_symmetrized_structure()
    sg_number = sga.get_space_group_number()

    sites = []
    for wp in sym_struct.wyckoff_positions:
        letter = wp.letter
        multiplicity = wp.multiplicity
        site_sym = wp.site_symmetry if hasattr(wp, "site_symmetry") else "1"

        # Use the first specie of the site composition
        first_specie = wp.species.elements[0]
        element = str(first_specie.symbol)
        # Real occupancy fraction from the Composition (defaults to 1.0)
        try:
            occupancy = float(wp.species.get(first_specie, 1.0))
        except Exception:
            occupancy = 1.0

        # Get all equivalent positions and the representative
        all_frac = np.array([s.frac_coords for s in wp])
        rep_coord = all_frac[0]

        sites.append({
            "wyckoff_letter": letter,
            "multiplicity": multiplicity,
            "site_symmetry": site_sym,
            "element": element,
            "occupancy": occupancy,
            "representative_coord": rep_coord.astype(np.float32),
            "all_positions": all_frac.astype(np.float32),
            "space_group": sg_number,
        })

    return sites
