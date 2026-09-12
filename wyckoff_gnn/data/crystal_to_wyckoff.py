"""Convert crystal structures to Wyckoff orbit representation.

Extracts occupied Wyckoff orbit instances from a pymatgen Structure using
spglib for symmetry analysis.  Each occupied orbit becomes one node in the
instance graph G_I.

Key improvements over the legacy path:
- Per-orbit symmetry operations (rotation + translation) for equivalent atoms.
- atom_to_orbit mapping for multiplicity-weighted readout.
- SG-local letter indexing (letter_in_sg_idx) — not shared across space groups.
- Standardised conventional cell from spglib for correct Wyckoff assignment.
- P1 fallback: when symmetry detection fails, every atom is its own orbit.
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

import spglib
from pymatgen.core.structure import Structure

from wyckoff_gnn.utils.wyckoff_utils import (
    encode_site_symmetry,
    encode_dof_pattern,
    get_wyckoff_letter_index,
    get_degree_of_freedom,
    SITE_SYM_TO_IDX,
)


# ---------------------------------------------------------------------------
# WyckoffOrbit — a node in the instance graph
# ---------------------------------------------------------------------------


class WyckoffOrbit:
    """An occupied Wyckoff orbit — one node in the Wyckoff instance graph.

    Represents ONE set of symmetry-equivalent atoms in the (conventional) unit
    cell.  All equivalent positions are generated from a single representative
    coordinate via the space-group operations.

    Attributes
    ----------
    space_group : int
        International space group number (1–230).
    wyckoff_letter : str
        Wyckoff letter ('a'–'z', 'A') as returned by spglib.
    wyckoff_symbol : str
        Human-readable label, e.g. ``"4a"``.
    multiplicity : int
        Number of equivalent atoms in the conventional cell (m_p).
    site_symmetry : str
        Oriented site-symmetry group label from spglib.
    site_symmetry_idx : int
        Categorical index into ``SITE_SYMMETRY_GROUPS`` (0–31).
    dof_pattern : str
        Coordinate degree-of-freedom pattern, e.g. ``"(x,y,z)"``, ``"(x,x,0)"``.
    num_free_params : int
        Number of independent DOF (0–3).
    element : str
        Chemical element symbol.
    atomic_number : int
        Proton number Z.
    occupancy : float
        Site occupancy fraction (1.0 for fully occupied).
    representative_coord : np.ndarray
        (3,) fractional coordinate of the representative atom in the
        **standardised conventional cell**.
    all_positions : np.ndarray
        (multiplicity, 3) fractional coordinates of ALL equivalent atoms.
    sym_ops_rotations : np.ndarray
        (multiplicity, 3, 3) integer rotation matrices that generate each
        equivalent position from the representative.
    sym_ops_translations : np.ndarray
        (multiplicity, 3) fractional translation vectors.
    atom_indices : List[int]
        Original atom indices in the input structure that belong to this orbit.
    letter_in_sg_idx : int
        SG-local Wyckoff letter index (0-based within this space group).
        For the global (sg,letter) token see ``letter_in_sg_token``.
    letter_in_sg_token : int
        Global (sg, letter) token id assigned at preprocessing time;
        -1 means "not yet attached".
    """

    __slots__ = (
        "space_group",
        "wyckoff_letter",
        "wyckoff_symbol",
        "multiplicity",
        "site_symmetry",
        "site_symmetry_idx",
        "dof_pattern",
        "num_free_params",
        "element",
        "atomic_number",
        "occupancy",
        "representative_coord",
        "all_positions",
        "sym_ops_rotations",
        "sym_ops_translations",
        "stabilizer_W_frac",
        "stabilizer_w_frac",
        "atom_indices",
        "letter_in_sg_idx",
        "letter_in_sg_token",
    )

    def __init__(
        self,
        space_group: int,
        wyckoff_letter: str,
        multiplicity: int,
        site_symmetry: str,
        element: str,
        atomic_number: int,
        representative_coord: np.ndarray,
        all_positions: np.ndarray,
        sym_ops_rotations: np.ndarray,
        sym_ops_translations: np.ndarray,
        atom_indices: List[int],
        letter_in_sg_idx: int = -1,
        occupancy: float = 1.0,
        dof_pattern: str = "(x,y,z)",
        letter_in_sg_token: int = -1,
        stabilizer_W_frac: Optional[np.ndarray] = None,
        stabilizer_w_frac: Optional[np.ndarray] = None,
    ):
        self.space_group = int(space_group)
        self.wyckoff_letter = str(wyckoff_letter)
        self.wyckoff_symbol = f"{multiplicity}{wyckoff_letter}"
        self.multiplicity = int(multiplicity)
        self.site_symmetry = str(site_symmetry)
        self.site_symmetry_idx = encode_site_symmetry(site_symmetry)
        self.element = str(element)
        self.atomic_number = int(atomic_number)
        self.occupancy = float(occupancy)
        self.representative_coord = np.asarray(representative_coord, dtype=np.float32)
        self.all_positions = np.asarray(all_positions, dtype=np.float32)
        self.sym_ops_rotations = np.asarray(sym_ops_rotations, dtype=np.float32)
        self.sym_ops_translations = np.asarray(sym_ops_translations, dtype=np.float32)
        # Full site stabilizer H_p (may be None until populated by
        # structure_to_wyckoff_orbits). NOT the m_p image generators — those
        # live in ``sym_ops_rotations`` / ``sym_ops_translations``.
        if stabilizer_W_frac is None:
            self.stabilizer_W_frac = np.eye(3, dtype=np.float32).reshape(1, 3, 3)
            self.stabilizer_w_frac = np.zeros((1, 3), dtype=np.float32)
        else:
            self.stabilizer_W_frac = np.asarray(stabilizer_W_frac, dtype=np.float32)
            self.stabilizer_w_frac = np.asarray(stabilizer_w_frac, dtype=np.float32)
        self.atom_indices = list(atom_indices)
        self.letter_in_sg_idx = int(letter_in_sg_idx)
        self.dof_pattern = str(dof_pattern)
        self.num_free_params = get_degree_of_freedom(dof_pattern)
        self.letter_in_sg_token = int(letter_in_sg_token)


# ---------------------------------------------------------------------------
# Pure source-image operation selection
# ---------------------------------------------------------------------------


def select_source_image_operation(
    rep_frac: np.ndarray,
    target_frac: np.ndarray,
    rotations: np.ndarray,
    translations: np.ndarray,
    lattice: np.ndarray,
) -> Tuple[int, float]:
    """Find the symmetry operation that best maps rep_frac to target_frac.

    Enumerates all symmetry operations and returns the one with the smallest
    minimum-image Cartesian residual.

    Args:
        rep_frac: (3,) representative fractional coordinate.
        target_frac: (3,) target fractional coordinate.
        rotations: (N_ops, 3, 3) integer rotation matrices.
        translations: (N_ops, 3) fractional translation vectors.
        lattice: (3, 3) lattice matrix (row vectors), in Å.

    Returns:
        (best_idx, best_residual_cart): index of the best operation and its
        Cartesian residual in Å.  best_idx=-1 if no operations provided.
    """
    from wyckoff_gnn.data.pbc_residual import (
        batch_minimum_image_cartesian_residual,
    )

    if len(rotations) == 0:
        return -1, float("inf")

    rotations = np.asarray(rotations, dtype=np.float64)
    transformed = np.einsum("nij,j->ni", rotations,
                            np.asarray(rep_frac, dtype=np.float64))
    transformed = transformed + np.asarray(translations, dtype=np.float64)

    residual = np.atleast_1d(
        batch_minimum_image_cartesian_residual(
            transformed, target_frac, lattice,
        ).residual_cart
    )
    # Smallest residual, lowest index on an exact tie: the same rule as the
    # original per-operation loop, which used a strict `<`.
    best_idx = int(np.argmin(residual))
    return best_idx, float(residual[best_idx])


# ---------------------------------------------------------------------------
# Cell standardization (no quotient side effects)
# ---------------------------------------------------------------------------


def standardize_structure_cell(
    structure: Structure,
    symprec: float = 1e-3,
) -> Dict:
    """Standardize a crystal structure to the conventional cell via spglib.

    This is a pure standardization step with **no** Wyckoff quotient logic.
    It is shared by both the full quotient path and the P1 matched baseline.

    Args:
        structure: Input pymatgen Structure.
        symprec: Symmetry tolerance for spglib (Å).

    Returns:
        Dict with keys:

        - ``standardized_lattice``: (3, 3) float64 — conventional cell lattice.
        - ``standardized_positions``: (M, 3) float64 — fractional coords.
        - ``standardized_numbers``: (M,) int32 — atomic numbers.
        - ``std_rotation_matrix``: (3, 3) float64 — input→std Cartesian rotation.
        - ``standardization_success``: bool — whether refine_cell succeeded.
        - ``input_spg_number``: int — SG detected on the **input** cell (0 if None).
        - ``input_international_symbol``: str — international symbol from input.
        - ``std_spg_dataset``: spglib dataset on the standardized cell (or None).
        - ``symmetry_rotations``: (N_ops, 3, 3) int32 from std dataset (or None).
        - ``symmetry_translations``: (N_ops, 3) float64 from std dataset (or None).
    """
    lattice = np.array(structure.lattice.matrix, dtype=np.float64)
    frac_coords = np.array(structure.frac_coords, dtype=np.float64)
    atomic_numbers = np.array(
        [s.number for s in structure.species], dtype=np.int32,
    )
    cell_in = (lattice, frac_coords, atomic_numbers)

    # --- Input-cell symmetry detection (for std_rotation_matrix + SG info) ---
    std_rotation_matrix = np.eye(3, dtype=np.float64)
    input_spg_number = 0
    input_international_symbol = ""
    try:
        ds_input = spglib.get_symmetry_dataset(cell_in, symprec=symprec)
        if ds_input is not None:
            input_spg_number = int(ds_input.number)
            input_international_symbol = ds_input.international or f"SG{input_spg_number}"
            if hasattr(ds_input, "std_rotation_matrix"):
                std_rotation_matrix = np.array(
                    ds_input.std_rotation_matrix, dtype=np.float64,
                )
    except Exception:
        pass

    # --- Standardize to conventional cell ---
    standardization_success = True
    try:
        std_cell = spglib.refine_cell(cell_in, symprec=symprec)
        if std_cell is None:
            std_cell = cell_in
            standardization_success = False
    except Exception:
        std_cell = cell_in
        standardization_success = False

    std_lattice = np.array(std_cell[0], dtype=np.float64)
    std_positions = np.array(std_cell[1], dtype=np.float64)
    std_numbers = np.array(std_cell[2], dtype=np.int32)

    # --- Symmetry dataset on the standardized cell ---
    std_spg_dataset = None
    symmetry_rotations = None
    symmetry_translations = None
    try:
        std_spg_dataset = spglib.get_symmetry_dataset(
            (std_lattice, std_positions, std_numbers),
            symprec=symprec,
        )
        if std_spg_dataset is not None:
            symmetry_rotations = np.array(
                std_spg_dataset.rotations, dtype=np.int32,
            )
            symmetry_translations = np.array(
                std_spg_dataset.translations, dtype=np.float64,
            )
    except Exception:
        pass

    return {
        "standardized_lattice": std_lattice,
        "standardized_positions": std_positions,
        "standardized_numbers": std_numbers,
        "std_rotation_matrix": std_rotation_matrix,
        "standardization_success": standardization_success,
        "input_spg_number": input_spg_number,
        "input_international_symbol": input_international_symbol,
        "std_spg_dataset": std_spg_dataset,
        "symmetry_rotations": symmetry_rotations,
        "symmetry_translations": symmetry_translations,
    }


# ---------------------------------------------------------------------------
# Main conversion: Structure → list of WyckoffOrbit
# ---------------------------------------------------------------------------


def structure_to_wyckoff_orbits(
    structure: Structure,
    tol: float = 1e-3,
    fallback_p1: bool = True,
    symop_match_tol: float = None,
    symop_match_tol_cart: Optional[float] = None,
    site_stabilizer_tol_cart: Optional[float] = None,
    symmetry_match_failure: str = "fallback_p1",
) -> Tuple[List[WyckoffOrbit], Dict]:
    """Convert a pymatgen Structure to occupied Wyckoff orbit instances.

    Args:
        structure: Input pymatgen Structure.
        tol: Symmetry tolerance for spglib (``symprec``), in Å.
        fallback_p1: If True, create one orbit per atom when spglib fails.
        symop_match_tol: DEPRECATED. Use ``symop_match_tol_cart`` instead.
            If provided, raises an error directing users to the new parameter.
        symop_match_tol_cart: Cartesian threshold (Å) for source-image
            symmetry operation matching.  Default: ``min(tol, 1e-4)``.
            An operation (W,w) is accepted only if the minimum-image
            Cartesian distance between ``W @ rep + w`` and ``target`` is
            at most ``symop_match_tol_cart``.
        site_stabilizer_tol_cart: Cartesian threshold (Å) for site stabilizer
            extraction.  Default: ``min(tol, 1e-4)``.  Independent of
            source-image threshold.
        symmetry_match_failure: What to do when a source image cannot be
            matched within threshold.  ``"fallback_p1"`` (default): fall back
            to P1 for the entire structure.  ``"raise"``: raise RuntimeError.

    Steps
    -----
    1. Standardise the cell using ``spglib.refine_cell`` to get the
       conventional setting.  This ensures Wyckoff letters and
       multiplicities match the ITC-A tables.
    2. Run ``spglib.get_symmetry_dataset`` on the standardised cell to
       obtain Wyckoff letters, site-symmetry symbols, equivalent-atom
       mapping, and symmetry operations.
    3. For each unique orbit, collect:
       - representative coordinate,
       - all equivalent positions,
       - the specific symmetry operations that map the representative
         to each equivalent atom,
       - original atom indices.

    If spglib cannot reliably identify the space group (e.g. heavily
    distorted structure), this function can **fall back to P1**: every
    atom becomes its own orbit with multiplicity 1.
    """
    from wyckoff_gnn.data.pbc_residual import minimum_image_cartesian_residual

    if symop_match_tol is not None:
        raise ValueError(
            "symop_match_tol (fractional, dimensionless) is deprecated. "
            "Use symop_match_tol_cart (Cartesian, Å) instead. "
            "Recommended: set symop_match_tol_cart=None to use min(symprec, 1e-4)."
        )

    if symop_match_tol_cart is None:
        symop_match_tol_cart = min(tol, 1e-4)
    if site_stabilizer_tol_cart is None:
        site_stabilizer_tol_cart = min(tol, 1e-4)

    numerical_eps = max(1e-12, 1e-7 * abs(symop_match_tol_cart))
    effective_tol = symop_match_tol_cart + numerical_eps

    if symmetry_match_failure not in ("fallback_p1", "raise"):
        raise ValueError(
            f"symmetry_match_failure must be 'fallback_p1' or 'raise', "
            f"got '{symmetry_match_failure}'"
        )
    do_fallback = (symmetry_match_failure == "fallback_p1") and fallback_p1

    # ---- Step 1: standardise to conventional cell ----
    std_info = standardize_structure_cell(structure, symprec=tol)

    std_lattice = std_info["standardized_lattice"]
    std_positions = std_info["standardized_positions"]
    std_numbers = std_info["standardized_numbers"]
    std_rotation_matrix = std_info["std_rotation_matrix"]

    spg_dataset = std_info["std_spg_dataset"]
    input_spg_number = std_info["input_spg_number"]
    input_international = std_info["input_international_symbol"]

    if spg_dataset is None:
        if fallback_p1:
            return _p1_fallback(
                structure, std_lattice, std_positions, std_numbers,
                std_rotation_matrix=std_rotation_matrix,
                fallback_stage="symmetry_dataset",
                fallback_reason="spglib.get_symmetry_dataset returned None on standardized cell",
                detected_sg_number=input_spg_number,
                detected_international_symbol=input_international,
            )
        else:
            raise RuntimeError(
                "spglib.get_symmetry_dataset returned None. "
                "The structure may be too distorted for symmetry detection "
                "with the current tolerance. Set fallback_p1=True to treat "
                "every atom as its own orbit, or increase tol."
            )

    sg_number = int(spg_dataset.number)
    international = spg_dataset.international or f"SG{sg_number}"
    wyckoff_letters = list(spg_dataset.wyckoffs)
    site_sym_symbols = list(spg_dataset.site_symmetry_symbols)
    equiv_atoms = np.array(spg_dataset.equivalent_atoms, dtype=np.int32)

    # Symmetry operations from the standardised cell.
    rotations_all = np.array(spg_dataset.rotations, dtype=np.int32)  # (N_ops, 3, 3)
    translations_all = np.array(spg_dataset.translations, dtype=np.float64)  # (N_ops, 3)

    # ---- Step 3: group atoms by orbit and extract per-orbit data ----
    unique_orbits = sorted(set(int(e) for e in equiv_atoms))
    orbits: List[WyckoffOrbit] = []
    atom_to_orbit = -np.ones(len(std_numbers), dtype=np.int32)
    atom_image_index = -np.ones(len(std_numbers), dtype=np.int32)

    for orbit_idx_local, orbit_id in enumerate(unique_orbits):
        mask = equiv_atoms == orbit_id
        atom_indices = np.where(mask)[0]
        m_p = len(atom_indices)

        letter = wyckoff_letters[orbit_id]
        site_sym = site_sym_symbols[orbit_id]
        rep_frac = std_positions[atom_indices[0]].copy()

        # All equivalent positions (in standardised cell).
        all_frac = std_positions[atom_indices].copy()

        # Per-equivalent symmetry operations: find the sym-op that BEST maps the
        # representative to each equivalent position, using periodic residual.
        per_op_rots = np.zeros((m_p, 3, 3), dtype=np.float32)
        per_op_trans = np.zeros((m_p, 3), dtype=np.float32)
        per_op_rots[0] = np.eye(3, dtype=np.float32)
        per_op_trans[0] = 0.0

        max_residual_cart = 0.0

        for a in range(1, m_p):
            target_frac = all_frac[a]
            best_idx, best_residual_cart = select_source_image_operation(
                rep_frac, target_frac, rotations_all, translations_all,
                std_lattice,
            )

            if best_idx >= 0 and best_residual_cart <= effective_tol:
                per_op_rots[a] = rotations_all[best_idx].astype(np.float32)
                per_op_trans[a] = translations_all[best_idx].astype(np.float32)
                max_residual_cart = max(max_residual_cart, best_residual_cart)
                if best_residual_cart > symop_match_tol_cart:
                    warnings.warn(
                        f"SG {sg_number}, orbit '{letter}' (mult={m_p}), "
                        f"image {a}: residual_cart={best_residual_cart:.2e} Å "
                        f"exceeds symop_match_tol_cart={symop_match_tol_cart:.2e} Å "
                        f"but within numerical_eps. Accepted."
                    )
            else:
                msg = (
                    f"SG {sg_number}, orbit '{letter}' (mult={m_p}), "
                    f"image {a}: no sym-op maps rep to target within "
                    f"symop_match_tol_cart={symop_match_tol_cart:.2e} Å "
                    f"(best residual_cart={best_residual_cart:.2e} Å)."
                )
                if do_fallback:
                    warnings.warn(
                        f"{msg}  Falling back to P1 (every atom its own orbit)."
                    )
                    return _p1_fallback(
                        structure, std_lattice, std_positions, std_numbers,
                        std_rotation_matrix=std_rotation_matrix,
                        fallback_stage="source_image",
                        fallback_reason=msg,
                        detected_sg_number=sg_number,
                        detected_international_symbol=international,
                    )
                else:
                    raise RuntimeError(
                        f"{msg}  The structure may not be symmetric, or "
                        f"spglib's tolerance (symprec={tol}) is too strict. "
                        f"Set symmetry_match_failure='fallback_p1' to treat "
                        f"every atom as its own orbit."
                    )

        # Element: use the first (representative) atom.
        z_p = int(std_numbers[atom_indices[0]])
        element_str = _atomic_number_to_symbol(z_p)

        # DOF pattern from the representative coordinate.
        dof = _classify_dof_pattern(rep_frac)

        # SG-local letter index: order within this specific space group.
        # This is the index of the letter among all letters present in this SG
        # in the ITC-A tables.  We compute it from the Wyckoff letter itself.
        letter_in_sg_idx = get_wyckoff_letter_index(letter)

        orbit = WyckoffOrbit(
            space_group=sg_number,
            wyckoff_letter=letter,
            multiplicity=m_p,
            site_symmetry=site_sym,
            element=element_str,
            atomic_number=z_p,
            representative_coord=rep_frac,
            all_positions=all_frac,
            sym_ops_rotations=per_op_rots,
            sym_ops_translations=per_op_trans,
            atom_indices=list(int(i) for i in atom_indices),
            letter_in_sg_idx=letter_in_sg_idx,
            occupancy=1.0,
            dof_pattern=dof,
        )
        # Populate the FULL site stabilizer H_p = { g in G : g·rep = rep (mod 1) }.
        # This is the complete set of point-group operations that fix the
        # representative fractional coordinate; the Reynolds projector at this
        # site is the average of D(R_g) over exactly this set. Do NOT confuse
        # with orbit.sym_ops_rotations, which holds only the m_p image
        # generators (coset representatives of H_p in G).
        stab_numerical_eps = max(1e-12, 1e-7 * abs(site_stabilizer_tol_cart))
        stab_effective_tol = site_stabilizer_tol_cart + stab_numerical_eps
        stab_W, stab_w = _compute_stab_cartesian(
            rep_frac.astype(np.float64),
            rotations_all,
            translations_all,
            std_lattice,
            tol_cart=stab_effective_tol,
        )
        orbit.stabilizer_W_frac = np.asarray(stab_W, dtype=np.float32)
        orbit.stabilizer_w_frac = np.asarray(stab_w, dtype=np.float32)
        orbits.append(orbit)

        # Fill atom_to_orbit and atom_image_index for original atoms.
        for a, idx in enumerate(atom_indices):
            atom_to_orbit[int(idx)] = len(orbits) - 1
            atom_image_index[int(idx)] = a

    meta = {
        "atom_to_orbit": atom_to_orbit,
        "atom_image_index": atom_image_index,
        "standardized_lattice": std_lattice,
        "standardized_positions": std_positions,
        "standardized_numbers": std_numbers,
        "sg_number": sg_number,
        "international_symbol": str(international),
        # Store the complete symmetry operations from the standardized cell.
        # These MUST be used for stabilizer extraction and projection building,
        # NOT re-computed from a second spglib call with incomplete positions.
        "symmetry_rotations": np.array(spg_dataset.rotations, dtype=np.int32),
        "symmetry_translations": np.array(spg_dataset.translations, dtype=np.float64),
        "std_rotation_matrix": std_rotation_matrix,
        # --- Fallback provenance metadata ---
        "preprocess_mode": "native_p1" if sg_number == 1 else "quotient",
        "used_p1_fallback": False,
        "fallback_stage": "none",
        "fallback_reason": None,
        "detected_sg_number": input_spg_number,
        "detected_international_symbol": input_international,
        "output_sg_number": sg_number,
    }
    return orbits, meta


# ---------------------------------------------------------------------------
# Site stabilizer extraction (Cartesian PBC residual)
# ---------------------------------------------------------------------------


def _compute_stab_cartesian(
    rep_frac: np.ndarray,
    all_rotations: np.ndarray,
    all_translations: np.ndarray,
    lattice: np.ndarray,
    tol_cart: float = 1e-3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Find symmetry operations that fix rep_frac within Cartesian tolerance.

    Uses minimum-image Cartesian distance (not fractional norm) to determine
    which operations belong to the site stabilizer.

    Args:
        rep_frac: (3,) representative fractional coordinate.
        all_rotations: (N_ops, 3, 3) integer rotation matrices.
        all_translations: (N_ops, 3) fractional translation vectors.
        lattice: (3, 3) lattice matrix (row vectors), in Å.
        tol_cart: Cartesian threshold in Å.

    Returns:
        (stab_W, stab_w): rotations and translations of stabilizer operations.
    """
    from wyckoff_gnn.data.pbc_residual import (
        batch_minimum_image_cartesian_residual,
    )

    n_ops = len(all_rotations)
    if n_ops == 0:
        mask = np.zeros(0, dtype=bool)
    else:
        transformed = np.einsum(
            "nij,j->ni",
            np.asarray(all_rotations, dtype=np.float64),
            np.asarray(rep_frac, dtype=np.float64),
        ) + np.asarray(all_translations, dtype=np.float64)
        residual = np.atleast_1d(
            batch_minimum_image_cartesian_residual(
                transformed, rep_frac, lattice,
            ).residual_cart
        )
        mask = residual <= tol_cart

    if not mask.any():
        return (
            np.eye(3, dtype=np.float32)[None],
            np.zeros((1, 3), dtype=np.float32),
        )

    return (
        all_rotations[mask].astype(np.float32),
        all_translations[mask].astype(np.float32),
    )


# ---------------------------------------------------------------------------
# P1 fallback — every atom is its own orbit
# ---------------------------------------------------------------------------


def _p1_fallback(
    structure: Structure,
    lattice: np.ndarray,
    positions: np.ndarray,
    numbers: np.ndarray,
    std_rotation_matrix: Optional[np.ndarray] = None,
    fallback_stage: str = "explicit",
    fallback_reason: Optional[str] = None,
    detected_sg_number: int = 0,
    detected_international_symbol: str = "",
) -> Tuple[List[WyckoffOrbit], Dict]:
    """Fallback: every atom becomes a separate orbit (multiplicity = 1).

    This handles distorted structures or cases where spglib cannot reliably
    detect higher symmetry.  The instance graph degenerates to an atom-level
    graph, which is always correct albeit without Wyckoff compression.

    **Critical**: In P1 (SG=1), site symmetry is "1" (trivial), so ALL atoms
    have full 3D coordinate freedom: dof_pattern="(x,y,z)", num_free_params=3.
    This is true regardless of whether numerical coordinates happen to be 0,
    1/2, etc. — those are just coordinate values, not symmetry constraints.
    """
    if std_rotation_matrix is None:
        std_rotation_matrix = np.eye(3, dtype=np.float64)
    orbits = []
    M = len(numbers)
    atom_to_orbit = np.arange(M, dtype=np.int32)
    atom_image_index = np.zeros(M, dtype=np.int32)  # P1: all image_index=0

    for i in range(M):
        z = int(numbers[i])
        element_str = _atomic_number_to_symbol(z)
        rep = positions[i].copy()

        # P1: site symmetry is "1" (trivial), so ALL coordinates are free
        # Do NOT call _classify_dof_pattern — that's for higher-symmetry sites
        dof = "(x,y,z)"

        orbit = WyckoffOrbit(
            space_group=1,  # P1
            wyckoff_letter="a",
            multiplicity=1,
            site_symmetry="1",
            element=element_str,
            atomic_number=z,
            representative_coord=rep,
            all_positions=rep.reshape(1, 3),
            sym_ops_rotations=np.eye(3, dtype=np.float32).reshape(1, 3, 3),
            sym_ops_translations=np.zeros((1, 3), dtype=np.float32),
            atom_indices=[i],
            letter_in_sg_idx=0,
            occupancy=1.0,
            dof_pattern=dof,
        )
        orbits.append(orbit)

    meta = {
        "atom_to_orbit": atom_to_orbit,
        "atom_image_index": atom_image_index,
        "standardized_lattice": lattice,
        "standardized_positions": positions,
        "standardized_numbers": numbers,
        "sg_number": 1,
        "international_symbol": "P1",
        # P1: the only symmetry operation is identity.
        "symmetry_rotations": np.eye(3, dtype=np.int32)[None, :, :],
        "symmetry_translations": np.zeros((1, 3), dtype=np.float64),
        "std_rotation_matrix": std_rotation_matrix,
        # --- Fallback provenance metadata ---
        "preprocess_mode": "explicit_p1" if fallback_stage == "explicit_p1" else "fallback_p1",
        "used_p1_fallback": True,
        "fallback_stage": fallback_stage,
        "fallback_reason": fallback_reason,
        "detected_sg_number": detected_sg_number,
        "detected_international_symbol": detected_international_symbol,
        "output_sg_number": 1,
    }
    return orbits, meta


# ---------------------------------------------------------------------------
# Node feature extraction
# ---------------------------------------------------------------------------


def orbits_to_node_features(
    orbits: List[WyckoffOrbit],
    target_max_mult: int = 0,
) -> Dict[str, torch.Tensor]:
    """Convert a list of WyckoffOrbit instances to node feature tensors.

    Args:
        orbits: List of WyckoffOrbit objects.
        target_max_mult: If > 0 and > local max_mult, pad sym_ops to this
            size for batch collation.  Set to 0 for per-graph sizing.

    Returns a dict with keys:
      - orbit_element: (K,) long — atomic number Z.
      - orbit_multiplicity: (K,) long — m_p.
      - orbit_rep_frac: (K, 3) — representative fractional coordinates.
      - orbit_letter_in_sg: (K,) long — SG-local Wyckoff letter index.
      - orbit_site_sym: (K,) long — site-symmetry categorical index.
      - letter_in_sg_token: (K,) long — global (sg, letter) token id.
      - orbit_sym_ops_rotations: (K, max_mult, 3, 3) padded IMAGE generators
        (one op per atomic image; NOT the site stabilizer).
      - orbit_sym_ops_translations: (K, max_mult, 3) padded translations.
      - orbit_mult_mask: (K, max_mult) bool — True for valid equivalent
        positions.
      - orbit_stabilizer_W_frac: (K, max_H, 3, 3) padded FULL site stabilizer
        H_p in fractional coordinates. This is the set of operations
        satisfying ``W·rep + w ≡ rep (mod 1)`` and is what the Reynolds
        projector must average over.
      - orbit_stabilizer_w_frac: (K, max_H, 3) fractional translations of H_p.
      - orbit_stabilizer_mask: (K, max_H) bool — True for valid stabilizer
        entries (H_p may be shorter than max_H due to per-orbit variation).
    """
    K = len(orbits)
    if K == 0:
        m = max(1, target_max_mult)
        return {
            "orbit_element": torch.empty(0, dtype=torch.long),
            "orbit_multiplicity": torch.empty(0, dtype=torch.long),
            "orbit_rep_frac": torch.empty(0, 3),
            "orbit_letter_in_sg": torch.empty(0, dtype=torch.long),
            "orbit_site_sym": torch.empty(0, dtype=torch.long),
            "letter_in_sg_token": torch.empty(0, dtype=torch.long),
            "orbit_sym_ops_rotations": torch.empty(0, m, 3, 3),
            "orbit_sym_ops_translations": torch.empty(0, m, 3),
            "orbit_mult_mask": torch.empty(0, m, dtype=torch.bool),
            "orbit_stabilizer_W_frac": torch.empty(0, 1, 3, 3),
            "orbit_stabilizer_w_frac": torch.empty(0, 1, 3),
            "orbit_stabilizer_mask": torch.empty(0, 1, dtype=torch.bool),
        }

    max_mult = max(o.multiplicity for o in orbits)
    pad_mult = max(max_mult, target_max_mult)  # ensure uniform across dataset
    # Pad stabilizer to the maximum possible site-symmetry order (m-3m = 48)
    # so that batch collation across graphs from different SGs is uniform.
    local_max_H = max(int(o.stabilizer_W_frac.shape[0]) for o in orbits)
    max_H = max(local_max_H, 48)

    orbit_element = torch.zeros(K, dtype=torch.long)
    orbit_multiplicity = torch.zeros(K, dtype=torch.long)
    orbit_rep_frac = torch.zeros(K, 3)
    orbit_letter_in_sg = torch.zeros(K, dtype=torch.long)
    orbit_site_sym = torch.zeros(K, dtype=torch.long)
    letter_in_sg_token = torch.zeros(K, dtype=torch.long)

    sym_ops_rot = torch.zeros(K, pad_mult, 3, 3)
    sym_ops_trans = torch.zeros(K, pad_mult, 3)
    mult_mask = torch.zeros(K, pad_mult, dtype=torch.bool)

    stab_W_pad = torch.zeros(K, max_H, 3, 3)
    stab_w_pad = torch.zeros(K, max_H, 3)
    stab_mask = torch.zeros(K, max_H, dtype=torch.bool)
    # Identity padding for stab slots so any downstream code that accidentally
    # reads padded rows produces an identity operation rather than zero.
    stab_W_pad[:, :] = torch.eye(3)

    for i, orb in enumerate(orbits):
        orbit_element[i] = orb.atomic_number
        orbit_multiplicity[i] = orb.multiplicity
        orbit_rep_frac[i] = torch.from_numpy(orb.representative_coord)
        orbit_letter_in_sg[i] = orb.letter_in_sg_idx
        orbit_site_sym[i] = orb.site_symmetry_idx
        letter_in_sg_token[i] = orb.letter_in_sg_token

        m = orb.multiplicity
        sym_ops_rot[i, :m] = torch.from_numpy(orb.sym_ops_rotations)
        sym_ops_trans[i, :m] = torch.from_numpy(orb.sym_ops_translations)
        mult_mask[i, :m] = True

        h = int(orb.stabilizer_W_frac.shape[0])
        stab_W_pad[i, :h] = torch.from_numpy(orb.stabilizer_W_frac)
        stab_w_pad[i, :h] = torch.from_numpy(orb.stabilizer_w_frac)
        stab_mask[i, :h] = True

    # Compute DOF and free-parameter tangent basis for each orbit.
    orbit_dof = torch.zeros(K, dtype=torch.long)
    max_dof = max(o.num_free_params for o in orbits) if K > 0 else 0
    orbit_param_basis_frac = torch.zeros(K, max(max_dof, 1), 3)
    for i, orb in enumerate(orbits):
        orbit_dof[i] = orb.num_free_params
        basis = _dof_basis_from_pattern(orb.dof_pattern)
        for a, b in enumerate(basis):
            if a < orbit_param_basis_frac.shape[1]:
                orbit_param_basis_frac[i, a] = torch.tensor(b, dtype=torch.float32)

    result = {
        "orbit_element": orbit_element,
        "orbit_multiplicity": orbit_multiplicity,
        "orbit_rep_frac": orbit_rep_frac,
        "orbit_letter_in_sg": orbit_letter_in_sg,
        "orbit_site_sym": orbit_site_sym,
        "letter_in_sg_token": letter_in_sg_token,
        "orbit_sym_ops_rotations": sym_ops_rot,
        "orbit_sym_ops_translations": sym_ops_trans,
        "orbit_mult_mask": mult_mask,
        "orbit_stabilizer_W_frac": stab_W_pad,
        "orbit_stabilizer_w_frac": stab_w_pad,
        "orbit_stabilizer_mask": stab_mask,
        "orbit_dof": orbit_dof,
        "orbit_param_basis_frac": orbit_param_basis_frac,
    }
    return result


def _dof_basis_from_pattern(pattern: str):
    """Parse DOF pattern into free-parameter tangent basis vectors.

    E.g. ``"(x,y,z)"`` → [[1,0,0], [0,1,0], [0,0,1]]
         ``"(x,x,0)"`` → [[1,1,0]]
         ``"(0,0,z)"`` → [[0,0,1]]
         ``"(0,0,0)"`` → []
    """
    # Extract the pattern inside parentheses.
    inner = pattern.strip("()")
    parts = [p.strip() for p in inner.split(",")]

    # Collect unique free variables.
    free_vars = []
    seen = set()
    for p in parts:
        if p == "0" or p == "1/2":
            continue
        # Normalize: x, y, z, -x, -y, -z, 2x, 2y, 2z
        var = p.replace("-", "").replace("2", "")
        if var in ("x", "y", "z") and var not in seen:
            free_vars.append(var)
            seen.add(var)

    # Build basis: for each free variable, create a vector [∂x/∂u, ∂y/∂u, ∂z/∂u].
    basis = []
    for var in free_vars:
        vec = [0.0, 0.0, 0.0]
        for dim_idx, p in enumerate(parts):
            p_clean = p.strip()
            if p_clean == var:
                vec[dim_idx] = 1.0
            elif p_clean == f"-{var}":
                vec[dim_idx] = -1.0
            elif p_clean == f"2{var}":
                vec[dim_idx] = 2.0
        basis.append(vec)
    return basis


# ---------------------------------------------------------------------------
# Token attachment for template prior
# ---------------------------------------------------------------------------


def attach_letter_tokens(
    orbits: List[WyckoffOrbit],
    token_lookup: Dict[Tuple[int, str], int],
    strict: bool = True,
) -> int:
    """Fill ``orbit.letter_in_sg_token`` from a (sg, letter) → id table.

    Args:
        orbits: Orbit list to mutate in-place.
        token_lookup: Mapping built by ``load_template_graph`` /
            ``build_wyckoff_templates``.
        strict: If True, raise on a missing (sg, letter); else leave -1.

    Returns:
        Number of orbits successfully tagged.
    """
    n_ok = 0
    for orb in orbits:
        key = (int(orb.space_group), str(orb.wyckoff_letter))
        tok = token_lookup.get(key, None)
        if tok is None:
            if strict:
                raise KeyError(
                    f"(sg={key[0]}, letter='{key[1]}') missing from template "
                    "vocabulary; rebuild templates with the latest pyxtal."
                )
            continue
        orb.letter_in_sg_token = int(tok)
        n_ok += 1
    return n_ok


# ---------------------------------------------------------------------------
# DOF pattern classifier
# ---------------------------------------------------------------------------


def _classify_dof_pattern(coord: np.ndarray, frac_tol: float = 1e-3) -> str:
    """Classify the DOF pattern of a representative coordinate.

    Compares x, y, z against the special values {0, 1/2} and against each
    other (including x=-y, y=2x) to decide which coordinates are free, tied,
    or fixed.  Comparisons against special values take precedence.

    Args:
        coord: (3,) fractional coordinate.
        frac_tol: Dimensionless fractional tolerance for comparing against
            special values {0, 1/2} and for equality checks.  This is NOT
            an Å tolerance — it operates directly on fractional coordinates.
    """
    x, y, z = (float(v % 1.0) for v in coord)

    def base(v):
        if abs(v) < frac_tol or abs(v - 1.0) < frac_tol:
            return "0"
        if abs(v - 0.5) < frac_tol:
            return "1/2"
        return None

    def eq(a, b):
        return abs(a - b) < frac_tol

    def neg_eq(a, b):
        return abs((a + b) % 1.0) < frac_tol or abs((a + b) % 1.0 - 1.0) < frac_tol

    def double_eq(a, b):
        diff = (b - 2.0 * a) % 1.0
        return diff < frac_tol or diff > 1.0 - frac_tol

    bx, by, bz = base(x), base(y), base(z)
    vx = bx if bx is not None else "x"
    vy = by if by is not None else "y"
    vz = bz if bz is not None else "z"

    # Couple free variables only; never touch components already pinned.
    if bx is None and by is None:
        if eq(x, y):
            vy = "x"
        elif neg_eq(x, y):
            vy = "-x"
        elif double_eq(x, y):
            vy = "2x"
    if bx is None and bz is None:
        if eq(x, z):
            vz = "x"
        elif neg_eq(x, z):
            vz = "-x"
    if by is None and bz is None and vy == "y" and vz == "z":
        if eq(y, z):
            vz = "y"
        elif neg_eq(y, z):
            vz = "-y"

    return f"({vx},{vy},{vz})"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ATOMIC_NUMBER_TABLE: Dict[int, str] = {
    0: "X", 1: "H", 2: "He", 3: "Li", 4: "Be", 5: "B", 6: "C", 7: "N",
    8: "O", 9: "F", 10: "Ne", 11: "Na", 12: "Mg", 13: "Al", 14: "Si",
    15: "P", 16: "S", 17: "Cl", 18: "Ar", 19: "K", 20: "Ca",
    21: "Sc", 22: "Ti", 23: "V", 24: "Cr", 25: "Mn", 26: "Fe",
    27: "Co", 28: "Ni", 29: "Cu", 30: "Zn", 31: "Ga", 32: "Ge",
    33: "As", 34: "Se", 35: "Br", 36: "Kr", 37: "Rb", 38: "Sr",
    39: "Y", 40: "Zr", 41: "Nb", 42: "Mo", 43: "Tc", 44: "Ru",
    45: "Rh", 46: "Pd", 47: "Ag", 48: "Cd", 49: "In", 50: "Sn",
    51: "Sb", 52: "Te", 53: "I", 54: "Xe", 55: "Cs", 56: "Ba",
    57: "La", 58: "Ce", 59: "Pr", 60: "Nd", 61: "Pm", 62: "Sm",
    63: "Eu", 64: "Gd", 65: "Tb", 66: "Dy", 67: "Ho", 68: "Er",
    69: "Tm", 70: "Yb", 71: "Lu", 72: "Hf", 73: "Ta", 74: "W",
    75: "Re", 76: "Os", 77: "Ir", 78: "Pt", 79: "Au", 80: "Hg",
    81: "Tl", 82: "Pb", 83: "Bi", 84: "Po", 85: "At", 86: "Rn",
    87: "Fr", 88: "Ra", 89: "Ac", 90: "Th", 91: "Pa", 92: "U",
    93: "Np", 94: "Pu", 95: "Am", 96: "Cm", 97: "Bk", 98: "Cf",
    99: "Es", 100: "Fm", 101: "Md", 102: "No", 103: "Lr",
    104: "Rf", 105: "Db", 106: "Sg", 107: "Bh", 108: "Hs",
    109: "Mt", 110: "Ds", 111: "Rg", 112: "Cn", 113: "Nh",
    114: "Fl", 115: "Mc", 116: "Lv", 117: "Ts", 118: "Og",
}


def _atomic_number_to_symbol(z: int) -> str:
    return _ATOMIC_NUMBER_TABLE.get(z, "X")


# ---------------------------------------------------------------------------
# pyxtal-based alternative path (kept for reference / edge cases)
# ---------------------------------------------------------------------------


def get_orbits_via_pyxtal(
    structure: Structure,
) -> Optional[Tuple[List[WyckoffOrbit], Dict]]:
    """Alternative path using pyxtal for Wyckoff analysis.

    pyxtal provides explicit Wyckoff letter, multiplicity, and DOF from the
    International Tables data, which is more complete than spglib for some
    edge cases (e.g. rhombohedral settings).
    """
    try:
        from pyxtal import pyxtal as pyxtal_mod
    except ImportError:
        return None

    try:
        lattice = structure.lattice.matrix
        species = [str(s) for s in structure.species]
        coords = structure.frac_coords

        xtal = pyxtal_mod()
        xtal.from_seed(lattice, species, coords)

        orbits = []
        atom_to_orbit = -np.ones(len(species), dtype=np.int32)
        total_atoms = 0

        for site in xtal.atom_sites:
            letter = site.wyckoff_letter
            mult = site.multiplicity
            site_sym = str(site.get_site_symmetry())
            elem = site.specie
            atomic_num = int(getattr(elem, "number", 0))
            rep = site.position.astype(np.float32)
            try:
                all_pos = np.array(
                    [pos for pos in site.get_all_positions()], dtype=np.float32
                )
            except Exception:
                all_pos = rep.reshape(1, 3)

            dof = _classify_dof_pattern(rep, 1e-3)

            orbit = WyckoffOrbit(
                space_group=xtal.group.number,
                wyckoff_letter=letter,
                multiplicity=mult,
                site_symmetry=site_sym,
                element=str(elem),
                atomic_number=atomic_num,
                representative_coord=rep,
                all_positions=all_pos,
                sym_ops_rotations=np.eye(3, dtype=np.float32).reshape(1, 3, 3).repeat(mult, axis=0),
                sym_ops_translations=np.zeros((mult, 3), dtype=np.float32),
                atom_indices=list(range(total_atoms, total_atoms + mult)),
                letter_in_sg_idx=get_wyckoff_letter_index(letter),
                occupancy=1.0,
                dof_pattern=dof,
            )
            orbits.append(orbit)
            total_atoms += mult

        atom_to_orbit[:] = -1  # pyxtal path doesn't maintain original atom indices
        meta = {
            "atom_to_orbit": atom_to_orbit,
            "standardized_lattice": lattice,
            "sg_number": xtal.group.number,
            "international_symbol": str(xtal.group.symbol),
        }
        return orbits, meta
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Fractional → Cartesian rotation matrix conversion
# ---------------------------------------------------------------------------


def fractional_rotation_to_cartesian(
    W: np.ndarray, lattice: np.ndarray
) -> np.ndarray:
    """Convert a fractional-space rotation W to the e3nn Cartesian O(3) matrix.

    spglib fractional operations use **column convention**::

        r'_col = W @ r_col + w    (W integer, w fractional)

    Our Cartesian convention is row-vectors::

        x_cart_row = r_frac_row @ A    (A rows = lattice vectors)

    The e3nn-correct Cartesian rotation acting on column vectors is::

        R_e3nn = A^T @ W @ A^{-T}

    This satisfies R^T @ R = I (orthogonal) for all lattice types,
    and transforms correctly under global O(3) rotations:
    R → Q @ R @ Q^T.

    This is the **single authoritative formula** used throughout
    WyckoffGNN for fractional→Cartesian rotation conversion.

    Args:
        W: (..., 3, 3) integer matrix in fractional space.
        lattice: (3, 3) lattice matrix (row vectors).

    Returns:
        (..., 3, 3) Cartesian e3nn rotation matrix (float32).
    """
    A = np.asarray(lattice, dtype=np.float64)
    # R_e3nn = A^T @ W @ A^{-T}
    A_T = A.T
    A_inv_T = np.linalg.inv(A).T
    R = A_T @ W.astype(np.float64) @ A_inv_T
    return R.astype(np.float32)


def fractional_rotation_to_cartesian_torch(
    W: torch.Tensor, lattice: torch.Tensor
) -> torch.Tensor:
    """Torch version of ``fractional_rotation_to_cartesian``.

    Uses the same e3nn convention:  R = A^T @ W @ A^{-T}

    Args:
        W: (..., 3, 3) fractional rotation matrices.
        lattice: (3, 3) or (B, 3, 3) or (..., 3, 3) lattice matrix.

    Returns:
        (..., 3, 3) Cartesian e3nn rotation matrices.
    """
    A = lattice.to(W.dtype)
    A_T = A.transpose(-2, -1)
    A_inv_T = torch.inverse(A).transpose(-2, -1)
    # Expand lattice to match W's leading dims if needed.
    while A_T.dim() < W.dim():
        A_T = A_T.unsqueeze(0)
        A_inv_T = A_inv_T.unsqueeze(0)
    return torch.matmul(torch.matmul(A_T, W.to(A.dtype)), A_inv_T)


__all__ = [
    "WyckoffOrbit",
    "select_source_image_operation",
    "standardize_structure_cell",
    "structure_to_wyckoff_orbits",
    "orbits_to_node_features",
    "attach_letter_tokens",
    "fractional_rotation_to_cartesian",
    "fractional_rotation_to_cartesian_torch",
    "get_orbits_via_pyxtal",
]
