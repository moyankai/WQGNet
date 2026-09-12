"""GMTNet piezoelectric tensor dataset adapter.

Reads the pre-filtered GMTNet piezoelectric pickle and produces
StructureRecords whose targets are (3, 6) Voigt-form rank-3 tensors.

Target convention (verified against GMTNet_piezo/data.py):
  - key ``piezoelectric_C_m2``, units C/m^2 (piezoelectric *stress* tensor
    e_ij, not the strain tensor d_ij)
  - stored as a (3, 6) Voigt matrix with VASP PIEZO column ordering
    ``xx, yy, zz, xy, yz, zx`` and NO factor of 2
  - symmetric in the last two Cartesian indices (e_ijk = e_ikj)
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np

from wyckoff_gnn.data.adapters.base import DatasetAdapter
from wyckoff_gnn.data.records import StructureRecord

log = logging.getLogger(__name__)


def _gmtnet_atoms_to_pymatgen_structure(atoms: Dict[str, Any], jid: str) -> Any:
    """Convert a GMTNet atoms dict to a pymatgen Structure.

    The GMTNet atoms dicts carry an explicit ``cartesian`` flag; honour it
    rather than assuming a coordinate convention.
    """
    from pymatgen.core.structure import Structure
    from pymatgen.core.lattice import Lattice

    lattice_mat = atoms.get("lattice_mat")
    coords = atoms.get("coords", [])
    elements = atoms.get("elements", [])

    if lattice_mat is None:
        raise ValueError(f"GMTNet entry {jid}: missing 'lattice_mat' in atoms")

    lattice = Lattice(np.array(lattice_mat))
    species = [
        el["element"] if isinstance(el, dict) else str(el)
        for el in elements
    ]
    return Structure(
        lattice, species, coords,
        coords_are_cartesian=bool(atoms.get("cartesian", False)),
    )


class GMTNetPiezoAdapter(DatasetAdapter):
    """Adapter for the GMTNet piezoelectric tensor dataset.

    Config keys
    -----------
    pkl_path : str (required)
        Path to gmtnet_piezo_filtered.pkl (a dict jid -> entry).
    split_file : str (required)
        Path to split_seed32.json with train/val/test JID lists.
    max_entries : int (optional)
        Limit total entries (for smoke tests).
    fail_closed : bool
        If True, raise when a JID is absent from the split file instead of
        skipping it.
    """

    def __init__(
        self,
        pkl_path: str,
        split_file: str,
        max_entries: Optional[int] = None,
        fail_closed: bool = False,
    ):
        self.pkl_path = Path(pkl_path).resolve()
        if not self.pkl_path.exists():
            raise FileNotFoundError(f"GMTNet piezo pickle not found: {self.pkl_path}")

        self.split_path = Path(split_file).resolve()
        if not self.split_path.exists():
            raise FileNotFoundError(f"Split file not found: {self.split_path}")

        self.max_entries = max_entries
        self.fail_closed = fail_closed

        with open(self.pkl_path, "rb") as f:
            self._data = pickle.load(f)

        with open(self.split_path, "r") as f:
            self._split = json.load(f)

        self._split_map: Dict[str, str] = {}
        for name in ("train", "val", "test"):
            for jid in self._split.get(name, []):
                self._split_map[str(jid)] = name

        log.info(
            "GMTNet piezo adapter: %d entries, split: %d train / %d val / %d test",
            len(self._data),
            len(self._split.get("train", [])),
            len(self._split.get("val", [])),
            len(self._split.get("test", [])),
        )

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "GMTNetPiezoAdapter":
        return cls(
            pkl_path=config["pkl_path"],
            split_file=config["split_file"],
            max_entries=config.get("max_entries"),
            fail_closed=config.get("fail_closed", False),
        )

    def iter_records(self) -> Iterable[StructureRecord]:
        items = list(self._data.items())
        if self.max_entries:
            items = items[: self.max_entries]

        for jid, entry in items:
            jid_str = str(jid)

            split = self._split_map.get(jid_str)
            if split is None:
                if self.fail_closed:
                    raise KeyError(
                        f"GMTNet JID {jid_str} is absent from split file "
                        f"{self.split_path}. Split matching must not default "
                        f"to 'train'."
                    )
                log.warning("GMTNet JID %s not in split file, skipping.", jid_str)
                continue

            structure = _gmtnet_atoms_to_pymatgen_structure(entry["atoms"], jid_str)
            piezo = np.array(entry["piezoelectric_C_m2"], dtype=np.float32)
            if piezo.shape != (3, 6):
                raise ValueError(
                    f"GMTNet JID {jid_str}: expected piezo shape (3, 6), "
                    f"got {piezo.shape}"
                )

            yield StructureRecord(
                material_id=jid_str,
                structure=structure,
                target=piezo,
                target_type="graph_tensor",
                split=split,
                metadata={
                    "jid": jid_str,
                    "target_frame": "input_cartesian",
                    "target_tensor_rank": 3,
                    "target_tensor_symmetry": "symmetric_last_two",
                    "target_layout": "voigt_3x6_vasp_piezo",
                    "target_semantics": "piezoelectric_stress_e_C_per_m2",
                },
            )


__all__ = ["GMTNetPiezoAdapter"]
