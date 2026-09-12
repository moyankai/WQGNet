"""JARVIS-DFT dataset adapter.

Reads the JARVIS-DFT JSON database (``jdft_3d-*.json``) and converts
each entry into a :class:`StructureRecord`.

Optionally aligns with official JARVIS-Leaderboard split files.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from wyckoff_gnn.data.adapters.base import DatasetAdapter
from wyckoff_gnn.data.records import StructureRecord

log = logging.getLogger(__name__)


class JARVISAdapter(DatasetAdapter):
    """Adapter for JARVIS-DFT JSON databases.

    Config keys
    -----------
    json_path : str (required)
        Path to the ``jdft_3d-*.json`` file.
    id_key : str (default ``"jid"``)
        JSON key for the material identifier.
    atoms_key : str (default ``"atoms"``)
        JSON key for the atoms dict (JARVIS format).
    target_key : str (optional)
        JSON key for the scalar target value.
    split_file : str (optional)
        Path to a JARVIS-Leaderboard split JSON.
    max_entries : int (optional)
        Limit to the first N entries.
    """

    def __init__(
        self,
        json_path: str,
        id_key: str = "jid",
        atoms_key: str = "atoms",
        target_key: Optional[str] = None,
        split_file: Optional[str] = None,
        max_entries: Optional[int] = None,
    ):
        self.json_path = Path(json_path).resolve()
        if not self.json_path.exists():
            raise FileNotFoundError(f"JARVIS JSON not found: {self.json_path}")
        self.id_key = id_key
        self.atoms_key = atoms_key
        self.target_key = target_key
        self.max_entries = max_entries

        # Load split if provided. The JARVIS-Leaderboard split file maps
        # jid -> target_value; we only use the key set to identify test samples.
        self._official_test_ids: set = set()
        if split_file:
            split_path = Path(split_file).resolve()
            if not split_path.exists():
                raise FileNotFoundError(f"Split file not found: {split_path}")
            with open(split_path, "r") as f:
                split_data = json.load(f)
            self._official_test_ids = {str(k) for k in split_data}
            log.info(
                f"Loaded {len(self._official_test_ids)} test ids from {split_path}"
            )

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "JARVISAdapter":
        return cls(
            json_path=config["json_path"],
            id_key=config.get("id_key", "jid"),
            atoms_key=config.get("atoms_key", "atoms"),
            target_key=config.get("target_key"),
            split_file=config.get("split_file"),
            max_entries=config.get("max_entries"),
        )

    def iter_records(self) -> Iterable[StructureRecord]:
        with open(self.json_path, "r") as f:
            data = json.load(f)

        entries = data if isinstance(data, list) else []
        if self.max_entries:
            entries = entries[: self.max_entries]

        for entry in entries:
            yield self._entry_to_record(entry)

    def _entry_to_record(self, entry: Dict[str, Any]) -> StructureRecord:
        jid = str(entry.get(self.id_key, ""))
        if not jid:
            raise ValueError(f"Empty JARVIS id (key={self.id_key!r})")

        atoms = entry.get(self.atoms_key, {})
        structure = _atoms_to_pymatgen_structure(atoms, jid)

        target, target_type = self._extract_target(entry)

        split = "test" if jid in self._official_test_ids else None

        # Preserve useful JARVIS metadata.
        metadata: Dict[str, Any] = {
            "jid": jid,
            "formula": entry.get("formula", ""),
            "spg_number": entry.get("spg_number", 0),
        }
        _copy_keys = [
            "desc", "natoms", "nspecies", "formation_energy_peratom",
            "optb88vdw_bandgap", "optb88vdw_total_energy",
            "ehull", "mbj_bandgap", "slme", "spg_number",
            "epsx", "epsy", "epsz", "mepsx", "mepsy", "mepsz",
            "kv", "gv", "elasticity",
        ]
        for k in _copy_keys:
            if k in entry and k not in metadata:
                metadata[k] = entry[k]

        return StructureRecord(
            material_id=jid,
            structure=structure,
            target=target,
            target_type=target_type,
            split=split,
            metadata=metadata,
        )

    def _extract_target(self, entry: Dict[str, Any]):
        """Extract target value based on target_key and infer target_type.

        Returns (target, target_type) where target is float/ndarray/None.
        """
        if not self.target_key:
            return None, None

        # Composite tensor targets built from multiple JARVIS fields.
        _TENSOR_COMPOSITES = {
            "dielectric_diagonal": {
                "keys": ["epsx", "epsy", "epsz"],
                "target_type": "graph_tensor",
            },
            "magnetic_dielectric_diagonal": {
                "keys": ["mepsx", "mepsy", "mepsz"],
                "target_type": "graph_tensor",
            },
        }
        if self.target_key in _TENSOR_COMPOSITES:
            spec = _TENSOR_COMPOSITES[self.target_key]
            vals = [entry.get(k) for k in spec["keys"]]
            if any(v is None for v in vals):
                return None, None
            try:
                arr = np.array([float(v) for v in vals], dtype=np.float64)
            except (TypeError, ValueError):
                return None, None
            return arr, spec["target_type"]

        # Full elasticity tensor (6x6 Voigt).
        if self.target_key == "elasticity":
            val = entry.get("elasticity")
            if val is None or not isinstance(val, list):
                return None, None
            arr = np.array(val, dtype=np.float64)
            if arr.shape == (6, 6):
                return arr, "graph_tensor"
            return None, None

        # Require key present for scalar targets.
        if self.target_key not in entry:
            return None, None

        val = entry[self.target_key]
        if val is None:
            return None, None
        try:
            return float(val), "graph_scalar_intensive"
        except (TypeError, ValueError):
            return None, None


def _atoms_to_pymatgen_structure(
    atoms: Dict[str, Any], jid: str,
) -> Any:
    """Convert a JARVIS atoms dict to a pymatgen Structure.

    JARVIS atoms dict format::

        {
            "lattice_mat": [[a1,a2,a3],[b1,b2,b3],[c1,c2,c3]],
            "coords": [[x,y,z], ...],
            "elements": [{"element": "Si"}, ...],
        }
    """
    from pymatgen.core.structure import Structure
    from pymatgen.core.lattice import Lattice

    lattice_mat = atoms.get("lattice_mat")
    coords = atoms.get("coords", [])
    elements = atoms.get("elements", [])

    if lattice_mat is None:
        raise ValueError(f"JARVIS entry {jid}: missing 'lattice_mat' in atoms")

    lattice = Lattice(np.array(lattice_mat))
    species = [
        el["element"] if isinstance(el, dict) else str(el)
        for el in elements
    ]
    return Structure(lattice, species, coords, coords_are_cartesian=True)


__all__ = ["JARVISAdapter"]
