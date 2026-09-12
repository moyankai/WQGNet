"""GMTNet dielectric tensor dataset adapter.

Reads the pre-filtered GMTNet dielectric tensor pickle file and produces
StructureRecords with full 3x3 symmetric tensor targets.
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

    GMTNet atoms dict has fractional coordinates (0-1 range), unlike JARVIS
    which uses cartesian coordinates.
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
    # GMTNet uses fractional coordinates
    return Structure(lattice, species, coords, coords_are_cartesian=False)


class GMTNetDielectricAdapter(DatasetAdapter):
    """Adapter for GMTNet dielectric tensor dataset.

    Config keys
    -----------
    pkl_path : str (required)
        Path to gmtnet_dielectric_filtered.pkl.
    split_file : str (required)
        Path to split_seed32.json with train/val/test JID lists.
    max_entries : int (optional)
        Limit total entries (for smoke tests).
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
            raise FileNotFoundError(f"GMTNet pickle not found: {self.pkl_path}")

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
        for jid in self._split.get("train", []):
            self._split_map[str(jid)] = "train"
        for jid in self._split.get("val", []):
            self._split_map[str(jid)] = "val"
        for jid in self._split.get("test", []):
            self._split_map[str(jid)] = "test"

        log.info(
            f"GMTNet adapter: {len(self._data)} entries, "
            f"split: {len(self._split.get('train', []))} train / "
            f"{len(self._split.get('val', []))} val / "
            f"{len(self._split.get('test', []))} test"
        )

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "GMTNetDielectricAdapter":
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
            atoms = entry["atoms"]
            structure = _gmtnet_atoms_to_pymatgen_structure(atoms, jid_str)

            dielectric = np.array(entry["dielectric"], dtype=np.float32)

            split = self._split_map.get(jid_str)
            if split is None:
                if self.fail_closed:
                    raise KeyError(
                        f"GMTNet JID {jid_str} is absent from split file "
                        f"{self.split_path}. Split matching must not default to 'train'."
                    )
                else:
                    log.warning(
                        f"GMTNet JID {jid_str} not in split file, skipping."
                    )
                    continue

            yield StructureRecord(
                material_id=jid_str,
                structure=structure,
                target=dielectric,
                target_type="graph_tensor",
                split=split,
                metadata={
                    "jid": jid_str,
                    "target_frame": "input_cartesian",
                    "target_tensor_rank": 2,
                    "target_tensor_symmetry": "symmetric",
                    "target_semantics": "dielectric",
                },
            )


__all__ = ["GMTNetDielectricAdapter"]
