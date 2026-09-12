"""DeepH HDF5 dataset adapter.

Reads the DeepH public datasets (Si, Bi2Se3, MoS2, etc.) from their
standard HDF5 directory layout and converts each structure + Hamiltonian
into a :class:`StructureRecord`.

DeepH directory layout (per-structure)::

    dataset_dir/
    ├── 000000/
    │   ├── lat.dat          # lattice vectors (3x3, Bohr or Angstrom)
    │   ├── rlat.dat         # reciprocal lattice
    │   ├── site_positions.dat  # fractional or Cartesian positions
    │   ├── element.dat      # atomic species
    │   ├── orbital_types.dat # orbital type per atom
    │   └── hamiltonians_pred.h5 / hamiltonians.h5
    │       /H  (group)
    │          /0_0_0    dataset (n_orb, n_orb)  # H block for R=(0,0,0)
    │          /1_0_0    dataset (n_orb, n_orb)  # H block for R=(1,0,0)
    │          ...
    │       /S  (group, optional)
    │          /0_0_0    dataset  # overlap S(R)
    └── 000001/
        ...

Alternative simpler format (from Zenodo downloads)::

    dataset.h5
       /structures/000000/lattice   (3, 3)
       /structures/000000/positions (N, 3) Cartesian Angstrom
       /structures/000000/species   (N,) int
       /hamiltonians/000000/R_list  (n_R, 3) int
       /hamiltonians/000000/H       (n_R, n_orb, n_orb) float

This adapter supports BOTH layouts (auto-detected).

Usage in config::

    dataset:
      adapter: deeph
      adapter_kwargs:
        data_dir: data/deeph_raw/Si
        format: directory   # or "single_hdf5"
        target_key: hamiltonian
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import numpy as np

from wyckoff_gnn.data.adapters.base import DatasetAdapter
from wyckoff_gnn.data.records import StructureRecord


__all__ = ["DeepHAdapter"]


# Bohr to Angstrom
_BOHR_TO_ANG = 0.529177249


class DeepHAdapter(DatasetAdapter):
    """Adapter for DeepH-style Hamiltonian datasets.

    Args:
        data_dir: path to the dataset root (directory of per-structure folders,
            or a single HDF5 file).
        format: "directory" (many per-structure folders) or "single_hdf5".
        units: "bohr" or "angstrom" for lattice/positions.
        max_entries: limit number of structures (for debugging).
        split_file: optional JSON with train/val/test splits.
    """

    def __init__(
        self,
        data_dir: str,
        format: str = "directory",
        units: str = "angstrom",
        max_entries: Optional[int] = None,
        split_file: Optional[str] = None,
        **kwargs,
    ):
        self.data_dir = data_dir
        self.format = format
        self.units = units
        self.max_entries = max_entries
        self.split_file = split_file
        self._split_map: Dict[str, str] = {}
        if split_file and os.path.exists(split_file):
            import json
            with open(split_file) as f:
                self._split_map = json.load(f)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "DeepHAdapter":
        kw = config.get("adapter_kwargs", {})
        return cls(**kw)

    def iter_records(self) -> Iterator[StructureRecord]:
        if self.format == "single_hdf5":
            yield from self._iter_single_hdf5()
        else:
            yield from self._iter_directory()

    def _iter_directory(self) -> Iterator[StructureRecord]:
        root = Path(self.data_dir)
        subdirs = sorted(
            p for p in root.iterdir()
            if p.is_dir() and (p / "lat.dat").exists()
        )
        if self.max_entries:
            subdirs = subdirs[: self.max_entries]

        for struct_dir in subdirs:
            try:
                record = self._parse_directory_entry(struct_dir)
                if record is not None:
                    yield record
            except Exception:
                continue

    def _parse_directory_entry(self, d: Path) -> Optional[StructureRecord]:
        from pymatgen.core import Lattice, Structure

        mid = d.name
        lat = np.loadtxt(d / "lat.dat")
        if self.units == "bohr":
            lat = lat * _BOHR_TO_ANG

        pos_file = d / "site_positions.dat"
        if pos_file.exists():
            pos = np.loadtxt(pos_file)
            if self.units == "bohr":
                pos = pos * _BOHR_TO_ANG
        else:
            return None

        elem_file = d / "element.dat"
        if elem_file.exists():
            elements = [line.strip() for line in open(elem_file) if line.strip()]
        else:
            return None

        lattice = Lattice(lat)
        structure = Structure(lattice, elements, pos, coords_are_cartesian=True)

        # Load Hamiltonian blocks.
        h_blocks = self._load_hamiltonian_blocks(d)
        if h_blocks is None:
            return None

        split = self._split_map.get(mid)
        return StructureRecord(
            material_id=mid,
            structure=structure,
            target=h_blocks,
            target_type="hamiltonian",
            split=split,
            metadata={"source": "deeph", "path": str(d)},
        )

    def _load_hamiltonian_blocks(self, d: Path) -> Optional[Dict]:
        """Load H(R) blocks from HDF5 in the directory."""
        for fname in ["hamiltonians.h5", "hamiltonians_pred.h5", "hamiltonian.h5"]:
            hpath = d / fname
            if hpath.exists():
                return self._read_h5_blocks(hpath)
        return None

    def _read_h5_blocks(self, path: Path) -> Dict:
        """Read H blocks from an HDF5 file. Returns dict {(i,j,(Rx,Ry,Rz)): ndarray}."""
        import h5py
        blocks = {}
        with h5py.File(path, "r") as f:
            h_group = f.get("H") or f.get("hamiltonian") or f.get("hamiltonians")
            if h_group is None:
                return blocks
            for key in h_group:
                data = np.array(h_group[key])
                parts = key.replace("-", "n").split("_")
                if len(parts) == 3:
                    R = tuple(int(p.replace("n", "-")) for p in parts)
                    blocks[(0, 0, R)] = data
                elif len(parts) == 5:
                    i, j = int(parts[0]), int(parts[1])
                    R = tuple(int(p.replace("n", "-")) for p in parts[2:])
                    blocks[(i, j, R)] = data
        return blocks

    def _iter_single_hdf5(self) -> Iterator[StructureRecord]:
        import h5py
        from pymatgen.core import Lattice, Structure

        with h5py.File(self.data_dir, "r") as f:
            structs = f.get("structures")
            hams = f.get("hamiltonians")
            if structs is None or hams is None:
                return

            keys = sorted(structs.keys())
            if self.max_entries:
                keys = keys[: self.max_entries]

            for mid in keys:
                try:
                    sg = structs[mid]
                    lat = np.array(sg["lattice"])
                    pos = np.array(sg["positions"])
                    species = [int(s) for s in np.array(sg["species"])]
                    from pymatgen.core.periodic_table import Element
                    elements = [Element.from_Z(z).symbol for z in species]

                    lattice = Lattice(lat)
                    structure = Structure(
                        lattice, elements, pos, coords_are_cartesian=True,
                    )

                    hg = hams[mid]
                    R_list = np.array(hg["R_list"]) if "R_list" in hg else None
                    H_data = np.array(hg["H"]) if "H" in hg else None
                    if R_list is None or H_data is None:
                        continue

                    blocks = {}
                    for r_idx, R in enumerate(R_list):
                        blocks[(0, 0, tuple(int(x) for x in R))] = H_data[r_idx]

                    split = self._split_map.get(mid)
                    yield StructureRecord(
                        material_id=mid,
                        structure=structure,
                        target=blocks,
                        target_type="hamiltonian",
                        split=split,
                        metadata={"source": "deeph"},
                    )
                except Exception:
                    continue
