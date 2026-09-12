"""CIF directory dataset adapter.

Reads a directory of ``.cif`` files, each representing one material.
Optionally aligns targets and splits from a companion CSV.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from wyckoff_gnn.data.adapters.base import DatasetAdapter
from wyckoff_gnn.data.records import StructureRecord

log = logging.getLogger(__name__)


class CIFDirAdapter(DatasetAdapter):
    """Adapter for a directory of ``.cif`` files.

    Config keys
    -----------
    cif_dir : str (required)
        Directory containing ``.cif`` files.
    target_csv : str (optional)
        CSV mapping ``id_column`` -> ``target_column`` (and optionally
        ``split_column``).
    id_column : str (default ``"material_id"``)
        Column name for material id in the target CSV.
    target_column : str (optional)
        Column name for the scalar target in the target CSV.
    split_column : str (optional)
        Column name for the split label in the target CSV.
    recursive : bool (default False)
        If True, search subdirectories recursively.
    skip_failed : bool (default False)
        If True, skip CIFs that cannot be parsed.
    """

    def __init__(
        self,
        cif_dir: str,
        target_csv: Optional[str] = None,
        id_column: str = "material_id",
        target_column: Optional[str] = None,
        split_column: Optional[str] = None,
        recursive: bool = False,
        skip_failed: bool = False,
    ):
        self.cif_dir = Path(cif_dir).resolve()
        if not self.cif_dir.is_dir():
            raise NotADirectoryError(f"Not a directory: {self.cif_dir}")
        self.target_csv = Path(target_csv).resolve() if target_csv else None
        self.id_column = id_column
        self.target_column = target_column
        self.split_column = split_column
        self.recursive = recursive
        self.skip_failed = skip_failed

        # Pre-load target map if provided.
        self._target_map: Dict[str, Dict[str, Any]] = {}
        if self.target_csv is not None:
            if not self.target_csv.exists():
                raise FileNotFoundError(f"Target CSV not found: {self.target_csv}")
            with open(self.target_csv, "r", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    mid = row[id_column].strip()
                    entry: Dict[str, Any] = {}
                    if target_column and target_column in row:
                        val = row[target_column].strip()
                        entry["target"] = float(val) if val else None
                    if split_column and split_column in row:
                        val = row[split_column].strip().lower()
                        if val in ("train", "val", "test"):
                            entry["split"] = val
                    self._target_map[mid] = entry

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "CIFDirAdapter":
        return cls(
            cif_dir=config["cif_dir"],
            target_csv=config.get("target_csv"),
            id_column=config.get("id_column", "material_id"),
            target_column=config.get("target_column"),
            split_column=config.get("split_column"),
            recursive=config.get("recursive", False),
            skip_failed=config.get("skip_failed", False),
        )

    def iter_records(self) -> Iterable[StructureRecord]:
        glob_pattern = "**/*.cif" if self.recursive else "*.cif"
        cif_paths = sorted(self.cif_dir.glob(glob_pattern))
        for cif_path in cif_paths:
            material_id = cif_path.stem
            try:
                yield self._cif_to_record(cif_path, material_id)
            except Exception as e:
                msg = f"CIF {cif_path}: {e}"
                if self.skip_failed:
                    log.warning(msg)
                    continue
                raise RuntimeError(msg) from e

    def _cif_to_record(
        self, cif_path: Path, material_id: str,
    ) -> StructureRecord:
        from pymatgen.core.structure import Structure
        structure = Structure.from_file(str(cif_path))

        info = self._target_map.get(material_id, {})
        target = info.get("target")
        split = info.get("split")

        return StructureRecord(
            material_id=material_id,
            structure=structure,
            target=target,
            split=split,
            metadata={
                "cif_path": str(cif_path),
                "source": "cif_dir",
            },
        )

    @property
    def cif_files(self) -> List[Path]:
        glob_pattern = "**/*.cif" if self.recursive else "*.cif"
        return sorted(self.cif_dir.glob(glob_pattern))


__all__ = ["CIFDirAdapter"]
