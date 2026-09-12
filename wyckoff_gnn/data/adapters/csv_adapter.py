"""CSV dataset adapter.

Reads a CSV file where each row describes one material.  The structure
can be provided as an embedded CIF string, a file path (CIF or POSCAR),
or a pymatgen JSON dict.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from wyckoff_gnn.data.adapters.base import DatasetAdapter
from wyckoff_gnn.data.records import StructureRecord

log = logging.getLogger(__name__)


class CSVAdapter(DatasetAdapter):
    """Adapter for CSV datasets with embedded or file-path structures.

    Config keys
    -----------
    csv_path : str (required)
        Path to the CSV file.
    id_column : str (required)
        Column name for ``material_id``.
    structure_column : str (required)
        Column name containing the structure data.
    structure_format : str (required)
        One of ``"cif_string"``, ``"cif_path"``, ``"poscar_path"``,
        ``"pymatgen_json"``.
    target_column : str (optional)
        Column name for the scalar regression target.
    split_column : str (optional)
        Column name for ``split`` label (must be train/val/test).
    skip_failed : bool (default False)
        If True, skip rows that fail to parse and log a warning.
        If False, raise on the first failure.
    metadata_columns : list[str] (optional)
        Additional columns to store in ``record.metadata``.
    """

    def __init__(
        self,
        csv_path: str,
        id_column: str,
        structure_column: str,
        structure_format: str = "cif_string",
        target_column: Optional[str] = None,
        split_column: Optional[str] = None,
        skip_failed: bool = False,
        metadata_columns: Optional[list] = None,
    ):
        self.csv_path = Path(csv_path).resolve()
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")
        self.id_column = id_column
        self.structure_column = structure_column
        self.structure_format = structure_format
        self.target_column = target_column
        self.split_column = split_column
        self.skip_failed = skip_failed
        self.metadata_columns = metadata_columns or []
        self._base_dir = self.csv_path.parent

        _valid_formats = {"cif_string", "cif_path", "poscar_path", "pymatgen_json"}
        if structure_format not in _valid_formats:
            raise ValueError(
                f"structure_format must be one of {_valid_formats}, "
                f"got {structure_format!r}"
            )

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "CSVAdapter":
        return cls(
            csv_path=config["csv_path"],
            id_column=config["id_column"],
            structure_column=config["structure_column"],
            structure_format=config.get("structure_format", "cif_string"),
            target_column=config.get("target_column"),
            split_column=config.get("split_column"),
            skip_failed=config.get("skip_failed", False),
            metadata_columns=config.get("metadata_columns", []),
        )

    def iter_records(self) -> Iterable[StructureRecord]:
        with open(self.csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row_idx, row in enumerate(reader):
                try:
                    yield self._row_to_record(row)
                except Exception as e:
                    mid = row.get(self.id_column, f"row_{row_idx}")
                    msg = (
                        f"CSV row {row_idx} (id={mid}): {e}"
                    )
                    if self.skip_failed:
                        log.warning(msg)
                        continue
                    raise RuntimeError(msg) from e

    def _row_to_record(self, row: Dict[str, str]) -> StructureRecord:
        mid = row[self.id_column].strip()
        if not mid:
            raise ValueError(f"empty material_id in column {self.id_column!r}")

        structure = self._parse_structure(row)
        target = self._parse_target(row)
        split = self._parse_split(row)
        metadata = {
            col: row.get(col, "")
            for col in self.metadata_columns
        }
        # Always include original row keys for traceability.
        metadata["_csv_row_keys"] = list(row.keys())

        return StructureRecord(
            material_id=mid,
            structure=structure,
            target=target,
            split=split,
            metadata=metadata,
        )

    def _parse_structure(self, row: Dict[str, str]):
        raw = row[self.structure_column]
        fmt = self.structure_format

        if fmt == "cif_string":
            from pymatgen.core.structure import Structure
            return Structure.from_str(raw, fmt="cif")

        elif fmt == "cif_path":
            path = self._resolve_path(raw)
            from pymatgen.core.structure import Structure
            return Structure.from_file(str(path))

        elif fmt == "poscar_path":
            path = self._resolve_path(raw)
            from pymatgen.core.structure import Structure
            return Structure.from_file(str(path))

        elif fmt == "pymatgen_json":
            data = json.loads(raw)
            from pymatgen.core.structure import Structure
            return Structure.from_dict(data)

        raise ValueError(f"Unknown structure_format: {fmt!r}")

    def _parse_target(self, row: Dict[str, str]) -> Optional[float]:
        if self.target_column is None:
            return None
        val = row.get(self.target_column, "").strip()
        if not val:
            return None
        try:
            return float(val)
        except ValueError:
            log.warning(
                f"Non-numeric target '{val}' for {row.get(self.id_column, '?')}"
            )
            return None

    def _parse_split(self, row: Dict[str, str]) -> Optional[str]:
        if self.split_column is None:
            return None
        val = row.get(self.split_column, "").strip().lower()
        if val in ("train", "val", "test"):
            return val
        if not val:
            return None
        log.warning(
            f"Unexpected split value '{val}' for {row.get(self.id_column, '?')}"
        )
        return None

    def _resolve_path(self, raw: str) -> Path:
        p = Path(raw.strip())
        if p.is_absolute():
            return p
        return (self._base_dir / p).resolve()


__all__ = ["CSVAdapter"]
