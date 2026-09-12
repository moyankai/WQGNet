"""Unified structure record for dataset adapters.

All external data sources are converted into :class:`StructureRecord`
objects, which carry a material identifier, a pymatgen Structure (or
constructible equivalent), a target value (scalar, vector, tensor, dict, or
None), an optional property-type discriminator, an optional split label, and
arbitrary metadata.

This is the contract between *adapters* (which read raw data) and
*graph builders* (which turn records into graphs).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import numpy as np


_VALID_SPLITS = {None, "train", "val", "test"}


@dataclass
class StructureRecord:
    """A single material sample with its structure, target, and metadata.

    Attributes
    ----------
    material_id : str
        Unique non-empty identifier.
    structure : Any
        A pymatgen :class:`~pymatgen.core.structure.Structure`, or an
        object that can be converted to one by a graph builder.
    target : Union[float, np.ndarray, Dict[str, Any], None]
        Regression target. For graph-level scalars this is a float; for
        tensors/vectors it is an ndarray; for Hamiltonians it is a dict of
        block arrays; None for unlabelled data.
    target_type : Optional[str]
        A :class:`~wyckoff_gnn.data.property_types.PropertyType` string
        (e.g. ``"graph_scalar_intensive"``). ``None`` means legacy scalar
        (backward-compat).
    split : Optional[str]
        One of ``"train"``, ``"val"``, ``"test"``, or ``None``.
    metadata : Dict[str, Any]
        Arbitrary key-value pairs from the source dataset (formula,
        space group, extra columns, etc.).
    """

    material_id: str
    structure: Any
    target: Union[float, np.ndarray, Dict[str, Any], None] = None
    target_type: Optional[str] = None
    split: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.material_id, str) or not self.material_id.strip():
            raise ValueError(
                f"material_id must be a non-empty string, got {self.material_id!r}"
            )
        if self.split not in _VALID_SPLITS:
            raise ValueError(
                f"split must be one of {_VALID_SPLITS}, got {self.split!r}"
            )


def validate_record(record: StructureRecord) -> None:
    """Validate a StructureRecord, raising ``ValueError`` on problems.

    Calls ``__post_init__`` checks and additionally verifies the
    structure appears to be a pymatgen Structure.
    """
    # Re-run dataclass validation.
    record.__post_init__()
    # Light structure check: duck-type for common pymatgen attributes.
    if not hasattr(record.structure, "lattice") or not hasattr(record.structure, "sites"):
        raise ValueError(
            f"Structure for {record.material_id} does not look like a "
            f"pymatgen Structure (missing .lattice or .sites)"
        )


def record_to_summary(record: StructureRecord) -> Dict[str, Any]:
    """Return a short human-readable summary dict for a record."""
    num_sites = (
        len(record.structure)
        if hasattr(record.structure, "__len__")
        else "?"
    )
    return {
        "material_id": record.material_id,
        "target": record.target,
        "split": record.split,
        "num_sites": num_sites,
        "metadata_keys": sorted(record.metadata.keys()),
    }


def validate_records(records: List[StructureRecord]) -> List[str]:
    """Validate a list of records, returning a list of error messages.

    Returns an empty list if all records are valid.
    """
    errors: List[str] = []
    for i, rec in enumerate(records):
        try:
            validate_record(rec)
        except ValueError as e:
            errors.append(f"Record[{i}] ({rec.material_id}): {e}")
    return errors


__all__ = [
    "StructureRecord",
    "validate_record",
    "validate_records",
    "record_to_summary",
]
