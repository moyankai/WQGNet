"""Split loading, generation, and validation utilities.

Supports:
- Official JARVIS-Leaderboard style splits (JSON mapping id -> target)
- Random train/val/test splits
- Leak validation
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


def load_official_split(split_path: str) -> Dict[str, float]:
    """Load an official split file (JSON mapping material_id -> target value).

    Returns: ``{material_id: target_value}``.
    """
    with open(split_path, "r") as f:
        raw = json.load(f)
    # Normalise: values may be dicts (JARVIS format) with "value" key.
    result: Dict[str, float] = {}
    for key, val in raw.items():
        if isinstance(val, dict):
            result[str(key)] = float(val.get("value", 0.0))
        else:
            result[str(key)] = float(val)
    return result


def generate_random_split(
    material_ids: List[str],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    """Generate a random train/val/test split from a list of material IDs.

    Returns three dicts mapping material_id -> 0.0 (placeholder target).
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-8
    rng = np.random.RandomState(seed)
    ids = sorted(material_ids)
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_ids = ids[:n_train]
    val_ids = ids[n_train:n_train + n_val]
    test_ids = ids[n_train + n_val:]
    return (
        {mid: 0.0 for mid in train_ids},
        {mid: 0.0 for mid in val_ids},
        {mid: 0.0 for mid in test_ids},
    )


def validate_split_no_leak(
    train_ids: Set[str],
    val_ids: Set[str],
    test_ids: Set[str],
) -> bool:
    """Return True if there is no overlap between splits."""
    return (
        len(train_ids & val_ids) == 0
        and len(train_ids & test_ids) == 0
        and len(val_ids & test_ids) == 0
    )


def split_map_from_metas(
    train_meta: Dict, val_meta: Dict, test_meta: Dict,
) -> Dict[str, str]:
    """Build a unified ``{material_id: split_name}`` dict from three split metas."""
    result: Dict[str, str] = {}
    for mid in train_meta:
        result[str(mid)] = "train"
    for mid in val_meta:
        result[str(mid)] = "val"
    for mid in test_meta:
        result[str(mid)] = "test"
    return result


__all__ = [
    "load_official_split",
    "generate_random_split",
    "validate_split_no_leak",
    "split_map_from_metas",
]
