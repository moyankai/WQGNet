"""Official JARVIS-Leaderboard split loading.

Reads the official ``dft_3d_*.json.zip`` split files distributed with the
JARVIS-Leaderboard project and applies the ``train/val/test`` partition to
a local LMDB manifest so results are directly comparable to the leaderboard.

Contract:
- The zip contains a single JSON with keys ``train``, ``val``, ``test``,
  each mapping ``JID -> target_value`` (target may be a float or a dict
  with a ``"value"`` key, handled uniformly).
- Local manifest entries are matched by ``material_id`` == JID.
- Only matched entries survive; every matched entry has ``split`` and
  ``target`` overwritten by the official values.
- JIDs present in the official split but missing locally are reported.
"""

from __future__ import annotations

import io
import json
import logging
import zipfile
from typing import Any, Dict, List, Tuple

log = logging.getLogger(__name__)


def load_leaderboard_zip(zip_path: str) -> Dict[str, Dict[str, float]]:
    """Load an official JARVIS-Leaderboard split archive.

    Accepts either a zip archive containing one ``.json`` file, or a raw
    ``.json`` file directly (some mirrors serve broken zips).

    Returns:
        Dict with keys ``'train'``, ``'val'``, ``'test'``. Each value is a
        dict ``{jid: target_value_float}``.

    Raises:
        FileNotFoundError, KeyError, ValueError on malformed archives.
    """
    if zip_path.endswith(".json"):
        with open(zip_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        json_name = zip_path
    else:
        with zipfile.ZipFile(zip_path, "r") as zf:
            json_names = [n for n in zf.namelist() if n.endswith(".json")]
            if not json_names:
                raise ValueError(f"No .json file inside {zip_path}")
            json_name = json_names[0]
            with zf.open(json_name) as f:
                raw = json.load(io.TextIOWrapper(f, encoding="utf-8"))

    if not isinstance(raw, dict):
        raise ValueError(f"Root of {json_name} must be an object")
    for split_name in ("train", "val", "test"):
        if split_name not in raw:
            raise KeyError(
                f"Official split zip {zip_path} missing '{split_name}' key; "
                f"found keys={list(raw.keys())}"
            )

    def _to_float(v: Any) -> float:
        if isinstance(v, dict):
            return float(v.get("value", 0.0))
        return float(v)

    out: Dict[str, Dict[str, float]] = {}
    for split_name in ("train", "val", "test"):
        section = raw[split_name]
        if not isinstance(section, dict):
            raise ValueError(f"'{split_name}' section must be a JID->value mapping")
        out[split_name] = {str(jid): _to_float(v) for jid, v in section.items()}
    return out


def apply_leaderboard_split_to_manifest(
    manifest_entries: List[Dict[str, Any]],
    leaderboard: Dict[str, Dict[str, float]],
    override_targets: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, List[str]]]:
    """Filter and relabel local manifest entries by an official split.

    For every JID in ``leaderboard[split]``: if a local entry with the same
    ``material_id`` exists, keep it and overwrite its ``split`` field with
    the official value. If ``override_targets=True``, also overwrite the
    ``target`` field with the official value. Local entries whose JID is
    not in any official split are dropped.

    Args:
        manifest_entries: Raw local manifest rows (list of dicts).
        leaderboard: Output of ``load_leaderboard_zip``.
        override_targets: If True, overwrite target values with leaderboard
            values. If False, keep original target values from manifest.

    Returns:
        (kept_entries, missing_jids) where:
        - ``kept_entries`` is a fresh list of dicts (never mutating the input);
        - ``missing_jids`` is ``{split: [jids in official split but missing locally]}``.
    """
    local_by_jid: Dict[str, Dict[str, Any]] = {
        str(e["material_id"]): e for e in manifest_entries
    }
    kept: List[Dict[str, Any]] = []
    missing: Dict[str, List[str]] = {"train": [], "val": [], "test": []}

    for split_name in ("train", "val", "test"):
        for jid, y_val in leaderboard[split_name].items():
            src = local_by_jid.get(jid)
            if src is None:
                missing[split_name].append(jid)
                continue
            new_entry = dict(src)  # shallow copy — do not mutate input
            new_entry["split"] = split_name
            if override_targets:
                new_entry["target"] = float(y_val)
            kept.append(new_entry)

    return kept, missing


def summarize_split_application(
    leaderboard: Dict[str, Dict[str, float]],
    kept_entries: List[Dict[str, Any]],
    missing: Dict[str, List[str]],
    zip_path: str,
) -> Dict[str, Any]:
    """Build a JSON-serialisable report of the official split application."""
    matched = {
        "train": sum(1 for e in kept_entries if e["split"] == "train"),
        "val":   sum(1 for e in kept_entries if e["split"] == "val"),
        "test":  sum(1 for e in kept_entries if e["split"] == "test"),
    }
    official = {k: len(v) for k, v in leaderboard.items()}
    missing_counts = {k: len(v) for k, v in missing.items()}

    for split_name in ("train", "val", "test"):
        miss_frac = missing_counts[split_name] / max(official[split_name], 1)
        if miss_frac > 0.05:
            log.warning(
                f"Official split '{split_name}': {missing_counts[split_name]} "
                f"of {official[split_name]} JIDs ({miss_frac*100:.1f}%) missing "
                f"from local dataset — coverage below 95%."
            )

    return {
        "source": "official_jarvis_leaderboard",
        "zip": zip_path,
        "official_counts": official,
        "matched_counts": matched,
        "missing_counts": missing_counts,
        "missing_jids": missing,
    }


__all__ = [
    "load_leaderboard_zip",
    "apply_leaderboard_split_to_manifest",
    "summarize_split_application",
]
