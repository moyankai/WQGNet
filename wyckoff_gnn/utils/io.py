"""Simple IO utilities — JSON and YAML read/write with error handling."""

from __future__ import annotations

import json
from typing import Any


def load_json(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def save_json(path: str, obj: Any, indent: int = 2) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=indent, default=str)


def load_yaml(path: str) -> dict:
    try:
        import yaml
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        raise ImportError("PyYAML is required for YAML configs. Install: pip install pyyaml")


def save_yaml(path: str, obj: dict) -> None:
    try:
        import yaml
        with open(path, "w") as f:
            yaml.safe_dump(obj, f, default_flow_style=False)
    except ImportError:
        save_json(path, obj)  # fallback


__all__ = ["load_json", "save_json", "load_yaml", "save_yaml"]
