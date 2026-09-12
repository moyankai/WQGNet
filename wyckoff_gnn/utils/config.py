"""Config loading with YAML merge and CLI override.

No Hydra dependency — simple nested dict merge.
Supports::

    wyckoffgnn train --config configs/train.yaml
    wyckoffgnn train --config configs/train.yaml --override "hidden_irreps=96x0e+48x1o+24x2e"
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional


def load_config(
    config_path: str,
    overrides: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Load a YAML config, resolve includes, and apply CLI overrides.

    ``$include`` keys are special: if a config file has::

        $include: [configs/default.yaml, configs/datasets/jarvis_fe.yaml]

    those files are loaded and merged in order, with later files
    overriding earlier ones.  The main config overrides all includes.
    """
    from wyckoff_gnn.utils.io import load_yaml

    base = load_yaml(config_path)

    # Resolve $include.
    includes = base.pop("$include", [])
    merged: Dict[str, Any] = {}
    for inc_path in includes:
        inc = load_yaml(inc_path)
        merged = _deep_merge(merged, inc)
    merged = _deep_merge(merged, base)

    # CLI overrides: key.sub.key=value
    overrides = overrides or []
    for ov in overrides:
        _apply_override(merged, ov)

    # Normalize YAML string numerics (e.g. "1e-3" → 0.001).
    _normalize_numerics(merged)

    return merged


def _apply_override(config: Dict[str, Any], override: str) -> None:
    """Apply a dotted-path CLI override like 'model.hidden_irreps=96x0e'."""
    if "=" not in override:
        return
    path, value = override.split("=", 1)
    keys = path.split(".")
    d = config
    for key in keys[:-1]:
        if key not in d:
            d[key] = {}
        d = d[key]
    # Try to parse value.
    d[keys[-1]] = _parse_value(value)


def _parse_value(s: str) -> Any:
    s = s.strip()
    if s.lower() == "true":
        return True
    if s.lower() == "false":
        return False
    if s.lower() == "none" or s.lower() == "null":
        return None
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


# Keys whose values should be float.
_FLOAT_KEYS = frozenset({
    "lr", "learning_rate", "weight_decay", "cutoff", "rbf_max",
    "grad_clip", "gradient_clip_val", "train_ratio", "val_ratio",
    "test_ratio", "target_mean", "target_std", "ema_decay",
    "dropout", "force_weight", "symprec", "angle_tolerance",
    "scheduler_patience", "early_stop_patience", "min_lr",
})

# Keys whose values should be int.
_INT_KEYS = frozenset({
    "epochs", "num_epochs", "batch_size", "val_batch_size",
    "num_layers", "num_rbf", "patience", "seed", "num_workers",
    "build_workers", "shard_size", "max_samples", "max_train",
    "max_val", "max_test", "max_train_samples", "max_val_samples",
    "max_test_samples", "global_max_mult", "edge_state_dim",
    "edge_state_layers", "radial_mlp_width", "max_pbc_images",
})

# Keys that must stay as strings (never coerced).
_STRING_KEYS = frozenset({
    "hidden_irreps", "init_irreps", "model_type", "graph_type", "loss",
    "scheduler", "pool", "readout_mode", "radial_gate_mode",
    "tp_mode", "edge_pool_mode", "material_id", "adapter",
    "structure_column", "structure_format", "id_column",
    "target_column", "split_column", "id_key", "atoms_key",
    "target_key", "name", "cache_dir", "output_dir", "output_root",
    "manifest_path", "shard_dir", "csv_path", "json_path",
    "cif_dir", "split_file", "benchmark", "device",
})


def _normalize_numerics(config: Dict[str, Any]) -> None:
    """Recursively coerce YAML string numerics to float/int in-place."""
    for key, val in list(config.items()):
        if isinstance(val, dict):
            _normalize_numerics(val)
        elif isinstance(val, str):
            # Only coerce known numeric keys — never guess.
            new_val: Any = val
            if key in _INT_KEYS:
                try:
                    new_val = int(float(val))  # float first handles "1e-3"
                except (ValueError, OverflowError):
                    pass
            elif key in _FLOAT_KEYS:
                try:
                    new_val = float(val)
                except (ValueError, OverflowError):
                    pass
            # Otherwise leave as string (including _STRING_KEYS).
            if new_val is not val:
                config[key] = new_val


def _deep_merge(base: Dict, override: Dict) -> Dict:
    """Recursively merge override into base."""
    result = deepcopy(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = deepcopy(val)
    return result


__all__ = ["load_config"]
