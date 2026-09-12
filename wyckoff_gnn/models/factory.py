"""Model factory — create models from config dict.

Supported models:
    wyckoff_gnn            WyckoffGNN equivariant (legacy)
    composition_ridge      Ridge regression on composition features
    composition_rf         Random forest on composition features
    mean_baseline          Mean predictor (always predicts train target mean)

Usage::

    from wyckoff_gnn.models.factory import create_model
    model = create_model({"model_type": "unified_quotient_equivariant",
                        "hidden_irreps": "128x0e + 4x1o"})
"""

from __future__ import annotations

from typing import Any, Callable, Dict

import torch
import torch.nn as nn


MODEL_REGISTRY: Dict[str, Callable[[Dict[str, Any]], Any]] = {}

MODEL_DESCRIPTIONS: Dict[str, str] = {
    "wyckoff_gnn": (
        "WyckoffGNN equivariant model (dynamic_v2 + EdgeState) — "
        "production model holding all best JARVIS results "
        "(bandgap test_mae=0.158, 5.38M params)."
    ),
    "unified_quotient_equivariant": (
        "Unified crystallographic quotient-equivariant backbone with an "
        "expressive scalar 0e sector and optional higher-order sectors."
    ),
    "composition_ridge": "Ridge regression on composition features",
    "composition_rf": "Random forest on composition features",
    "mean_baseline": "Mean predictor",
}


def register_model(name: str):
    def decorator(fn: Callable[[Dict[str, Any]], Any]) -> Callable:
        MODEL_REGISTRY[name] = fn
        return fn
    return decorator


def create_model(config: Dict[str, Any]) -> Any:
    model_type = config.get("model_type", "unified_quotient_equivariant")
    builder = MODEL_REGISTRY.get(model_type)
    if builder is None:
        raise ValueError(
            f"Unknown model_type '{model_type}'. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )
    model = builder(config)
    if isinstance(model, nn.Module):
        model._n_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
    return model


@register_model("wyckoff_gnn")
def _create_wyckoff_gnn(config: Dict[str, Any]) -> nn.Module:
    from wyckoff_gnn.models.wyckoff_equiv import WyckoffGNN
    return WyckoffGNN(
        num_layers=config.get("num_layers", 3),
        num_rbf=config.get("num_rbf", 16),
        rbf_max=config.get("rbf_max", config.get("cutoff", 5.0)),
        init_irreps=config.get("init_irreps", "64x0e"),
        hidden_irreps=config.get("hidden_irreps", "512x0e+16x1o"),
        dropout=config.get("dropout", 0.05),
        tp_mode=config.get("tp_mode", "dynamic_v2"),
        radial_mlp_width=config.get("radial_mlp_width", 32),
        use_edge_state=config.get("use_edge_state", True),
        edge_state_dim=config.get("edge_state_dim", 64),
        edge_state_layers=config.get("edge_state_layers", 2),
        edge_readout=config.get("edge_readout", True),
        edge_pool_mode=config.get("edge_pool_mode", "normalized"),
        use_site_projection=config.get("use_site_projection", False),
        per_type_shift_learnable=config.get("per_type_shift", False),
        per_type_scale_learnable=config.get("per_type_scale", False),
        max_atomic_number=config.get("max_atomic_number", 118),
        use_atom_props=config.get("use_atom_props", False),
        property_name=config.get("property_name"),
        property_type=config.get("property_type", "graph_scalar_intensive"),
    )


def _normalize_hidden_irreps(config: Dict[str, Any]) -> str:
    """Resolve hidden_irreps from config, handling legacy formats.

    New format: ``hidden_irreps: "128x0e + 8x1o + 4x2e"`` (contains 0e).
    Legacy format: ``hidden_scalar: 128, hidden_irreps: "8x1o + 4x2e", lmax: 2``.
    """
    hi = config.get("hidden_irreps", "")
    if hi and "0e" in hi:
        return hi

    import warnings
    scalar = config.get("hidden_scalar", config.get("hidden_dim", 128))
    high = hi if hi else ""
    result = f"{scalar}x0e + {high}".strip(" +") if high else f"{scalar}x0e"
    warnings.warn(
        f"Legacy config detected (hidden_scalar={scalar}, hidden_irreps='{hi}', "
        f"lmax={config.get('lmax', 0)}). Converted to hidden_irreps='{result}'. "
        "Please update your config to use hidden_irreps directly.",
        DeprecationWarning,
        stacklevel=3,
    )
    return result


@register_model("unified_quotient_equivariant")
def _create_unified_quotient_equivariant(config: Dict[str, Any]) -> nn.Module:
    from wyckoff_gnn.models.unified_equivariant import UnifiedQuotientEquivariantGNN
    hidden_irreps = _normalize_hidden_irreps(config)
    property_type = config.get("property_type", "graph_scalar_intensive")
    pool_mode = "sum" if property_type == "graph_scalar_extensive" else "mean"
    return UnifiedQuotientEquivariantGNN(
        hidden_irreps=hidden_irreps,
        num_layers=config.get("num_layers", 5),
        num_rbf=config.get("num_rbf", 32),
        rbf_max=config.get("rbf_max", config.get("cutoff", 8.0)),
        output_type=config.get("output_type", "scalar"),
        use_source_image_transport=config.get("use_source_image_transport", True),
        global_point_group_projection=config.get(
            "tensor_global_point_group_projection", False
        ),
        block_type=config.get("block_type", "scalar"),
        use_scalar_to_high_l=config.get("use_scalar_to_high_l", True),
        use_high_l_propagation=config.get("use_high_l_propagation", True),
        use_site_irrep_projection=config.get("use_site_irrep_projection", False),
        site_projector_path=config.get("site_projector_path", ""),
        site_irrep_table_path=config.get("site_irrep_table_path", ""),
        site_irrep_lmax=config.get("site_irrep_lmax", 4),
        apply_site_projection=config.get("apply_site_projection", "none"),
        unknown_projector_policy=config.get("unknown_projector_policy", "identity"),
        missing_irrep_policy=config.get("missing_irrep_policy", "identity"),
        high_l_feedback_type=config.get("high_l_feedback_type", "norm2"),
        high_l_aggregation=config.get("high_l_aggregation", "mean"),
        pool_mode=pool_mode,
    )


@register_model("composition_ridge")
def _create_composition_ridge(config: Dict[str, Any]):
    from sklearn.linear_model import Ridge
    return Ridge(alpha=config.get("ridge_alpha", 1.0))


@register_model("composition_rf")
def _create_composition_rf(config: Dict[str, Any]):
    from sklearn.ensemble import RandomForestRegressor
    return RandomForestRegressor(
        n_estimators=config.get("rf_n_estimators", 100),
        random_state=config.get("seed", 42),
    )


@register_model("mean_baseline")
def _create_mean_baseline(config: Dict[str, Any]):
    target_mean = config.get("target_mean", 0.0)

    class MeanPredictor(nn.Module):
        def __init__(self, mean_val: float):
            super().__init__()
            self.register_buffer("mean", torch.tensor([float(mean_val)]))

        def forward(self, data):
            n = data.num_graphs if hasattr(data, "num_graphs") else 1
            return self.mean.expand(n)

    return MeanPredictor(target_mean)


MODEL_GRAPH_TYPES = {

    "wyckoff_gnn": "wyckoff",

    "unified_quotient_equivariant": "wyckoff_or_p1",
    "composition_ridge": "composition",
    "composition_rf": "composition",
    "mean_baseline": "any",
}

TRAINABLE_TORCH_MODELS = {
    "wyckoff_gnn",
    "unified_quotient_equivariant",
    "mean_baseline",
}


def get_required_graph_type(model_type: str) -> str:
    return MODEL_GRAPH_TYPES.get(model_type, "unknown")


def validate_model_graph_type(model_type: str, graph_type: str) -> None:
    required = get_required_graph_type(model_type)
    if required == "any":
        return
    if required == "wyckoff_or_p1" and graph_type in ("wyckoff", "p1"):
        return
    if required != graph_type:
        raise ValueError(
            f"Model '{model_type}' requires '{required}' graphs, "
            f"but datamodule provides '{graph_type}' graphs."
        )


__all__ = [
    "create_model",
    "register_model",
    "MODEL_REGISTRY",
    "MODEL_DESCRIPTIONS",
    "MODEL_GRAPH_TYPES",
    "TRAINABLE_TORCH_MODELS",
    "get_required_graph_type",
    "validate_model_graph_type",
]
