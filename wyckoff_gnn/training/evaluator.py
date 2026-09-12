"""Standardised evaluation and experiment output.

Produces: predictions.csv, results.json, config.json, status.json,
          dataset_summary.json, error_bins.json, and optionally
          jarvis_leaderboard_submission.zip.
"""

from __future__ import annotations

import csv
import io
import json
import os
import time
import zipfile
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from wyckoff_gnn.data.normalization import TargetNormalizer
from wyckoff_gnn.utils.io import save_json


# ---------------------------------------------------------------------------
# Prediction collection
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    normalizer: TargetNormalizer,
    device: torch.device,
    metadata_map: Optional[Dict[str, Dict[str, Any]]] = None,
    task: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Run inference on a dataloader and collect per-sample predictions.

    Property-type-aware: dispatches based on model._property_spec.output_head.
    Returns a list of dicts, one per sample, suitable for writing as CSV/JSON.

    Args:
        task: Optional task label recorded on every row (e.g.,
            'formation_energy_peratom', 'optb88vdw_bandgap'). Used for
            downstream JARVIS-Leaderboard comparisons.
    """
    model.eval()
    rows: List[Dict[str, Any]] = []

    spec = getattr(model, "_property_spec", None)
    output_head = spec.output_head if spec else "ScalarReadout"

    for batch in loader:
        batch = batch.to(device)
        pred = model(batch)

        if output_head == "ScalarReadout":
            _collect_scalar_predictions(
                pred, batch, normalizer, metadata_map, rows, task
            )
        elif output_head in ("TensorPropertyHead", "DiagonalCanonicalHead",
                             "PiezoRank3Readout"):
            _collect_tensor_predictions(
                pred, batch, normalizer, metadata_map, rows, task
            )
        elif output_head == "WyckoffHamiltonianHead":
            _collect_hamiltonian_predictions(
                pred, batch, metadata_map, rows, task
            )
        else:
            _collect_scalar_predictions(
                pred, batch, normalizer, metadata_map, rows, task
            )

    return rows


def _collect_scalar_predictions(pred, batch, normalizer, metadata_map, rows, task=None):
    """Original scalar path."""
    pred_norm = pred.view(-1).cpu()
    target_norm = batch.y.view(-1).float().cpu()

    for i in range(len(pred_norm)):
        y_norm = float(target_norm[i])
        y_hat_norm = float(pred_norm[i])
        y_raw = normalizer.denormalize(y_norm)
        y_hat_raw = normalizer.denormalize(y_hat_norm)
        error = y_hat_raw - y_raw
        abs_error = abs(error)

        mid = _get_material_id(batch, i)
        meta = metadata_map.get(mid, {}) if metadata_map else {}

        rows.append({
            "material_id": mid,
            "jid": mid,  # leaderboard-style alias
            "task": task or "",
            "split": meta.get("split", "test"),
            "y_true_raw": y_raw,
            "y_pred_raw": y_hat_raw,
            "target": y_raw,           # leaderboard-style alias for y_true_raw
            "prediction": y_hat_raw,   # leaderboard-style alias for y_pred_raw
            "y_true_norm": y_norm,
            "y_pred_norm": y_hat_norm,
            "error": error,
            "abs_error": abs_error,
            "absolute_error": abs_error,  # leaderboard-style alias
            "num_atoms": int(meta.get("num_atoms", 0)),
            "num_orbits": int(meta.get("num_orbits", 0)),
            "compression_ratio": int(meta.get("num_atoms", 0)) / max(int(meta.get("num_orbits", 1)), 1),
            "space_group": int(meta.get("space_group", 0)),
            "num_geo_edges": int(meta.get("num_geo_edges", 0)),
        })


def _collect_tensor_predictions(pred, batch, normalizer, metadata_map, rows, task=None):
    """Tensor/vector path: per-sample, per-component comparison."""
    # Handle dict predictions (e.g., from symmetric_rank2 output)
    if isinstance(pred, dict):
        tensor_out = pred.get("tensor", pred)
        if isinstance(tensor_out, dict):
            # Rank-3 heads expose "voigt" (G,3,6) matching y_tensor;
            # rank-2 heads only have "cartesian".
            pred_cpu = tensor_out.get(
                "voigt", tensor_out.get("cartesian", tensor_out)
            ).cpu()
            raw_cpu = (tensor_out["voigt_raw"].cpu()
                       if "voigt_raw" in tensor_out else None)
        else:
            pred_cpu = tensor_out.cpu()
            raw_cpu = None
    else:
        pred_cpu = pred.cpu()
        raw_cpu = None
    
    n_graphs = int(batch.num_graphs) if hasattr(batch, "num_graphs") else pred_cpu.shape[0]
    target = getattr(batch, "y_tensor", None)
    if target is not None:
        target_cpu = target.cpu()
    else:
        target_cpu = torch.zeros_like(pred_cpu)

    for i in range(n_graphs):
        mid = _get_material_id(batch, i)
        meta = metadata_map.get(mid, {}) if metadata_map else {}
        # Flatten to 1D for consistent error computation
        p_flat = pred_cpu[i].flatten().numpy().tolist()
        t_flat = target_cpu[i].flatten().numpy().tolist()
        err = [abs(pi - ti) for pi, ti in zip(p_flat, t_flat)]
        row = {
            "material_id": mid,
            "jid": mid,
            "task": task or "",
            "split": meta.get("split", "test"),
            "y_true_components": t_flat,
            "y_pred_components": p_flat,
            "component_errors": err,
            "mae": float(np.mean(err)),
            "abs_error": float(np.mean(err)),
            "absolute_error": float(np.mean(err)),
            "num_atoms": int(meta.get("num_atoms", 0)),
            "space_group": int(meta.get("space_group", 0)),
        }
        if raw_cpu is not None:
            row["y_pred_raw_components"] = raw_cpu[i].flatten().numpy().tolist()
        rows.append(row)


def _collect_hamiltonian_predictions(pred, batch, metadata_map, rows, task=None):
    """Hamiltonian path: report block-Frobenius norm."""
    onsite = pred.get("onsite")
    offsite = pred.get("offsite")
    mid = _get_material_id(batch, 0)
    meta = metadata_map.get(mid, {}) if metadata_map else {}

    onsite_frob = float(onsite.norm().item()) if onsite is not None else 0.0
    offsite_frob = float(offsite.norm().item()) if offsite is not None else 0.0

    rows.append({
        "material_id": mid,
        "jid": mid,
        "task": task or "",
        "split": meta.get("split", "test"),
        "onsite_frobenius_norm": onsite_frob,
        "offsite_frobenius_norm": offsite_frob,
        "n_onsite_blocks": int(onsite.shape[0]) if onsite is not None else 0,
        "n_offsite_blocks": int(offsite.shape[0]) if offsite is not None else 0,
        "abs_error": 0.0,
        "absolute_error": 0.0,
        "num_atoms": int(meta.get("num_atoms", 0)),
        "space_group": int(meta.get("space_group", 0)),
    })


def _get_material_id(batch, idx: int) -> str:
    """Extract material_id for sample idx from a batch."""
    if hasattr(batch, "material_id"):
        mids = batch.material_id
        if isinstance(mids, list) and idx < len(mids):
            return str(mids[idx])
        elif isinstance(mids, str):
            return mids
    return ""


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

def compute_results(
    predictions: List[Dict[str, Any]],
    normalizer: TargetNormalizer,
    model: torch.nn.Module,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Compute summary metrics from prediction rows.

    Dispatches based on prediction schema:
    - Scalar: y_true_raw / y_pred_raw → MAE, RMSE, R²
    - Tensor: y_true_components / y_pred_components → per-component MAE
    - Hamiltonian: onsite/offsite Frobenius norms
    """
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    base = {
        "model": config.get("model_type", "unknown"),
        "dataset": config.get("dataset", "unknown"),
        "task": config.get("benchmark", config.get("task", "unknown")),
        "seed": config.get("seed", 0),
        "train_size": config.get("train_size", 0),
        "val_size": config.get("val_size", 0),
        "test_size": len(predictions),
        "target_mean": normalizer.mean,
        "target_std": normalizer.std,
        "params": n_params,
    }

    if not predictions:
        return {**base, "test_mae": float("nan"), "test_rmse": float("nan"), "test_r2": 0.0}

    first = predictions[0]

    # Scalar path
    if "y_true_raw" in first:
        y_true = np.array([r["y_true_raw"] for r in predictions])
        y_pred = np.array([r["y_pred_raw"] for r in predictions])
        y_true_n = np.array([r["y_true_norm"] for r in predictions])
        y_pred_n = np.array([r["y_pred_norm"] for r in predictions])

        mae = float(np.mean(np.abs(y_true - y_pred)))
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        ss_res = float(((y_true - y_pred) ** 2).sum())
        ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
        r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
        mae_norm = float(np.mean(np.abs(y_true_n - y_pred_n)))
        rmse_norm = float(np.sqrt(np.mean((y_true_n - y_pred_n) ** 2)))

        return {
            **base,
            "test_mae": mae, "test_rmse": rmse, "test_r2": r2,
            "test_mae_norm": mae_norm, "test_rmse_norm": rmse_norm,
        }

    # Tensor path
    if "y_true_components" in first:
        component_maes = [r["mae"] for r in predictions]
        return {
            **base,
            "test_mae": float(np.mean(component_maes)),
            "test_rmse": float(np.sqrt(np.mean([r.get("abs_error", 0) ** 2 for r in predictions]))),
            "test_r2": 0.0,
            "prediction_schema": "tensor_components",
        }

    # Hamiltonian path
    if "onsite_frobenius_norm" in first:
        return {
            **base,
            "test_mae": 0.0,
            "test_rmse": 0.0,
            "test_r2": 0.0,
            "prediction_schema": "hamiltonian_frobenius",
            "avg_onsite_frobenius": float(np.mean([r["onsite_frobenius_norm"] for r in predictions])),
            "avg_offsite_frobenius": float(np.mean([r["offsite_frobenius_norm"] for r in predictions])),
        }

    return {**base, "test_mae": float("nan"), "test_rmse": float("nan"), "test_r2": 0.0}


# ---------------------------------------------------------------------------
# Output writing
# ---------------------------------------------------------------------------

def _save_csv(rows: List[Dict[str, Any]], path: str) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_training_curve(
    rows: List[Dict[str, Any]], path: str,
) -> None:
    """Write training_curve.csv."""
    _save_csv(rows, path)


def save_predictions_csv(
    predictions: List[Dict[str, Any]], path: str,
) -> None:
    """Write predictions.csv."""
    _save_csv(predictions, path)


def write_status(path: str, stage: str) -> None:
    save_json(path, {"stage": stage, "timestamp": time.time()})


# ---------------------------------------------------------------------------
# Effective-config introspection (Phase-3 edge readout)
# ---------------------------------------------------------------------------

def extract_edge_readout_actuals(model: torch.nn.Module) -> Dict[str, Any]:
    """Introspect a WyckoffGNN (or its encoder) and return the actual
    edge-readout wiring: mode, features, edge_pool_dim, use_edge_state,
    and message_irreps_used_for_edge_readout.

    Safe on non-WyckoffGNN models — returns an empty dict.
    """
    enc = getattr(model, "encoder", None) or model
    if not hasattr(enc, "_irreps_message"):
        return {}

    mode = getattr(enc, "edge_readout_mode", None)
    use_edge_state = bool(getattr(enc, "use_edge_state", False))
    inv = getattr(enc, "edge_invariant_readout", None)

    features_eff = getattr(enc, "_edge_invariant_features_effective", None)
    if features_eff is None and inv is not None:
        features_eff = list(getattr(inv, "features", []))

    msg_irreps = getattr(enc, "_edge_readout_message_irreps", None)
    if msg_irreps is None and inv is not None:
        msg_irreps = getattr(inv.bank, "message_irreps", None) if hasattr(inv, "bank") else None

    if inv is not None:
        edge_pool_dim = int(inv.output_dim)
    elif getattr(enc, "edge_pool", None) is not None:
        edge_pool_dim = int(getattr(enc, "edge_state_dim", 0))
    else:
        edge_pool_dim = 0

    return {
        "edge_readout_mode_actual": mode,
        "edge_invariant_features_actual": features_eff if features_eff is not None else [],
        "edge_pool_dim": edge_pool_dim,
        "use_edge_state_actual": use_edge_state,
        "message_irreps_used_for_edge_readout": str(msg_irreps) if msg_irreps is not None else None,
    }


def extract_property_metadata(model: torch.nn.Module) -> Dict[str, Any]:
    """Extract PropertySpec metadata from the model for results.json recording.

    Every training run records exactly which property was predicted, which head
    was used, and what physical claims are attached. Safe on non-WyckoffGNN.
    """
    spec = getattr(model, "_property_spec", None)
    if spec is None:
        return {}

    # Covariance is claimed only for equivariant tensor heads (not diagonal/scalar)
    covariance_claimed = (
        spec.output_head in ("TensorPropertyHead", "PiezoRank3Readout")
        and spec.target_irreps is not None
        and spec.physical_type not in (
            "graph_rank2_diagonal_canonical",
            "graph_scalar_intensive",
            "graph_scalar_extensive",
        )
    )

    return {
        "property_name": spec.name,
        "physical_type": spec.physical_type,
        "output_head": spec.output_head,
        "target_irreps": spec.target_irreps,
        "property_status": spec.status,
        "pooling": spec.pooling,
        "physics_notes": spec.notes,
        "covariance_claimed": covariance_claimed,
    }


# ---------------------------------------------------------------------------
# Error bins
# ---------------------------------------------------------------------------

_DEFAULT_ATOM_BIN_EDGES = [1, 4, 8, 16, 32, 64, 128, 256, 512, 1024]


def _bucketize(value: float, edges: List[float]) -> str:
    lo = None
    for e in edges:
        if value <= e:
            hi = e
            return f"{lo if lo is not None else '-inf'}-{hi}"
        lo = e
    return f"{lo}-inf"


def compute_error_bins(predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Bucket |error| by three axes: absolute-error deciles, num_atoms bins,
    and space_group.

    Returns a JSON-serialisable dict with counts and MAE per bin.
    """
    if not predictions:
        return {"n_samples": 0, "by_abs_error_decile": {},
                "by_num_atoms_bin": {}, "by_space_group": {}}

    abs_err = np.array([r.get("abs_error", 0.0) for r in predictions], dtype=float)
    n_atoms = np.array([int(r.get("num_atoms", 0)) for r in predictions], dtype=int)
    sgs = np.array([int(r.get("space_group", 0)) for r in predictions], dtype=int)

    # Decile bins over |error|
    deciles = np.quantile(abs_err, np.linspace(0.1, 0.9, 9)).tolist()
    by_err: Dict[str, Any] = {}
    edges = [-float("inf")] + deciles + [float("inf")]
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        mask = (abs_err > lo) & (abs_err <= hi)
        if not mask.any():
            continue
        by_err[f"q{i}"] = {
            "lo": float(lo if lo != -float("inf") else abs_err.min()),
            "hi": float(hi if hi != float("inf") else abs_err.max()),
            "n": int(mask.sum()),
            "mae": float(abs_err[mask].mean()),
        }

    # Num-atoms bin
    by_atoms: Dict[str, Any] = {}
    for k, atoms in enumerate(n_atoms):
        key = _bucketize(int(atoms), _DEFAULT_ATOM_BIN_EDGES)
        b = by_atoms.setdefault(key, {"n": 0, "sum_abs_err": 0.0})
        b["n"] += 1
        b["sum_abs_err"] += float(abs_err[k])
    for key, b in by_atoms.items():
        b["mae"] = b["sum_abs_err"] / b["n"] if b["n"] else 0.0
        b.pop("sum_abs_err", None)

    # Per space-group (top 20 by count so the file stays small)
    sg_stats: Dict[int, Dict[str, float]] = {}
    for k, sg in enumerate(sgs):
        s = sg_stats.setdefault(int(sg), {"n": 0, "sum_abs_err": 0.0})
        s["n"] += 1
        s["sum_abs_err"] += float(abs_err[k])
    for s in sg_stats.values():
        s["mae"] = s["sum_abs_err"] / s["n"] if s["n"] else 0.0
        s.pop("sum_abs_err", None)
    top_sgs = dict(sorted(sg_stats.items(), key=lambda kv: -kv[1]["n"])[:20])

    return {
        "n_samples": int(len(predictions)),
        "overall_mae": float(abs_err.mean()),
        "overall_rmse": float(np.sqrt((abs_err ** 2).mean())),
        "by_abs_error_decile": by_err,
        "by_num_atoms_bin": by_atoms,
        "by_space_group_top20": top_sgs,
    }


# ---------------------------------------------------------------------------
# JARVIS submission zip
# ---------------------------------------------------------------------------

_JARVIS_BENCH_MAP = {
    "formation_energy_peratom": "AI-SinglePropertyPrediction-formation_energy_peratom-dft_3d-test-mae.csv",
    "optb88vdw_bandgap":        "AI-SinglePropertyPrediction-optb88vdw_bandgap-dft_3d-test-mae.csv",
    "mbj_bandgap":              "AI-SinglePropertyPrediction-mbj_bandgap-dft_3d-test-mae.csv",
    "ehull":                    "AI-SinglePropertyPrediction-ehull-dft_3d-test-mae.csv",
    "slme":                     "AI-SinglePropertyPrediction-slme-dft_3d-test-mae.csv",
}


def write_jarvis_submission_zip(
    predictions: List[Dict[str, Any]],
    zip_path: str,
    benchmark: str,
    id_column: str = "material_id",
) -> Optional[str]:
    """Package predictions into a JARVIS-leaderboard submission zip.

    The zip contains a single CSV named after the leaderboard entry, with
    two columns: `id,prediction`. Returns the path on success, or None if
    ``benchmark`` is not a known JARVIS leaderboard entry.
    """
    if benchmark not in _JARVIS_BENCH_MAP:
        return None
    csv_name = _JARVIS_BENCH_MAP[benchmark]

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["id", "prediction"])
    for r in predictions:
        mid = str(r.get(id_column, "")) or str(r.get("material_id", ""))
        if not mid:
            continue
        writer.writerow([mid, float(r.get("y_pred_raw", 0.0))])

    os.makedirs(os.path.dirname(zip_path) or ".", exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(csv_name, buf.getvalue())
    return zip_path


def write_experiment_outputs(
    output_dir: str,
    config: Dict[str, Any],
    predictions: List[Dict[str, Any]],
    results: Dict[str, Any],
    training_curve: Optional[List[Dict[str, Any]]] = None,
    dataset_summary: Optional[Dict[str, Any]] = None,
    model: Optional[torch.nn.Module] = None,
) -> None:
    """Write all standard experiment outputs to ``output_dir``.

    If ``model`` is provided, the effective edge-readout wiring is
    introspected and merged into ``config`` **and** ``results`` under the
    Phase-3-agreed field names (``edge_readout_mode_actual`` etc.). The
    same actuals also appear in ``results.json`` so downstream benchmark
    aggregators can filter runs without re-parsing configs.

    ``error_bins.json`` is always written. A JARVIS submission zip is
    written when ``config['benchmark']`` (or ``config['submission_benchmark']``)
    is a recognised leaderboard entry.
    """
    os.makedirs(output_dir, exist_ok=True)

    if model is not None:
        actuals = extract_edge_readout_actuals(model)
        if actuals:
            config = {**config, **actuals}
            results = {**results, **actuals}
        prop_meta = extract_property_metadata(model)
        if prop_meta:
            config = {**config, **prop_meta}
            results = {**results, **prop_meta}

    write_status(os.path.join(output_dir, "status.json"), "outputs_written")
    save_json(os.path.join(output_dir, "config.json"), config)
    save_json(os.path.join(output_dir, "results.json"), results)
    save_predictions_csv(predictions, os.path.join(output_dir, "predictions.csv"))
    if training_curve is not None:
        save_training_curve(training_curve, os.path.join(output_dir, "training_curve.csv"))
    if dataset_summary is not None:
        save_json(os.path.join(output_dir, "dataset_summary.json"), dataset_summary)

    # Error bins are always written.
    error_bins = compute_error_bins(predictions)
    save_json(os.path.join(output_dir, "error_bins.json"), error_bins)

    # Optional JARVIS submission zip.
    bench = config.get("submission_benchmark") or config.get("benchmark")
    if bench and bench in _JARVIS_BENCH_MAP:
        write_jarvis_submission_zip(
            predictions,
            os.path.join(output_dir, "jarvis_leaderboard_submission.zip"),
            bench,
        )


# ---------------------------------------------------------------------------
# Checkpoint evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model: torch.nn.Module,
    test_loader: DataLoader,
    normalizer: TargetNormalizer,
    config: Dict[str, Any],
    output_dir: str,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Evaluate a model on a dataloader and write standard outputs.

    Does NOT load a checkpoint — the caller must load weights first.
    For checkpoint loading, use :meth:`ExperimentRunner.run_evaluate`.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    predictions = collect_predictions(model, test_loader, normalizer, device)
    results = compute_results(predictions, normalizer, model, config)
    write_experiment_outputs(output_dir, config, predictions, results)
    return results


__all__ = [
    "collect_predictions",
    "compute_results",
    "save_training_curve",
    "save_predictions_csv",
    "write_status",
    "write_experiment_outputs",
    "evaluate_model",
]
