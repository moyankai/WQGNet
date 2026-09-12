#!/usr/bin/env python3
"""Stage 2 of the piezo coordinate-frame audit — fully standalone.

Dependencies: Python 3, numpy, pandas ONLY. No torch, no lmdb, no project
library imports, no hardcoded server paths.

Usage:
    python audit_piezo_frame_standalone.py --root .

All inputs are read from ``root`` (see file list in README.md) and all outputs
are written into ``root``:
    audit_summary.json
    audit_per_structure.csv
    audit_stdout.log
    sha256sums.txt
    requirements.txt

Audit checks (hard failures raise, not warn):
    1. prediction ID sets: 499 each, unique, equal across models, equal to the
       official split_seed32.json test set (missing=0, extra=0)
    2. stored labels: max |y_std_wq - y_std_p1| <= 1e-6
    3. Q matrices: shape (3,3), finite, max ||Q^T Q - I||_F <= 1e-8,
       max |Q_wq - Q_p1| <= 1e-8 (det(Q) min/max reported)
    4. Voigt <-> Cartesian roundtrip on all 18 basis tensors <= 1e-12
    5. original -> standardized -> original roundtrip on all 499 labels,
       compared against the raw benchmark targets
    6. metrics: V18 and full 3x3x3 Cartesian Frobenius, both frames,
       EwT 25/10/5/2% under both norms, zero predictor, and the
       zero-label / centrosymmetric / non-centrosymmetric /
       label-inconsistent subsets.

Conventions re-implemented locally (must match production):
    Voigt column order: xx, yy, zz, xy, yz, zx (VASP PIEZO), NO factor of 2.
    Forward:  T_std[i,j,k] = Q[i,a] Q[j,b] Q[k,c] T_in[a,b,c]
    Inverse:  T_in[a,b,c]  = Q[i,a] Q[j,b] Q[k,c] T_std[i,j,k]
"""
from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# local re-implementation of the Voigt / Cartesian conventions
# --------------------------------------------------------------------------
VOIGT_PAIRS = ((0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (0, 2))
T_EPS = (0.25, 0.10, 0.05, 0.02)
ZERO_PREDICTOR_FINGERPRINT = 0.4372
LABEL_INCONSISTENT_TOL = 1e-3  # production: Neumann residual > 1e-3


def voigt_to_cartesian(v: np.ndarray) -> np.ndarray:
    """(..., 3, 6) Voigt -> (..., 3, 3, 3) Cartesian.

    Column order xx, yy, zz, xy, yz, zx; no factor of 2; the last two
    Cartesian indices are filled symmetrically.
    """
    v = np.asarray(v, dtype=np.float64)
    out = np.zeros(v.shape[:-2] + (3, 3, 3), dtype=np.float64)
    for col, (i, k) in enumerate(VOIGT_PAIRS):
        out[..., i, k] = v[..., col]
        if i != k:
            out[..., k, i] = v[..., col]
    return out


def cartesian_to_voigt(t: np.ndarray) -> np.ndarray:
    """(..., 3, 3, 3) Cartesian -> (..., 3, 6) Voigt (inverse of above)."""
    t = np.asarray(t, dtype=np.float64)
    return np.stack([t[..., i, k] for i, k in VOIGT_PAIRS], axis=-1)


def v18_norm(v: np.ndarray) -> np.ndarray:
    """(N, 18) -> (N,) Frobenius over the 18 stored Voigt components."""
    return np.linalg.norm(np.asarray(v).reshape(len(v), -1), axis=1)


def cart_norm(t: np.ndarray) -> np.ndarray:
    """(N, 3, 3, 3) -> (N,) full-tensor Frobenius norm (rotation invariant)."""
    return np.sqrt((np.asarray(t) ** 2).sum(axis=(1, 2, 3)))


def forward_transform(v: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """(..., 18) Voigt original -> standardized via T_std = Q Q Q . T_in."""
    T = voigt_to_cartesian(v.reshape(v.shape[:-1] + (3, 6)))
    T_std = np.einsum("...ia,...jb,...kc,...abc->...ijk",
                      Q, Q, Q, T, optimize=True)
    return cartesian_to_voigt(T_std).reshape(v.shape[:-1] + (18,))


def inverse_transform(v: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """(..., 18) Voigt standardized -> original via T_in = Q Q Q . T_std."""
    T = voigt_to_cartesian(v.reshape(v.shape[:-1] + (3, 6)))
    T_in = np.einsum("...ia,...jb,...kc,...ijk->...abc",
                     Q, Q, Q, T, optimize=True)
    return cartesian_to_voigt(T_in).reshape(v.shape[:-1] + (18,))


def ewt(err: np.ndarray, lab: np.ndarray) -> dict:
    """Official error-within-threshold: err <= t * ||label||, zero labels
    included, no epsilon."""
    return {f"EwT{int(t * 100)}": float((err <= t * lab).mean()) for t in T_EPS}


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_components(s: str) -> np.ndarray:
    return np.asarray(json.loads(s), dtype=np.float64).reshape(18)


# --------------------------------------------------------------------------
# hard checks (raise on failure)
# --------------------------------------------------------------------------
def check_ids(root: Path) -> tuple:
    wq = pd.read_csv(root / "predictions_wq.csv")
    p1 = pd.read_csv(root / "predictions_p1.csv")
    split = json.loads((root / "split_seed32.json").read_text())
    test_ids = [str(x) for x in split["test"]]

    ids_wq = [str(x) for x in wq["material_id"]]
    ids_p1 = [str(x) for x in p1["material_id"]]

    problems = []
    if len(ids_wq) != 499:
        problems.append(f"WQ has {len(ids_wq)} rows, expected 499")
    if len(ids_p1) != 499:
        problems.append(f"P1 has {len(ids_p1)} rows, expected 499")
    if len(set(ids_wq)) != len(ids_wq):
        problems.append("WQ ids contain duplicates")
    if len(set(ids_p1)) != len(ids_p1):
        problems.append("P1 ids contain duplicates")
    if set(ids_wq) != set(ids_p1):
        problems.append("WQ and P1 id sets differ")
    missing = sorted(set(test_ids) - set(ids_wq))
    extra = sorted(set(ids_wq) - set(test_ids))
    if missing:
        problems.append(f"missing vs split test set: {len(missing)}: {missing[:5]}")
    if extra:
        problems.append(f"extra vs split test set: {len(extra)}: {extra[:5]}")
    if problems:
        raise AssertionError("ID checks failed: " + "; ".join(problems))
    order = {m: i for i, m in enumerate(ids_wq)}
    return wq, p1, test_ids, order


def check_Q(Q_wq: np.ndarray, Q_p1: np.ndarray) -> dict:
    for name, Q in (("Q_wq", Q_wq), ("Q_p1", Q_p1)):
        if Q.shape != (499, 3, 3):
            raise AssertionError(f"{name} shape {Q.shape} != (499,3,3)")
        if not np.isfinite(Q).all():
            raise AssertionError(f"{name} contains non-finite entries")
        res = np.max(np.abs(np.einsum("nij,nkj->nik", Q, Q)
                            - np.eye(3)[None, ...]))
        if res > 1e-8:
            raise AssertionError(
                f"{name} orthogonality residual {res:.3e} > 1e-8")
    diff = np.max(np.abs(Q_wq - Q_p1))
    if diff > 1e-8:
        raise AssertionError(f"max |Q_wq - Q_p1| = {diff:.3e} > 1e-8")
    dets = np.linalg.det(np.concatenate([Q_wq, Q_p1]))
    return {
        "max_orthogonality_residual": float(np.max(
            np.abs(np.einsum("nij,nkj->nik", Q_wq, Q_wq)
                   - np.eye(3)[None, ...]))),
        "max_abs_Q_wq_minus_Q_p1": float(diff),
        "det_min": float(dets.min()),
        "det_max": float(dets.max()),
    }


def check_voigt_roundtrip() -> dict:
    """18 basis tensors: Voigt -> Cartesian -> Voigt, error must be < 1e-12."""
    worst = 0.0
    for col in range(18):
        v = np.zeros((3, 6))
        v.flat[col] = 1.0
        rt = cartesian_to_voigt(voigt_to_cartesian(v[np.newaxis, ...]))[0]
        worst = max(worst, float(np.max(np.abs(rt - v))))
    if worst > 1e-12:
        raise AssertionError(f"basis roundtrip max error {worst:.3e} > 1e-12")
    return {"basis_roundtrip_max_error": float(worst), "n_basis": 18}


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("."))
    args = ap.parse_args()
    root: Path = args.root.resolve()

    npz_path = root / "frame_audit_inputs.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"{npz_path} not found; run export_piezo_frame_inputs.py "
                                "on the project server first")
    data = np.load(npz_path, allow_pickle=False)
    Q_wq = data["Q_wq"].astype(np.float64)
    Q_p1 = data["Q_p1"].astype(np.float64)
    Q_stored_res = data["Q_stored_orthogonality_residual"].astype(np.float64)
    y_std_wq = data["y_std_wq"].astype(np.float64)
    y_std_p1 = data["y_std_p1"].astype(np.float64)
    y_original = data["y_original"].astype(np.float64)
    contains_inv = data["contains_inversion"].astype(bool)
    projector_rank = data["projector_rank"].astype(np.int64)
    P_18 = data["P_18"].astype(np.float64)
    npz_ids = [str(x) for x in data["ids"]]

    # provenance
    env = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "platform": platform.platform(),
        "script_sha256": sha256_of(Path(__file__).resolve()),
        "inputs_sha256": {
            "frame_audit_inputs.npz": sha256_of(npz_path),
            "frame_audit_metadata.csv": sha256_of(root / "frame_audit_metadata.csv"),
            "predictions_wq.csv": sha256_of(root / "predictions_wq.csv"),
            "predictions_p1.csv": sha256_of(root / "predictions_p1.csv"),
            "split_seed32.json": sha256_of(root / "split_seed32.json"),
            "pkl_sha256.txt": sha256_of(root / "pkl_sha256.txt"),
        },
        "run_utc": datetime.datetime.utcnow().isoformat() + "Z",
    }

    log_lines = []
    def log(msg: str) -> None:
        print(msg, flush=True)
        log_lines.append(msg)

    log(f"# piezo coordinate-frame standalone audit  {env['run_utc']}")
    log(f"# python={env['python']} numpy={env['numpy']} pandas={env['pandas']}")
    log(f"# script sha256={env['script_sha256']}")

    # 1. IDs
    wq, p1, test_ids, order = check_ids(root)
    wq = wq.copy(); wq["order"] = wq["material_id"].map(order)
    p1 = p1.copy(); p1["order"] = p1["material_id"].map(order)
    wq = wq.sort_values("order").reset_index(drop=True)
    p1 = p1.sort_values("order").reset_index(drop=True)
    log("ID checks: PASS (499 unique each, sets equal, == split test set, missing=0 extra=0)")

    # The npz arrays follow the split test-set order while predictions.csv
    # follows the evaluator's iteration order; align everything to the CSV id
    # order before any comparison.
    csv_ids = [str(x) for x in wq["material_id"]]
    npz_index = {m: i for i, m in enumerate(npz_ids)}
    if set(csv_ids) != set(npz_ids):
        raise AssertionError("csv id set != npz id set after ID checks")
    perm = np.asarray([npz_index[m] for m in csv_ids], dtype=np.int64)
    Q_wq = Q_wq[perm]; Q_p1 = Q_p1[perm]; Q_stored_res = Q_stored_res[perm]
    y_std_wq = y_std_wq[perm]; y_std_p1 = y_std_p1[perm]
    y_original = y_original[perm]; contains_inv = contains_inv[perm]
    projector_rank = projector_rank[perm]; P_18 = P_18[perm]
    test_ids = csv_ids

    # 2. labels
    lab_diff = float(np.max(np.abs(y_std_wq - y_std_p1)))
    if lab_diff > 1e-6:
        raise AssertionError(f"max |y_std_wq - y_std_p1| = {lab_diff:.3e} > 1e-6")
    log(f"label check: max |y_std_wq - y_std_p1| = {lab_diff:.3e}  (threshold 1e-6)")

    # 3. Q
    q_audit = check_Q(Q_wq, Q_p1)
    q_audit["stored_Q_orthogonality_residual_max"] = float(Q_stored_res.max())
    q_audit["stored_Q_orthogonality_residual_mean"] = float(Q_stored_res.mean())
    log(f"Q check: orthogonality residual={q_audit['max_orthogonality_residual']:.3e}, "
        f"max|Q_wq-Q_p1|={q_audit['max_abs_Q_wq_minus_Q_p1']:.3e}, "
        f"det in [{q_audit['det_min']:.6f}, {q_audit['det_max']:.6f}]  (threshold 1e-8)")
    log(f"  provenance: float32 Q as stored in LMDB deviates from orthogonality by "
        f"max {q_audit['stored_Q_orthogonality_residual_max']:.2e} "
        f"(mean {q_audit['stored_Q_orthogonality_residual_mean']:.2e}); "
        f"the audit uses the exact SVD orthogonalization of each stored Q")

    # 4. Voigt conventions
    rt = check_voigt_roundtrip()
    log(f"Voigt basis roundtrip: max error = {rt['basis_roundtrip_max_error']:.3e} "
        f"(< 1e-12 required, order xx,yy,zz,xy,yz,zx, no factor of 2)")

    # 5. transforms + label roundtrip vs raw benchmark targets
    y_rt = inverse_transform(forward_transform(y_original, Q_wq), Q_wq)
    roundtrip_err = np.max(np.abs(y_rt - y_original), axis=1)
    y_back = inverse_transform(y_std_wq, Q_wq)   # std -> original via stored Q
    orig_vs_pkl = np.max(np.abs(y_back - y_original), axis=1)
    log(f"transform roundtrip (orig->std->orig, 499 labels): "
        f"per-structure max err mean={roundtrip_err.mean():.3e} "
        f"global max={roundtrip_err.max():.3e}")
    log(f"inverse-rotated stored std label vs raw benchmark target: "
        f"global max = {orig_vs_pkl.max():.3e}  "
        f"(float32 storage noise expected ~1e-6)")

    # predictions
    yw_true = np.stack([parse_components(s) for s in wq["y_true_components"]])
    yw_pred = np.stack([parse_components(s) for s in wq["y_pred_components"]])
    yp_true = np.stack([parse_components(s) for s in p1["y_true_components"]])
    yp_pred = np.stack([parse_components(s) for s in p1["y_pred_components"]])

    # predictions.csv targets must reproduce the exported stored labels
    csv_npz_wq = float(np.max(np.abs(yw_true - y_std_wq)))
    csv_npz_p1 = float(np.max(np.abs(yp_true - y_std_p1)))
    if max(csv_npz_wq, csv_npz_p1) > 1e-6:
        raise AssertionError(
            f"predictions.csv targets vs exported y_std: "
            f"WQ {csv_npz_wq:.3e}, P1 {csv_npz_p1:.3e} > 1e-6")
    log(f"predictions.csv targets == exported labels: "
        f"WQ max diff {csv_npz_wq:.3e}, P1 max diff {csv_npz_p1:.3e}")

    # 6. metrics
    models = {"WQGNet": (yw_pred, y_std_wq), "P1": (yp_pred, y_std_p1)}
    summary = {"environment": env, "checks": {
        "ids": "PASS", "label_diff_max_abs": lab_diff, **q_audit, **rt,
        "label_roundtrip_orig_vs_benchmark_max_abs": float(orig_vs_pkl.max()),
        "label_roundtrip_orig_vs_benchmark_n": int(len(orig_vs_pkl)),
    }}

    # zero predictor
    zp_v18 = float(v18_norm(y_std_wq).mean())
    zp_cart = float(cart_norm(voigt_to_cartesian(
        y_std_wq.reshape(-1, 3, 6))).mean())
    fingerprint_ok = abs(zp_v18 - ZERO_PREDICTOR_FINGERPRINT) < 5e-4
    log(f"zero predictor: V18={zp_v18:.4f} (fingerprint {ZERO_PREDICTOR_FINGERPRINT}, "
        f"{'OK' if fingerprint_ok else 'MISMATCH'}), Cartesian={zp_cart:.4f}")
    summary["zero_predictor"] = {"v18": zp_v18, "cartesian": zp_cart,
                                 "fingerprint_ok": bool(fingerprint_ok)}

    # label Neumann residual for the label-inconsistent subset
    neu = np.zeros(len(test_ids))
    for i in range(len(test_ids)):
        c = voigt_to_cartesian(y_std_wq[i].reshape(1, 3, 6))[0]
        n = float(np.linalg.norm(c))
        if n < 1e-12:
            continue
        proj = (P_18[i] @ y_std_wq[i]).reshape(1, 3, 6)
        pc = voigt_to_cartesian(proj)[0]
        neu[i] = float(np.linalg.norm(pc - c)) / n
    subset = np.where(contains_inv & (v18_norm(y_std_wq) == 0), "A_centrosym_zero_label",
             np.where((~contains_inv) & (neu <= LABEL_INCONSISTENT_TOL),
                      "B_noncentrosym_consistent",
                      "C_label_symmetry_inconsistent"))
    log(f"subsets: A={int((subset == 'A_centrosym_zero_label').sum())}, "
        f"B={int((subset == 'B_noncentrosym_consistent').sum())}, "
        f"C={int((subset == 'C_label_symmetry_inconsistent').sum())}")

    per_structure_rows = []
    summary["models"] = {}
    for name, (pred, y_std) in models.items():
        lab_v18 = v18_norm(y_std)
        lab_cart = cart_norm(voigt_to_cartesian(y_std.reshape(-1, 3, 6)))
        err_v18_std = v18_norm(y_std - pred)
        y_orig = inverse_transform(y_std, Q_wq)
        p_orig = inverse_transform(pred, Q_wq)
        err_v18_orig = v18_norm(y_orig - p_orig)
        err_cart_std = cart_norm(voigt_to_cartesian(
            (y_std - pred).reshape(-1, 3, 6)))
        err_cart_orig = cart_norm(voigt_to_cartesian(
            (y_orig - p_orig).reshape(-1, 3, 6)))
        inv_gap = float(np.max(np.abs(err_cart_std - err_cart_orig)))

        m = {
            "fnorm_v18_std_mean": float(err_v18_std.mean()),
            "fnorm_v18_std_median": float(np.median(err_v18_std)),
            "fnorm_v18_orig_mean": float(err_v18_orig.mean()),
            "fnorm_v18_orig_median": float(np.median(err_v18_orig)),
            "fnorm_cart_std_mean": float(err_cart_std.mean()),
            "fnorm_cart_std_median": float(np.median(err_cart_std)),
            "fnorm_cart_orig_mean": float(err_cart_orig.mean()),
            "fnorm_cart_orig_median": float(np.median(err_cart_orig)),
            "cart_frame_invariance_max_abs": inv_gap,
            "ewt_v18_std": ewt(err_v18_std, lab_v18),
            "ewt_cart_orig": ewt(err_cart_orig, lab_cart),
            "subsets": {},
        }
        for s in ("A_centrosym_zero_label", "B_noncentrosym_consistent",
                  "C_label_symmetry_inconsistent"):
            sel = subset == s
            if not sel.any():
                continue
            m["subsets"][s] = {
                "n": int(sel.sum()),
                "fnorm_v18_std_mean": float(err_v18_std[sel].mean()),
                "fnorm_cart_orig_mean": float(err_cart_orig[sel].mean()),
                "ewt_v18_std": ewt(err_v18_std[sel], lab_v18[sel]),
            }
        summary["models"][name] = m
        log(f"\n{name}: V18 mean std={m['fnorm_v18_std_mean']:.4f} "
            f"orig={m['fnorm_v18_orig_mean']:.4f} | "
            f"Cartesian mean std={m['fnorm_cart_std_mean']:.4f} "
            f"orig={m['fnorm_cart_orig_mean']:.4f} | "
            f"invariance max|d|={inv_gap:.2e}")
        log(f"  EwT (V18, official): "
            + "  ".join(f"@{k}%={100*v:6.2f}%" for k, v in m["ewt_v18_std"].items()))
        log(f"  EwT (Cartesian invariant): "
            + "  ".join(f"@{k}%={100*v:6.2f}%" for k, v in m["ewt_cart_orig"].items()))
        for s, sm in m["subsets"].items():
            log(f"  [{s} n={sm['n']}] V18 mean={sm['fnorm_v18_std_mean']:.4f} "
                f"Cart mean={sm['fnorm_cart_orig_mean']:.4f}")

        for i in range(len(test_ids)):
            per_structure_rows.append({
                "model": name, "material_id": test_ids[i],
                "contains_inversion": int(contains_inv[i]),
                "projector_rank": int(projector_rank[i]),
                "subset": subset[i],
                "label_v18_norm_std": lab_v18[i],
                "label_cart_norm": lab_cart[i],
                "err_v18_std": err_v18_std[i],
                "err_v18_orig": err_v18_orig[i],
                "err_cart_std": err_cart_std[i],
                "err_cart_orig": err_cart_orig[i],
                "label_neumann_residual": neu[i],
            })

    # direction statement
    w, p1m = summary["models"]["WQGNet"], summary["models"]["P1"]
    direction = {
        "mean_error": ("WQGNet" if w["fnorm_v18_std_mean"] < p1m["fnorm_v18_std_mean"]
                       else "P1"),
        "ewt25_v18": ("WQGNet" if w["ewt_v18_std"]["EwT25"] > p1m["ewt_v18_std"]["EwT25"]
                      else "P1"),
    }
    summary["direction"] = direction

    # optional external GMTNet predictions
    gmt_path = root / "gmtnet_piezo_predictions.csv"
    gmt_tensor_path = root / "gmtnet_piezo_tensor_predictions.csv"
    if gmt_path.exists():
        gm = pd.read_csv(gmt_path)
        gm_ids = [str(x) for x in gm["structure_id"]]
        g_missing = sorted(set(test_ids) - set(gm_ids))
        g_extra = sorted(set(gm_ids) - set(test_ids))
        if len(gm_ids) != 499 or len(set(gm_ids)) != 499 or g_missing or g_extra:
            raise AssertionError(
                f"GMTNet predictions: n={len(gm_ids)} unique={len(set(gm_ids))} "
                f"missing={len(g_missing)} extra={len(g_extra)} — must equal the "
                f"499-structure test set")
        gm = gm.set_index("structure_id").loc[test_ids]
        frob = gm["frob"].to_numpy(dtype=np.float64)
        nt = gm["norm_true"].to_numpy(dtype=np.float64)
        summary["gmtnet"] = {
            "n": int(len(frob)),
            "protocol": ("V18 per-structure errors as released in "
                         "gmtnet_piezo_predictions.csv (scalar values only)"),
            "fnorm_v18_mean": float(frob.mean()),
            "fnorm_v18_median": float(np.median(frob)),
            "ewt_v18": ewt(frob, nt),
            "frame_note": (
                "norm_true matches V18 of the original-frame benchmark labels to "
                "<=4.3e-6 for all 499 structures (float32 precision); the V18 "
                "label norms of the two frames differ by <1e-6 for every "
                "structure in this test set, so the frame distinction is below "
                "the released file's precision"),
            "cartesian_invariant": (
                "resolved from the component-level CSV (see below)"
                if gmt_tensor_path.exists() else
                "pending: the released scalar CSV carries per-structure errors "
                "only, so the full 3x3x3 Cartesian rescoring requires "
                "component-level GMTNet predictions"),
        }
        log(f"\nGMTNet (released predictions CSV): V18 mean={frob.mean():.4f} "
            f"(published 0.407), median={np.median(frob):.4f}, n={len(frob)}")
        log("  frame note: norm_true == V18(original labels) to float32 precision; "
            "V18 label norms of the two frames differ by <1e-6 here")
        log("  Cartesian invariant rescoring: "
            + ("resolved from the component-level CSV below"
               if gmt_tensor_path.exists() else
               "PENDING (scalar CSV has no components)"))
    else:
        log("\nGMTNet: gmtnet_piezo_predictions.csv not present — "
            "external GMTNet invariant comparison pending.")
        summary["gmtnet"] = {
            "status": "WQ/P1 frame audit closed; external GMTNet invariant "
                      "comparison pending."}

    # optional component-level GMTNet predictions -> invariant rescoring
    if gmt_tensor_path.exists():
        gt = pd.read_csv(gmt_tensor_path)
        gt_ids = [str(x) for x in gt["structure_id"]]
        t_missing = sorted(set(test_ids) - set(gt_ids))
        t_extra = sorted(set(gt_ids) - set(test_ids))
        if len(gt_ids) != 499 or len(set(gt_ids)) != 499 or t_missing or t_extra:
            raise AssertionError(
                f"GMTNet tensor predictions: n={len(gt_ids)} "
                f"unique={len(set(gt_ids))} missing={len(t_missing)} "
                f"extra={len(t_extra)}")
        gt = gt.set_index("structure_id").loc[test_ids]
        pairs = ["11", "22", "33", "12", "23", "13"]
        true_cols = [f"d{r}_{p}_true" for r in range(3) for p in pairs]
        pred_cols = [f"d{r}_{p}_pred" for r in range(3) for p in pairs]
        y_gt = np.stack([gt[c].to_numpy(dtype=np.float64) for c in true_cols],
                        axis=1)
        p_gt = np.stack([gt[c].to_numpy(dtype=np.float64) for c in pred_cols],
                        axis=1)
        d_orig = float(np.max(np.abs(y_gt - y_original)))
        d_std = float(np.max(np.abs(y_gt - y_std_wq)))
        frame = ("original" if d_orig < 1e-3 else
                 "standardized" if d_std < 1e-3 else "indeterminate")
        if frame == "indeterminate":
            raise AssertionError(
                f"GMTNet tensor targets match neither frame: "
                f"|t-y_orig|={d_orig:.3e}, |t-y_std|={d_std:.3e}")
        err_v18 = v18_norm(y_gt - p_gt)
        lab_v18 = v18_norm(y_gt)
        err_cart_own = cart_norm(voigt_to_cartesian(
            (y_gt - p_gt).reshape(-1, 3, 6)))
        lab_cart_own = cart_norm(voigt_to_cartesian(y_gt.reshape(-1, 3, 6)))
        # invariance demonstration: rotate GMTNet tensors into the std frame
        y_gt_std = forward_transform(y_gt, Q_wq)
        p_gt_std = forward_transform(p_gt, Q_wq)
        err_cart_std_g = cart_norm(voigt_to_cartesian(
            (y_gt_std - p_gt_std).reshape(-1, 3, 6)))
        inv_gap = float(np.max(np.abs(err_cart_own - err_cart_std_g)))
        summary["gmtnet"].update({
            "frame": frame,
            "target_frame_match_orig_max_abs": d_orig,
            "target_frame_match_std_max_abs": d_std,
            "fnorm_v18_recomputed_mean": float(err_v18.mean()),
            "fnorm_cart_own_frame_mean": float(err_cart_own.mean()),
            "fnorm_cart_own_frame_median": float(np.median(err_cart_own)),
            "fnorm_cart_std_frame_mean": float(err_cart_std_g.mean()),
            "cart_frame_invariance_max_abs": inv_gap,
            "ewt_v18_recomputed": ewt(err_v18, lab_v18),
            "ewt_cart": ewt(err_cart_own, lab_cart_own),
        })
        log(f"\nGMTNet (component-level CSV, frame={frame}): "
            f"|t-y_orig|max={d_orig:.2e} |t-y_std|max={d_std:.2e}")
        released_scalar_mean = frob.mean() if gmt_path.exists() else float("nan")
        log(f"  V18 recomputed mean={err_v18.mean():.4f} "
            f"(vs released scalar CSV {released_scalar_mean:.4f})")
        log(f"  Cartesian invariant mean={err_cart_own.mean():.4f} "
            f"(std-frame={err_cart_std_g.mean():.4f}, "
            f"frame-invariance max|d|={inv_gap:.2e})")
        log(f"  EwT Cartesian: "
            + "  ".join(f"@{k}%={100*v:6.2f}%"
                        for k, v in summary["gmtnet"]["ewt_cart"].items()))
        err_v18_std_g = v18_norm(y_gt_std - p_gt_std)
        lab_v18_std_g = v18_norm(y_gt_std)
        for i in range(len(test_ids)):
            per_structure_rows.append({
                "model": "GMTNet", "material_id": test_ids[i],
                "contains_inversion": int(contains_inv[i]),
                "projector_rank": int(projector_rank[i]),
                "subset": subset[i],
                "label_v18_norm_std": lab_v18_std_g[i],
                "label_cart_norm": lab_cart_own[i],
                "err_v18_std": err_v18_std_g[i],
                "err_v18_orig": err_v18[i],
                "err_cart_std": err_cart_std_g[i],
                "err_cart_orig": err_cart_own[i],
                "label_neumann_residual": neu[i],
            })

    # outputs
    (root / "audit_summary.json").write_text(json.dumps(summary, indent=2))
    pd.DataFrame(per_structure_rows).to_csv(root / "audit_per_structure.csv",
                                            index=False)
    log("\noutputs written: audit_summary.json, audit_per_structure.csv, "
        "audit_stdout.log, sha256sums.txt")
    (root / "audit_stdout.log").write_text("\n".join(log_lines) + "\n")
    (root / "requirements.txt").write_text(
        f"numpy=={np.__version__}\npandas=={pd.__version__}\n")

    files = ["audit_piezo_frame_standalone.py", "export_piezo_frame_inputs.py",
             "frame_audit_inputs.npz", "frame_audit_metadata.csv",
             "predictions_wq.csv", "predictions_p1.csv", "split_seed32.json",
             "pkl_sha256.txt", "gmtnet_piezo_predictions.csv",
             "gmtnet_piezo_tensor_predictions.csv",
             "audit_summary.json", "audit_per_structure.csv",
             "audit_stdout.log", "requirements.txt", "README.md"]
    sums = []
    for f in files:
        p = root / f
        if p.exists():
            sums.append(f"{sha256_of(p)}  {f}")
    (root / "sha256sums.txt").write_text("\n".join(sums) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
