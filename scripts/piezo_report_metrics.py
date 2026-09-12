#!/usr/bin/env python3
"""P7 test metrics for the projected piezo runs.

Official rules, locked:
  Fnorm : V18 -- Frobenius over the 18 stored Voigt components.
  EwT   : err <= threshold * ||label||, over all 499 test samples, no epsilon
          and no zero-label exception. A zero label therefore demands a
          mathematically exact zero prediction.
A nonzero-only EwT is printed too, but strictly as a diagnostic.

Usage:
    python scripts/piezo_report_metrics.py RUN_DIR [RUN_DIR ...]
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/public/home/moyk/wyckoff_gnn")

from wyckoff_gnn.data.lmdb_cache import LMDBReader
from wyckoff_gnn.models.unified_equivariant.crystal_tensor_projection import (
    project_cartesian_tensor,
)
from wyckoff_gnn.models.unified_equivariant.tensor_adapter import (
    voigt36_to_cartesian333,
)
from wyckoff_gnn.data.tensor_frame import (
    _piezo_voigt_to_cartesian,
    _piezo_cartesian_to_voigt,
    _PIEZO_VOIGT_PAIRS,
)

CACHE = "data/processed/gmtnet_piezo_wyckoff_stdframe"
P1CACHE = "data/processed/gmtnet_piezo_p1_stdframe"
ORIGINAL_PKL = "data/processed/gmtnet_piezo/gmtnet_piezo_filtered.pkl"
T = (0.25, 0.10, 0.05, 0.02)
ZERO_FINGERPRINT = 0.4372


def v18(a):
    return np.linalg.norm(np.asarray(a).reshape(len(a), -1), axis=1)


def v18_matrix(a):
    """(N, 18) -> (N,) V18 norm."""
    return np.linalg.norm(np.asarray(a).reshape(len(a), -1), axis=1)


def cart333(v):
    """(N, 18) Voigt -> (N, 3, 3, 3) Cartesian, symmetric in last two axes."""
    v = np.asarray(v, dtype=np.float64).reshape(len(v), 3, 6)
    out = np.zeros((len(v), 3, 3, 3), dtype=np.float64)
    for col, (i, k) in enumerate(_PIEZO_VOIGT_PAIRS):
        out[:, :, i, k] = v[:, :, col]
        if i != k:
            out[:, :, k, i] = v[:, :, col]
    return out


def cart_frobenius(a):
    """Rotation-invariant full 3x3x3 Frobenius norm of (N, 3, 3, 3)."""
    return np.sqrt((a ** 2).sum(axis=(1, 2, 3)))


def inverse_rotate_to_original(voigt_std, Q_list):
    """Voigt (N,18) in std frame -> Voigt (N,18) in original input frame.

    Inverse of ``transform_piezo_rank3_voigt``: with Q orthogonal,
    T_orig[a,b,c] = Q[i,a] Q[j,b] Q[k,c] T_std[i,j,k].
    """
    out = np.empty_like(np.asarray(voigt_std, dtype=np.float64).reshape(len(voigt_std), 3, 6))
    for n, (v, Q) in enumerate(zip(voigt_std, Q_list)):
        T = _piezo_voigt_to_cartesian(np.asarray(v, dtype=np.float64).reshape(3, 6))
        T_orig = np.einsum("ia,jb,kc,ijk->abc", Q, Q, Q, T, optimize=True)
        out[n] = _piezo_cartesian_to_voigt(T_orig)
    return out


def rotation_invariant_block(ids, y_std, p_std, cache):
    """Rescore in the original frame and with the invariant Cartesian norm.

    Returns dict with V18-original and full-tensor Frobenius metrics plus the
    label round-trip audit against the original benchmark targets.
    """
    Q_list = [np.asarray(cache.get(m)["tensor_frame_rotation"], dtype=np.float64)
              for m in ids]
    y_orig = inverse_rotate_to_original(y_std, Q_list)
    p_orig = inverse_rotate_to_original(p_std, Q_list)

    # invariant full-tensor error, computed in BOTH frames to demonstrate
    # frame independence
    err_cart_std = cart_frobenius(cart333(y_std) - cart333(p_std))
    err_cart_orig = cart_frobenius(cart333(y_orig) - cart333(p_orig))
    lab_cart_orig = cart_frobenius(cart333(y_orig))

    res = {
        "fnorm_v18_std": float(v18_matrix(y_std - p_std).mean()),
        "fnorm_v18_orig": float(v18_matrix(y_orig - p_orig).mean()),
        "fnorm_cart_std": float(err_cart_std.mean()),
        "fnorm_cart_orig": float(err_cart_orig.mean()),
        "cart_frame_invariance_max_abs": float(
            np.max(np.abs(err_cart_std - err_cart_orig))),
        "cart_median": float(np.median(err_cart_orig)),
        "ewt_cart": {t: float((err_cart_orig <= t * lab_cart_orig).mean())
                     for t in T},
    }

    # label round-trip audit against the raw benchmark targets (original frame)
    with open(ORIGINAL_PKL, "rb") as f:
        import pickle
        raw = pickle.load(f)
    resid, n_checked, n_missing = [], 0, 0
    for m, y_o in zip(ids, y_orig):
        e = raw.get(m)
        if e is None:
            n_missing += 1
            continue
        target = np.asarray(e.get("piezoelectric_C_m2"), dtype=np.float64)
        if target.size != 18:
            continue
        resid.append(float(np.max(np.abs(target.reshape(18) - y_o.reshape(18)))))
        n_checked += 1
    res["label_roundtrip_max_abs"] = float(max(resid)) if resid else float("nan")
    res["label_roundtrip_n_checked"] = n_checked
    res["label_roundtrip_n_missing"] = n_missing
    return res


def official_ewt(err, ln):
    return {t: float((err <= t * ln).mean()) for t in T}


def diag_ewt(err, ln):
    m = ln > 0
    return {t: float((err[m] / ln[m] < t).mean()) for t in T}


def show(tag, err, ln, n_all):
    o = official_ewt(err, ln)
    d = diag_ewt(err, ln)
    print(f"  {tag:<22} Fnorm={err.mean():.4f} median={np.median(err):.4f}")
    print(f"    {'official EwT':<20} " +
          "  ".join(f"@{int(t*100)}%={100*o[t]:6.2f}%" for t in T) + f"   (n={n_all})")
    print(f"    {'diagnostic nonzero':<20} " +
          "  ".join(f"@{int(t*100)}%={100*d[t]:6.2f}%" for t in T)
          + f"   (n={int((ln>0).sum())})")
    return o


def main():
    runs = sys.argv[1:]
    rw, rp = LMDBReader(CACHE), LMDBReader(P1CACHE)

    for run in runs:
        rows = list(csv.DictReader(open(Path(run) / "predictions.csv")))
        ids = [r["material_id"] for r in rows]
        y = np.array([json.loads(r["y_true_components"]) for r in rows])
        p = np.array([json.loads(r["y_pred_components"]) for r in rows])
        has_raw = "y_pred_raw_components" in rows[0]
        praw = (np.array([json.loads(r["y_pred_raw_components"]) for r in rows])
                if has_raw else None)
        ln = v18(y)
        err = v18(y - p)

        print(f"\n######## {run}   n={len(ids)}")
        zp = v18(y - np.zeros_like(y)).mean()
        print(f"  zero-predictor Fnorm fingerprint = {zp:.4f} "
              f"(must be {ZERO_FINGERPRINT})"
              f"  [{'OK' if abs(zp - ZERO_FINGERPRINT) < 5e-4 else 'MISMATCH -> STOP'}]")

        print("\n  --- OFFICIAL, all test samples ---")
        show("projected", err, ln, len(ids))
        if has_raw:
            show("raw (pre-projection)", v18(y - praw), ln, len(ids))

        # groups
        inv = np.array([bool(rw.get(m)["contains_inversion"]) for m in ids])
        neu = np.zeros(len(ids))
        for i, m in enumerate(ids):
            c = voigt36_to_cartesian333(
                torch.tensor(y[i].reshape(1, 3, 6), dtype=torch.float64))[0]
            n = float(c.norm())
            if n < 1e-12:
                continue
            R = torch.tensor(
                np.asarray(rw.get(m)["tensor_point_group_projector"]),
                dtype=torch.float64)
            proj = (R @ torch.tensor(y[i].reshape(-1), dtype=torch.float64)
                    ).reshape(1, 3, 6)
            pc = voigt36_to_cartesian333(proj)[0]
            neu[i] = float((pc - c).norm()) / n
        gA = inv & (ln == 0)
        gB = (~inv) & (neu <= 1e-3)
        gC = neu > 1e-3

        print(f"\n  --- GROUP A centrosymmetric (n={int(gA.sum())}) ---")
        pn = v18(p[gA])
        n_exact = int(np.all(p[gA] == 0.0, axis=1).sum())
        print(f"    Fnorm={v18(y[gA]-p[gA]).mean():.6f} "
              f"max |prediction|={pn.max():.3e}  "
              f"element-wise exact zero: {n_exact}/{int(gA.sum())}"
              f"  [{'PASS' if n_exact == int(gA.sum()) else 'FAIL'}]")

        for name, mask in (("B non-centrosym, label-consistent", gB),
                           ("C label symmetry-inconsistent", gC)):
            if not mask.any():
                continue
            print(f"\n  --- GROUP {name} (n={int(mask.sum())}) ---")
            show("projected", v18(y[mask] - p[mask]), ln[mask], int(mask.sum()))

        if "JVASP-5197" in ids:
            i = ids.index("JVASP-5197")
            g = rw.get("JVASP-5197")
            print(f"\n  --- JVASP-5197 (SG19, class 222) ---")
            print(f"    projector_rank={int(g['projector_rank'])} "
                  f"contains_inversion={bool(g['contains_inversion'])} "
                  f"label||.||={ln[i]:.4f} pred||.||={v18(p[i:i+1])[0]:.4e} "
                  f"Fnorm={v18(y[i:i+1]-p[i:i+1])[0]:.4e}")

        if has_raw:
            fb = v18(praw - p) / np.maximum(v18(praw), 1e-30)
            print(f"\n  --- forbidden-subspace fraction of the raw output ---")
            print(f"    ||raw - Pi(raw)|| / ||raw||: median={np.median(fb):.4f} "
                  f"mean={fb.mean():.4f}  (post-hoc, pre-retrain WQ median was 0.951)")
            nz = ln > 0
            print(f"    on nonzero-label subset: median={np.median(fb[nz]):.4f}")

        # --- rotation-invariant rescoring (original frame + full Cartesian) ---
        inv = rotation_invariant_block(ids, y, p, rw)
        print("\n  --- ROTATION-INVARIANT RESCORING ---")
        print(f"    Fnorm V18  std frame (reported): {inv['fnorm_v18_std']:.4f}")
        print(f"    Fnorm V18  original frame:       {inv['fnorm_v18_orig']:.4f}")
        print(f"    Fnorm full Cartesian (std):      {inv['fnorm_cart_std']:.4f}")
        print(f"    Fnorm full Cartesian (original): {inv['fnorm_cart_orig']:.4f}")
        print(f"    Cartesian frame-invariance max |d|: {inv['cart_frame_invariance_max_abs']:.2e}")
        print(f"    EwT with invariant norm:         "
              + "  ".join(f"@{int(t*100)}%={100*inv['ewt_cart'][t]:6.2f}%"
                         for t in T))
        print(f"    label roundtrip vs original pkl: "
              f"max abs = {inv['label_roundtrip_max_abs']:.2e} "
              f"(n={inv['label_roundtrip_n_checked']}, "
              f"missing={inv['label_roundtrip_n_missing']})")

        # Neumann residual of the prediction itself
        res = []
        for i, m in enumerate(ids):
            c = voigt36_to_cartesian333(
                torch.tensor(p[i].reshape(1, 3, 6), dtype=torch.float64))[0]
            n = float(c.norm())
            if n < 1e-12:
                continue
            R = torch.tensor(np.asarray(rw.get(m)["tensor_point_group_projector"]),
                             dtype=torch.float64)
            q = (R @ torch.tensor(p[i].reshape(-1), dtype=torch.float64)).reshape(1, 3, 6)
            res.append(float((voigt36_to_cartesian333(q)[0] - c).norm()) / n)
        res = np.array(res)
        print(f"\n  --- Neumann residual of the PREDICTION (n={len(res)}) ---")
        print(f"    median={np.median(res):.3e} max={res.max():.3e}"
              f"  [{'PASS' if res.max() < 1e-6 else 'FAIL'}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
