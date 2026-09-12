#!/usr/bin/env python3
"""Stage 1 of the piezo coordinate-frame audit: export LMDB fields to plain files.

Runs ONLY on the project server (requires the project environment). After this
script completes, the downstream standalone audit
(``audit_piezo_frame_standalone.py``) must not depend on LMDB, torch or
wyckoff_gnn.

Exported files (all under OUT_DIR):
    frame_audit_inputs.npz
    frame_audit_metadata.csv
    predictions_wq.csv          (copy of the WQ run predictions.csv)
    predictions_p1.csv          (copy of the P1 run predictions.csv)
    split_seed32.json           (copy of the official split)
    pkl_sha256.txt              (sha256 of the raw GMTNet benchmark pickle)

npz arrays, all indexed by the 499 official test ids in split order:
    ids                    (499,) str
    split                  (499,) str          always "test" here
    Q_wq                   (499,3,3) float64   tensor_frame_rotation from WQ cache
    Q_p1                   (499,3,3) float64   tensor_frame_rotation from P1 cache
    y_std_wq               (499,18) float64    stored y_tensor, WQ stdframe cache
    y_std_p1               (499,18) float64    stored y_tensor, P1 stdframe cache
    y_original             (499,18) float64    piezoelectric_C_m2 from the raw
                                               GMTNet benchmark pickle (the
                                               original input-Cartesian frame)
    contains_inversion     (499,) bool
    projector_rank         (499,) int64
    P_18                   (499,18,18) float64 crystal point-group projector
                                               (needed to build the
                                               label-symmetry-consistency subset)

Conventions (documented so the standalone audit can re-implement them):
    Voigt column order: xx, yy, zz, xy, yz, zx (VASP PIEZO), no factor of 2.
    Forward transform (production): T_std[i,j,k] = Q[i,a]Q[j,b]Q[k,c]T_in[a,b,c]
    Inverse: T_in[a,b,c] = Q[i,a]Q[j,b]Q[k,c]T_std[i,j,k]  (Q orthogonal)
"""
from __future__ import annotations

import csv
import hashlib
import json
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wyckoff_gnn.data.lmdb_cache import LMDBReader  # noqa: E402

WQ_CACHE = "data/processed/gmtnet_piezo_wyckoff_stdframe"
P1_CACHE = "data/processed/gmtnet_piezo_p1_stdframe"
ORIGINAL_PKL = "data/processed/gmtnet_piezo/gmtnet_piezo_filtered.pkl"
SPLIT = "data/processed/gmtnet_piezo/split_seed32.json"
WQ_PRED = "results/wqgnet_piezo_wyckoff_s2/predictions.csv"
P1_PRED = "results/p1_piezo_s2/predictions.csv"
OUT_DIR = Path("results/piezo_frame_audit/standalone_inputs")


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    out = Path(OUT_DIR)
    out.mkdir(parents=True, exist_ok=True)

    with open(SPLIT) as f:
        split = json.load(f)
    test_ids = [str(x) for x in split["test"]]
    if len(test_ids) != 499:
        raise RuntimeError(f"split test set has {len(test_ids)} ids, expected 499")

    with open(ORIGINAL_PKL, "rb") as f:
        raw_pkl = pickle.load(f)

    rw, rp = LMDBReader(WQ_CACHE), LMDBReader(P1_CACHE)

    arrays = {
        "ids": test_ids,
        "split": ["test"] * len(test_ids),
        "Q_wq": np.zeros((len(test_ids), 3, 3)),
        "Q_p1": np.zeros((len(test_ids), 3, 3)),
        "Q_stored_orthogonality_residual": np.zeros(len(test_ids)),
        "y_std_wq": np.zeros((len(test_ids), 18)),
        "y_std_p1": np.zeros((len(test_ids), 18)),
        "y_original": np.zeros((len(test_ids), 18)),
        "contains_inversion": np.zeros(len(test_ids), dtype=bool),
        "projector_rank": np.zeros(len(test_ids), dtype=np.int64),
        "P_18": np.zeros((len(test_ids), 18, 18)),
    }

    def _orthogonalize(Q: np.ndarray) -> np.ndarray:
        """Nearest orthogonal matrix (SVD), sign of det preserved.

        The LMDB stores Q as float32, so the raw matrix is orthogonal only to
        ~1e-7.  The audit uses the exact orthogonalization; the per-id residual
        of the stored matrix is exported separately as provenance.
        """
        U, _, Vt = np.linalg.svd(Q)
        return U @ Vt

    missing = []
    for i, mid in enumerate(test_ids):
        gw, gp = rw.get(mid), rp.get(mid)
        if gw is None or gp is None:
            missing.append(mid)
            continue
        Qw_raw = np.asarray(gw["tensor_frame_rotation"], dtype=np.float64)
        Qp_raw = np.asarray(gp["tensor_frame_rotation"], dtype=np.float64)
        arrays["Q_wq"][i] = _orthogonalize(Qw_raw)
        arrays["Q_p1"][i] = _orthogonalize(Qp_raw)
        res = float(np.max(np.abs(Qw_raw.T @ Qw_raw - np.eye(3))))
        arrays["Q_stored_orthogonality_residual"][i] = res
        arrays["y_std_wq"][i] = np.asarray(gw["y_tensor"], dtype=np.float64).reshape(18)
        arrays["y_std_p1"][i] = np.asarray(gp["y_tensor"], dtype=np.float64).reshape(18)
        arrays["contains_inversion"][i] = bool(gw["contains_inversion"])
        arrays["projector_rank"][i] = int(gw["projector_rank"])
        arrays["P_18"][i] = np.asarray(gw["tensor_point_group_projector"],
                                      dtype=np.float64).reshape(18, 18)
        entry = raw_pkl.get(mid)
        if entry is None:
            missing.append(mid)
            continue
        arrays["y_original"][i] = np.asarray(entry["piezoelectric_C_m2"],
                                             dtype=np.float64).reshape(18)
    if missing:
        raise RuntimeError(f"{len(missing)} test ids missing from caches/pkl: {missing[:10]}")

    np.savez_compressed(out / "frame_audit_inputs.npz", **arrays)

    with open(out / "frame_audit_metadata.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["material_id", "split", "contains_inversion", "projector_rank",
                    "Q_stored_orthogonality_residual"])
        for i, mid in enumerate(test_ids):
            w.writerow([mid, "test", int(arrays["contains_inversion"][i]),
                        int(arrays["projector_rank"][i]),
                        float(arrays["Q_stored_orthogonality_residual"][i])])

    for src, dst in ((WQ_PRED, "predictions_wq.csv"), (P1_PRED, "predictions_p1.csv")):
        shutil.copyfile(src, out / dst)
    shutil.copyfile(SPLIT, out / "split_seed32.json")
    (out / "pkl_sha256.txt").write_text(
        f"{sha256_of(Path(ORIGINAL_PKL))}  {Path(ORIGINAL_PKL).name}\n")

    print(f"exported to {out}")
    print(f"  n_test={len(test_ids)}  missing=0")
    print(f"  Q_wq finite: {np.isfinite(arrays['Q_wq']).all()}, "
          f"Q_p1 finite: {np.isfinite(arrays['Q_p1']).all()}")
    print(f"  max |Q_wq - Q_p1| = {np.max(np.abs(arrays['Q_wq'] - arrays['Q_p1'])):.3e}")
    print(f"  max |y_std_wq - y_std_p1| = "
          f"{np.max(np.abs(arrays['y_std_wq'] - arrays['y_std_p1'])):.3e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
