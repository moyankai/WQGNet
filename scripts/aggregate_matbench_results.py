#!/usr/bin/env python
"""Aggregate 5-fold Matbench benchmark results into mean±std MAE.

Usage:
    python scripts/aggregate_matbench_results.py <bench_dir>

    # Example:
    python scripts/aggregate_matbench_results.py results/bench-matbench_mp_e_form_5fold-*
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np

# Matbench v0.1 published SOTA (regression MAE) — from
# https://matbench.materialsproject.org/ (as of 2024-2025).
SOTA_REGRESSION = {
    "matbench_mp_e_form": {"MODNet": 0.033, "coGN": 0.017, "coNGN": 0.018, "ALIGNN": 0.022},
    "matbench_mp_gap":    {"coGN": 0.156, "MegNet": 0.199, "MODNet": 0.211, "ALIGNN": 0.186},
    "matbench_perovskites": {"coGN": 0.027, "MODNet": 0.091, "ALIGNN": 0.029},
    "matbench_jdft2d":    {"MODNet": 33.19, "coGN": 36.16},
    "matbench_dielectric": {"MODNet": 0.271, "coGN": 0.309},
    "matbench_log_gvrh":  {"MODNet": 0.056, "coGN": 0.066},
    "matbench_log_kvrh":  {"MODNet": 0.050, "coGN": 0.053},
    "matbench_phonons":   {"MODNet": 34.28, "coGN": 29.71},
}


def aggregate(bench_dir: Path):
    fold_results = []
    for fold_dir in sorted(bench_dir.glob("fold*")):
        result_path = fold_dir / "seed_0" / "results.json"
        if not result_path.exists():
            print(f"  ⚠ {fold_dir.name}: results.json missing (not finished)")
            continue
        with open(result_path) as f:
            r = json.load(f)
        mae = r.get("test_mae")
        r2 = r.get("test_r2")
        params = r.get("params", 0)
        task = r.get("task") or r.get("dataset", "unknown")
        epoch = r.get("best_epoch", "?")
        fold_results.append({
            "fold": fold_dir.name, "test_mae": mae, "test_r2": r2,
            "params": params, "task": task, "best_epoch": epoch,
        })

    if not fold_results:
        print(f"❌ No completed folds in {bench_dir}")
        return

    task_name = fold_results[0]["task"]
    maes = np.array([r["test_mae"] for r in fold_results if r["test_mae"] is not None])
    r2s = np.array([r["test_r2"] for r in fold_results if r["test_r2"] is not None])

    print(f"\n{'=' * 60}")
    print(f"Task: {task_name}")
    print(f"Benchmark dir: {bench_dir}")
    print(f"Completed folds: {len(fold_results)}/5")
    print(f"{'=' * 60}")

    print(f"{'Fold':<10} {'test_MAE':>12} {'test_R2':>10} {'params':>12} {'epoch':>8}")
    print("-" * 55)
    for r in fold_results:
        mae_str = f"{r['test_mae']:.4f}" if r["test_mae"] is not None else "—"
        r2_str = f"{r['test_r2']:.4f}" if r["test_r2"] is not None else "—"
        print(f"{r['fold']:<10} {mae_str:>12} {r2_str:>10} {r['params']:>12,} {r['best_epoch']:>8}")

    if len(maes) > 0:
        mean_mae = maes.mean()
        std_mae = maes.std()
        mean_r2 = r2s.mean() if len(r2s) > 0 else float("nan")
        print("-" * 55)
        print(f"{'MEAN':<10} {mean_mae:>12.4f} {mean_r2:>10.4f}")
        print(f"{'STD':<10} {std_mae:>12.4f}")

        # Compare to SOTA.
        if task_name in SOTA_REGRESSION:
            print(f"\n{'=' * 60}")
            print(f"Comparison to Matbench SOTA (as of 2024):")
            print(f"{'=' * 60}")
            sota_dict = SOTA_REGRESSION[task_name]
            best_sota_name = min(sota_dict, key=sota_dict.get)
            best_sota_mae = sota_dict[best_sota_name]
            print(f"  Our MAE:    {mean_mae:.4f} ± {std_mae:.4f}")
            print(f"  Best SOTA:  {best_sota_mae:.4f}  ({best_sota_name})")
            print(f"  Gap:        {(mean_mae - best_sota_mae):+.4f} "
                  f"({100 * (mean_mae - best_sota_mae) / best_sota_mae:+.1f}%)")
            print(f"  All published:")
            for name, m in sorted(sota_dict.items(), key=lambda kv: kv[1]):
                marker = " ★" if name == best_sota_name else ""
                print(f"    {name:15s} {m:.4f}{marker}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    # Support glob patterns.
    paths = []
    for arg in sys.argv[1:]:
        matched = glob.glob(arg)
        paths.extend(matched or [arg])

    for p in paths:
        bench_dir = Path(p)
        if not bench_dir.exists():
            print(f"❌ Not found: {bench_dir}")
            continue
        aggregate(bench_dir)


if __name__ == "__main__":
    main()
