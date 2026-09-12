#!/usr/bin/env python3
"""Compute Fnorm (mean Frobenius norm of error tensors) from predictions CSV.

Usage:
    python scripts/compute_fnorm.py RESULTS_DIR [--by-crystal-system]

Examples:
    python scripts/compute_fnorm.py results/eval-jarvis_leaderboard-...-2026/
    python scripts/compute_fnorm.py results/eval-jarvis_leaderboard-...-2026/ --by-crystal-system
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

CRYSTAL_SYSTEMS = {
    "triclinic":  (1, 2),
    "monoclinic": (3, 15),
    "orthorhombic": (16, 74),
    "tetragonal": (75, 142),
    "hexagonal":  (143, 194),  # includes trigonal (143-167) and hexagonal (168-194)
    "cubic":      (195, 230),
}


EWT_THRESHOLDS = (0.25, 0.10, 0.05, 0.02)


def crystal_system(sg: int) -> str:
    for name, (lo, hi) in CRYSTAL_SYSTEMS.items():
        if lo <= sg <= hi:
            return name
    return "unknown"


def compute_fnorm(predictions_csv: Path, by_crystal_system: bool = False):
    fnorms = []
    label_norms = []
    sg_list = []

    with open(predictions_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            y_true = json.loads(row["y_true_components"])
            y_pred = json.loads(row["y_pred_components"])
            true_arr = np.array(y_true)
            diff = true_arr - np.array(y_pred)
            fnorm = np.sqrt((diff ** 2).sum())
            fnorms.append(fnorm)
            label_norms.append(np.sqrt((true_arr ** 2).sum()))
            sg = int(row.get("space_group", 0))
            sg_list.append(sg)

    fnorms = np.array(fnorms)
    label_norms = np.array(label_norms)
    print(f"Samples: {len(fnorms)}")
    print(f"Fnorm (mean): {fnorms.mean():.4f}")
    print(f"Fnorm (std):  {fnorms.std():.4f}")
    print(f"Fnorm (median): {np.median(fnorms):.4f}")

    nonzero = label_norms > 0
    n_zero = int((~nonzero).sum())
    print(f"\n--- EwT (relative Frobenius error) ---")
    print(f"Zero-norm labels excluded: {n_zero} / {len(fnorms)}")
    if nonzero.any():
        rel = fnorms[nonzero] / label_norms[nonzero]
        for t in EWT_THRESHOLDS:
            frac = float((rel < t).mean())
            print(f"EwT@{t * 100:g}%: {frac * 100:>6.2f}%  ({int((rel < t).sum())}/{len(rel)})")

    if by_crystal_system:
        print("\n--- By crystal system ---")
        print(f"{'System':<15} {'Count':>6} {'Fnorm':>8} {'Std':>8} {'EwT@25%':>9}")
        for name in ["cubic", "tetragonal", "hexagonal", "orthorhombic", "monoclinic", "triclinic"]:
            mask = np.array([crystal_system(sg) == name for sg in sg_list])
            if mask.any():
                vals = fnorms[mask]
                sub = mask & nonzero
                if sub.any():
                    rel_sub = fnorms[sub] / label_norms[sub]
                    ewt = f"{float((rel_sub < 0.25).mean()) * 100:>8.2f}%"
                else:
                    ewt = f"{'n/a':>9}"
                print(f"{name:<15} {len(vals):>6} {vals.mean():>8.4f} {vals.std():>8.4f} {ewt}")


def main():
    parser = argparse.ArgumentParser(description="Compute Fnorm from predictions CSV")
    parser.add_argument("results_dir", type=Path, help="Evaluation results directory")
    parser.add_argument("--by-crystal-system", action="store_true",
                        help="Break down Fnorm by crystal system")
    args = parser.parse_args()

    csv_path = args.results_dir / "predictions.csv"
    if not csv_path.exists():
        print(f"Error: {csv_path} not found", file=sys.stderr)
        sys.exit(1)

    compute_fnorm(csv_path, args.by_crystal_system)


if __name__ == "__main__":
    main()
