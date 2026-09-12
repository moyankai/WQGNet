#!/usr/bin/env python3
"""Prepare GMTNet piezoelectric tensor dataset for WyckoffGNN.

Converts the GMTNet raw pickle (jarvis_diele_piezo.pkl, which contains BOTH
dielectric and piezoelectric labels) into the intermediate format required by
the standard preprocessing pipeline.

Protocol reproduced from GMTNet_piezo/data.py + utils.py + train.py:
  1. Keep entries with |max(piezoelectric_C_m2)| < 100      -> 4998 of 5000
  2. Split with Python random, seed=32, ratios 0.8/0.1/0.1
       n = int(ratio * total)  =>  3998 / 499 / 499
       train = ids[:n_train]
       val   = ids[-(n_val+n_test):-n_test]
       test  = ids[-n_test:]
       (the two ids between the train block and the tail block are silently
        dropped, exactly as in GMTNet utils.py)
  3. Drop zero-norm targets (||T||_F <= 1e-5) from TRAIN ONLY
     (GMTNet train.py:169); val/test keep them.

Target tensor: piezoelectric_C_m2, shape (3, 6) Voigt (VASP PIEZO ordering
xx, yy, zz, xy, yz, zx), units C/m^2, symmetry ijk = ikj (piezoelectric
stress tensor e_ij, not the strain tensor d_ij).

Output:
  - data/processed/gmtnet_piezo/gmtnet_piezo_filtered.pkl   (dict jid -> entry)
  - data/processed/gmtnet_piezo/split_seed32.json           (train/val/test)
"""

import argparse
import json
import pickle
import random
from pathlib import Path

import numpy as np

ENTRY_ID_KEYS = ("JARVIS_ID", "jid")


def load_raw(path: str) -> list:
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def jid_of(entry) -> str:
    for k in ENTRY_ID_KEYS:
        v = entry.get(k)
        if v is not None:
            return str(v)
    raise KeyError(f"entry has none of {ENTRY_ID_KEYS}: {list(entry.keys())}")


def filter_by_max(data: list, max_val: float = 100.0) -> list:
    kept = []
    for entry in data:
        t = np.asarray(entry["piezoelectric_C_m2"], dtype=np.float64)
        if np.abs(t).max() < max_val:
            kept.append(entry)
    return kept


def zero_norm(entry) -> bool:
    t = np.asarray(entry["piezoelectric_C_m2"], dtype=np.float64)
    return float((t ** 2).sum() ** 0.5) <= 1e-5


def create_split(ids, seed=32, train_ratio=0.8, val_ratio=0.1):
    """GMTNet protocol: shuffled tail-slice for val/test, two ids dropped."""
    random.seed(seed)
    random.shuffle(ids)
    n = len(ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = int(n * val_ratio)
    train = ids[:n_train]
    val = ids[-(n_val + n_test):-n_test]
    test = ids[-n_test:]
    return {"train": train, "val": val, "test": test}


def main():
    parser = argparse.ArgumentParser(description="Prepare GMTNet piezo tensor data")
    parser.add_argument("--raw-data", required=True, help="Path to jarvis_diele_piezo.pkl")
    parser.add_argument("--output-dir", default="data/processed/gmtnet_piezo")
    parser.add_argument("--max-val", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=32)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    data = load_raw(args.raw_data)
    print(f"Loaded {len(data)} samples")

    filtered = filter_by_max(data, args.max_val)
    print(f"Kept {len(filtered)} after |max| < {args.max_val}")

    ids = [jid_of(e) for e in filtered]
    split = create_split(ids, seed=args.seed)
    print(f"Split (seed={args.seed}): train={len(split['train'])} "
          f"val={len(split['val'])} test={len(split['test'])}")

    zero_ids = {jid_of(e) for e in filtered if zero_norm(e)}
    n_zero_train = sum(1 for i in split["train"] if i in zero_ids)
    split["train"] = [i for i in split["train"] if i not in zero_ids]
    print(f"Dropped {n_zero_train} zero-norm targets from train only "
          f"(val/test keep all); train now {len(split['train'])}")

    entry_map = {jid_of(e): e for e in filtered}
    pkl_path = out / "gmtnet_piezo_filtered.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(entry_map, f)
    print(f"Saved filtered dict -> {pkl_path}")

    split_path = out / f"split_seed{args.seed}.json"
    with open(split_path, "w") as f:
        json.dump(split, f, indent=2)
    print(f"Saved split -> {split_path}")

    print("\nNext steps:")
    print("  1. wyckoffgnn preprocess --config configs/preprocess_gmtnet_piezo_wyckoff.yaml")
    print("  2. wyckoffgnn preprocess --config configs/preprocess_gmtnet_piezo_p1.yaml")


if __name__ == "__main__":
    main()
