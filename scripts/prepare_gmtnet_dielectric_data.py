#!/usr/bin/env python3
"""Prepare GMTNet dielectric tensor dataset for WyckoffGNN.

This script converts the GMTNet dielectric tensor data into the intermediate
format required by the standard preprocessing pipeline.

Workflow:
  1. Download GMTNet data from https://github.com/gmtnet/gmtnet (or paper supplement)
  2. Run this script to create filtered pickle + split JSON
  3. Run: wyckoffgnn preprocess --config configs/preprocess_gmtnet_dielectric_wyckoff.yaml
  4. Run: wyckoffgnn preprocess --config configs/preprocess_gmtnet_dielectric_p1.yaml

Output:
  - data/processed/gmtnet_dielectric/gmtnet_dielectric_filtered.pkl
  - data/processed/gmtnet_dielectric/split_seed32.json

The filtered pickle contains 4713 samples with |max(dielectric)| < 100.
The split follows GMTNet's official protocol: seed=32, 80/10/10 ratio.
"""

import sys
import json
import pickle
import random
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jarvis.core.atoms import Atoms


def load_gmtnet_dielectric_data(raw_data_path: str):
    """Load GMTNet dielectric tensor data.
    
    Args:
        raw_data_path: Path to GMTNet's dielectric tensor pickle/json file
    
    Returns:
        List of dicts with keys: jid, atoms, dielectric, dielectric_ionic
    """
    # GMTNet provides data as a list of dicts
    # Each dict has: 'jid', 'atoms', 'dielectric' (3x3 tensor), 'dielectric_ionic'
    with open(raw_data_path, "rb") as f:
        data = pickle.load(f)
    
    return data


def filter_by_max_value(data, max_val=100.0):
    """Filter samples where |max(dielectric)| < max_val.
    
    Removes outliers with extremely large dielectric constants.
    """
    filtered = []
    for entry in data:
        diag = np.array(entry["dielectric"])
        if np.abs(diag).max() < max_val:
            filtered.append(entry)
    return filtered


def create_split(data, seed=32, train_ratio=0.8, val_ratio=0.1):
    """Create train/val/test split following GMTNet protocol.
    
    Args:
        data: List of entries
        seed: Random seed (GMTNet uses 32)
        train_ratio: Fraction for training
        val_ratio: Fraction for validation (test = 1 - train - val)
    
    Returns:
        Dict with keys: train, val, test (lists of JIDs)
    """
    ids = [entry["jid"] for entry in data]
    random.seed(seed)
    random.shuffle(ids)
    
    n = len(ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    
    split = {
        "train": ids[:n_train],
        "val": ids[n_train:n_train + n_val],
        "test": ids[n_train + n_val:],
    }
    
    return split


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare GMTNet dielectric tensor data")
    parser.add_argument("--raw-data", required=True, help="Path to GMTNet raw pickle file")
    parser.add_argument("--output-dir", default="data/processed/gmtnet_dielectric")
    parser.add_argument("--max-val", type=float, default=100.0, help="Filter threshold")
    parser.add_argument("--seed", type=int, default=32, help="Random seed for split")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading GMTNet data from {args.raw_data}")
    data = load_gmtnet_dielectric_data(args.raw_data)
    print(f"  Loaded {len(data)} samples")
    
    print(f"Filtering by |max| < {args.max_val}")
    filtered = filter_by_max_value(data, args.max_val)
    print(f"  Kept {len(filtered)} samples ({len(filtered)/len(data)*100:.1f}%)")
    
    print(f"Creating split (seed={args.seed})")
    split = create_split(filtered, seed=args.seed)
    print(f"  Train: {len(split['train'])}")
    print(f"  Val:   {len(split['val'])}")
    print(f"  Test:  {len(split['test'])}")
    
    # Save filtered data
    pkl_path = output_dir / "gmtnet_dielectric_filtered.pkl"
    print(f"Saving filtered data to {pkl_path}")
    with open(pkl_path, "wb") as f:
        pickle.dump(filtered, f)
    
    # Save split
    split_path = output_dir / f"split_seed{args.seed}.json"
    print(f"Saving split to {split_path}")
    with open(split_path, "w") as f:
        json.dump(split, f, indent=2)
    
    print("\nDone! Next steps:")
    print(f"  1. wyckoffgnn preprocess --config configs/preprocess_gmtnet_dielectric_wyckoff.yaml")
    print(f"  2. wyckoffgnn preprocess --config configs/preprocess_gmtnet_dielectric_p1.yaml")


if __name__ == "__main__":
    main()
