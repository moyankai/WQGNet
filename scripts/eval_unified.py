"""Simple evaluation script for unified model checkpoints."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import json
from pathlib import Path

from wyckoff_gnn.models.unified_equivariant import UnifiedQuotientEquivariantGNN
from wyckoff_gnn.data.datamodule import PropertyDataModule
from wyckoff_gnn.training.evaluator import collect_predictions, compute_results


def evaluate_checkpoint(
    checkpoint_path: str,
    config_path: str,
    split: str = "test",
    device: str = "cuda",
):
    """Evaluate a unified model checkpoint."""
    print(f"\n{'='*60}")
    print(f"Evaluating: {Path(checkpoint_path).name}")
    print(f"Split: {split}")
    print(f"{'='*60}\n")

    # Load config
    with open(config_path) as f:
        import yaml
        config = yaml.safe_load(f)

    # Build data
    manifest = config["manifest_path"]
    shard_dir = config.get("shard_dir", os.path.dirname(manifest))

    dm = PropertyDataModule(
        manifest_path=manifest,
        shard_dir=shard_dir,
        batch_size=config.get("batch_size", 64),
        num_workers=config.get("num_workers", 4),
        pin_memory=True,
        persistent_workers=False,
        leaderboard_split_zip=config.get("leaderboard_split_zip"),
    )
    dm.setup()

    # Load model
    model = UnifiedQuotientEquivariantGNN(
        hidden_scalar=config.get("hidden_dim", 128),
        num_layers=config.get("num_layers", 5),
        num_rbf=config.get("num_rbf", 32),
        rbf_max=config.get("rbf_max", 8.0),
        lmax=0,
    )

    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    print(f"Model loaded: {sum(p.numel() for p in model.parameters())} parameters")

    # Get dataloader
    if split == "test":
        loader = dm.test_dataloader()
    elif split == "val":
        loader = dm.val_dataloader()
    else:
        loader = dm.train_dataloader()

    # Evaluate
    predictions = collect_predictions(model, loader, dm.normalizer, torch.device(device))
    results = compute_results(predictions, dm.normalizer, model, config)

    print(f"\nResults:")
    print(f"  Keys: {list(results.keys())}")
    for k, v in results.items():
        if isinstance(v, (int, float)):
            print(f"  {k}: {v:.6f}")
        else:
            print(f"  {k}: {v}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    results = evaluate_checkpoint(args.checkpoint, args.config, args.split, args.device)
