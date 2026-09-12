"""Property prediction DataModule.

Creates train/val/test dataloaders from shard-based Wyckoff datasets.
Handles target normalization (fit on train, apply to all).
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from torch.utils.data import DataLoader

from wyckoff_gnn.data.shard_dataset import WyckoffShardDataset
from wyckoff_gnn.data.normalization import TargetNormalizer


def _collate_wyckoff(batch):
    """Collate a list of PyG Data objects into a batch."""
    from torch_geometric.data import Batch
    return Batch.from_data_list(batch)


class PropertyDataModule:
    """Wraps train/val/test WyckoffShardDataset + DataLoaders.

    Args:
        manifest_path: Path to ``manifest.jsonl``.
        shard_dir: Directory containing shard files.
        batch_size: Batch size for training.
        val_batch_size: Batch size for val/test (defaults to batch_size).
        num_workers: Number of dataloader workers.
        pin_memory: Whether to pin GPU memory.
        persistent_workers: Keep workers alive between epochs.
        prefetch_factor: Number of batches to prefetch per worker.
        max_train_samples: Limit train set size (debug mode).
        max_val_samples: Limit val set size.
        max_test_samples: Limit test set size.
        normalizer: Optional pre-fitted normalizer (bypass auto-fit).
        leaderboard_split_zip: Optional path to an official JARVIS-Leaderboard
            split zip. When set, the manifest's ``split`` and ``target`` fields
            are overridden by the official values, and only JIDs present in
            the official split are kept. Populates ``self.split_report``.
    """

    def __init__(
        self,
        manifest_path: str,
        shard_dir: str,
        batch_size: int = 32,
        val_batch_size: Optional[int] = None,
        num_workers: int = 0,
        pin_memory: bool = True,
        persistent_workers: bool = False,
        prefetch_factor: Optional[int] = None,
        max_train_samples: Optional[int] = None,
        max_val_samples: Optional[int] = None,
        max_test_samples: Optional[int] = None,
        normalizer: Optional[TargetNormalizer] = None,
        global_max_mult: int = 192,
        leaderboard_split_zip: Optional[str] = None,
        leaderboard_override_targets: bool = True,
    ):
        self.manifest_path = manifest_path
        self.shard_dir = shard_dir
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size or batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.prefetch_factor = prefetch_factor
        self.max_train_samples = max_train_samples
        self.max_val_samples = max_val_samples
        self.max_test_samples = max_test_samples
        self._global_max_mult = global_max_mult
        self._normalizer = normalizer
        self.leaderboard_split_zip = leaderboard_split_zip
        self.leaderboard_override_targets = leaderboard_override_targets
        self.split_report: Optional[dict] = None

        self._train_ds: Optional[WyckoffShardDataset] = None
        self._val_ds: Optional[WyckoffShardDataset] = None
        self._test_ds: Optional[WyckoffShardDataset] = None

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self) -> "PropertyDataModule":
        """Create train/val/test datasets and fit normalizer.

        Normalizer is fit from train-split targets first, then injected into
        datasets so ``__getitem__`` returns ``data.y = normalized_target``
        and ``data.y_raw = raw_target``.

        If ``leaderboard_split_zip`` is set, the manifest's split/target
        fields are overridden by the official JARVIS-Leaderboard values
        before dataset construction.
        """
        import json as _json
        import logging as _logging
        _log = _logging.getLogger(__name__)

        gm = getattr(self, "_global_max_mult", 192)

        # Read manifest once — needed for both the plain and the leaderboard path.
        raw_entries: list = []
        with open(self.manifest_path, "r") as f:
            for line in f:
                raw_entries.append(_json.loads(line))

        if self.leaderboard_split_zip:
            from wyckoff_gnn.data.leaderboard_split import (
                load_leaderboard_zip,
                apply_leaderboard_split_to_manifest,
                summarize_split_application,
            )
            leaderboard = load_leaderboard_zip(self.leaderboard_split_zip)
            kept_entries, missing = apply_leaderboard_split_to_manifest(
                raw_entries, leaderboard,
                override_targets=self.leaderboard_override_targets,
            )
            self.split_report = summarize_split_application(
                leaderboard, kept_entries, missing, self.leaderboard_split_zip,
            )
            _log.info(
                f"Official JARVIS-Leaderboard split from "
                f"{self.leaderboard_split_zip}: "
                f"matched={self.split_report['matched_counts']}, "
                f"missing={self.split_report['missing_counts']}, "
                f"override_targets={self.leaderboard_override_targets}"
            )
            entries_to_use = kept_entries
        else:
            self.split_report = None
            entries_to_use = raw_entries

        train_entries = [e for e in entries_to_use if e.get("split") == "train"]
        val_entries   = [e for e in entries_to_use if e.get("split") == "val"]
        test_entries  = [e for e in entries_to_use if e.get("split") == "test"]

        # Fit normalizer from train targets (only valid, non-null ones).
        if self._normalizer is None:
            train_targets = np.array(
                [float(e["target"]) for e in train_entries if e.get("target") is not None],
                dtype=np.float32,
            )
            self._normalizer = TargetNormalizer.from_targets(train_targets)

        self._train_ds = WyckoffShardDataset(
            self.manifest_path, self.shard_dir,
            global_max_mult=gm, normalizer=self._normalizer,
            entries_override=train_entries,
        )
        self._val_ds = WyckoffShardDataset(
            self.manifest_path, self.shard_dir,
            global_max_mult=gm, normalizer=self._normalizer,
            entries_override=val_entries,
        )
        self._test_ds = WyckoffShardDataset(
            self.manifest_path, self.shard_dir,
            global_max_mult=gm, normalizer=self._normalizer,
            entries_override=test_entries,
        )

        # Apply sample limits.
        if self.max_train_samples:
            self._train_ds.truncate(self.max_train_samples)
        if self.max_val_samples:
            self._val_ds.truncate(self.max_val_samples)
        if self.max_test_samples:
            self._test_ds.truncate(self.max_test_samples)

        # Filter entries with null targets (common for tensor properties
        # where some JARVIS entries lack all required components).
        for ds in (self._train_ds, self._val_ds, self._test_ds):
            n_removed = ds.filter_valid_targets()
            if n_removed > 0:
                _log.info(f"Filtered {n_removed} entries with null targets")

        return self

    # ------------------------------------------------------------------
    # Dataloaders
    # ------------------------------------------------------------------

    @property
    def _loader_kwargs(self) -> dict:
        kw: dict = {
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "persistent_workers": self.persistent_workers and self.num_workers > 0,
            "collate_fn": _collate_wyckoff,
        }
        if self.num_workers > 0 and self.prefetch_factor is not None:
            kw["prefetch_factor"] = self.prefetch_factor
        return kw

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            **self._loader_kwargs,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self._val_ds,
            batch_size=self.val_batch_size,
            shuffle=False,
            **self._loader_kwargs,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self._test_ds,
            batch_size=self.val_batch_size,
            shuffle=False,
            **self._loader_kwargs,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def normalizer(self) -> TargetNormalizer:
        if self._normalizer is None:
            raise RuntimeError("setup() must be called first")
        return self._normalizer

    @property
    def train_dataset(self) -> WyckoffShardDataset:
        return self._train_ds

    @property
    def val_dataset(self) -> WyckoffShardDataset:
        return self._val_ds

    @property
    def test_dataset(self) -> WyckoffShardDataset:
        return self._test_ds

    @property
    def train_size(self) -> int:
        return len(self._train_ds) if self._train_ds else 0

    @property
    def val_size(self) -> int:
        return len(self._val_ds) if self._val_ds else 0

    @property
    def test_size(self) -> int:
        return len(self._test_ds) if self._test_ds else 0


def _read_targets_from_manifest(
    manifest_path: str, split: Optional[str] = None,
) -> np.ndarray:
    """Read target values from a manifest JSONL file."""
    import json
    targets = []
    with open(manifest_path, "r") as f:
        for line in f:
            entry = json.loads(line)
            if split is None or entry.get("split") == split:
                y = entry.get("target")
                if y is not None:
                    targets.append(float(y))
    return np.array(targets, dtype=np.float32)


__all__ = ["PropertyDataModule"]
