"""Matbench (v0.1) dataset adapter for traditional/static property tasks.

Matbench is a benchmark suite for materials property prediction with strict
5-fold nested cross-validation. This adapter loads a single fold at preprocess
time; run preprocess 5 times (fold=0..4) to build all fold caches, then run
5-fold benchmark training.

The matbench train set for each fold is further split 90/10 into local
train/val (using a deterministic seed derived from the fold index), so val
metrics track generalization without touching the official test set.

Config keys:
    task_name : str          # e.g. "matbench_mp_gap", "matbench_mp_e_form"
    fold : int (0..4)        # which of the 5 CV folds to load
    val_fraction : float     # fraction of matbench-train used as our val (default 0.1)
    max_entries : int (opt)  # cap total records (train+val+test) for debugging
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, Optional

import numpy as np

from wyckoff_gnn.data.adapters.base import DatasetAdapter
from wyckoff_gnn.data.records import StructureRecord

log = logging.getLogger(__name__)


class MatbenchAdapter(DatasetAdapter):
    """Load a single fold of a Matbench v0.1 task."""

    def __init__(
        self,
        task_name: str,
        fold: int = 0,
        val_fraction: float = 0.1,
        max_entries: Optional[int] = None,
    ):
        try:
            from matbench.bench import MatbenchBenchmark
        except ImportError as e:
            raise ImportError(
                "matbench not installed. Run `pip install matbench --no-deps` "
                "and `pip install matminer --no-deps` in the wyckoff_gnn env."
            ) from e

        self.task_name = task_name
        self.fold = int(fold)
        self.val_fraction = float(val_fraction)
        self.max_entries = max_entries

        if not (0 <= self.fold <= 4):
            raise ValueError(f"fold must be in [0..4], got {self.fold}")

        # Load just this task (subset saves memory).
        mb = MatbenchBenchmark(autoload=False, subset=[task_name])
        task = getattr(mb, task_name)
        task.load()
        self._task = task
        self._target_col = task.metadata["target"]
        self._task_type = task.metadata["task_type"]
        if self._task_type != "regression":
            raise NotImplementedError(
                f"Matbench task '{task_name}' has task_type='{self._task_type}'; "
                "only 'regression' is currently supported."
            )

        # Get official train and test splits for this fold.
        tr_X, tr_y = task.get_train_and_val_data(self.fold)
        te_X, te_y = task.get_test_data(self.fold, include_target=True)

        # Sub-split matbench-train into local train (90%) + val (10%).
        # Deterministic by fold: seed = fold + 1000.
        rng = np.random.RandomState(self.fold + 1000)
        n_train = len(tr_X)
        n_val = int(round(n_train * self.val_fraction))
        perm = rng.permutation(n_train)
        val_idx = set(perm[:n_val].tolist())

        # Store as (mid, structure, y, split) tuples.
        self._records: list = []
        for i, (mid, y) in enumerate(zip(tr_X.index, tr_y)):
            split = "val" if i in val_idx else "train"
            self._records.append((str(mid), tr_X.iloc[i], float(y), split))
        for mid, y in zip(te_X.index, te_y):
            i = len(self._records)  # not used
            structure = te_X.loc[mid]
            self._records.append((str(mid), structure, float(y), "test"))

        if self.max_entries is not None:
            self._records = self._records[: int(self.max_entries)]

        n_by_split = {"train": 0, "val": 0, "test": 0}
        for _, _, _, s in self._records:
            n_by_split[s] += 1
        log.info(
            f"MatbenchAdapter: task={task_name} fold={self.fold} "
            f"train={n_by_split['train']} val={n_by_split['val']} test={n_by_split['test']}"
        )

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "MatbenchAdapter":
        return cls(
            task_name=config["task_name"],
            fold=config.get("fold", 0),
            val_fraction=config.get("val_fraction", 0.1),
            max_entries=config.get("max_entries"),
        )

    def iter_records(self) -> Iterable[StructureRecord]:
        for mid, structure, target, split in self._records:
            yield StructureRecord(
                material_id=mid,
                structure=structure,
                target=target,
                target_type="graph_scalar_intensive",
                split=split,
                metadata={
                    "task_name": self.task_name,
                    "fold": self.fold,
                    "target_col": self._target_col,
                },
            )


__all__ = ["MatbenchAdapter"]
