"""Target normalization utilities.

Computes mean/std from training targets, applies normalization,
and supports inverse transformation for raw-space predictions.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


class TargetNormalizer:
    """Stores target mean/std and provides normalize / denormalize.

    Fit on training targets only (to avoid data leakage).
    """

    def __init__(self, mean: float = 0.0, std: float = 1.0):
        self.mean = float(mean)
        self.std = max(float(std), 1e-8)

    @classmethod
    def from_targets(cls, targets: np.ndarray) -> "TargetNormalizer":
        """Fit from a numpy array of training target values."""
        mean = float(np.mean(targets))
        std = float(np.std(targets))
        return cls(mean=mean, std=std)

    def normalize(self, value: float) -> float:
        """Normalize a single target value."""
        return (value - self.mean) / self.std

    def denormalize(self, value: float) -> float:
        """Reverse normalization to raw space."""
        return value * self.std + self.mean

    def normalize_array(
        self, arr: np.ndarray
    ) -> np.ndarray:
        """Normalize a numpy array."""
        return (arr - self.mean) / self.std

    def denormalize_array(
        self, arr: np.ndarray
    ) -> np.ndarray:
        """Denormalize a numpy array."""
        return arr * self.std + self.mean

    def to_dict(self) -> dict:
        return {"mean": self.mean, "std": self.std}

    @classmethod
    def from_dict(cls, d: dict) -> "TargetNormalizer":
        return cls(mean=d["mean"], std=d["std"])


def compute_target_stats(
    targets: np.ndarray,
) -> Tuple[float, float]:
    """Compute (mean, std) from a numpy array. Convenience function."""
    return float(np.mean(targets)), float(np.std(targets))


__all__ = [
    "TargetNormalizer",
    "compute_target_stats",
]
