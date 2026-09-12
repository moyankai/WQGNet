"""Deterministic seed setting."""

import random
import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Set Python, NumPy, and PyTorch random seeds.

    Args:
        seed: Random seed value.
        deterministic: If True, disable cudnn.benchmark for strict
            reproducibility (slower). If False, allow cudnn auto-tuning.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic


__all__ = ["set_seed"]
