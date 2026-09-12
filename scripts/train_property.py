#!/usr/bin/env python
# Backward-compatible thin wrapper. Recommended: wyckoffgnn train
"""Unified property prediction training — thin CLI wrapper.

Usage::

    python scripts/train_property.py --config configs/train.yaml
    wyckoffgnn train --config configs/train.yaml

All training logic is in :class:`wyckoff_gnn.experiments.runner.ExperimentRunner`.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wyckoff_gnn.cli.train import main

if __name__ == "__main__":
    raise SystemExit(main())
