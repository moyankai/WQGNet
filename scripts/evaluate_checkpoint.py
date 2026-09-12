# Backward-compatible thin wrapper. Recommended: wyckoffgnn <subcommand>
#!/usr/bin/env python
"""Evaluate a trained checkpoint — thin CLI wrapper.

Usage::

    python scripts/evaluate_checkpoint.py --config configs/evaluate.yaml --checkpoint best.pt
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wyckoff_gnn.cli.evaluate import main

if __name__ == "__main__":
    raise SystemExit(main())
