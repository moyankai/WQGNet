# Backward-compatible thin wrapper. Recommended: wyckoffgnn <subcommand>
#!/usr/bin/env python
"""Unified dataset preprocessing — thin CLI wrapper.

Usage::

    python scripts/preprocess_dataset.py --config configs/preprocess/mp20_wyckoff.yaml
    python scripts/preprocess_dataset.py --config configs/preprocess/mp20_wyckoff.yaml --max-records 100

All core logic lives in :mod:`wyckoff_gnn.cli.preprocess`.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wyckoff_gnn.cli.preprocess import main

if __name__ == "__main__":
    main()
