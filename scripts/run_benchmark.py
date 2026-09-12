# Backward-compatible thin wrapper. Recommended: wyckoffgnn <subcommand>
#!/usr/bin/env python
"""Run multi-model, multi-seed benchmark — thin CLI wrapper."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wyckoff_gnn.cli.benchmark import main

if __name__ == "__main__":
    raise SystemExit(main())
