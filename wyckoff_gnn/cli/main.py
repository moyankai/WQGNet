#!/usr/bin/env python
"""WyckoffGNN unified CLI entry point.

Usage::

    wyckoffgnn preprocess --config configs/preprocess.yaml
    wyckoffgnn train --config configs/train.yaml
    wyckoffgnn evaluate --config configs/evaluate.yaml --checkpoint best.pt
    wyckoffgnn benchmark --config configs/benchmark.yaml
"""

from __future__ import annotations

import sys
from typing import List, Optional

SUBCOMMANDS = {
    "train": "wyckoff_gnn.cli.train:run_train_cli",
    "preprocess": "wyckoff_gnn.cli.preprocess:run_preprocess_cli",
    "evaluate": "wyckoff_gnn.cli.evaluate:run_evaluate_cli",
    "benchmark": "wyckoff_gnn.cli.benchmark:run_benchmark_cli",
    "resolve-output-dir": "wyckoff_gnn.cli.resolve_output_dir:run_resolve_output_dir_cli",
}


def main(argv: Optional[List[str]] = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    if not argv or argv[0] in ("-h", "--help"):
        print("WyckoffGNN CLI")
        print()
        print("Available subcommands:")
        for name, _ in SUBCOMMANDS.items():
            print(f"  {name}")
        print()
        print("Use: wyckoffgnn <subcommand> --help for details")
        return 0

    subcommand = argv[0]
    rest = argv[1:]

    if subcommand not in SUBCOMMANDS:
        print(f"Unknown subcommand: {subcommand}", file=sys.stderr)
        print(f"Available: {list(SUBCOMMANDS.keys())}", file=sys.stderr)
        return 1

    import importlib
    mod_path, func_name = SUBCOMMANDS[subcommand].split(":")
    mod = importlib.import_module(mod_path)
    func = getattr(mod, func_name)
    return func(rest)


if __name__ == "__main__":
    raise SystemExit(main())
