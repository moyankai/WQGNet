"""Helper CLI: print the resolved output directory for a config.

Slurm scripts use this to determine the result folder path before starting
the actual command, so job logs can be routed directly into that folder.

Usage::

    wyckoffgnn resolve-output-dir --config configs/train.yaml \
        [--command train] [--override key=val ...]

Prints the resolved absolute path to stdout and exits.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional


def run_resolve_output_dir_cli(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Resolve output directory from config")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--command", type=str, default="train")
    p.add_argument("--override", nargs="*", default=[])
    p.add_argument("--output-dir", type=str, default="")
    p.add_argument("--absolute", action="store_true",
                   help="Print absolute path (default: relative to cwd)")
    args, _ = p.parse_known_args(argv)

    try:
        if args.output_dir and args.output_dir != "auto":
            path = args.output_dir
        elif args.command == "benchmark":
            from wyckoff_gnn.utils.config import load_config
            from wyckoff_gnn.utils.auto_config import resolve_benchmark_output_root

            config = load_config(args.config, overrides=args.override)
            path = resolve_benchmark_output_root(config)
        else:
            from wyckoff_gnn.utils.config import load_config
            from wyckoff_gnn.utils.auto_config import resolve_output_dir

            config = load_config(args.config, overrides=args.override)
            path = resolve_output_dir(config, command=args.command)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    if args.absolute:
        path = os.path.abspath(path)
    print(path)
    return 0


def main():
    raise SystemExit(run_resolve_output_dir_cli())


__all__ = ["run_resolve_output_dir_cli", "main"]
