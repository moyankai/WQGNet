"""evaluate CLI — thin wrapper around ExperimentRunner.run_evaluate()."""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from wyckoff_gnn.experiments.runner import ExperimentRunner


def run_evaluate_cli(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Evaluate a WyckoffGNN checkpoint")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--output-dir", type=str, default="")
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--override", nargs="*", default=[])
    args = p.parse_args(argv)

    try:
        runner = ExperimentRunner.from_config_file(args.config, overrides=args.override)
        runner.run_evaluate(
            checkpoint_path=args.checkpoint,
            output_dir=args.output_dir or None,
            split=args.split,
            device=args.device,
        )
        return 0
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Fatal: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 2


def main():
    raise SystemExit(run_evaluate_cli())


__all__ = ["run_evaluate_cli", "main"]
