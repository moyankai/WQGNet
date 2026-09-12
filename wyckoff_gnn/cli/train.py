"""train CLI — thin wrapper around ExperimentRunner."""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from wyckoff_gnn.experiments.runner import ExperimentRunner


def run_train_cli(argv: Optional[List[str]] = None) -> int:
    """Parse args and run training. Returns exit code."""
    p = argparse.ArgumentParser(
        description="Train WyckoffGNN for property prediction",
    )
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--override", nargs="*", default=[])
    p.add_argument("--output-dir", type=str, default="")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--resume", action="store_true",
                   help="Resume from checkpoint.pt in the output directory")
    p.add_argument("--resume-from", type=str, default="",
                   help="Explicit path to a checkpoint.pt file to resume from")
    args = p.parse_args(argv)

    # Determine resume path
    resume_from = None
    if args.resume_from:
        resume_from = args.resume_from
    elif args.resume:
        resume_from = "auto"  # runner will resolve to output_dir/checkpoint.pt

    try:
        runner = ExperimentRunner.from_config_file(args.config, overrides=args.override)
        runner.run_train(
            output_dir=args.output_dir or None,
            device=args.device,
            resume_from=resume_from,
            config_path=args.config,
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
    raise SystemExit(run_train_cli())


__all__ = ["run_train_cli", "main"]
