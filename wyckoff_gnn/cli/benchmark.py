"""benchmark CLI — thin wrapper around BenchmarkRunner."""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from wyckoff_gnn.experiments.benchmark import BenchmarkRunner


def run_benchmark_cli(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Run WyckoffGNN benchmark")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--override", nargs="*", default=[])
    p.add_argument("--output-root", type=str, default="")
    p.add_argument("--models", nargs="*", default=None)
    p.add_argument("--seeds", type=int, nargs="*", default=None)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--no-resume", action="store_true")
    args = p.parse_args(argv)

    try:
        runner = BenchmarkRunner.from_config_file(args.config, overrides=args.override)

        # Override from CLI.
        if args.output_root:
            runner.config.setdefault("benchmark", {})["output_root"] = args.output_root
        if args.no_resume:
            runner.config.setdefault("benchmark", {})["skip_completed"] = False
        if args.models is not None:
            available = runner.config.get("models", [])
            seen: set = set()
            filtered: list = []
            for m in args.models:
                matched = False
                for entry in available:
                    if m == entry.get("name"):
                        if entry.get("name") not in seen:
                            filtered.append(entry)
                            seen.add(entry.get("name"))
                        matched = True
                    elif m == entry.get("model_type"):
                        if entry.get("name") not in seen:
                            filtered.append(entry)
                            seen.add(entry.get("name"))
                        matched = True
                if not matched:
                    available_names = [e.get("name", "?") for e in available]
                    available_types = [e.get("model_type", "?") for e in available]
                    raise ValueError(
                        f"No model matches '{m}'. "
                        f"Available names: {available_names}, types: {available_types}"
                    )
            if not filtered:
                available_names = [e.get("name", "?") for e in available]
                available_types = [e.get("model_type", "?") for e in available]
                raise ValueError(
                    f"No models match {args.models}. "
                    f"Available names: {available_names}, types: {available_types}"
                )
            runner.config["models"] = filtered
        if args.seeds is not None:
            runner.config["seeds"] = args.seeds
        if args.device != "auto":
            runner.config.setdefault("base", {})["device"] = args.device

        runner.run()
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


def main():
    raise SystemExit(run_benchmark_cli())


__all__ = ["run_benchmark_cli", "main"]
