"""BenchmarkRunner — multi-model, multi-seed experiment orchestration.

Calls :class:`ExperimentRunner` for each (model, seed) combination.
Handles skip_completed with config-hash verification, failure logging,
run_info.json writing, and summary generation.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from wyckoff_gnn.utils.config import load_config as _load_config
from wyckoff_gnn.utils.logging import get_logger
from wyckoff_gnn.experiments.runner import ExperimentRunner
from wyckoff_gnn.experiments.summary import (
    SummaryBuilder, is_completed_status,
)

log = get_logger(__name__)


def _config_hash(config: Dict[str, Any]) -> str:
    """Stable hash of a config dict for skip_completed detection."""
    raw = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


def _read_status_stage(output_dir: str) -> str:
    """Read status.json stage, or 'unknown'."""
    path = os.path.join(output_dir, "status.json")
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f).get("stage", "unknown")
        except Exception:
            return "unknown"
    return "unknown"


class BenchmarkRunner:
    """Orchestrate multi-model, multi-seed experiments."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @classmethod
    def from_config_file(
        cls, path: str, overrides: Optional[List[str]] = None,
    ) -> "BenchmarkRunner":
        config = _load_config(path, overrides=overrides)
        return cls(config)

    def run(self) -> Dict[str, Any]:
        bm_cfg = self.config.get("benchmark", {})
        base = self.config.get("base", {})
        models = self.config.get("models", [])
        seeds = self.config.get("seeds", [0])

        from wyckoff_gnn.utils.auto_config import resolve_benchmark_output_root
        output_root = resolve_benchmark_output_root(self.config)
        skip_completed = bm_cfg.get("skip_completed", True)
        fail_fast = bm_cfg.get("fail_fast", False)

        os.makedirs(output_root, exist_ok=True)

        # Clean stale failures.jsonl at start.
        fail_path = os.path.join(output_root, "failures.jsonl")
        if os.path.exists(fail_path):
            os.remove(fail_path)

        # Collect runs.
        runs: List[Dict[str, Any]] = []
        for model_entry in models:
            model_name = model_entry.get("name", model_entry.get("model_type", "unknown"))
            model_overrides = {
                k: v for k, v in model_entry.items() if k != "name"
            }
            for seed in seeds:
                run_dir = os.path.join(output_root, model_name, f"seed_{seed}")
                run_config = copy.deepcopy(base)
                run_config.update(model_overrides)
                run_config["seed"] = seed
                run_config["output_dir"] = run_dir
                run_config["model_name"] = model_name
                runs.append({
                    "config": run_config,
                    "output_dir": run_dir,
                    "model_name": model_name,
                    "seed": seed,
                })

        # Execute.
        completed = 0
        skipped = 0
        failures: List[Dict[str, Any]] = []
        t0 = time.time()

        for run in runs:
            out = run["output_dir"]
            ch = _config_hash(run["config"])

            # Check skip.
            if skip_completed:
                results_ok = os.path.exists(os.path.join(out, "results.json"))
                stage = _read_status_stage(out)
                run_info_path = os.path.join(out, "run_info.json")
                old_hash = "missing"
                if os.path.exists(run_info_path):
                    try:
                        with open(run_info_path) as f:
                            saved = json.load(f)
                        old_hash = saved.get("config_hash", "missing")
                    except Exception:
                        old_hash = "corrupt"
                hash_match = (old_hash == ch)

                if results_ok and is_completed_status(stage) and hash_match:
                    log.info(f"Skip {run['model_name']}/seed_{run['seed']} (completed, hash={ch})")
                    skipped += 1
                    continue
                elif results_ok and not hash_match:
                    log.warning(
                        f"Re-run {run['model_name']}/seed_{run['seed']}: "
                        f"config hash changed ({old_hash} → {ch})"
                    )

            # Write run_info before train.
            os.makedirs(out, exist_ok=True)
            run_info = {
                "benchmark": bm_cfg.get("name", Path(output_root).name),
                "model_name": run["model_name"],
                "model_type": run["config"].get("model_type", "unknown"),
                "seed": run["seed"],
                "output_dir": out,
                "config_hash": ch,
                "run_config": run["config"],
            }
            with open(os.path.join(out, "run_info.json"), "w") as f:
                json.dump(run_info, f, indent=2, default=str)

            try:
                log.info(f"Run {run['model_name']}/seed_{run['seed']} → {out}")
                runner = ExperimentRunner(run["config"])
                runner.run_train(
                    device=run["config"].get("device", "auto"),
                    resume_from="auto",
                )
                completed += 1
            except Exception as e:
                failures.append({
                    "model_name": run["model_name"],
                    "seed": run["seed"],
                    "output_dir": out,
                    "error": str(e),
                })
                log.error(f"FAILED {run['model_name']}/seed_{run['seed']}: {e}")
                if fail_fast:
                    break

        elapsed = time.time() - t0

        # Write failures (empty file for no failures → clean state).
        with open(fail_path, "w") as f:
            for entry in failures:
                f.write(json.dumps(entry) + "\n")

        # Benchmark status.
        status = {
            "total": len(runs),
            "completed": completed,
            "skipped": skipped,
            "failed": len(failures),
            "completed_this_run": completed,
            "skipped_existing": skipped,
            "failed_this_run": len(failures),
            "elapsed_sec": elapsed,
        }
        with open(os.path.join(output_root, "benchmark_status.json"), "w") as f:
            json.dump(status, f, indent=2)

        SummaryBuilder(output_root).write_summary_csv()
        SummaryBuilder(output_root).write_summary_md()

        log.info(
            f"Benchmark done: {completed} ok, {skipped} skipped, "
            f"{len(failures)} failed in {elapsed:.0f}s"
        )
        return status


__all__ = ["BenchmarkRunner"]
