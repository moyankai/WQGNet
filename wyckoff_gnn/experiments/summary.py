"""Benchmark summary builder — collects results and writes summary.csv/md."""

from __future__ import annotations

import csv
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

log = logging.getLogger(__name__)

COMPLETED_STAGES = {"completed", "outputs_written"}


def is_completed_status(status: str) -> bool:
    """Return True if the status string indicates a completed run."""
    return status in COMPLETED_STAGES


def _safe_fmt(val) -> str:
    """Format a numeric value for markdown, handling None/non-numeric."""
    if val is None:
        return "N/A"
    try:
        return f"{float(val):.4f}"
    except (TypeError, ValueError):
        return str(val)


class SummaryBuilder:
    """Collect per-run results and produce summary tables."""

    def __init__(self, benchmark_dir: str):
        self.benchmark_dir = Path(benchmark_dir)

    def _read_run_info(self, run_dir: str) -> Dict[str, Any]:
        """Read run_info.json if it exists, else return empty dict."""
        path = os.path.join(run_dir, "run_info.json")
        if os.path.exists(path):
            try:
                with open(path) as f:
                    return json.load(f)
            except (OSError, json.JSONDecodeError) as e:
                log.warning(f"Could not read {path}: {e}")
        return {}

    def collect_results(self) -> List[Dict[str, Any]]:
        """Walk benchmark tree, reading results.json + run_info.json."""
        rows: List[Dict[str, Any]] = []
        for root, dirs, files in os.walk(self.benchmark_dir):
            if "results.json" in files:
                res_path = os.path.join(root, "results.json")
                try:
                    with open(res_path) as f:
                        res = json.load(f)
                except (OSError, json.JSONDecodeError) as e:
                    log.warning(f"Could not read {res_path}: {e}")
                    res = {}

                info = self._read_run_info(root)

                # Prefer run_info.json fields; fall back to path inference.
                rel = os.path.relpath(root, self.benchmark_dir)
                parts = Path(rel).parts
                model_name = info.get("model_name") or (
                    parts[0] if len(parts) >= 1 else "unknown"
                )
                seed = info.get("seed")
                if seed is None:
                    for p in parts:
                        if p.startswith("seed_"):
                            try:
                                seed = int(p.split("_")[1])
                            except ValueError:
                                pass

                status_path = os.path.join(root, "status.json")
                status = "unknown"
                if os.path.exists(status_path):
                    try:
                        with open(status_path) as f:
                            s = json.load(f)
                        status = s.get("stage", "unknown")
                    except (OSError, json.JSONDecodeError) as e:
                        log.warning(f"Could not read {status_path}: {e}")

                rows.append({
                    "benchmark": info.get("benchmark", self.benchmark_dir.name),
                    "model_name": model_name,
                    "model_type": info.get("model_type") or res.get("model", "unknown"),
                    "seed": seed,
                    "run_dir": rel,
                    "status": status,
                    "config_hash": info.get("config_hash", ""),
                    "params": res.get("params", None),
                    "train_size": res.get("train_size", None),
                    "val_size": res.get("val_size", None),
                    "test_size": res.get("test_size", None),
                    "target_mean": res.get("target_mean", None),
                    "target_std": res.get("target_std", None),
                    "test_mae": res.get("test_mae", None),
                    "test_rmse": res.get("test_rmse", None),
                    "test_r2": res.get("test_r2", None),
                    "test_mae_norm": res.get("test_mae_norm", None),
                    "test_rmse_norm": res.get("test_rmse_norm", None),
                    "best_epoch": res.get("best_epoch", None),
                    "best_val_mae": res.get("best_val_mae", None),
                    "train_time_sec": res.get("train_time_sec", None),
                    "avg_epoch_time_sec": res.get("avg_epoch_time_sec", None),
                })
        rows.sort(key=lambda r: (r["model_name"], r["seed"] or 0))
        return rows

    def write_summary_csv(self, path: Optional[str] = None) -> str:
        if path is None:
            path = str(self.benchmark_dir / "summary.csv")
        rows = self.collect_results()
        if not rows:
            with open(path, "w", encoding="utf-8") as f:
                f.write("")
            return path
        fieldnames = list(rows[0].keys())
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        return path

    def write_summary_md(self, path: Optional[str] = None) -> str:
        if path is None:
            path = str(self.benchmark_dir / "summary.md")
        rows = self.collect_results()
        n_completed = sum(1 for r in rows if is_completed_status(r["status"]))

        # Read benchmark_status and failures for complete counts.
        bm_status = {}
        bm_path = os.path.join(self.benchmark_dir, "benchmark_status.json")
        if os.path.exists(bm_path):
            try:
                with open(bm_path) as f:
                    bm_status = json.load(f)
            except (OSError, json.JSONDecodeError) as e:
                log.warning(f"Could not read {bm_path}: {e}")
        failures = []
        fail_path = os.path.join(self.benchmark_dir, "failures.jsonl")
        if os.path.exists(fail_path):
            try:
                with open(fail_path) as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            failures.append(json.loads(line))
            except (OSError, json.JSONDecodeError) as e:
                log.warning(f"Could not read {fail_path}: {e}")

        total = bm_status.get("total", len(rows))
        n_skipped = bm_status.get("skipped", 0)
        n_failed = bm_status.get("failed", len(failures))

        lines = [f"# Benchmark: {self.benchmark_dir.name}", ""]
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Total runs | {total} |")
        lines.append(f"| Completed (with results) | {n_completed} |")
        lines.append(f"| Skipped | {n_skipped} |")
        if n_failed > 0:
            lines.append(f"| **Failed** | **{n_failed}** |")
        lines.append(f"| Rows in summary | {len(rows)} |")
        lines.append("")
        lines.append("")

        # Aggregate by model (completed only).
        groups: Dict[str, List[Dict]] = {}
        for r in rows:
            if not is_completed_status(r["status"]):
                continue
            groups.setdefault(r["model_name"], []).append(r)

        if groups:
            lines.append("## Per-model summary")
            lines.append("")
            header = "| Model | N | MAE | RMSE | R² | Params |"
            lines.append(header)
            lines.append("|" + "---|" * 6)
            for name, grp in sorted(groups.items()):
                maes = [r["test_mae"] for r in grp if r["test_mae"] is not None]
                rmses = [r["test_rmse"] for r in grp if r["test_rmse"] is not None]
                r2s = [r["test_r2"] for r in grp if r["test_r2"] is not None]
                params = grp[0].get("params", "?")
                if len(maes) > 1:
                    mae_s = f"{np.mean(maes):.4f}±{np.std(maes):.4f}"
                    rmse_s = f"{np.mean(rmses):.4f}±{np.std(rmses):.4f}"
                    r2_s = f"{np.mean(r2s):.4f}±{np.std(r2s):.4f}"
                elif maes:
                    mae_s = _safe_fmt(maes[0])
                    rmse_s = _safe_fmt(rmses[0])
                    r2_s = _safe_fmt(r2s[0])
                else:
                    mae_s = rmse_s = r2_s = "N/A"
                lines.append(f"| {name} | {len(grp)} | {mae_s} | {rmse_s} | {r2_s} | {params} |")
            lines.append("")

        lines.append("## Per-run table")
        lines.append("")
        header = "| Model | Seed | MAE | R² | Status |"
        lines.append(header)
        lines.append("|" + "---|" * 5)
        for r in rows:
            mae_str = _safe_fmt(r.get("test_mae"))
            r2_str = _safe_fmt(r.get("test_r2"))
            lines.append(
                f"| {r['model_name']} | {r['seed']} | {mae_str} | {r2_str} | "
                f"{r['status']} |"
            )

        # Failed runs section (from failures.jsonl, not results.json).
        if failures:
            lines.append("## Failed runs")
            lines.append("")
            for f in failures:
                lines.append(
                    f"- **{f.get('model_name', '?')}** seed={f.get('seed','?')}: "
                    f"`{f.get('error','?')}`"
                )
            lines.append("")

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        return path


def summarize_benchmark(benchmark_dir: str) -> Dict[str, str]:
    sb = SummaryBuilder(benchmark_dir)
    return {
        "csv": sb.write_summary_csv(),
        "md": sb.write_summary_md(),
    }


__all__ = [
    "SummaryBuilder",
    "summarize_benchmark",
    "COMPLETED_STAGES",
    "is_completed_status",
]
