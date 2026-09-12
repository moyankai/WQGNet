"""Preprocess pipeline: raw dataset → LMDB cache.

    DatasetAdapter → StructureRecord → GraphBuilder → LMDBWriter

Usage::

    wyckoffgnn preprocess --config configs/preprocess.yaml
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import time
from pathlib import Path
from typing import Any, Dict

from wyckoff_gnn.data.adapters.registry import create_adapter
from wyckoff_gnn.data.graph_builders.registry import create_graph_builder
from wyckoff_gnn.data.lmdb_cache import LMDBWriter
from wyckoff_gnn.data.split import generate_random_split, load_official_split
from wyckoff_gnn.utils.io import save_json

log = logging.getLogger(__name__)


# --- Worker-side globals (populated by _worker_init) ---
_WORKER_BUILDER = None
_WORKER_SKIP_FAILED = True


def _worker_init(graph_cfg: Dict[str, Any], skip_failed: bool) -> None:
    """Initializer for each preprocess worker process."""
    global _WORKER_BUILDER, _WORKER_SKIP_FAILED
    _WORKER_BUILDER = create_graph_builder(graph_cfg)
    _WORKER_SKIP_FAILED = skip_failed


def _worker_build(job):
    """Build one graph in a worker. Returns (mid, split, graph_or_None, err_or_None)."""
    mid, split, record = job
    try:
        graph_dict = _WORKER_BUILDER.build(record)
        return (mid, split, graph_dict, None)
    except Exception as e:
        return (mid, split, None, f"{type(e).__name__}: {e}")


def run_preprocess(config: Dict[str, Any]) -> Dict[str, Any]:
    """Run the full preprocess pipeline.

    Args:
        config: Full config dict with ``dataset``, ``graph``, and
            ``preprocess`` sub-blocks.

    Returns:
        Stats dict with ``n_ok``, ``n_failed``, ``cache_dir``, etc.
    """
    dataset_cfg = config.get("dataset", config)
    graph_cfg = config.get("graph", {})
    preproc_cfg = config.get("preprocess", {})

    cache_dir = Path(preproc_cfg.get("cache_dir", "data/processed/default"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    resume = preproc_cfg.get("resume", True)
    max_records = preproc_cfg.get("max_records")
    skip_failed = preproc_cfg.get("skip_failed", True)
    num_workers = int(preproc_cfg.get("num_workers", 1))
    chunksize = int(preproc_cfg.get("chunksize", 8))
    fail_closed_splits = preproc_cfg.get("fail_closed_splits", False)

    # --- Create adapter and builder ---
    if fail_closed_splits:
        dataset_cfg["fail_closed"] = True
    adapter = create_adapter(dataset_cfg)
    builder = create_graph_builder(graph_cfg)

    # --- Split generation ---
    split_cfg = dataset_cfg.get("split", {})
    split_type = split_cfg.get("type")
    split_map: Dict[str, str] = {}

    if split_type == "random":
        all_records = list(adapter.iter_records())
        if max_records and max_records < len(all_records):
            all_records = all_records[:max_records]
        ids = [r.material_id for r in all_records]
        train_r = split_cfg.get("train_ratio", 0.8)
        val_r = split_cfg.get("val_ratio", 0.1)
        seed = split_cfg.get("seed", 42)
        train_m, val_m, test_m = generate_random_split(
            ids, train_ratio=train_r, val_ratio=val_r,
            test_ratio=1.0 - train_r - val_r, seed=seed,
        )
        for mid in train_m:
            split_map[mid] = "train"
        for mid in val_m:
            split_map[mid] = "val"
        for mid in test_m:
            split_map[mid] = "test"
        log.info(
            f"Random split: {len(train_m)} train / {len(val_m)} val "
            f"/ {len(test_m)} test (seed={seed})"
        )
    elif split_type == "file":
        split_path = split_cfg.get("split_file", "")
        raw_split = load_official_split(split_path)
        split_map = {mid: "test" for mid in raw_split}
        log.info(f"File split: {len(split_map)} ids loaded from {split_path}")
    else:
        all_records = list(adapter.iter_records())
        if max_records and max_records < len(all_records):
            all_records = all_records[:max_records]

    if split_type == "random":
        records_iter = iter(all_records)
    else:
        records_iter = adapter.iter_records()

    # --- Build and write to LMDB ---
    if resume:
        writer = LMDBWriter.resume_from_existing(str(cache_dir))
    else:
        writer = LMDBWriter(output_dir=str(cache_dir))

    existing_ids = {e["material_id"] for e in writer.manifest_entries}

    t0 = time.time()
    n_processed = 0

    def _job_stream():
        """Yield (mid, split, record) tuples, filtering existing/max_records."""
        n = 0
        for record in records_iter:
            if max_records and n >= max_records:
                break
            mid = record.material_id
            if mid in existing_ids:
                continue
            split = record.split
            if split is None and mid in split_map:
                split = split_map[mid]
            if split is None:
                if fail_closed_splits:
                    raise KeyError(
                        f"Record {mid} has no split assignment. "
                        f"Refusing to default to 'train'."
                    )
                split = "train"
            n += 1
            yield (mid, split, record)

    def _handle_result(mid, split, graph_dict, err):
        nonlocal n_processed
        if err is None:
            writer.add_ok(graph_dict, material_id=mid, split=split)
            n_processed += 1
        else:
            if skip_failed:
                log.warning(f"Failed {mid}: {err}")
                writer.add_failed(mid, err)
            else:
                raise RuntimeError(f"Failed to process {mid}: {err}")
        if n_processed % 100 == 0 and n_processed > 0:
            log.info(f"  {n_processed} records processed")

    if num_workers > 1:
        log.info(f"Preprocessing with {num_workers} workers (chunksize={chunksize})")
        ctx = mp.get_context("spawn")
        with ctx.Pool(
            processes=num_workers,
            initializer=_worker_init,
            initargs=(graph_cfg, skip_failed),
        ) as pool:
            for mid, split, graph_dict, err in pool.imap_unordered(
                _worker_build, _job_stream(), chunksize=chunksize
            ):
                _handle_result(mid, split, graph_dict, err)
    else:
        for mid, split, record in _job_stream():
            try:
                graph_dict = builder.build(record)
                _handle_result(mid, split, graph_dict, None)
            except Exception as e:
                msg = f"{type(e).__name__}: {e}"
                _handle_result(mid, split, None, msg)

    writer.finalize()
    elapsed = time.time() - t0

    stats = {
        "cache_dir": str(cache_dir),
        "n_ok": writer.stats["n_ok"],
        "n_failed": writer.stats["n_failed"],
        "elapsed_sec": elapsed,
        "records_per_sec": writer.stats["n_ok"] / max(elapsed, 1),
    }

    save_json(str(cache_dir / "resolved_config.json"), config)
    save_json(str(cache_dir / "preprocess_status.json"), {"status": "completed", **stats})

    log.info(f"Preprocess done: {stats['n_ok']} ok, {stats['n_failed']} failed "
             f"in {elapsed:.0f}s")
    return stats


def run_preprocess_cli(argv=None) -> int:
    """CLI entry point for preprocess. Returns exit code."""
    import argparse
    import sys
    import traceback
    from wyckoff_gnn.utils.config import load_config

    p = argparse.ArgumentParser(description="Preprocess dataset into LMDB cache")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--override", nargs="*", default=[])
    p.add_argument("--max-records", type=int, default=None)
    args = p.parse_args(argv)

    try:
        config = load_config(args.config, overrides=args.override)
        if args.max_records:
            config.setdefault("preprocess", {})["max_records"] = args.max_records
        run_preprocess(config)
        return 0
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Fatal: {e}", file=sys.stderr)
        traceback.print_exc()
        return 2


def main():
    import sys
    sys.exit(run_preprocess_cli())


__all__ = ["run_preprocess", "run_preprocess_cli", "main"]
