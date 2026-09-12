"""Experiment archival utilities for reproducibility.

Snapshots the project source tree (as directories, not a tarball) plus
run metadata and the original config into the experiment output directory,
so the whole run can be inspected — and re-run — in place.

Layout produced inside ``output_dir``::

    <output_dir>/
        run_metadata.json      # env, git, slurm, command
        original_config.yaml   # unresolved user config
        data -> ../../data     # symlink so relative paths still resolve
        source/                # exact project source at run time
            wyckoff_gnn/
            configs/
            jobs/
            scripts/
            tests/
            setup.py
            requirements.txt
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
from datetime import datetime
from typing import Any, Dict, Optional


def save_run_metadata(
    output_dir: str,
    config: Dict[str, Any],
    start_time: datetime,
    command: Optional[str] = None,
) -> None:
    """Save environment and run metadata to output_dir/run_metadata.json."""
    import torch

    meta: Dict[str, Any] = {
        "hostname": platform.node(),
        "user": os.environ.get("USER", "unknown"),
        "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
        "end_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "pid": os.getpid(),
        "cwd": os.getcwd(),
    }

    try:
        import e3nn
        meta["e3nn_version"] = e3nn.__version__
    except (ImportError, AttributeError):
        pass

    try:
        import wyckoff_gnn
        meta["wyckoff_gnn_version"] = wyckoff_gnn.__version__
    except (ImportError, AttributeError):
        pass

    if torch.cuda.is_available():
        meta["cuda_device"] = torch.cuda.get_device_name(0)
        meta["cuda_memory_gb"] = round(
            torch.cuda.get_device_properties(0).total_memory / 1024**3, 1
        )

    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    if slurm_job_id:
        meta["slurm_job_id"] = slurm_job_id
        meta["slurm_job_name"] = os.environ.get("SLURM_JOB_NAME", "")
        meta["slurm_nodelist"] = os.environ.get("SLURM_NODELIST", "")

    if command:
        meta["command"] = command

    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        meta["git_commit"] = git_commit
        git_branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        meta["git_branch"] = git_branch
        git_dirty = subprocess.call(
            ["git", "diff", "--quiet"], stderr=subprocess.DEVNULL
        ) != 0
        meta["git_dirty"] = git_dirty
    except (subprocess.SubprocessError, FileNotFoundError):
        pass

    path = os.path.join(output_dir, "run_metadata.json")
    with open(path, "w") as f:
        json.dump(meta, f, indent=2, default=str)


# Directories/files to copy into <output_dir>/source/ mirroring project layout.
_SOURCE_DIRS = ("wyckoff_gnn", "configs", "jobs", "scripts", "tests")
_SOURCE_FILES = ("setup.py", "requirements.txt", "README.md")
_IGNORE_PATTERNS = (
    "__pycache__", "*.pyc", "*.pyo", ".pytest_cache", ".mypy_cache",
    "*.egg-info", ".DS_Store",
)


def save_source_snapshot(output_dir: str, project_root: Optional[str] = None) -> None:
    """Copy source directories into ``<output_dir>/source/``.

    Mirrors the project layout so the snapshot is directly readable and can
    be executed in place. Skips caches and build artifacts.
    """
    if project_root is None:
        project_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )

    dst_root = os.path.join(output_dir, "source")
    os.makedirs(dst_root, exist_ok=True)
    ignore = shutil.ignore_patterns(*_IGNORE_PATTERNS)

    for name in _SOURCE_DIRS:
        src = os.path.join(project_root, name)
        if not os.path.isdir(src):
            continue
        dst = os.path.join(dst_root, name)
        if os.path.exists(dst):
            shutil.rmtree(dst)
        try:
            shutil.copytree(src, dst, ignore=ignore, symlinks=False)
        except OSError:
            pass

    for name in _SOURCE_FILES:
        src = os.path.join(project_root, name)
        if not os.path.isfile(src):
            continue
        try:
            shutil.copy2(src, os.path.join(dst_root, name))
        except OSError:
            pass


def link_data_dir(output_dir: str, project_root: Optional[str] = None) -> None:
    """Create ``<output_dir>/data`` symlink pointing at the project data dir.

    Lets a user reproduce the run from within ``output_dir`` — configs that
    reference ``data/...`` still resolve because the symlink bridges into the
    real dataset location one level above the results tree.
    """
    if project_root is None:
        project_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    data_src = os.path.join(project_root, "data")
    if not os.path.isdir(data_src):
        return
    link_path = os.path.join(output_dir, "data")
    if os.path.islink(link_path) or os.path.exists(link_path):
        return
    try:
        rel = os.path.relpath(data_src, output_dir)
        os.symlink(rel, link_path)
    except OSError:
        pass


def copy_original_config(output_dir: str, config_path: Optional[str] = None) -> None:
    """Copy the original (unresolved) config YAML into output_dir."""
    if config_path and os.path.isfile(config_path):
        dst = os.path.join(output_dir, "original_config.yaml")
        shutil.copy2(config_path, dst)


def archive_experiment(
    output_dir: str,
    config: Dict[str, Any],
    config_path: Optional[str] = None,
    start_time: Optional[datetime] = None,
    command: Optional[str] = None,
) -> None:
    """Run all archival steps after experiment completes."""
    if start_time is None:
        start_time = datetime.now()
    save_run_metadata(output_dir, config, start_time, command=command)
    save_source_snapshot(output_dir)
    copy_original_config(output_dir, config_path)
    link_data_dir(output_dir)
