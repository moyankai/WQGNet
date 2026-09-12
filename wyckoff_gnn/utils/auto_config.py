"""Hardware-aware auto-configuration for training parameters.

Resolves "auto" values in config based on available hardware and model
architecture. Goes beyond simple memory-tier heuristics by:

- Estimating model memory footprint from irreps/layers/TP mode
- Scaling batch_size to target GPU memory utilization (~70%)
- Detecting SLURM CPU allocation for optimal num_workers
- Auto-enabling persistent_workers, prefetch_factor, torch.compile
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any, Dict

log = logging.getLogger(__name__)


def _parse_irreps_dim(irreps_str: str) -> int:
    """Compute total dimension from an irreps string like '64x0e+32x1o+16x2e'.

    Pure-Python parser: each term NxLp contributes N*(2L+1) dimensions.
    Falls back to e3nn.o3.Irreps if parsing fails.
    """
    import re
    total = 0
    terms = re.split(r'[+\s]+', irreps_str.strip())
    for term in terms:
        if not term:
            continue
        m = re.match(r'(\d+)x(\d+)[eo]', term)
        if m:
            mult, l_val = int(m.group(1)), int(m.group(2))
            total += mult * (2 * l_val + 1)
        else:
            # Non-standard format, delegate to e3nn
            from e3nn import o3
            return o3.Irreps(irreps_str).dim
    return total if total > 0 else 240


def _estimate_params(config: Dict[str, Any]) -> int:
    """Estimate model parameter count from config without constructing the model.

    This is a rough analytical estimate. Dominant costs:
    - Per layer: TP weights ~ hidden_dim * message_dim + radial MLP
    - Edge state modules (if enabled)
    - Readout head
    """
    hidden_irreps = config.get("hidden_irreps", "64x0e+16x1o+8x2e")
    num_layers = config.get("num_layers", 4)
    num_rbf = config.get("num_rbf", 32)
    radial_mlp_width = config.get("radial_mlp_width", 128)
    use_edge_state = config.get("use_edge_state", False)
    edge_state_dim = config.get("edge_state_dim", 64)
    edge_state_layers = config.get("edge_state_layers", 2)

    try:
        hidden_dim = _parse_irreps_dim(hidden_irreps)
    except Exception:
        hidden_dim = 240  # fallback

    # TP weights per layer: roughly hidden_dim * hidden_dim (weight matrix equivalent)
    # plus radial MLP: num_rbf -> radial_mlp_width -> radial_mlp_width -> n_paths
    n_tp_paths = hidden_dim  # approximate
    radial_mlp_params = (
        num_rbf * radial_mlp_width
        + radial_mlp_width * radial_mlp_width
        + radial_mlp_width * n_tp_paths
    )
    layer_params = hidden_dim * hidden_dim + radial_mlp_params + hidden_dim * 2  # norms
    total = layer_params * num_layers

    # Node encoder + readout
    total += hidden_dim * 128  # encoder embedding
    total += hidden_dim * 64 + 64  # readout MLP

    # Edge state modules
    if use_edge_state:
        es_params = (
            num_rbf * edge_state_dim
            + edge_state_dim * edge_state_dim * edge_state_layers
            + edge_state_dim * hidden_dim
        )
        total += es_params * num_layers

    return int(total)


def _estimate_per_sample_bytes(config: Dict[str, Any]) -> float:
    """Estimate peak activation memory per graph sample (bytes).

    For e3nn equivariant GNNs, memory is dominated by tensor product (TP)
    operations on edges. The TP creates large intermediate tensors proportional
    to hidden_dim × SH_dim per edge, and autograd retains the full graph.

    Calibrated against empirical measurement:
    - Model: 32x0e+16x1o (dim=80), 2 layers, batch=256 → 2.4GB peak
    - Per-sample empirical: ~9.4MB

    Formula: hidden_dim * num_layers * AVG_EDGES * CALIBRATION_BYTES
    where CALIBRATION_BYTES ≈ 294 accounts for TP intermediates, autograd
    overhead, and message passing buffers in e3nn.
    """
    hidden_irreps = config.get("hidden_irreps", "64x0e+16x1o+8x2e")
    num_layers = config.get("num_layers", 4)

    try:
        hidden_dim = _parse_irreps_dim(hidden_irreps)
    except Exception:
        hidden_dim = 240

    # Typical Wyckoff-compressed crystal graph: 4-8 orbit nodes, each with
    # 20-40 geometric neighbors → 100-300 edges per graph (cutoff 8Å).
    avg_edges = 200

    # Empirical calibration constant (bytes per hidden_dim * layer * edge).
    # Derived from: 9.4MB = C * 80 * 2 * 200 → C ≈ 294
    # Accounts for: TP intermediates, CG coefficients, gate activations,
    # autograd graph overhead, and backward pass retention.
    CALIBRATION = 294

    per_sample = CALIBRATION * hidden_dim * num_layers * avg_edges

    # Edge state modules add ~30% overhead when enabled
    if config.get("use_edge_state", False):
        per_sample *= 1.3

    return per_sample


def _get_available_cpus() -> int:
    """Get number of CPUs available to this process.

    Priority order:
    1. SLURM_CPUS_PER_TASK (most accurate for slurm jobs)
    2. SLURM_CPUS_ON_NODE / SLURM_NTASKS_PER_NODE
    3. len(os.sched_getaffinity(0)) (cgroup-aware)
    4. os.cpu_count() (fallback)
    """
    # SLURM environment
    cpus_per_task = os.environ.get("SLURM_CPUS_PER_TASK")
    if cpus_per_task:
        return int(cpus_per_task)

    cpus_on_node = os.environ.get("SLURM_CPUS_ON_NODE")
    ntasks = os.environ.get("SLURM_NTASKS_PER_NODE", "1")
    if cpus_on_node:
        return int(cpus_on_node) // int(ntasks)

    # cgroup-aware count (respects taskset, docker, etc.)
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        pass

    return os.cpu_count() or 1


def _get_gpu_info() -> Dict[str, Any]:
    """Get GPU hardware info for the currently selected device.

    Returns both total and *actually free* memory so callers can budget
    correctly when the GPU is shared (multi-user nodes) or already holds
    another process's allocations. Works uniformly across any CUDA GPU
    (A100/A800/H100/V100/P100/RTX/…): all sizing decisions downstream
    depend on the reported ``free_memory_gb``, not the card model.
    """
    import torch
    if not torch.cuda.is_available():
        return {"available": False}

    device_idx = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device_idx)

    # mem_get_info requires the CUDA context to be initialized on this device.
    # Allocate a tiny tensor to force lazy init if it hasn't happened yet.
    try:
        _ = torch.empty(1, device=f"cuda:{device_idx}")
        torch.cuda.empty_cache()
        free_bytes, total_bytes = torch.cuda.mem_get_info(device_idx)
    except Exception as e:
        log.warning(
            f"torch.cuda.mem_get_info failed ({e}); "
            "falling back to props.total_memory (may over-estimate on shared GPUs)."
        )
        free_bytes, total_bytes = props.total_memory, props.total_memory

    return {
        "available": True,
        "device_index": device_idx,
        "name": props.name,
        "total_memory_gb": total_bytes / 1024**3,
        "free_memory_gb": free_bytes / 1024**3,
        "major": props.major,
        "minor": props.minor,
        "multi_processor_count": props.multi_processor_count,
        "is_ampere_plus": props.major >= 8,  # sm_80+ = Ampere, Ada, Hopper
    }


def resolve_auto_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve 'auto' values in config based on detected hardware and model size.

    Modifies config in-place and returns it. Only overrides values
    that are explicitly set to "auto" or not present.
    """
    import torch

    gpu_info = _get_gpu_info()
    has_cuda = gpu_info["available"]
    available_cpus = _get_available_cpus()

    # --- batch_size ---
    bs = config.get("batch_size", "auto")
    if bs == "auto":
        if has_cuda:
            # Use ACTUALLY FREE memory, not card total. This is the only
            # value that generalizes across (a) different GPU models and
            # (b) shared multi-user nodes where other processes already
            # hold allocations. On dedicated GPUs, free ≈ total.
            free_gb = gpu_info["free_memory_gb"]
            total_gb = gpu_info["total_memory_gb"]

            # Safety headroom: fragmentation, PyTorch caching allocator
            # overhead, workspace allocations for cuDNN/cuBLAS, etc.
            SAFETY_HEADROOM_GB = 1.0
            usable_gb = max(0.0, free_gb - SAFETY_HEADROOM_GB)

            # Refuse to run if less than 2 GB is actually free. Fall back
            # to a small, safe batch and let the user notice from logs.
            MIN_USABLE_GB = 2.0
            if usable_gb < MIN_USABLE_GB:
                config["batch_size"] = 32
                log.warning(
                    f"Only {free_gb:.2f} GB free of {total_gb:.1f} GB on "
                    f"{gpu_info['name']} — GPU is likely shared. "
                    f"Using conservative batch_size=32."
                )
            else:
                # Target 70% of usable (free) memory for data + activations.
                target_util = 0.70
                usable_bytes = usable_gb * 1024**3

                # Model base memory: params + grads + Adam m/v (fp32).
                est_params = _estimate_params(config)
                model_base_bytes = est_params * 4 * 4

                available_for_data = usable_bytes * target_util - model_base_bytes
                per_sample = _estimate_per_sample_bytes(config)

                if per_sample > 0 and available_for_data > 0:
                    optimal_bs = int(available_for_data / per_sample)
                    optimal_bs = max(32, min(1024, optimal_bs))
                    optimal_bs = (optimal_bs // 32) * 32
                    config["batch_size"] = optimal_bs
                else:
                    config["batch_size"] = 64

                log.info(
                    f"Auto batch_size={config['batch_size']} "
                    f"(GPU: {gpu_info['name']}, free={free_gb:.1f}/{total_gb:.1f}GB, "
                    f"est_params={est_params:,}, per_sample≈{per_sample/1024:.1f}KB)"
                )
        else:
            config["batch_size"] = 16

    # --- val_batch_size ---
    vbs = config.get("val_batch_size", "auto")
    if vbs == "auto":
        # Validation has no gradients/optimizer → can use ~2x train batch
        train_bs = config.get("batch_size", 64)
        config["val_batch_size"] = min(train_bs * 2, 1024)

    # --- num_workers ---
    nw = config.get("num_workers", "auto")
    if nw == "auto":
        if has_cuda:
            # Heuristic: use most available CPUs, but leave 1-2 for main process
            # and cap based on batch_size (diminishing returns if workers > bs/4)
            train_bs = config.get("batch_size", 64)
            max_useful = max(1, train_bs // 4)
            workers = min(available_cpus - 2, max_useful, 16)
            config["num_workers"] = max(2, workers)
        else:
            config["num_workers"] = min(2, available_cpus)
        log.info(
            f"Auto num_workers={config['num_workers']} "
            f"(available_cpus={available_cpus})"
        )

    # --- pin_memory ---
    pm = config.get("pin_memory", "auto")
    if pm == "auto":
        config["pin_memory"] = has_cuda

    # --- persistent_workers ---
    pw = config.get("persistent_workers", "auto")
    if pw == "auto":
        nw_resolved = config.get("num_workers", 0)
        config["persistent_workers"] = nw_resolved > 0

    # --- prefetch_factor ---
    pf = config.get("prefetch_factor", "auto")
    if pf == "auto":
        nw_resolved = config.get("num_workers", 0)
        if nw_resolved > 0:
            # More workers → lower prefetch needed. Fewer workers → prefetch more
            if nw_resolved >= 8:
                config["prefetch_factor"] = 2
            else:
                config["prefetch_factor"] = 4
        else:
            config["prefetch_factor"] = None  # not applicable with 0 workers

    # --- torch_compile ---
    tc = config.get("torch_compile", "auto")
    if tc == "auto":
        # e3nn models use Wigner D-matrices with dynamic shapes that are
        # incompatible with torch.compile (dynamo graph breaks on D_from_matrix).
        config["torch_compile"] = False

    # --- Log resolved hardware context ---
    config.setdefault("_hardware", {}).update({
        "gpu_name": gpu_info.get("name", "none"),
        "gpu_index": gpu_info.get("device_index"),
        "gpu_memory_gb": round(gpu_info.get("total_memory_gb", 0), 1),
        "gpu_free_memory_gb": round(gpu_info.get("free_memory_gb", 0), 2),
        "gpu_sm": f"{gpu_info.get('major', 0)}.{gpu_info.get('minor', 0)}" if has_cuda else None,
        "available_cpus": available_cpus,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    })

    return config


def generate_run_name(config: Dict[str, Any], command: str = "train") -> str:
    """Generate a descriptive run directory name from config parameters.

    Format: [command]-[dataset]-[irreps]-L[layers]-[tp_mode]-[task]-s[seed]-[YYYYMMDD-HHmm]
    Example: train-mp20-32x0e_16x1o-L2-dynamic_v2-formation_energy-s0-20260704-1510
    
    The seed is included to ensure different random seeds produce different
    directories, preventing collisions when multiple seeds are run simultaneously.
    """
    parts = [command]

    dataset = config.get("dataset", "data")
    parts.append(dataset)

    model_type = config.get("model_type", "")

    irreps = config.get("hidden_irreps", "")
    if irreps:
        parts.append(irreps.replace(" ", "").replace("+", "_"))

    # Angular interaction config
    n_l1 = config.get("n_channels_l1", 0)
    n_l2 = config.get("n_channels_l2", 0)
    if n_l1 or n_l2:
        ang_parts = []
        if n_l1:
            ang_parts.append(f"{n_l1}x1o")
        if n_l2:
            ang_parts.append(f"{n_l2}x2e")
        parts.append(f"ANG{'_'.join(ang_parts)}")

    parts.append(f"L{config.get('num_layers', '?')}")

    tp = config.get("tp_mode", "")
    if tp:
        parts.append(tp)

    task = config.get("task", "")
    if task:
        parts.append(task)

    seed = config.get("seed")
    if seed is not None:
        parts.append(f"s{seed}")

    timestamp_output = config.get("timestamp_output", True)
    if timestamp_output:
        parts.append(datetime.now().strftime("%Y%m%d-%H%M"))

    return "-".join(parts)


def resolve_output_dir(config: Dict[str, Any], command: str = "train") -> str:
    """Resolve output_dir from config. If 'auto' or absent, generate path.

    Structure: results/<run_name>   (single flat directory)

    Args:
        config: Full config dict.
        command: Command name prefix (train, eval, bench).

    Config keys:
        output_dir: "auto" or explicit path. If explicit, used as-is.
        output_root: base directory (default "results")
        timestamp_output: whether to append timestamp (default True)
    """
    output_dir = config.get("output_dir", "auto")
    if output_dir != "auto":
        return output_dir

    output_root = config.get("output_root", "results")
    run_name = generate_run_name(config, command=command)
    path = os.path.join(output_root, run_name)
    config["output_dir"] = path
    return path


def resolve_benchmark_output_root(config: Dict[str, Any]) -> str:
    """Resolve benchmark output_root from config.

    Format: results/bench-{name}-{YYYYMMDD-HHmm}

    Args:
        config: Full benchmark config dict (with 'benchmark' key).

    Returns:
        Resolved output root path.
    """
    bm_cfg = config.get("benchmark", {})
    output_root = bm_cfg.get("output_root", "auto")
    if output_root != "auto":
        return output_root

    bench_name = bm_cfg.get("name", "default")
    ts = datetime.now().strftime("%Y%m%d-%H%M")
    path = os.path.join("results", f"bench-{bench_name}-{ts}")
    bm_cfg["output_root"] = path
    return path
