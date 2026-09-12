"""ExperimentRunner — single training run from config.

Replaces the core logic of ``scripts/train_property.py``.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import torch

from wyckoff_gnn.utils.config import load_config as _load_config
from wyckoff_gnn.utils.seed import set_seed
from wyckoff_gnn.utils.logging import get_logger
from wyckoff_gnn.utils.io import save_json
from wyckoff_gnn.data.datamodule import PropertyDataModule
from wyckoff_gnn.data.normalization import TargetNormalizer
from wyckoff_gnn.models.factory import (
    create_model, validate_model_graph_type, TRAINABLE_TORCH_MODELS,
)
from wyckoff_gnn.training.loop import train_and_evaluate
from wyckoff_gnn.training.evaluator import (
    collect_predictions,
    compute_results,
    write_experiment_outputs,
    write_status,
)

log = get_logger(__name__)


class ExperimentRunner:
    """Orchestrate a single training run from a config dict.

    Usage::

        runner = ExperimentRunner(config)
        results = runner.run_train()
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = dict(config)

    @classmethod
    def from_config_file(
        cls, path: str, overrides: Optional[List[str]] = None,
    ) -> "ExperimentRunner":
        config = _load_config(path, overrides=overrides)
        return cls(config)

    def run_train(
        self,
        output_dir: Optional[str] = None,
        device: Optional[str] = None,
        resume_from: Optional[str] = None,
        config_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute full training pipeline.

        Args:
            output_dir: Override output directory. If None, uses config.
            device: Override device. If None, uses config.
            resume_from: Path to checkpoint.pt for resuming.
            config_path: Path to original config file (for archival).

        Returns a dict with ``test_mae``, ``test_r2``, ``params``, etc.
        """
        from datetime import datetime
        start_time = datetime.now()

        config = self.config
        if output_dir:
            out = output_dir
        else:
            from wyckoff_gnn.utils.auto_config import resolve_output_dir
            out = resolve_output_dir(config, command="train")

        os.makedirs(out, exist_ok=True)

        # Snapshot source + config into the run folder immediately so the
        # experiment is inspectable (and reproducible in place) from the
        # very start — before training even begins.
        try:
            from wyckoff_gnn.utils.experiment_archive import (
                save_source_snapshot,
                copy_original_config,
                link_data_dir,
            )
            save_source_snapshot(out)
            copy_original_config(out, config_path)
            link_data_dir(out)
        except Exception as _e:  # noqa: BLE001
            log.warning(f"Initial source snapshot failed: {_e}")

        # Auto-detect resume checkpoint if resume_from is "auto"
        if resume_from == "auto":
            ckpt_path = os.path.join(out, "checkpoint.pt")
            resume_from = ckpt_path if os.path.exists(ckpt_path) else None

        write_status(os.path.join(out, "status.json"), "initializing")

        # Seed.
        seed = config.get("seed", 42)
        set_seed(seed)
        config["seed"] = seed

        # Auto-configure hardware-dependent parameters.
        from wyckoff_gnn.utils.auto_config import resolve_auto_config
        resolve_auto_config(config)

        # Persist the RESOLVED config (post-auto) so users can see actual
        # batch_size, num_workers, pin_memory, etc. selected for this run.
        try:
            save_json(os.path.join(out, "resolved_config.json"), config)
        except Exception as _e:  # noqa: BLE001
            log.warning(f"Saving resolved_config.json failed: {_e}")

        # Emit a compact summary so key values are visible in stdout/slurm_job.out.
        hw = config.get("_hardware", {}) or {}
        log.info(
            "Resolved auto-config: "
            f"batch_size={config.get('batch_size')}, "
            f"val_batch_size={config.get('val_batch_size')}, "
            f"num_workers={config.get('num_workers')}, "
            f"pin_memory={config.get('pin_memory')}, "
            f"persistent_workers={config.get('persistent_workers')}, "
            f"prefetch_factor={config.get('prefetch_factor')}, "
            f"torch_compile={config.get('torch_compile')} | "
            f"GPU={hw.get('gpu_name')} "
            f"(free={hw.get('gpu_free_memory_gb')}/{hw.get('gpu_memory_gb')}GB), "
            f"cpus={hw.get('available_cpus')}"
        )

        # Device.
        dev_str = device or config.get("device", "auto")
        if dev_str == "auto":
            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            dev = torch.device(dev_str)
        log.info(f"Using device: {dev}")

        # Model validation.
        model_type = config.get("model_type", "wyckoff_gnn")
        if model_type not in TRAINABLE_TORCH_MODELS:
            raise ValueError(
                f"Model '{model_type}' is not a trainable PyTorch model. "
                f"Trainable types: {sorted(TRAINABLE_TORCH_MODELS)}."
            )
        validate_model_graph_type(model_type, config.get("graph_type", "wyckoff"))

        # Data.
        write_status(os.path.join(out, "status.json"), "building_data")
        manifest = config.get("manifest_path", "")
        shard_dir = config.get("shard_dir", os.path.dirname(manifest) if manifest else "")
        if not manifest or not os.path.exists(manifest):
            raise FileNotFoundError(
                f"manifest_path not found: {manifest}. "
                f"Run preprocess_dataset.py first."
            )

        dm = PropertyDataModule(
            manifest_path=manifest,
            shard_dir=shard_dir,
            batch_size=config.get("batch_size", 32),
            val_batch_size=config.get("val_batch_size"),
            num_workers=config.get("num_workers", 0),
            pin_memory=config.get("pin_memory", True),
            persistent_workers=config.get("persistent_workers", False),
            prefetch_factor=config.get("prefetch_factor"),
            max_train_samples=config.get("max_train_samples"),
            max_val_samples=config.get("max_val_samples"),
            max_test_samples=config.get("max_test_samples"),
            global_max_mult=config.get("global_max_mult", 192),
            leaderboard_split_zip=config.get("leaderboard_split_zip"),
            leaderboard_override_targets=config.get("leaderboard_override_targets", True),
            normalizer=(
                TargetNormalizer.from_dict(config["target_stats"])
                if config.get("target_stats") else None
            ),
        ).setup()

        # Persist official-split provenance if used.
        if dm.split_report is not None:
            save_json(os.path.join(out, "split_report.json"), dm.split_report)
            config["split_source"] = dm.split_report["source"]
            config["split_matched_counts"] = dm.split_report["matched_counts"]
            config["split_missing_counts"] = dm.split_report["missing_counts"]
            log.info(
                f"Split source: {dm.split_report['source']} | "
                f"train={dm.train_size}, val={dm.val_size}, test={dm.test_size}"
            )
        else:
            config["split_source"] = "manifest"

        config["train_size"] = dm.train_size
        config["val_size"] = dm.val_size
        config["test_size"] = dm.test_size
        config["target_mean"] = dm.normalizer.mean
        config["target_std"] = dm.normalizer.std

        # Dataset summary.
        all_entries = (
            dm.train_dataset.entries + dm.val_dataset.entries
            + dm.test_dataset.entries
        )
        n = len(all_entries)
        avg_atoms = sum(e.get("num_atoms", 0) for e in all_entries) / max(n, 1)
        avg_orbits = sum(e.get("num_orbits", 0) for e in all_entries) / max(n, 1)
        avg_edges = sum(e.get("num_geo_edges", 0) for e in all_entries) / max(n, 1)
        all_targets = [e["target"] for e in all_entries if e.get("target") is not None]
        sg_counts: Dict[str, int] = {}
        for e in all_entries:
            sg = str(e.get("space_group", 0))
            sg_counts[sg] = sg_counts.get(sg, 0) + 1
        save_json(os.path.join(out, "dataset_summary.json"), {
            "n_samples": n,
            "n_train": dm.train_size,
            "n_val": dm.val_size,
            "n_test": dm.test_size,
            "target_mean": dm.normalizer.mean,
            "target_std": dm.normalizer.std,
            "target_min": min(all_targets) if all_targets else None,
            "target_max": max(all_targets) if all_targets else None,
            "avg_atoms": avg_atoms,
            "avg_orbits": avg_orbits,
            "avg_edges": avg_edges,
            "compression_ratio": avg_atoms / max(avg_orbits, 1),
            "space_group_histogram": sg_counts,
        })

        # Model.
        write_status(os.path.join(out, "status.json"), "training")
        model = create_model(config)
        n_params = sum(p.numel() for p in model.parameters())
        log.info(f"Model: {model_type}, params={n_params:,}")

        # Log block details for unified_quotient_equivariant
        if model_type == "unified_quotient_equivariant":
            block_type = config.get("block_type", "scalar")
            block_class = type(model.blocks[0]).__name__ if hasattr(model, 'blocks') and len(model.blocks) > 0 else "N/A"
            hidden_irreps = config.get("hidden_irreps", "N/A")
            log.info(f"block_type={block_type}")
            log.info(f"block_class={block_class}")
            log.info(f"hidden_irreps={hidden_irreps}")
            # Assert block class matches expectation
            if block_type == "dynamic_tp":
                assert block_class == "UnifiedDynamicTPBlock", (
                    f"Expected UnifiedDynamicTPBlock for block_type='dynamic_tp', got {block_class}"
                )

        # Apply reference (Keras-compatible) initialization for Unified models.
        # This ensures the scalar subset (128x0e) matches CoGN init exactly.
        if model_type == "unified_quotient_equivariant" and hasattr(model, "apply_reference_init"):
            model.apply_reference_init()
            log.info("Reference initialization applied (Keras-compatible)")

        if config.get("torch_compile", False):
            try:
                model = torch.compile(model)
                log.info("torch.compile enabled")
            except Exception as e:
                log.warning(f"torch.compile failed, falling back to eager: {e}")

        # Resolve property_type for loss dispatch from the model's PropertySpec.
        # The registry uses fine-grained PhysicalType strings (e.g.
        # "graph_rank2_diagonal_canonical") but the loss function expects
        # coarser PropertyType values (e.g. "graph_tensor").
        _PHYSICAL_TO_LOSS_TYPE = {
            "graph_scalar_intensive": "graph_scalar_intensive",
            "graph_scalar_extensive": "graph_scalar_extensive",
            "graph_rank2_symmetric_tensor": "graph_tensor",
            "graph_rank2_diagonal_canonical": "graph_tensor",
            "graph_rank4_elastic_tensor": "graph_tensor",
            "graph_rank3_piezo_tensor": "graph_tensor",
            "graph_vector_polar": "graph_vector",
            "graph_vector_axial": "graph_vector",
            "atom_scalar": "atom_scalar",
            "atom_vector_polar": "atom_vector",
            "atom_vector_axial": "atom_vector",
            "atom_rank2_symmetric_tensor": "atom_tensor",
            "hamiltonian": "hamiltonian",
        }
        spec = getattr(model, "_property_spec", None)
        if spec is not None:
            physical = spec.physical_type
            config["property_type"] = _PHYSICAL_TO_LOSS_TYPE.get(physical, physical)
            log.info(
                f"Property: {spec.name} (physical={physical}, "
                f"loss_type={config['property_type']})"
            )

        # Train.
        training_results = train_and_evaluate(
            model=model,
            train_loader=dm.train_dataloader(),
            val_loader=dm.val_dataloader(),
            test_loader=dm.test_dataloader(),
            config=config,
            output_dir=out,
            device=dev,
            resume_from=resume_from,
        )

        # Evaluate on test.
        write_status(os.path.join(out, "status.json"), "evaluating")
        best_path = os.path.join(out, "best.pt")
        if os.path.exists(best_path):
            model.load_state_dict(torch.load(best_path, map_location="cpu", weights_only=True))
        model = model.to(dev)

        metadata_map = _build_metadata_map(dm)
        # collect_predictions dispatches on _property_spec.output_head. Set it
        # here, after training, so tensor targets reach the tensor collector
        # without touching the property_type resolution used for the loss.
        if getattr(model, "_property_spec", None) is None:
            from wyckoff_gnn.properties.registry import get_property_spec
            try:
                model._property_spec = get_property_spec(config["property_name"])
            except (KeyError, TypeError):
                log.warning(
                    f"property_name={config.get('property_name')!r} not in the "
                    "registry; prediction dump falls back to the scalar path"
                )
        predictions = collect_predictions(
            model, dm.test_dataloader(), dm.normalizer, dev,
            metadata_map=metadata_map,
            task=config.get("task") or config.get("property_name"),
        )
        results_full = compute_results(predictions, dm.normalizer, model, config)
        training_results.update(results_full)
        results_full = training_results

        write_experiment_outputs(out, config, predictions, results_full, model=model)

        # Split.json.
        split_map = _build_split_map(dm)
        save_json(os.path.join(out, "split.json"), split_map)

        # Archive experiment artifacts for reproducibility.
        from wyckoff_gnn.utils.experiment_archive import archive_experiment
        archive_experiment(
            output_dir=out,
            config=config,
            config_path=config_path,
            start_time=start_time,
            command=f"wyckoffgnn train --config {config_path or '?'}",
        )

        write_status(os.path.join(out, "status.json"), "outputs_written")
        log.info(f"Done. Results saved to {out}")
        return results_full

    def run_evaluate(
        self,
        checkpoint_path: str,
        output_dir: Optional[str] = None,
        split: str = "test",
        device: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Evaluate a trained checkpoint on a specific data split.

        Args:
            checkpoint_path: Path to ``best.pt`` or ``last.pt``.
            output_dir: Where to save outputs (default: ``config.output_dir/eval_<split>``).
            split: Which split to evaluate (``train``/``val``/``test``).
            device: Torch device string.

        Returns:
            Dict with ``test_mae``, ``test_r2``, ``params``, etc.
        """
        config = self.config
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be train/val/test, got {split!r}")

        # Auto-configure hardware-dependent parameters.
        from wyckoff_gnn.utils.auto_config import resolve_auto_config
        resolve_auto_config(config)

        if output_dir and output_dir != "auto":
            out = output_dir
        else:
            from wyckoff_gnn.utils.auto_config import resolve_output_dir
            # Force fresh generation for eval
            config_for_eval = dict(config)
            config_for_eval["output_dir"] = "auto"
            out = resolve_output_dir(config_for_eval, command="eval")
        os.makedirs(out, exist_ok=True)
        try:
            from wyckoff_gnn.utils.experiment_archive import (
                save_source_snapshot,
                link_data_dir,
            )
            save_source_snapshot(out)
            link_data_dir(out)
        except Exception as _e:  # noqa: BLE001
            log.warning(f"Eval source snapshot failed: {_e}")

        # Persist the RESOLVED eval config with actual batch_size, workers, etc.
        try:
            save_json(os.path.join(out, "resolved_config.json"), config)
        except Exception as _e:  # noqa: BLE001
            log.warning(f"Saving resolved_config.json failed: {_e}")

        hw = config.get("_hardware", {}) or {}
        log.info(
            "Resolved auto-config (eval): "
            f"batch_size={config.get('batch_size')}, "
            f"num_workers={config.get('num_workers')}, "
            f"pin_memory={config.get('pin_memory')} | "
            f"GPU={hw.get('gpu_name')} "
            f"(free={hw.get('gpu_free_memory_gb')}/{hw.get('gpu_memory_gb')}GB), "
            f"cpus={hw.get('available_cpus')}"
        )
        write_status(os.path.join(out, "status.json"), "initializing")

        # Seed and device.
        set_seed(config.get("seed", 42))
        dev_str = device or config.get("device", "auto")
        dev = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if dev_str == "auto" else torch.device(dev_str)
        )

        # Model validation.
        model_type = config.get("model_type", "wyckoff_gnn")
        if model_type not in TRAINABLE_TORCH_MODELS:
            raise ValueError(f"Model '{model_type}' is not trainable.")
        validate_model_graph_type(model_type, config.get("graph_type", "wyckoff"))

        # Checkpoint.
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Data.
        write_status(os.path.join(out, "status.json"), "building_data")
        manifest = config.get("manifest_path", "")
        shard_dir = config.get("shard_dir", os.path.dirname(manifest) if manifest else "")
        if not manifest or not os.path.exists(manifest):
            raise FileNotFoundError(f"manifest_path not found: {manifest}")

        dm = PropertyDataModule(
            manifest_path=manifest, shard_dir=shard_dir,
            batch_size=config.get("batch_size", 32),
            num_workers=config.get("num_workers", 0),
            pin_memory=config.get("pin_memory", True),
            persistent_workers=config.get("persistent_workers", False),
            prefetch_factor=config.get("prefetch_factor"),
            global_max_mult=config.get("global_max_mult", 192),
            leaderboard_split_zip=config.get("leaderboard_split_zip"),
            normalizer=(
                TargetNormalizer.from_dict(config["target_stats"])
                if config.get("target_stats") else None
            ),
        ).setup()

        # Model + load checkpoint.
        write_status(os.path.join(out, "status.json"), "loading_checkpoint")
        model = create_model(config)
        
        # Set property spec for evaluator dispatch
        from wyckoff_gnn.properties.registry import get_property_spec
        property_name = config.get("property_name")
        if property_name:
            try:
                model._property_spec = get_property_spec(property_name)
            except KeyError:
                log.warning(f"Property '{property_name}' not in registry, using default")
        
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        state_dict = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state_dict)
        model = model.to(dev)
        model.eval()

        # Choose dataloader.
        loaders = {
            "train": dm.train_dataloader(),
            "val": dm.val_dataloader(),
            "test": dm.test_dataloader(),
        }
        loader = loaders[split]
        if split == "test":
            dataset = dm.test_dataset
        elif split == "val":
            dataset = dm.val_dataset
        else:
            dataset = dm.train_dataset

        # Dataset summary.
        all_entries = dataset.entries
        n = len(all_entries)
        save_json(os.path.join(out, "dataset_summary.json"), {
            "n_samples": n,
            "split": split,
            "target_mean": dm.normalizer.mean,
            "target_std": dm.normalizer.std,
        })

        # Evaluate.
        write_status(os.path.join(out, "status.json"), "evaluating")
        metadata_map = _build_metadata_map(dm)
        predictions = collect_predictions(
            model, loader, dm.normalizer, dev,
            metadata_map=metadata_map,
            task=config.get("task") or config.get("property_name"),
        )
        results = compute_results(predictions, dm.normalizer, model, config)
        results["checkpoint"] = checkpoint_path
        results["split"] = split

        write_experiment_outputs(out, config, predictions, results, model=model)
        write_status(os.path.join(out, "status.json"), "outputs_written")
        log.info(f"Evaluate done. Results saved to {out}")
        return results


def _build_metadata_map(dm: PropertyDataModule) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    for ds, split_name in [
        (dm.train_dataset, "train"),
        (dm.val_dataset, "val"),
        (dm.test_dataset, "test"),
    ]:
        for i in range(len(ds)):
            e = ds.entry(i)
            result[e["material_id"]] = {
                "num_atoms": e.get("num_atoms", 0),
                "num_orbits": e.get("num_orbits", 0),
                "num_geo_edges": e.get("num_geo_edges", 0),
                "space_group": e.get("space_group", 0),
                "split": split_name,
            }
    return result


def _build_split_map(dm: PropertyDataModule) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for ds, split_name in [
        (dm.train_dataset, "train"),
        (dm.val_dataset, "val"),
        (dm.test_dataset, "test"),
    ]:
        for e in ds.entries:
            result[e["material_id"]] = split_name
    return result


__all__ = ["ExperimentRunner"]
