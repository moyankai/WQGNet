"""Non-Lightning train/eval loop for WyckoffGNN property prediction.

Modeled after the proven ``jarvis_official_benchmark.py`` training logic.
Supports EMA, early stopping, plateau and cosine schedulers, gradient
clipping, and checkpoint saving.
"""

from __future__ import annotations

import csv
import os
import time
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from wyckoff_gnn.training.evaluator import save_training_curve
from wyckoff_gnn.utils.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------

def _get_loss_fn(
    loss_name: str,
    property_type: str = "graph_scalar_intensive",
    loss_config: Optional[Dict[str, Any]] = None,
) -> Callable:
    """Return a loss callable appropriate for the property type.

    For graph_scalar_* types: returns f(pred, target) — same as before.
    For everything else: returns f(pred, batch) which internally extracts
    the correct target field. The returned callable has a `.needs_batch`
    attribute so train_epoch knows which calling convention to use.
    """
    from wyckoff_gnn.data.property_types import PropertyType, validate_property_type
    pt = validate_property_type(property_type)
    loss_config = loss_config or {}

    if pt in (PropertyType.GRAPH_SCALAR_INTENSIVE, PropertyType.GRAPH_SCALAR_EXTENSIVE):
        if loss_name == "mse":
            fn = F.mse_loss
        elif loss_name == "mae":
            fn = F.l1_loss
        elif loss_name == "huber":
            fn = lambda pred, target: F.smooth_l1_loss(pred, target)
        else:
            raise ValueError(f"Unknown loss: {loss_name}")
        fn.needs_batch = False  # type: ignore[attr-defined]
        return fn

    from wyckoff_gnn.training.losses import compute_loss

    def _batch_loss(pred, batch):
        return compute_loss(pred, batch, pt, loss_name, loss_config)

    _batch_loss.needs_batch = True  # type: ignore[attr-defined]
    return _batch_loss


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable,
    grad_clip: Optional[float],
    device: torch.device,
    ema_model: Optional[torch.optim.swa_utils.AveragedModel] = None,
    log_every: int = 50,
    max_batches: int = 0,
    step_scheduler=None,
    angular_lr_factor: float = 0.0,
) -> float:
    """One training epoch. Returns average loss.

    Args:
        log_every: Print timing every N batches (0 to disable).
        max_batches: Stop after N batches (0 = full epoch). For debugging.
    """
    model.train()
    total_loss = 0.0
    n_batches = 0
    n_samples = 0
    use_cuda = device.type == "cuda"
    epoch_t0 = time.time()

    t_data_acc = 0.0
    t_fwd_acc = 0.0
    t_bwd_acc = 0.0
    t_step_acc = 0.0

    t_iter_start = time.time()
    t_batch_end = time.time()  # marks end of previous batch (start of data loading)
    for batch_idx, batch in enumerate(loader):
        if max_batches > 0 and batch_idx >= max_batches:
            break
        # --- Data loading + transfer ---
        # Time includes DataLoader __getitem__ + collate + .to(device)
        t0 = time.time()
        batch = batch.to(device)
        if use_cuda:
            torch.cuda.synchronize()
        t_data = time.time() - t_batch_end  # from end of previous step to now

        # --- Forward ---
        t0 = time.time()
        pred = model(batch)
        if getattr(loss_fn, "needs_batch", False):
            loss = loss_fn(pred, batch)
        else:
            target = batch.y.view(-1).float().to(device)
            loss = loss_fn(pred.view(-1), target)
        if use_cuda:
            torch.cuda.synchronize()
        t_fwd = time.time() - t0

        # --- Backward ---
        t0 = time.time()
        optimizer.zero_grad()
        loss.backward()
        if grad_clip is not None and grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        if use_cuda:
            torch.cuda.synchronize()
        t_bwd = time.time() - t0

        # --- Optimizer step ---
        t0 = time.time()
        optimizer.step()
        if step_scheduler is not None:
            step_scheduler.step()
            base_lr = step_scheduler.get_lr()
            for pg in optimizer.param_groups:
                if pg.get("_is_angular"):
                    pg["lr"] = base_lr * angular_lr_factor
                else:
                    pg["lr"] = base_lr
        if ema_model is not None:
            ema_model.update_parameters(model)
        if use_cuda:
            torch.cuda.synchronize()
        t_step = time.time() - t0

        total_loss += loss.item()
        n_batches += 1
        bs = batch.num_graphs if hasattr(batch, "num_graphs") else 1
        n_samples += bs

        t_data_acc += t_data
        t_fwd_acc += t_fwd
        t_bwd_acc += t_bwd
        t_step_acc += t_step
        t_batch_end = time.time()

        if log_every > 0 and batch_idx % log_every == 0 and batch_idx > 0:
            elapsed = time.time() - t_iter_start
            sps = n_samples / elapsed if elapsed > 0 else 0
            mem_str = ""
            if use_cuda:
                mem_mb = torch.cuda.max_memory_allocated() / 1024**2
                mem_str = f"  mem={mem_mb:.0f}MB"
            log.info(
                f"  batch {batch_idx}: "
                f"data={t_data_acc/n_batches*1000:.0f}ms  "
                f"fwd={t_fwd_acc/n_batches*1000:.0f}ms  "
                f"bwd={t_bwd_acc/n_batches*1000:.0f}ms  "
                f"step={t_step_acc/n_batches*1000:.0f}ms  "
                f"loss={total_loss/n_batches:.4f}  "
                f"{sps:.1f} samples/s{mem_str}"
            )

    epoch_time = time.time() - epoch_t0
    sps = n_samples / epoch_time if epoch_time > 0 else 0
    log.info(
        f"  epoch done: {n_batches} batches, {n_samples} samples, "
        f"{epoch_time:.1f}s ({sps:.1f} samples/s)"
    )
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: Callable,
    device: torch.device,
    normalizer=None,
) -> Tuple[float, float, float]:
    """Evaluate on val/test set. Returns (avg_loss, mae, r2).

    If *normalizer* is provided the MAE and R² are computed in the
    original (denormalized) target space so they are directly comparable
    across runs.  The loss is always computed in normalized space (it is
    used for optimisation).

    For non-scalar property types (needs_batch=True), mae and r2 are
    computed from the batch-level loss as a proxy.
    """
    model.eval()
    total_loss = 0.0
    preds, targets = [], []
    n_batches = 0
    needs_batch = getattr(loss_fn, "needs_batch", False)

    for batch in loader:
        batch = batch.to(device)
        pred = model(batch)

        if needs_batch:
            loss = loss_fn(pred, batch)
            total_loss += loss.item()
            # Collect predictions and targets for proper metric computation
            if isinstance(pred, dict):
                # Tensor prediction: extract cartesian tensor
                tensor_out = pred.get("tensor", pred)
                if isinstance(tensor_out, dict):
                    # Rank-3 heads expose "voigt" (G,3,6) matching y_tensor;
                    # rank-2 heads only have "cartesian".
                    pred_cart = tensor_out.get(
                        "voigt", tensor_out.get("cartesian", tensor_out)
                    )
                else:
                    pred_cart = tensor_out
                pred_flat = pred_cart.reshape(pred_cart.size(0), -1)
            else:
                pred_flat = pred.reshape(pred.size(0), -1)
            
            # Get target from batch
            if hasattr(batch, "y_tensor"):
                target = batch.y_tensor.reshape(pred_flat.size(0), -1).to(device)
            elif hasattr(batch, "y"):
                target = batch.y.reshape(pred_flat.size(0), -1).to(device)
            else:
                target = None
            
            if target is not None:
                preds.append(pred_flat.cpu())
                targets.append(target.cpu())
        else:
            pred_flat = pred.view(-1)
            target = batch.y.view(-1).float().to(device)
            loss = loss_fn(pred_flat, target)
            total_loss += loss.item()
            preds.append(pred_flat.cpu())
            targets.append(target.cpu())

        n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)

    if preds:
        all_preds = torch.cat(preds)
        all_targets = torch.cat(targets)
        if normalizer is not None and all_preds.dim() == 1:
            # Only denormalize scalar targets
            all_preds = all_preds * normalizer.std + normalizer.mean
            all_targets = all_targets * normalizer.std + normalizer.mean
        mae = F.l1_loss(all_preds, all_targets).item()
        ss_res = ((all_preds - all_targets) ** 2).sum()
        ss_tot = ((all_targets - all_targets.mean()) ** 2).sum()
        r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    else:
        mae = avg_loss
        r2 = 0.0
    return avg_loss, mae, r2


def train_and_evaluate(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: Optional[DataLoader],
    config: Dict[str, Any],
    output_dir: str,
    device: Optional[torch.device] = None,
    resume_from: Optional[str] = None,
) -> Dict[str, Any]:
    """Full training run: train, validate, save checkpoints, test.

    Args:
        model: Model instance.
        train_loader: Training dataloader.
        val_loader: Validation dataloader.
        test_loader: Test dataloader (evaluated once at best epoch).
        config: Full resolved config dict.
        output_dir: Where to save outputs.
        device: Torch device (auto-detected if None).
        resume_from: Path to checkpoint.pt for resuming training.

    Returns:
        Results dict with metrics.
    """
    os.makedirs(output_dir, exist_ok=True)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Config.
    epochs = config.get("epochs", 200)
    lr = config.get("lr", 1e-3)
    weight_decay = config.get("weight_decay", 1e-5)
    loss_name = config.get("loss", "mae")
    grad_clip = config.get("grad_clip", 5.0)
    scheduler_name = config.get("scheduler", "plateau")
    patience = config.get("patience", 60)
    use_ema = config.get("ema", False)
    ema_decay = config.get("ema_decay", 0.999)
    log_every = config.get("log_every_n_batches", 50)
    max_train_batches = config.get("max_train_batches", 0)

    loss_fn = _get_loss_fn(
        loss_name,
        config.get("property_type", "graph_scalar_intensive"),
        config.get("loss_config"),
    )

    # Build normalizer for reporting metrics in original target space.
    _t_mean = config.get("target_mean")
    _t_std = config.get("target_std")
    if _t_mean is not None and _t_std is not None:
        from wyckoff_gnn.data.normalization import TargetNormalizer
        _normalizer = TargetNormalizer(mean=_t_mean, std=_t_std)
    else:
        _normalizer = None

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable_params)
    n_frozen = sum(p.numel() for p in model.parameters()) - n_trainable
    if n_frozen > 0:
        log.info(
            f"Optimizer: {n_trainable:,} trainable params, "
            f"{n_frozen:,} frozen params"
        )

    # Angular param groups: if model exposes angular_parameters(), split into
    # scalar trunk (baseline lr/wd) and angular (reduced lr, higher wd).
    angular_lr_factor = config.get("angular_lr_factor", 0.0)
    angular_params_set: set = set()
    if angular_lr_factor > 0:
        for block in getattr(model, "blocks", []):
            if hasattr(block, "angular_parameters"):
                for p in block.angular_parameters():
                    if p.requires_grad:
                        angular_params_set.add(id(p))

    if angular_params_set:
        angular_wd = config.get("angular_weight_decay", 1e-3)
        scalar_group = []
        angular_group = []
        for p in trainable_params:
            if id(p) in angular_params_set:
                angular_group.append(p)
            else:
                scalar_group.append(p)
        n_angular = sum(p.numel() for p in angular_group)
        n_scalar = sum(p.numel() for p in scalar_group)
        log.info(
            f"Param groups: scalar={n_scalar:,} (lr={lr:.1e}, wd={weight_decay:.1e}), "
            f"angular={n_angular:,} (lr={lr * angular_lr_factor:.1e}, wd={angular_wd:.1e})"
        )
        optimizer = torch.optim.AdamW([
            {"params": scalar_group, "lr": lr, "weight_decay": weight_decay},
            {"params": angular_group, "lr": lr * angular_lr_factor,
             "weight_decay": angular_wd, "_is_angular": True},
        ])
    else:
        optimizer = torch.optim.AdamW(
            trainable_params, lr=lr, weight_decay=weight_decay,
        )

    # Scheduler.
    if scheduler_name == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=patience // 3,
            min_lr=1e-6,
        )
    elif scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-6,
        )
    elif scheduler_name == "polynomial":
        from wyckoff_gnn.training.schedules import PolynomialDecaySchedule
        _bs = train_loader.batch_size or config.get("batch_size", 64)
        scheduler = PolynomialDecaySchedule(
            dataset_size=len(train_loader.dataset),
            batch_size=_bs,
            epochs=epochs,
            lr_start=lr,
            lr_stop=config.get("lr_stop", 1e-5),
            power=config.get("lr_power", 1.0),
        )
    else:
        scheduler = None

    # For constant scheduler, we don't need to do anything — lr stays fixed.
    if scheduler_name == "constant":
        scheduler = None

    # EMA.
    ema_model = None
    if use_ema:
        ema_model = torch.optim.swa_utils.AveragedModel(
            model, avg_fn=lambda avg_p, p, _: ema_decay * avg_p + (1 - ema_decay) * p,
        )

    # Training state.
    # Selection criterion is val_loss (SAME objective as train_loss), so
    # best-model / scheduler / early-stopping all track the actual training
    # objective. val_mae is recorded separately as the reporting metric.
    best_val_loss = float("inf")
    best_val_mae = float("inf")
    best_epoch = 0
    best_state = None
    best_ema_state = None
    no_improve = 0
    curve_rows: list = []
    start_epoch = 1

    # --- Resume from checkpoint ---
    if resume_from and os.path.exists(resume_from):
        ckpt = torch.load(resume_from, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if scheduler and ckpt.get("scheduler_state_dict"):
            if hasattr(scheduler, "from_state_dict"):
                restored = type(scheduler).from_state_dict(ckpt["scheduler_state_dict"])
                scheduler.__dict__.update(restored.__dict__)
            else:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        # Prefer new best_val_loss key; fall back to best_val_mae for older ckpts.
        best_val_loss = ckpt.get("best_val_loss", ckpt.get("best_val_mae", float("inf")))
        best_val_mae = ckpt.get("best_val_mae", float("inf"))
        best_epoch = ckpt["best_epoch"]
        best_state = ckpt.get("best_state")
        no_improve = ckpt.get("no_improve", 0)
        curve_rows = ckpt.get("curve_rows", [])
        if ema_model and ckpt.get("ema_state_dict"):
            ema_model.load_state_dict(ckpt["ema_state_dict"])
        if ckpt.get("rng_state") is not None:
            torch.set_rng_state(ckpt["rng_state"].cpu().byte())
        if device.type == "cuda" and ckpt.get("cuda_rng_state"):
            cuda_states = [s.cpu().byte() for s in ckpt["cuda_rng_state"]]
            torch.cuda.set_rng_state_all(cuda_states)
        log.info(
            f"Resumed from epoch {ckpt['epoch']}, "
            f"best_val_loss={best_val_loss:.4f}, best_val_mae={best_val_mae:.4f}"
        )

    t_start = time.time()

    for epoch in range(start_epoch, epochs + 1):
        train_loss = train_epoch(
            model, train_loader, optimizer, loss_fn, grad_clip, device, ema_model,
            log_every=log_every, max_batches=max_train_batches,
            step_scheduler=scheduler if scheduler_name == "polynomial" else None,
            angular_lr_factor=angular_lr_factor,
        )
        val_loss, val_mae, val_r2 = evaluate(model, val_loader, loss_fn, device, _normalizer)

        lr_now = optimizer.param_groups[0]["lr"]
        # Best is decided by val_loss (same objective as training loss).
        is_best = val_loss < best_val_loss

        # Report train_mae in original space for comparability with val_mae.
        # For tensor targets, don't scale by normalizer.std (it's fit on scalar proxy).
        property_type = config.get("property_type", "graph_scalar_intensive")
        is_tensor_target = property_type in ("graph_tensor", "graph_vector")
        
        if _normalizer is not None and loss_name == "mae" and not is_tensor_target:
            train_mae_raw = train_loss * _normalizer.std
            train_str = f"train_mae={train_mae_raw:.4f}"
        else:
            train_mae_raw = train_loss
            train_str = f"train_loss={train_loss:.4f}"

        log.info(
            f"Epoch {epoch:4d}/{epochs}  "
            f"{train_str}  "
            f"val_loss={val_loss:.4f}  "
            f"val_mae={val_mae:.4f}  val_r2={val_r2:.4f}  "
            f"lr={lr_now:.1e}  {'*best' if is_best else ''}"
        )

        curve_rows.append({
            "epoch": epoch, "train_loss": train_loss,
            "train_mae": train_mae_raw,
            "val_loss": val_loss, "val_mae": val_mae, "val_r2": val_r2,
            "lr": lr_now, "best": 1 if is_best else 0,
        })

        # Scheduler follows the training objective, not the reporting metric.
        if scheduler_name == "plateau" and scheduler is not None:
            scheduler.step(val_loss)
        elif scheduler_name == "cosine" and scheduler is not None:
            scheduler.step()

        if is_best:
            best_val_loss = val_loss
            best_val_mae = val_mae  # MAE at the best-loss epoch, for reporting
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if ema_model is not None:
                best_ema_state = {k: v.cpu().clone() for k, v in ema_model.state_dict().items()}
            no_improve = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                "best_val_loss": best_val_loss,
                "best_val_mae": best_val_mae,
                "best_epoch": best_epoch,
                "best_state": best_state,
                "no_improve": no_improve,
                "curve_rows": curve_rows,
                "ema_state_dict": ema_model.state_dict() if ema_model else None,
                "rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
            }, os.path.join(output_dir, "checkpoint_best.pt"))
        else:
            no_improve += 1

        # Save checkpoint for resume.
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "best_val_loss": best_val_loss,
            "best_val_mae": best_val_mae,
            "best_epoch": best_epoch,
            "best_state": best_state,
            "no_improve": no_improve,
            "curve_rows": curve_rows,
            "ema_state_dict": ema_model.state_dict() if ema_model else None,
            "rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
        }, os.path.join(output_dir, "checkpoint.pt"))

        if patience > 0 and no_improve >= patience:
            log.info(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break

    train_time = time.time() - t_start

    # Save.
    save_training_curve(curve_rows, os.path.join(output_dir, "training_curve.csv"))
    if best_state is not None:
        torch.save(best_state, os.path.join(output_dir, "best.pt"))
    torch.save(model.state_dict(), os.path.join(output_dir, "last.pt"))
    if best_ema_state is not None:
        torch.save(best_ema_state, os.path.join(output_dir, "ema_best.pt"))

    # Test evaluation.
    results = {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_mae": best_val_mae,
        "train_time_sec": train_time,
        "avg_epoch_time_sec": train_time / max(epoch, 1),
    }

    if test_loader is not None and best_state is not None:
        model.load_state_dict(best_state)
        _, test_mae, test_r2 = evaluate(model, test_loader, loss_fn, device, _normalizer)
        results["test_mae"] = test_mae
        results["test_r2"] = test_r2

    return results
