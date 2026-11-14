from pathlib import Path
from typing import Any

import torch
from torch import nn

from patternsmith.training.early_stopping import EarlyStopping


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    epoch: int,
    best_val_loss: float,
    history: dict[str, list] | None = None,
    config: dict[str, Any] | None = None,
    early_stopping: EarlyStopping | None = None,
) -> None:
    """Save a complete training checkpoint."""

    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": (
            scheduler.state_dict() if scheduler is not None else None
        ),
        "best_val_loss": best_val_loss,
        "history": history or {},
        "config": config or {},
        "early_stopping_state": (
            early_stopping.state_dict() if early_stopping is not None else None
        ),
    }

    temporary_path = checkpoint_path.with_name(checkpoint_path.name + ".tmp")

    torch.save(checkpoint, temporary_path)
    temporary_path.replace(checkpoint_path)


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    map_location: str | torch.device = "cpu",
    early_stopping: EarlyStopping | None = None,
) -> dict[str, Any]:
    """Load model and optional training states from a checkpoint."""

    checkpoint_path = Path(path)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=map_location,
        weights_only=True,
    )

    required_keys = {
        "epoch",
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "best_val_loss",
        "history",
        "config",
        "early_stopping_state",
    }

    missing_keys = required_keys.difference(checkpoint)

    if missing_keys:
        raise ValueError(f"Invalid checkpoint. Missing keys: {sorted(missing_keys)}")

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None:
        scheduler_state = checkpoint["scheduler_state_dict"]

        if scheduler_state is None:
            raise ValueError("Checkpoint does not contain scheduler state.")

        scheduler.load_state_dict(scheduler_state)

    if early_stopping is not None:
        early_stopping_state = checkpoint["early_stopping_state"]

        if early_stopping_state is not None:
            early_stopping.load_state_dict(early_stopping_state)

    return checkpoint
