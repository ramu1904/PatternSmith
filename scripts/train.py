import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from patternsmith.data.collate import pattern_collate_fn
from patternsmith.data.dataset import PatternDataset
from patternsmith.model.gpt_model import GPTModel
from patternsmith.training.checkpoint import (
    load_checkpoint,
    save_checkpoint,
)
from patternsmith.training.early_stopping import EarlyStopping
from patternsmith.training.engine import (
    train_one_epoch,
    validate_one_epoch,
)
from patternsmith.training.optimizer import build_optimizer
from patternsmith.training.scheduler import build_warmup_cosine_scheduler

ROOT_DIR = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description="Train PatternSmith from scratch.")

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT_DIR / "data" / "splits" / "NL-RX-Turk",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT_DIR / "checkpoints",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=40,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Path to a checkpoint to resume training from.",
    )

    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_argument: str) -> torch.device:
    """Select the training device."""

    if device_argument == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device_argument == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but no CUDA-enabled GPU is available.")

    return torch.device(device_argument)


def create_dataloaders(
    data_dir: Path,
    batch_size: int,
    context_length: int,
    device: torch.device,
):
    """Create training and validation DataLoaders."""

    train_dataset = PatternDataset(
        str(data_dir / "train" / "src.txt"),
        str(data_dir / "train" / "targ.txt"),
        context_length=context_length,
    )

    val_dataset = PatternDataset(
        str(data_dir / "validation" / "src.txt"),
        str(data_dir / "validation" / "targ.txt"),
        context_length=context_length,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=pattern_collate_fn,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=pattern_collate_fn,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    return train_loader, val_loader


def print_epoch_metrics(
    epoch: int,
    total_epochs: int,
    train_metrics,
    val_metrics,
) -> None:
    """Print training and validation metrics."""

    print(f"\nEpoch {epoch}/{total_epochs}")
    print("-" * 65)

    print(
        f"Train | Loss: {train_metrics.loss:.4f} | "
        f"Perplexity: {train_metrics.perplexity:.4f}"
    )

    print(
        f"Valid | Loss: {val_metrics.loss:.4f} | "
        f"Perplexity: {val_metrics.perplexity:.4f}"
    )

    print(f"Learning rate: {train_metrics.learning_rate:.8f}")

    print(f"Mean gradient norm: " f"{train_metrics.mean_gradient_norm:.4f}")

    print(
        f"Training steps: {train_metrics.steps} | "
        f"Supervised tokens: {train_metrics.supervised_tokens}"
    )


def main():
    args = parse_args()

    if args.epochs <= 0:
        raise ValueError("Epochs must be greater than zero.")

    if args.batch_size <= 0:
        raise ValueError("Batch size must be greater than zero.")

    set_seed(42)

    device = get_device(args.device)

    print("=" * 65)
    print("PatternSmith Training")
    print("=" * 65)
    print(f"Device: {device}")
    print(f"Dataset directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ---------------------------------------------------------
    # 1. Model configuration
    # ---------------------------------------------------------

    model_config = {
        "vocab_size": 50257,
        "context_length": 128,
        "embedding_dim": 128,
        "num_layers": 4,
        "num_heads": 4,
        "hidden_dim": 512,
        "dropout": 0.1,
    }

    # ---------------------------------------------------------
    # 2. Training configuration
    # ---------------------------------------------------------

    training_config = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": 3e-4,
        "weight_decay": 0.01,
        "warmup_ratio": 0.05,
        "min_lr_ratio": 0.1,
        "max_grad_norm": 1.0,
        "early_stopping_patience": 5,
        "early_stopping_min_delta": 0.001,
        "seed": 42,
    }

    full_config = {
        "model": model_config,
        "training": training_config,
    }

    # ---------------------------------------------------------
    # 3. Create datasets and DataLoaders
    # ---------------------------------------------------------

    train_loader, val_loader = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        context_length=model_config["context_length"],
        device=device,
    )

    print(f"\nTraining examples: {len(train_loader.dataset)}")
    print(f"Validation examples: {len(val_loader.dataset)}")
    print(f"Training batches per epoch: {len(train_loader)}")
    print(f"Validation batches per epoch: {len(val_loader)}")

    # ---------------------------------------------------------
    # 4. Initialize GPT model
    # ---------------------------------------------------------

    model = GPTModel(**model_config).to(device)

    total_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )

    print(f"\nTrainable parameters: {total_parameters:,}")

    # ---------------------------------------------------------
    # 5. Optimizer and learning-rate scheduler
    # ---------------------------------------------------------

    optimizer = build_optimizer(
        model,
        learning_rate=training_config["learning_rate"],
        weight_decay=training_config["weight_decay"],
    )

    total_steps = len(train_loader) * args.epochs

    warmup_steps = max(
        1,
        int(total_steps * training_config["warmup_ratio"]),
    )

    if warmup_steps >= total_steps:
        raise ValueError("Warmup steps must be smaller than total training steps.")

    scheduler = build_warmup_cosine_scheduler(
        optimizer,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        min_lr_ratio=training_config["min_lr_ratio"],
    )

    print(f"Total optimizer steps: {total_steps}")
    print(f"Warmup steps: {warmup_steps}")

    # ---------------------------------------------------------
    # 6. Early stopping and checkpoint paths
    # ---------------------------------------------------------

    early_stopping = EarlyStopping(
        patience=training_config["early_stopping_patience"],
        min_delta=training_config["early_stopping_min_delta"],
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    last_checkpoint_path = args.output_dir / "last.pt"
    best_checkpoint_path = args.output_dir / "best.pt"

    best_val_loss = float("inf")
    start_epoch = 1

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_perplexity": [],
        "val_perplexity": [],
        "learning_rate": [],
        "gradient_norm": [],
    }

    # ---------------------------------------------------------
    # 7. Resume training if a checkpoint was provided
    # ---------------------------------------------------------

    if args.resume is not None:
        print(f"\nResuming from: {args.resume}")

        checkpoint = load_checkpoint(
            path=args.resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            map_location=device,
            early_stopping=early_stopping,
        )

        saved_config = checkpoint["config"]

        if saved_config != full_config:
            raise ValueError(
                "The current configuration does not match the "
                "checkpoint configuration. Use the same training "
                "settings when resuming."
            )

        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint["best_val_loss"]
        history = checkpoint["history"]

        print(f"Resuming from epoch {start_epoch}")
        print(f"Best validation loss so far: {best_val_loss:.4f}")

    # ---------------------------------------------------------
    # 8. Main training loop
    # ---------------------------------------------------------

    for epoch in range(start_epoch, args.epochs + 1):

        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            scheduler=scheduler,
            max_grad_norm=training_config["max_grad_norm"],
        )

        val_metrics = validate_one_epoch(
            model=model,
            dataloader=val_loader,
            device=device,
        )

        print_epoch_metrics(
            epoch,
            args.epochs,
            train_metrics,
            val_metrics,
        )

        # -----------------------------------------------------
        # 9. Update training history
        # -----------------------------------------------------

        history["train_loss"].append(train_metrics.loss)
        history["val_loss"].append(val_metrics.loss)

        history["train_perplexity"].append(train_metrics.perplexity)
        history["val_perplexity"].append(val_metrics.perplexity)

        history["learning_rate"].append(train_metrics.learning_rate)
        history["gradient_norm"].append(train_metrics.mean_gradient_norm)

        # -----------------------------------------------------
        # 10. Save best checkpoint
        # -----------------------------------------------------

        is_best = val_metrics.loss < best_val_loss

        if is_best:
            best_val_loss = val_metrics.loss

            save_checkpoint(
                path=best_checkpoint_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_val_loss=best_val_loss,
                history=history,
                config=full_config,
                early_stopping=early_stopping,
            )

            print(f"New best validation loss: {best_val_loss:.4f}")

        # -----------------------------------------------------
        # 11. Update early stopping
        # -----------------------------------------------------

        should_stop = early_stopping.step(val_metrics.loss)

        # -----------------------------------------------------
        # 12. Save latest checkpoint every epoch
        # -----------------------------------------------------

        save_checkpoint(
            path=last_checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            best_val_loss=best_val_loss,
            history=history,
            config=full_config,
            early_stopping=early_stopping,
        )

        print(
            f"Early stopping counter: "
            f"{early_stopping.counter}/"
            f"{early_stopping.patience}"
        )

        if should_stop:
            print("\nEarly stopping triggered.")
            break

    # ---------------------------------------------------------
    # 13. Restore the best model weights
    # ---------------------------------------------------------

    if best_checkpoint_path.is_file():
        best_checkpoint = load_checkpoint(
            path=best_checkpoint_path,
            model=model,
            map_location=device,
        )

        print("\nBest model restored.")

        print(f"Best epoch: {best_checkpoint['epoch']}")
        print(f"Best validation loss: " f"{best_checkpoint['best_val_loss']:.4f}")

    print("\nTraining completed.")
    print(f"Latest checkpoint: {last_checkpoint_path}")
    print(f"Best checkpoint: {best_checkpoint_path}")


if __name__ == "__main__":
    main()
