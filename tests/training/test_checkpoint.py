import pytest
import torch
from torch import nn

from patternsmith.training.checkpoint import (
    load_checkpoint,
    save_checkpoint,
)
from patternsmith.training.early_stopping import EarlyStopping


def create_components():
    model = nn.Linear(4, 3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: 1.0 / (step + 1),
    )

    return model, optimizer, scheduler


def test_checkpoint_saves_and_loads_model(tmp_path):
    model, optimizer, scheduler = create_components()
    path = tmp_path / "checkpoint.pt"

    original_parameters = [
        parameter.detach().clone() for parameter in model.parameters()
    ]

    save_checkpoint(
        path,
        model,
        optimizer,
        scheduler,
        epoch=4,
        best_val_loss=2.5,
    )

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(1.0)

    checkpoint = load_checkpoint(
        path,
        model,
        optimizer,
        scheduler,
    )

    for original, restored in zip(
        original_parameters,
        model.parameters(),
    ):
        assert torch.equal(original, restored)

    assert checkpoint["epoch"] == 4
    assert checkpoint["best_val_loss"] == 2.5


def test_checkpoint_preserves_training_history_and_config(tmp_path):
    model, optimizer, scheduler = create_components()
    path = tmp_path / "checkpoint.pt"

    history = {
        "train_loss": [3.0, 2.5],
        "val_loss": [3.2, 2.7],
    }
    config = {
        "learning_rate": 1e-3,
        "batch_size": 4,
    }

    save_checkpoint(
        path,
        model,
        optimizer,
        scheduler,
        epoch=2,
        best_val_loss=2.7,
        history=history,
        config=config,
    )

    checkpoint = load_checkpoint(path, model)

    assert checkpoint["history"] == history
    assert checkpoint["config"] == config


def test_checkpoint_restores_early_stopping_state(tmp_path):
    model, optimizer, scheduler = create_components()
    early_stopping = EarlyStopping(patience=3, min_delta=0.01)

    early_stopping.step(2.0)
    early_stopping.step(2.1)

    path = tmp_path / "checkpoint.pt"

    save_checkpoint(
        path,
        model,
        optimizer,
        scheduler,
        epoch=2,
        best_val_loss=2.0,
        early_stopping=early_stopping,
    )

    restored_early_stopping = EarlyStopping(patience=3)

    load_checkpoint(
        path,
        model,
        early_stopping=restored_early_stopping,
    )

    assert restored_early_stopping.best_loss == 2.0
    assert restored_early_stopping.counter == 1
    assert restored_early_stopping.min_delta == 0.01


def test_checkpoint_restores_scheduler_state(tmp_path):
    model, optimizer, scheduler = create_components()
    path = tmp_path / "checkpoint.pt"

    optimizer.step()
    scheduler.step()

    expected_lr = optimizer.param_groups[0]["lr"]

    save_checkpoint(
        path,
        model,
        optimizer,
        scheduler,
        epoch=1,
        best_val_loss=2.0,
    )

    _, restored_optimizer, restored_scheduler = create_components()

    load_checkpoint(
        path,
        model,
        restored_optimizer,
        restored_scheduler,
    )

    assert restored_optimizer.param_groups[0]["lr"] == expected_lr
    assert restored_scheduler.state_dict() == scheduler.state_dict()


def test_checkpoint_rejects_missing_file(tmp_path):
    model, _, _ = create_components()

    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path / "missing.pt", model)
