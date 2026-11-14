import pytest
import torch
from torch import nn

from patternsmith.training.scheduler import (
    build_warmup_cosine_scheduler,
)


def create_optimizer(learning_rate=3e-4):
    model = nn.Linear(10, 5)

    return torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
    )


def test_scheduler_returns_lambda_lr():
    optimizer = create_optimizer()

    scheduler = build_warmup_cosine_scheduler(
        optimizer,
        total_steps=100,
        warmup_steps=10,
    )

    assert isinstance(
        scheduler,
        torch.optim.lr_scheduler.LambdaLR,
    )


def test_scheduler_starts_with_warmup_learning_rate():
    optimizer = create_optimizer()

    scheduler = build_warmup_cosine_scheduler(
        optimizer,
        total_steps=100,
        warmup_steps=10,
    )

    assert optimizer.param_groups[0]["lr"] == pytest.approx(3e-5)


def test_scheduler_reaches_base_learning_rate_after_warmup():
    optimizer = create_optimizer(learning_rate=3e-4)

    scheduler = build_warmup_cosine_scheduler(
        optimizer,
        total_steps=100,
        warmup_steps=10,
    )

    for _ in range(10):
        optimizer.step()
        scheduler.step()

    assert optimizer.param_groups[0]["lr"] == pytest.approx(3e-4)


def test_scheduler_reaches_minimum_learning_rate():
    optimizer = create_optimizer(learning_rate=3e-4)

    scheduler = build_warmup_cosine_scheduler(
        optimizer,
        total_steps=100,
        warmup_steps=10,
        min_lr_ratio=0.1,
    )

    for _ in range(100):
        optimizer.step()
        scheduler.step()

    assert optimizer.param_groups[0]["lr"] == pytest.approx(3e-5)


def test_scheduler_lr_decreases_during_cosine_decay():
    optimizer = create_optimizer()

    scheduler = build_warmup_cosine_scheduler(
        optimizer,
        total_steps=100,
        warmup_steps=10,
    )

    for _ in range(10):
        optimizer.step()
        scheduler.step()

    lr_after_warmup = optimizer.param_groups[0]["lr"]

    for _ in range(20):
        optimizer.step()
        scheduler.step()

    lr_after_decay = optimizer.param_groups[0]["lr"]

    assert lr_after_decay < lr_after_warmup


@pytest.mark.parametrize(
    "total_steps,warmup_steps",
    [
        (0, 0),
        (100, -1),
        (100, 100),
        (100, 101),
    ],
)
def test_scheduler_rejects_invalid_steps(
    total_steps,
    warmup_steps,
):
    optimizer = create_optimizer()

    with pytest.raises(ValueError):
        build_warmup_cosine_scheduler(
            optimizer,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
        )


def test_scheduler_rejects_invalid_min_lr_ratio():
    optimizer = create_optimizer()

    with pytest.raises(ValueError):
        build_warmup_cosine_scheduler(
            optimizer,
            total_steps=100,
            warmup_steps=10,
            min_lr_ratio=1.5,
        )
