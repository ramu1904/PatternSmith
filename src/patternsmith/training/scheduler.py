import math

import torch


def build_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int,
    min_lr_ratio: float = 0.1,
) -> torch.optim.lr_scheduler.LambdaLR:
    """
    Create a linear-warmup + cosine-decay learning-rate scheduler.

    Args:
        optimizer: Optimizer whose learning rate will be scheduled.
        total_steps: Total number of optimizer updates.
        warmup_steps: Number of linear warmup updates.
        min_lr_ratio: Minimum LR as a fraction of the base LR.

    Returns:
        PyTorch LambdaLR scheduler.
    """

    if total_steps <= 0:
        raise ValueError("total_steps must be greater than zero.")

    if warmup_steps < 0:
        raise ValueError("warmup_steps cannot be negative.")

    if warmup_steps >= total_steps:
        raise ValueError("warmup_steps must be less than total_steps.")

    if not 0.0 <= min_lr_ratio <= 1.0:
        raise ValueError("min_lr_ratio must be between 0 and 1.")

    def lr_lambda(current_step: int) -> float:
        # Linear warmup.
        if warmup_steps > 0 and current_step < warmup_steps:
            return (current_step + 1) / warmup_steps

        # Cosine decay.
        decay_steps = total_steps - warmup_steps

        progress = (current_step - warmup_steps) / decay_steps
        progress = min(max(progress, 0.0), 1.0)

        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))

        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_factor

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lr_lambda,
    )
