import math

import torch
from torch import nn


def clip_gradients(
    model: nn.Module,
    max_norm: float = 1.0,
) -> torch.Tensor:
    """
    Clip model gradients using global norm clipping.

    Returns:
        Total gradient norm before clipping.
    """

    if not math.isfinite(max_norm) or max_norm <= 0:
        raise ValueError("max_norm must be a positive finite number.")

    return torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=max_norm,
    )
