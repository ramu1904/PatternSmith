import torch
from torch import nn


def build_optimizer(
    model: nn.Module,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.01,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
) -> torch.optim.AdamW:
    """
    Build AdamW with separate weight-decay and no-weight-decay groups.

    Matrix parameters receive weight decay.
    Biases and one-dimensional parameters (including LayerNorm
    parameters) do not receive weight decay.
    """

    decay_parameters = []
    no_decay_parameters = []

    for name, parameter in model.named_parameters():

        if not parameter.requires_grad:
            continue

        if parameter.ndim >= 2:
            decay_parameters.append(parameter)
        else:
            no_decay_parameters.append(parameter)

    parameter_groups = [
        {
            "params": decay_parameters,
            "weight_decay": weight_decay,
        },
        {
            "params": no_decay_parameters,
            "weight_decay": 0.0,
        },
    ]

    optimizer = torch.optim.AdamW(
        parameter_groups,
        lr=learning_rate,
        betas=betas,
        eps=eps,
    )

    return optimizer
