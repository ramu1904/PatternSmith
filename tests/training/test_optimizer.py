import torch
from torch import nn

from patternsmith.training.optimizer import build_optimizer


def test_optimizer_returns_adamw():
    model = nn.Linear(10, 5)

    optimizer = build_optimizer(model)

    assert isinstance(optimizer, torch.optim.AdamW)


def test_optimizer_parameter_groups():
    model = nn.Linear(10, 5)

    optimizer = build_optimizer(
        model,
        weight_decay=0.01,
    )

    assert len(optimizer.param_groups) == 2

    assert optimizer.param_groups[0]["weight_decay"] == 0.01
    assert optimizer.param_groups[1]["weight_decay"] == 0.0


def test_bias_receives_no_weight_decay():
    model = nn.Linear(10, 5)

    optimizer = build_optimizer(model)

    decay_parameters = optimizer.param_groups[0]["params"]
    no_decay_parameters = optimizer.param_groups[1]["params"]

    assert model.weight in decay_parameters
    assert model.bias in no_decay_parameters


def test_all_trainable_parameters_are_included():
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.LayerNorm(20),
        nn.Linear(20, 5),
    )

    optimizer = build_optimizer(model)

    optimizer_parameters = [
        parameter for group in optimizer.param_groups for parameter in group["params"]
    ]

    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]

    assert len(optimizer_parameters) == len(trainable_parameters)
    assert len({id(p) for p in optimizer_parameters}) == len(optimizer_parameters)


def test_optimizer_updates_parameters():
    model = nn.Linear(10, 5)

    optimizer = build_optimizer(model)

    input_tensor = torch.randn(4, 10)
    target = torch.randn(4, 5)

    original_weight = model.weight.detach().clone()

    optimizer.zero_grad(set_to_none=True)

    prediction = model(input_tensor)
    loss = nn.functional.mse_loss(prediction, target)

    loss.backward()
    optimizer.step()

    assert not torch.equal(original_weight, model.weight.detach())
