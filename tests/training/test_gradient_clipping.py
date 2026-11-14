import pytest
import torch
from torch import nn

from patternsmith.training.gradient_clipping import clip_gradients


def test_gradient_norm_is_clipped():
    model = nn.Linear(10, 5)

    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)

    original_norm = clip_gradients(model, max_norm=1.0)

    total_norm_after = torch.sqrt(
        sum(torch.sum(parameter.grad**2) for parameter in model.parameters())
    )

    assert original_norm > 1.0
    assert total_norm_after <= 1.0 + 1e-6


def test_gradient_below_threshold_is_not_scaled():
    model = nn.Linear(2, 2)

    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, 0.1)

    original_gradients = [parameter.grad.clone() for parameter in model.parameters()]

    original_norm = clip_gradients(model, max_norm=1.0)

    assert original_norm < 1.0

    for parameter, original in zip(
        model.parameters(),
        original_gradients,
    ):
        assert torch.allclose(parameter.grad, original)


def test_returns_original_gradient_norm():
    model = nn.Linear(2, 2)

    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)

    original_norm = clip_gradients(model, max_norm=1.0)

    expected_norm = torch.sqrt(
        sum(torch.sum(parameter.grad**2) for parameter in model.parameters())
    )

    # Clipping changes the gradients, but the returned value
    # represents the norm before clipping.
    assert original_norm > expected_norm


@pytest.mark.parametrize("max_norm", [0, -1, float("inf"), float("nan")])
def test_rejects_invalid_max_norm(max_norm):
    model = nn.Linear(2, 2)

    with pytest.raises(ValueError):
        clip_gradients(model, max_norm=max_norm)


def test_handles_parameters_without_gradients():
    model = nn.Linear(2, 2)

    original_norm = clip_gradients(model, max_norm=1.0)

    assert original_norm == 0
