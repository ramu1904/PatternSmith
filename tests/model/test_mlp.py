import pytest
import torch

from patternsmith.model.mlp import FeedForwardMLP


def test_output_shape():
    model = FeedForwardMLP()
    x = torch.randn(4, 52, 128)

    output = model(x)

    assert output.shape == (4, 52, 128)


def test_output_shape_with_different_sequence_length():
    model = FeedForwardMLP()
    x = torch.randn(2, 16, 128)

    output = model(x)

    assert output.shape == (2, 16, 128)


def test_hidden_dimension():
    model = FeedForwardMLP(
        embedding_dim=128,
        hidden_dim=512,
    )

    assert model.fc1.out_features == 512
    assert model.fc2.in_features == 512


def test_parameter_count():
    model = FeedForwardMLP(
        embedding_dim=128,
        hidden_dim=512,
    )

    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    assert parameter_count == 131_712


def test_gelu_activation_is_used():
    model = FeedForwardMLP()

    assert isinstance(model.activation, torch.nn.GELU)


def test_rejects_non_3d_input():
    model = FeedForwardMLP()
    x = torch.randn(4, 128)

    with pytest.raises(ValueError):
        model(x)


def test_rejects_wrong_embedding_dimension():
    model = FeedForwardMLP()
    x = torch.randn(4, 52, 64)

    with pytest.raises(ValueError):
        model(x)


def test_rejects_zero_sequence_length():
    model = FeedForwardMLP()
    x = torch.randn(4, 0, 128)

    with pytest.raises(ValueError):
        model(x)


def test_rejects_invalid_dimensions():
    with pytest.raises(ValueError):
        FeedForwardMLP(embedding_dim=0)

    with pytest.raises(ValueError):
        FeedForwardMLP(hidden_dim=0)


def test_rejects_invalid_dropout():
    with pytest.raises(ValueError):
        FeedForwardMLP(dropout=1.0)


def test_gradient_flows_to_all_layers():
    model = FeedForwardMLP()
    x = torch.randn(2, 8, 128, requires_grad=True)

    output = model(x)
    loss = output.mean()
    loss.backward()

    assert x.grad is not None
    assert model.fc1.weight.grad is not None
    assert model.fc2.weight.grad is not None


def test_eval_mode_is_deterministic():
    model = FeedForwardMLP(dropout=0.5)
    model.eval()

    x = torch.randn(2, 8, 128)

    output_1 = model(x)
    output_2 = model(x)

    assert torch.allclose(output_1, output_2)
