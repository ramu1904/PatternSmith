import pytest
import torch

from patternsmith.model.multi_head_attention import MultiHeadAttention


def test_output_shape():
    model = MultiHeadAttention()
    x = torch.randn(4, 52, 128)

    output = model(x)

    assert output.shape == (4, 52, 128)


def test_output_shape_with_different_sequence_length():
    model = MultiHeadAttention()
    x = torch.randn(2, 16, 128)

    output = model(x)

    assert output.shape == (2, 16, 128)


def test_head_dimension():
    model = MultiHeadAttention(
        embedding_dim=128,
        num_heads=4,
    )

    assert model.head_dim == 32


def test_split_heads_shape():
    model = MultiHeadAttention(
        embedding_dim=128,
        num_heads=4,
    )

    x = torch.randn(4, 52, 128)
    split = model._split_heads(x)

    assert split.shape == (4, 4, 52, 32)


def test_parameter_count():
    model = MultiHeadAttention(
        embedding_dim=128,
        num_heads=4,
    )

    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    assert parameter_count == 66_048


def test_all_projection_layers_are_trainable():
    model = MultiHeadAttention()

    assert model.query_projection.weight.requires_grad
    assert model.key_projection.weight.requires_grad
    assert model.value_projection.weight.requires_grad
    assert model.output_projection.weight.requires_grad


def test_rejects_non_3d_input():
    model = MultiHeadAttention()
    x = torch.randn(4, 128)

    with pytest.raises(ValueError):
        model(x)


def test_rejects_wrong_embedding_dimension():
    model = MultiHeadAttention(embedding_dim=128)
    x = torch.randn(4, 52, 64)

    with pytest.raises(ValueError):
        model(x)


def test_rejects_embedding_dimension_not_divisible_by_heads():
    with pytest.raises(ValueError):
        MultiHeadAttention(
            embedding_dim=128,
            num_heads=3,
        )


def test_rejects_zero_sequence_length():
    model = MultiHeadAttention()
    x = torch.randn(4, 0, 128)

    with pytest.raises(ValueError):
        model(x)


def test_rejects_invalid_dropout():
    with pytest.raises(ValueError):
        MultiHeadAttention(dropout=1.0)


def test_gradient_flows_to_all_projection_layers():
    model = MultiHeadAttention()
    x = torch.randn(2, 8, 128, requires_grad=True)

    output = model(x)
    loss = output.mean()
    loss.backward()

    assert x.grad is not None
    assert model.query_projection.weight.grad is not None
    assert model.key_projection.weight.grad is not None
    assert model.value_projection.weight.grad is not None
    assert model.output_projection.weight.grad is not None


def test_eval_mode_is_deterministic():
    model = MultiHeadAttention(dropout=0.5)
    model.eval()

    x = torch.randn(2, 8, 128)

    output_1 = model(x)
    output_2 = model(x)

    assert torch.allclose(output_1, output_2)
