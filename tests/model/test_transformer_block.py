import pytest
import torch
import torch.nn as nn

from patternsmith.model.transformer_block import TransformerBlock


def test_output_shape():
    model = TransformerBlock()
    x = torch.randn(4, 52, 128)

    output = model(x)

    assert output.shape == (4, 52, 128)


def test_output_shape_with_different_sequence_length():
    model = TransformerBlock()
    x = torch.randn(2, 16, 128)

    output = model(x)

    assert output.shape == (2, 16, 128)


def test_layer_norm_configuration():
    model = TransformerBlock()

    assert isinstance(model.norm1, nn.LayerNorm)
    assert isinstance(model.norm2, nn.LayerNorm)
    assert model.norm1.normalized_shape == (128,)
    assert model.norm2.normalized_shape == (128,)


def test_attention_and_mlp_configuration():
    model = TransformerBlock()

    assert model.attention.num_heads == 4
    assert model.attention.head_dim == 32
    assert model.mlp.hidden_dim == 512


def test_parameter_count():
    model = TransformerBlock()

    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    assert parameter_count == 198_272


def test_all_parameters_are_trainable():
    model = TransformerBlock()

    assert all(parameter.requires_grad for parameter in model.parameters())


def test_rejects_non_3d_input():
    model = TransformerBlock()
    x = torch.randn(4, 128)

    with pytest.raises(ValueError):
        model(x)


def test_rejects_wrong_embedding_dimension():
    model = TransformerBlock()
    x = torch.randn(4, 52, 64)

    with pytest.raises(ValueError):
        model(x)


def test_rejects_zero_sequence_length():
    model = TransformerBlock()
    x = torch.randn(4, 0, 128)

    with pytest.raises(ValueError):
        model(x)


def test_rejects_invalid_configuration():
    with pytest.raises(ValueError):
        TransformerBlock(embedding_dim=0)

    with pytest.raises(ValueError):
        TransformerBlock(num_heads=0)

    with pytest.raises(ValueError):
        TransformerBlock(embedding_dim=128, num_heads=3)

    with pytest.raises(ValueError):
        TransformerBlock(hidden_dim=0)

    with pytest.raises(ValueError):
        TransformerBlock(dropout=1.0)


def test_gradient_flows_through_entire_block():
    model = TransformerBlock()
    x = torch.randn(2, 8, 128, requires_grad=True)

    output = model(x)
    loss = output.mean()
    loss.backward()

    assert x.grad is not None

    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, f"Missing gradient: {name}"


def test_eval_mode_is_deterministic():
    model = TransformerBlock(dropout=0.5)
    model.eval()

    x = torch.randn(2, 8, 128)

    output_1 = model(x)
    output_2 = model(x)

    assert torch.allclose(output_1, output_2)
