import pytest
import torch

from patternsmith.model.attention import (
    causal_scaled_dot_product_attention,
)


def test_attention_output_and_weight_shapes():
    query = torch.randn(4, 52, 32)
    key = torch.randn(4, 52, 32)
    value = torch.randn(4, 52, 32)

    output, weights = causal_scaled_dot_product_attention(query, key, value)

    assert output.shape == (4, 52, 32)
    assert weights.shape == (4, 52, 52)


def test_future_positions_receive_zero_attention():
    query = torch.randn(2, 4, 8)
    key = torch.randn(2, 4, 8)
    value = torch.randn(2, 4, 8)

    _, weights = causal_scaled_dot_product_attention(query, key, value)

    # Every position above the main diagonal is a future position.
    future_mask = torch.triu(
        torch.ones(4, 4, dtype=torch.bool),
        diagonal=1,
    )

    assert torch.all(weights[:, future_mask] == 0)


def test_attention_weights_sum_to_one():
    query = torch.randn(2, 4, 8)
    key = torch.randn(2, 4, 8)
    value = torch.randn(2, 4, 8)

    _, weights = causal_scaled_dot_product_attention(query, key, value)

    row_sums = weights.sum(dim=-1)

    assert torch.allclose(
        row_sums,
        torch.ones_like(row_sums),
        atol=1e-6,
    )


def test_attention_output_is_weighted_sum_of_values():
    query = torch.randn(2, 4, 8)
    key = torch.randn(2, 4, 8)
    value = torch.randn(2, 4, 8)

    output, weights = causal_scaled_dot_product_attention(query, key, value)

    expected_output = torch.matmul(weights, value)

    assert torch.allclose(output, expected_output)


def test_single_token_sequence():
    query = torch.randn(2, 1, 8)
    key = torch.randn(2, 1, 8)
    value = torch.randn(2, 1, 8)

    output, weights = causal_scaled_dot_product_attention(query, key, value)

    assert output.shape == (2, 1, 8)
    assert weights.shape == (2, 1, 1)
    assert torch.allclose(weights, torch.ones_like(weights))


def test_rejects_non_3d_input():
    query = torch.randn(4, 8)
    key = torch.randn(4, 8)
    value = torch.randn(4, 8)

    with pytest.raises(ValueError):
        causal_scaled_dot_product_attention(query, key, value)


def test_rejects_mismatched_batch_sizes():
    query = torch.randn(2, 4, 8)
    key = torch.randn(3, 4, 8)
    value = torch.randn(2, 4, 8)

    with pytest.raises(ValueError):
        causal_scaled_dot_product_attention(query, key, value)


def test_rejects_mismatched_query_and_key_dimensions():
    query = torch.randn(2, 4, 8)
    key = torch.randn(2, 4, 16)
    value = torch.randn(2, 4, 8)

    with pytest.raises(ValueError):
        causal_scaled_dot_product_attention(query, key, value)


def test_rejects_different_sequence_lengths():
    query = torch.randn(2, 4, 8)
    key = torch.randn(2, 3, 8)
    value = torch.randn(2, 4, 8)

    with pytest.raises(ValueError):
        causal_scaled_dot_product_attention(query, key, value)


def test_attention_supports_gradient_flow():
    query = torch.randn(2, 4, 8, requires_grad=True)
    key = torch.randn(2, 4, 8, requires_grad=True)
    value = torch.randn(2, 4, 8, requires_grad=True)

    output, _ = causal_scaled_dot_product_attention(query, key, value)

    loss = output.mean()
    loss.backward()

    assert query.grad is not None
    assert key.grad is not None
    assert value.grad is not None
