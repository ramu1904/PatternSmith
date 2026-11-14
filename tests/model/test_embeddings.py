import pytest
import torch

from patternsmith.model.embeddings import PatternEmbedding


@pytest.fixture
def embedding_layer():
    return PatternEmbedding(
        vocab_size=50257,
        context_length=128,
        embedding_dim=128,
    )


def test_embedding_output_shape(embedding_layer):
    input_ids = torch.randint(0, 50257, (4, 52))

    output = embedding_layer(input_ids)

    assert output.shape == (4, 52, 128)


def test_embedding_output_is_float_tensor(embedding_layer):
    input_ids = torch.randint(0, 50257, (4, 52))

    output = embedding_layer(input_ids)

    assert output.dtype == torch.float32


def test_embedding_parameter_count(embedding_layer):
    parameter_count = sum(
        parameter.numel() for parameter in embedding_layer.parameters()
    )

    expected_count = (50257 * 128) + (128 * 128)

    assert parameter_count == expected_count
    assert parameter_count == 6449280


def test_token_and_position_embeddings_are_added():
    layer = PatternEmbedding(
        vocab_size=10,
        context_length=8,
        embedding_dim=2,
    )

    with torch.no_grad():
        layer.token_embedding.weight.zero_()
        layer.position_embedding.weight.zero_()

        layer.token_embedding.weight[3] = torch.tensor([1.0, 2.0])
        layer.position_embedding.weight[0] = torch.tensor([0.5, 0.5])

    input_ids = torch.tensor([[3]], dtype=torch.long)

    output = layer(input_ids)

    expected = torch.tensor([[[1.5, 2.5]]])

    assert torch.allclose(output, expected)


def test_sequence_at_maximum_context_length_is_supported(embedding_layer):
    input_ids = torch.randint(0, 50257, (2, 128))

    output = embedding_layer(input_ids)

    assert output.shape == (2, 128, 128)


def test_sequence_exceeding_context_length_is_rejected(embedding_layer):
    input_ids = torch.randint(0, 50257, (2, 129))

    with pytest.raises(ValueError, match="exceeds context length"):
        embedding_layer(input_ids)


def test_one_dimensional_input_is_rejected(embedding_layer):
    input_ids = torch.randint(0, 50257, (52,))

    with pytest.raises(ValueError, match="batch_size, sequence_length"):
        embedding_layer(input_ids)


def test_non_long_input_is_rejected(embedding_layer):
    input_ids = torch.randint(0, 50257, (2, 52)).float()

    with pytest.raises(TypeError, match="torch.long"):
        embedding_layer(input_ids)


def test_empty_sequence_is_rejected(embedding_layer):
    input_ids = torch.empty((2, 0), dtype=torch.long)

    with pytest.raises(ValueError, match="greater than zero"):
        embedding_layer(input_ids)
