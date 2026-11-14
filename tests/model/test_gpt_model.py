import pytest
import torch
import torch.nn as nn

from patternsmith.model.gpt_model import GPTModel


def create_small_model():
    return GPTModel(
        vocab_size=128,
        context_length=16,
        embedding_dim=32,
        num_layers=2,
        num_heads=4,
        hidden_dim=64,
        dropout=0.0,
    )


def test_output_shape():
    model = create_small_model()
    input_ids = torch.randint(0, 128, (2, 8))

    logits = model(input_ids)

    assert logits.shape == (2, 8, 128)


def test_output_shape_with_single_token():
    model = create_small_model()
    input_ids = torch.randint(0, 128, (1, 1))

    logits = model(input_ids)

    assert logits.shape == (1, 1, 128)


def test_transformer_block_count():
    model = create_small_model()

    assert len(model.transformer_blocks) == 2


def test_transformer_blocks_are_registered():
    model = create_small_model()

    assert isinstance(model.transformer_blocks, nn.ModuleList)


def test_embedding_and_lm_head_weights_are_tied():
    model = create_small_model()

    assert model.lm_head.weight is model.embedding.token_embedding.weight


def test_lm_head_has_no_bias():
    model = create_small_model()

    assert model.lm_head.bias is None


def test_default_model_parameter_count():
    model = GPTModel()

    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    assert parameter_count == 7_242_624


def test_default_model_configuration():
    model = GPTModel()

    assert model.vocab_size == 50257
    assert model.context_length == 128
    assert model.embedding_dim == 128
    assert model.num_layers == 4
    assert len(model.transformer_blocks) == 4


def test_rejects_non_2d_input():
    model = create_small_model()
    input_ids = torch.randint(0, 128, (2, 8, 1))

    with pytest.raises(ValueError):
        model(input_ids)


def test_rejects_non_long_input():
    model = create_small_model()
    input_ids = torch.randint(0, 128, (2, 8)).float()

    with pytest.raises(TypeError):
        model(input_ids)


def test_rejects_empty_batch():
    model = create_small_model()
    input_ids = torch.empty((0, 8), dtype=torch.long)

    with pytest.raises(ValueError):
        model(input_ids)


def test_rejects_empty_sequence():
    model = create_small_model()
    input_ids = torch.empty((2, 0), dtype=torch.long)

    with pytest.raises(ValueError):
        model(input_ids)


def test_rejects_sequence_exceeding_context_length():
    model = create_small_model()
    input_ids = torch.randint(0, 128, (2, 17))

    with pytest.raises(ValueError):
        model(input_ids)


def test_rejects_negative_token_ids():
    model = create_small_model()
    input_ids = torch.tensor([[1, -1, 3]])

    with pytest.raises(ValueError):
        model(input_ids)


def test_rejects_token_ids_outside_vocabulary():
    model = create_small_model()
    input_ids = torch.tensor([[1, 128, 3]])

    with pytest.raises(ValueError):
        model(input_ids)


def test_gradient_flows_through_model():
    model = create_small_model()
    input_ids = torch.randint(0, 128, (2, 8))

    logits = model(input_ids)
    loss = logits.mean()
    loss.backward()

    assert model.embedding.token_embedding.weight.grad is not None
    assert (
        model.transformer_blocks[0].attention.query_projection.weight.grad is not None
    )
    assert model.transformer_blocks[0].mlp.fc1.weight.grad is not None


def test_eval_mode_is_deterministic():
    model = GPTModel(
        vocab_size=128,
        context_length=16,
        embedding_dim=32,
        num_layers=2,
        num_heads=4,
        hidden_dim=64,
        dropout=0.5,
    )
    model.eval()

    input_ids = torch.randint(0, 128, (2, 8))

    logits_1 = model(input_ids)
    logits_2 = model(input_ids)

    assert torch.allclose(logits_1, logits_2)
