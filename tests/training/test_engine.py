import pytest
import torch
from torch import nn

from patternsmith.training.engine import (
    calculate_loss,
    train_one_epoch,
    validate_one_epoch,
)


class TinyLanguageModel(nn.Module):
    def __init__(self, vocab_size=20, embedding_dim=8):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.output = nn.Linear(embedding_dim, vocab_size)

    def forward(self, input_ids):
        return self.output(self.embedding(input_ids))


@pytest.fixture
def batches():
    return [
        {
            "input_ids": torch.tensor(
                [[1, 2, 3], [4, 5, 6]],
                dtype=torch.long,
            ),
            "labels": torch.tensor(
                [[2, 3, 4], [5, 6, -100]],
                dtype=torch.long,
            ),
        },
        {
            "input_ids": torch.tensor(
                [[7, 8, 9]],
                dtype=torch.long,
            ),
            "labels": torch.tensor(
                [[8, 9, 10]],
                dtype=torch.long,
            ),
        },
    ]


def test_calculate_loss_ignores_masked_labels():
    logits = torch.randn(2, 3, 20)
    labels = torch.tensor([[1, 2, -100], [3, 4, -100]])

    loss, token_count = calculate_loss(logits, labels)

    expected_loss = nn.functional.cross_entropy(
        logits[:, :2, :].reshape(-1, 20),
        labels[:, :2].reshape(-1),
    )

    assert token_count == 4
    assert torch.allclose(loss, expected_loss)


def test_calculate_loss_rejects_empty_supervision():
    logits = torch.randn(2, 3, 20)
    labels = torch.full((2, 3), -100)

    with pytest.raises(ValueError):
        calculate_loss(logits, labels)


def test_train_one_epoch_updates_parameters(batches):
    model = TinyLanguageModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    original_weight = model.output.weight.detach().clone()

    metrics = train_one_epoch(
        model=model,
        dataloader=batches,
        optimizer=optimizer,
        device=torch.device("cpu"),
    )

    assert not torch.equal(original_weight, model.output.weight)
    assert metrics.steps == 2
    assert metrics.supervised_tokens == 8
    assert metrics.loss > 0
    assert metrics.perplexity > 0
    assert metrics.mean_gradient_norm > 0
    assert model.training


def test_validation_does_not_update_parameters(batches):
    model = TinyLanguageModel()

    original_parameters = [
        parameter.detach().clone() for parameter in model.parameters()
    ]

    metrics = validate_one_epoch(
        model=model,
        dataloader=batches,
        device=torch.device("cpu"),
    )

    for original, current in zip(
        original_parameters,
        model.parameters(),
    ):
        assert torch.equal(original, current)

    assert metrics.steps == 2
    assert metrics.supervised_tokens == 8
    assert metrics.loss > 0
    assert metrics.perplexity > 0
    assert not model.training


def test_training_rejects_empty_dataloader():
    model = TinyLanguageModel()
    optimizer = torch.optim.AdamW(model.parameters())

    with pytest.raises(ValueError):
        train_one_epoch(
            model,
            [],
            optimizer,
            torch.device("cpu"),
        )


def test_validation_rejects_empty_dataloader():
    model = TinyLanguageModel()

    with pytest.raises(ValueError):
        validate_one_epoch(
            model,
            [],
            torch.device("cpu"),
        )
