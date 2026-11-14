import pytest
import torch

from patternsmith.data.collate import pattern_collate_fn


def make_example(input_ids, labels):
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def test_dynamic_padding_uses_longest_sequence():
    batch = [
        make_example([10, 11, 12], [-100, 11, 12]),
        make_example([20, 21, 22, 23], [-100, 21, 22, 23]),
    ]

    result = pattern_collate_fn(batch)

    assert result["input_ids"].shape == (2, 4)
    assert result["labels"].shape == (2, 4)
    assert result["padding_mask"].shape == (2, 4)


def test_input_padding_uses_eot_token():
    batch = [
        make_example([10, 11], [-100, 11]),
        make_example([20, 21, 22], [-100, 21, 22]),
    ]

    result = pattern_collate_fn(batch)

    assert result["input_ids"][0].tolist() == [10, 11, 50256]


def test_padding_labels_use_ignore_index():
    batch = [
        make_example([10, 11], [-100, 11]),
        make_example([20, 21, 22], [-100, 21, 22]),
    ]

    result = pattern_collate_fn(batch)

    assert result["labels"][0].tolist() == [-100, 11, -100]


def test_padding_mask_marks_real_tokens_and_padding():
    batch = [
        make_example([10, 11], [-100, 11]),
        make_example([20, 21, 22], [-100, 21, 22]),
    ]

    result = pattern_collate_fn(batch)

    expected_mask = torch.tensor(
        [
            [True, True, False],
            [True, True, True],
        ],
        dtype=torch.bool,
    )

    assert torch.equal(result["padding_mask"], expected_mask)


def test_padding_mask_has_boolean_dtype():
    batch = [
        make_example([10, 11], [-100, 11]),
        make_example([20, 21, 22], [-100, 21, 22]),
    ]

    result = pattern_collate_fn(batch)

    assert result["padding_mask"].dtype == torch.bool


def test_empty_batch_raises_error():
    with pytest.raises(ValueError, match="empty batch"):
        pattern_collate_fn([])


def test_non_one_dimensional_tensors_are_rejected():
    batch = [
        {
            "input_ids": torch.tensor([[10, 11]], dtype=torch.long),
            "labels": torch.tensor([10, 11], dtype=torch.long),
        }
    ]

    with pytest.raises(ValueError, match="1D tensors"):
        pattern_collate_fn(batch)


def test_input_and_label_length_mismatch_is_rejected():
    batch = [
        make_example([10, 11, 12], [-100, 11]),
    ]

    with pytest.raises(ValueError, match="lengths differ"):
        pattern_collate_fn(batch)
