import pytest
import torch
import tiktoken

from patternsmith.data.dataset import PatternDataset


def create_dataset_files(tmp_path, descriptions, regexes):
    """Create temporary, line-aligned source and target files."""
    source_path = tmp_path / "src.txt"
    target_path = tmp_path / "targ.txt"

    source_path.write_text("\n".join(descriptions) + "\n", encoding="utf-8")
    target_path.write_text("\n".join(regexes) + "\n", encoding="utf-8")

    return source_path, target_path


def test_dataset_length_matches_number_of_examples(tmp_path):
    source_path, target_path = create_dataset_files(
        tmp_path,
        descriptions=["Match five digits", "Match a lowercase letter"],
        regexes=[r"\d{5}", r"[a-z]"],
    )

    dataset = PatternDataset(source_path, target_path)

    assert len(dataset) == 2


def test_dataset_returns_expected_keys_and_tensor_types(tmp_path):
    source_path, target_path = create_dataset_files(
        tmp_path,
        descriptions=["Match five digits"],
        regexes=[r"\d{5}"],
    )

    dataset = PatternDataset(source_path, target_path)
    example = dataset[0]

    assert set(example.keys()) == {"input_ids", "labels"}
    assert isinstance(example["input_ids"], torch.Tensor)
    assert isinstance(example["labels"], torch.Tensor)

    assert example["input_ids"].dtype == torch.long
    assert example["labels"].dtype == torch.long


def test_input_and_label_shapes_match(tmp_path):
    source_path, target_path = create_dataset_files(
        tmp_path,
        descriptions=["Match five digits"],
        regexes=[r"\d{5}"],
    )

    dataset = PatternDataset(source_path, target_path)
    example = dataset[0]

    assert example["input_ids"].ndim == 1
    assert example["labels"].ndim == 1
    assert example["input_ids"].shape == example["labels"].shape


def test_prompt_labels_are_masked_and_target_is_supervised(tmp_path):
    description = "Match five digits"
    regex = r"\d{5}"

    source_path, target_path = create_dataset_files(
        tmp_path,
        descriptions=[description],
        regexes=[regex],
    )

    dataset = PatternDataset(source_path, target_path)
    example = dataset[0]

    tokenizer = tiktoken.get_encoding("gpt2")

    prompt = f"Description: {description}\nRegex:"
    prompt_ids = tokenizer.encode(prompt)
    target_ids = tokenizer.encode(regex)

    expected_labels = (
        [-100] * (len(prompt_ids) - 1) + target_ids + [tokenizer.eot_token]
    )

    assert example["labels"].tolist() == expected_labels


def test_input_ids_contain_prompt_target_and_eot(tmp_path):
    description = "Match five digits"
    regex = r"\d{5}"

    source_path, target_path = create_dataset_files(
        tmp_path,
        descriptions=[description],
        regexes=[regex],
    )

    dataset = PatternDataset(source_path, target_path)
    example = dataset[0]

    tokenizer = tiktoken.get_encoding("gpt2")

    prompt_ids = tokenizer.encode(f"Description: {description}\nRegex:")
    target_ids = tokenizer.encode(regex)

    expected_input_ids = prompt_ids + target_ids

    assert example["input_ids"].tolist() == expected_input_ids


def test_sequence_exceeding_context_length_is_rejected(tmp_path):
    source_path, target_path = create_dataset_files(
        tmp_path,
        descriptions=["Match five digits"],
        regexes=[r"\d{5}"],
    )

    with pytest.raises(ValueError):
        PatternDataset(
            source_path,
            target_path,
            context_length=8,
        )
