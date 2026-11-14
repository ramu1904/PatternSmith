from __future__ import annotations

import torch


def pattern_collate_fn(
    batch: list[dict[str, torch.Tensor]],
    pad_token_id: int = 50256,
) -> dict[str, torch.Tensor]:
    """
    Dynamically right-pad PatternSmith examples into a batch.

    Input:
        A list of examples, each containing:
            input_ids: 1D LongTensor
            labels:    1D LongTensor

    Output:
        input_ids:      [batch_size, max_length]
        labels:         [batch_size, max_length]
        attention_mask: [batch_size, max_length]

    padding_mask:
        True  = actual token
        False = padding
    """
    if not batch:
        raise ValueError("Cannot collate an empty batch.")

    batch_size = len(batch)

    # Find the longest sequence in this batch.
    max_length = max(example["input_ids"].size(0) for example in batch)

    # Allocate padded tensors.
    input_ids = torch.full(
        (batch_size, max_length),
        fill_value=pad_token_id,
        dtype=torch.long,
    )

    labels = torch.full(
        (batch_size, max_length),
        fill_value=-100,
        dtype=torch.long,
    )

    padding_mask = torch.zeros(
        (batch_size, max_length),
        dtype=torch.bool,
    )

    for index, example in enumerate(batch):
        example_input_ids = example["input_ids"]
        example_labels = example["labels"]

        if example_input_ids.ndim != 1 or example_labels.ndim != 1:
            raise ValueError("Each example must contain 1D tensors.")

        if example_input_ids.size(0) != example_labels.size(0):
            raise ValueError(f"Input and label lengths differ in example {index}.")

        sequence_length = example_input_ids.size(0)

        # Copy actual tokens to the left; padding remains on the right.
        input_ids[index, :sequence_length] = example_input_ids
        labels[index, :sequence_length] = example_labels

        # Mark actual tokens as True.
        padding_mask[index, :sequence_length] = True

    return {
        "input_ids": input_ids,
        "labels": labels,
        "padding_mask": padding_mask,
    }
