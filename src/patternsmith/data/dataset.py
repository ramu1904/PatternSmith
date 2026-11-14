from pathlib import Path

import torch
from torch.utils.data import Dataset
import tiktoken


class PatternDataset(Dataset):
    """Dataset for conditional natural-language-to-regex generation."""

    def __init__(
        self,
        source_file: str | Path,
        target_file: str | Path,
        context_length: int = 128,
    ) -> None:

        self.source_file = Path(source_file)
        self.target_file = Path(target_file)

        self.context_length = context_length

        self.encoding = tiktoken.get_encoding("gpt2")

        self.eot_token_id = self.encoding.eot_token
        self.vocab_size = self.encoding.n_vocab

        self.descriptions = self.source_file.read_text(encoding="utf-8").splitlines()

        self.targets = self.target_file.read_text(encoding="utf-8").splitlines()

        if len(self.descriptions) != len(self.targets):
            raise ValueError("Description and target counts do not match.")

        if not self.descriptions:
            raise ValueError("Dataset is empty.")

        self.examples = []

        self._prepare_examples()

    def _prepare_examples(self) -> None:

        for index, (description, target) in enumerate(
            zip(self.descriptions, self.targets)
        ):

            prompt = f"Description: {description}\n" "Regex:"

            prompt_ids = self.encoding.encode(prompt)
            target_ids = self.encoding.encode(target)

            prompt_length = len(prompt_ids)

            input_ids = prompt_ids + target_ids + [self.eot_token_id]

            if len(input_ids) > self.context_length:
                raise ValueError(
                    f"Example {index} exceeds context length: "
                    f"{len(input_ids)} > {self.context_length}"
                )

            # Shift the complete sequence by one position. (input - target pairs)
            x = input_ids[:-1]
            labels = input_ids[1:]

            # Ignore prompt-prediction positions.
            labels = [-100] * (prompt_length - 1) + labels[prompt_length - 1 :]

            self.examples.append(
                {
                    "input_ids": torch.tensor(
                        x,
                        dtype=torch.long,
                    ),
                    "labels": torch.tensor(
                        labels,
                        dtype=torch.long,
                    ),
                    "prompt_length": prompt_length,
                    "target_length": len(target_ids),
                    "total_length": len(input_ids),
                }
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:

        example = self.examples[index]

        return {
            "input_ids": example["input_ids"],
            "labels": example["labels"],
        }
