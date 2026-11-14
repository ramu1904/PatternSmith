import argparse
from pathlib import Path
from typing import Optional

import torch
import tiktoken

from patternsmith.model.gpt_model import GPTModel


class PatternSmithInference:
    """Generate regular expressions from natural-language descriptions."""

    def __init__(
        self,
        checkpoint_path: str,
        device: Optional[str] = None,
    ) -> None:

        self.checkpoint_path = Path(checkpoint_path)

        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        # Select inference device.
        if device is None or device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested, but no CUDA-enabled GPU is available."
            )

        self.device = torch.device(device)

        # Load the trained checkpoint.
        checkpoint = torch.load(
            self.checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )

        # Recover the exact model configuration used during training.
        model_config = checkpoint["config"]["model"]

        self.context_length = model_config["context_length"]

        # Recreate the GPT architecture.
        self.model = GPTModel(**model_config)

        # Restore trained weights.
        self.model.load_state_dict(
            checkpoint["model_state_dict"],
            strict=True,
        )

        self.model.to(self.device)
        self.model.eval()

        # Load the same GPT-2 BPE tokenizer used during training.
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.eot_token_id = self.tokenizer.eot_token

        self.epoch = checkpoint["epoch"]
        self.best_val_loss = checkpoint["best_val_loss"]

    @torch.inference_mode()
    def generate(
        self,
        description: str,
        max_new_tokens: int = 64,
    ) -> str:
        """Generate a regex using greedy autoregressive decoding."""

        if not isinstance(description, str):
            raise TypeError("Description must be a string.")

        description = description.strip()

        if not description:
            raise ValueError("Description cannot be empty.")

        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive.")

        # IMPORTANT: This must match the training prompt format.
        prompt = f"Description: {description}\nRegex:"

        # Encode the prompt with GPT-2 BPE.
        prompt_ids = self.tokenizer.encode(prompt)

        if len(prompt_ids) >= self.context_length:
            raise ValueError(
                f"Prompt contains {len(prompt_ids)} tokens. "
                f"Model context length is {self.context_length}."
            )

        # Prevent generation beyond the context window.
        max_new_tokens = min(
            max_new_tokens,
            self.context_length - len(prompt_ids),
        )

        # Create a batch containing one prompt.
        input_ids = torch.tensor(
            [prompt_ids],
            dtype=torch.long,
            device=self.device,
        )

        generated_ids = []

        # Generate one token at a time.
        for _ in range(max_new_tokens):

            # Forward pass through PatternSmith.
            logits = self.model(input_ids)

            # Select logits corresponding to the final position.
            next_token_logits = logits[:, -1, :]

            # Greedy decoding: select the highest-scoring token.
            next_token = torch.argmax(
                next_token_logits,
                dim=-1,
                keepdim=True,
            )

            next_token_id = next_token.item()

            # Stop when the model generates GPT-2 EOT.
            if next_token_id == self.eot_token_id:
                break

            generated_ids.append(next_token_id)

            # Append the new token for the next prediction.
            input_ids = torch.cat(
                [input_ids, next_token],
                dim=1,
            )

        # Decode generated tokens into a regex string.
        prediction = self.tokenizer.decode(generated_ids)

        return prediction.strip()


def load_patternsmith(
    checkpoint_path: Optional[str] = None,
    device: str = "auto",
) -> PatternSmithInference:
    """Load the trained PatternSmith model."""

    if checkpoint_path is None:
        project_root = Path(__file__).resolve().parents[2]
        checkpoint_path = str(project_root / "checkpoints" / "best.pt")

    return PatternSmithInference(
        checkpoint_path=checkpoint_path,
        device=device,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate regular expressions using the trained " "PatternSmith GPT model."
        )
    )

    parser.add_argument(
        "description",
        type=str,
        help="Natural-language description of the regex.",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to the trained checkpoint.",
    )

    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Inference device. Default: auto.",
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="Maximum number of generated tokens. Default: 64.",
    )

    args = parser.parse_args()

    patternsmith = load_patternsmith(
        checkpoint_path=args.checkpoint,
        device=args.device,
    )

    prediction = patternsmith.generate(
        description=args.description,
        max_new_tokens=args.max_new_tokens,
    )

    print("\nPatternSmith")
    print("=" * 50)
    print(f"Description : {args.description}")
    print(f"Regex       : {prediction}")
    print(f"Checkpoint  : epoch {patternsmith.epoch}")
    print(f"Device      : {patternsmith.device}")
    print("=" * 50)


if __name__ == "__main__":
    main()
