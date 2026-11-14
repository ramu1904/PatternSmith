import torch
import torch.nn as nn

from patternsmith.model.embeddings import PatternEmbedding
from patternsmith.model.transformer_block import TransformerBlock


class GPTModel(nn.Module):
    def __init__(
        self,
        vocab_size: int = 50257,
        context_length: int = 128,
        embedding_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        hidden_dim: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive.")

        if context_length <= 0:
            raise ValueError("context_length must be positive.")

        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive.")

        if num_layers <= 0:
            raise ValueError("num_layers must be positive.")

        if num_heads <= 0:
            raise ValueError("num_heads must be positive.")

        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")

        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads.")

        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the range [0, 1).")

        self.vocab_size = vocab_size
        self.context_length = context_length
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        # Convert token IDs into token + positional representations.
        self.embedding = PatternEmbedding(
            vocab_size=vocab_size,
            context_length=context_length,
            embedding_dim=embedding_dim,
        )

        # Stack decoder-only Transformer blocks.
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        # Normalize the final hidden representations.
        self.final_norm = nn.LayerNorm(embedding_dim)

        # Project hidden representations into vocabulary logits.
        # Bias is disabled because the output weights are tied to embeddings.
        self.lm_head = nn.Linear(
            embedding_dim,
            vocab_size,
            bias=False,
        )

        # Share the input token embedding matrix with the output head.
        self.lm_head.weight = self.embedding.token_embedding.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: Integer token IDs of shape [B, T].

        Returns:
            Vocabulary logits of shape [B, T, vocab_size].
        """
        if input_ids.ndim != 2:
            raise ValueError("Input IDs must have shape [B, T].")

        if input_ids.dtype != torch.long:
            raise TypeError("Input IDs must have dtype torch.long.")

        if input_ids.size(0) == 0:
            raise ValueError("Batch size must be greater than zero.")

        if input_ids.size(1) == 0:
            raise ValueError("Sequence length must be greater than zero.")

        if input_ids.size(1) > self.context_length:
            raise ValueError(
                f"Sequence length {input_ids.size(1)} exceeds "
                f"context length {self.context_length}."
            )

        if torch.any(input_ids < 0) or torch.any(input_ids >= self.vocab_size):
            raise ValueError("Input IDs contain invalid vocabulary indices.")

        # [B, T] -> [B, T, D]
        x = self.embedding(input_ids)

        # Refine token representations through the Transformer stack.
        for block in self.transformer_blocks:
            x = block(x)

        # Normalize final hidden representations.
        x = self.final_norm(x)

        # [B, T, D] -> [B, T, vocab_size]
        logits = self.lm_head(x)

        return logits
