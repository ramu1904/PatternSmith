import torch
import torch.nn as nn


class PatternEmbedding(nn.Module):
    """
    Combines learned token embeddings and learned positional embeddings.

    Input:
        input_ids: [batch_size, sequence_length]

    Output:
        embeddings: [batch_size, sequence_length, embedding_dim]
    """

    def __init__(
        self,
        vocab_size: int = 50257,
        context_length: int = 128,
        embedding_dim: int = 128,
    ) -> None:
        super().__init__()

        self.vocab_size = vocab_size
        self.context_length = context_length
        self.embedding_dim = embedding_dim

        self.token_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
        )

        self.position_embedding = nn.Embedding(
            num_embeddings=context_length,
            embedding_dim=embedding_dim,
        )

        # GPT-2-style small normal initialization.
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch_size, sequence_length].")

        sequence_length = input_ids.size(1)

        if sequence_length == 0:
            raise ValueError("Sequence length must be greater than zero.")

        if sequence_length > self.context_length:
            raise ValueError(
                f"Sequence length {sequence_length} exceeds "
                f"context length {self.context_length}."
            )

        if input_ids.dtype != torch.long:
            raise TypeError("input_ids must have dtype torch.long.")

        # [B, T] -> [B, T, D]
        token_embeddings = self.token_embedding(input_ids)

        # Select positional indices 0 through T-1.
        position_ids = torch.arange(
            sequence_length,
            device=input_ids.device,
        )

        # [T] -> [T, D]
        position_embeddings = self.position_embedding(position_ids)

        # Broadcasting: [B, T, D] + [T, D]
        embeddings = token_embeddings + position_embeddings

        return embeddings
