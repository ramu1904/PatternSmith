import torch
import torch.nn as nn

from patternsmith.model.attention import (
    causal_scaled_dot_product_attention,
)


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive.")

        if num_heads <= 0:
            raise ValueError("num_heads must be positive.")

        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads.")

        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the range [0, 1).")

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        # Independent trainable Q, K, and V projections.
        self.query_projection = nn.Linear(embedding_dim, embedding_dim)
        self.key_projection = nn.Linear(embedding_dim, embedding_dim)
        self.value_projection = nn.Linear(embedding_dim, embedding_dim)

        # Combines information from all attention heads.
        self.output_projection = nn.Linear(embedding_dim, embedding_dim)

        self.output_dropout = nn.Dropout(dropout)

    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Convert [B, T, D] into [B, H, T, head_dim].
        """
        batch_size, seq_length, _ = tensor.shape

        tensor = tensor.view(
            batch_size,
            seq_length,
            self.num_heads,
            self.head_dim,
        )

        return tensor.transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [B, T, embedding_dim].

        Returns:
            Output tensor of shape [B, T, embedding_dim].
        """
        if x.ndim != 3:
            raise ValueError("Input must be a 3D tensor [B, T, D].")

        if x.size(-1) != self.embedding_dim:
            raise ValueError(
                f"Expected embedding dimension {self.embedding_dim}, "
                f"received {x.size(-1)}."
            )

        if x.size(1) == 0:
            raise ValueError("Sequence length must be greater than zero.")

        # Step 1: Generate Q, K, and V.
        query = self.query_projection(x)
        key = self.key_projection(x)
        value = self.value_projection(x)

        # Step 2: Split each projection into multiple heads.
        query = self._split_heads(query)
        key = self._split_heads(key)
        value = self._split_heads(value)

        # Step 3: Apply causal attention independently to every head.
        attention_output, _ = causal_scaled_dot_product_attention(
            query,
            key,
            value,
        )

        # Step 4: Merge the attention heads.
        attention_output = attention_output.transpose(1, 2).contiguous()

        batch_size, seq_length, _, _ = attention_output.shape

        attention_output = attention_output.view(
            batch_size,
            seq_length,
            self.embedding_dim,
        )

        # Step 5: Combine head representations using output projection.
        output = self.output_projection(attention_output)

        # Step 6: Apply dropout.
        output = self.output_dropout(output)

        return output
