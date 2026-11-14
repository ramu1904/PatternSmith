import torch
import torch.nn as nn

from patternsmith.model.mlp import FeedForwardMLP
from patternsmith.model.multi_head_attention import MultiHeadAttention


class TransformerBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 128,
        num_heads: int = 4,
        hidden_dim: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive.")

        if num_heads <= 0:
            raise ValueError("num_heads must be positive.")

        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads.")

        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")

        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the range [0, 1).")

        self.embedding_dim = embedding_dim

        # Normalize before the attention sublayer.
        self.norm1 = nn.LayerNorm(embedding_dim)

        # Causal multi-head self-attention.
        self.attention = MultiHeadAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        # Normalize before the feed-forward sublayer.
        self.norm2 = nn.LayerNorm(embedding_dim)

        # Token-wise feed-forward network.
        self.mlp = FeedForwardMLP(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

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

        # Sublayer 1: Pre-LN attention + residual connection.
        x = x + self.attention(self.norm1(x))

        # Sublayer 2: Pre-LN MLP + residual connection.
        x = x + self.mlp(self.norm2(x))

        return x
