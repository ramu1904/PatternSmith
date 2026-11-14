import torch
import torch.nn as nn


class FeedForwardMLP(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 128,
        hidden_dim: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive.")

        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")

        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the range [0, 1).")

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        # Expand each token representation.
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)

        # Introduce nonlinearity.
        self.activation = nn.GELU()

        # Project back to the model's embedding dimension.
        self.fc2 = nn.Linear(hidden_dim, embedding_dim)

        # Regularize the MLP output before the residual addition.
        self.output_dropout = nn.Dropout(dropout)

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

        # [B, T, 128] -> [B, T, 512]
        x = self.fc1(x)

        # Apply GELU without changing tensor dimensions.
        x = self.activation(x)

        # [B, T, 512] -> [B, T, 128]
        x = self.fc2(x)

        # Dropout is active during training only.
        x = self.output_dropout(x)

        return x
