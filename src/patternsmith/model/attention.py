import math

import torch


def causal_scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute causal scaled dot-product attention.

    Supports:
        Single-head: [B, T, D]
        Multi-head:  [B, H, T, D]

    Args:
        query: Query tensor.
        key: Key tensor.
        value: Value tensor.

    Returns:
        attention_output:
            [B, T, value_dim] or [B, H, T, value_dim]

        attention_weights:
            [B, T, T] or [B, H, T, T]
    """
    if query.ndim not in (3, 4):
        raise ValueError("Q, K, and V must be 3D or 4D tensors.")

    if key.ndim != query.ndim or value.ndim != query.ndim:
        raise ValueError("Q, K, and V must have the same number of dimensions.")

    if query.shape[:-2] != key.shape[:-2]:
        raise ValueError("Query and key batch/head dimensions must match.")

    if query.shape[:-2] != value.shape[:-2]:
        raise ValueError("Query and value batch/head dimensions must match.")

    if query.size(-1) != key.size(-1):
        raise ValueError("Query and key feature dimensions must match.")

    if query.size(-2) != key.size(-2):
        raise ValueError("Query and key sequence lengths must match.")

    if key.size(-2) != value.size(-2):
        raise ValueError("Key and value sequence lengths must match.")

    if query.size(-2) == 0:
        raise ValueError("Sequence length must be greater than zero.")

    if not all(tensor.is_floating_point() for tensor in (query, key, value)):
        raise TypeError("Query, key, and value must be floating-point tensors.")

    if query.device != key.device or query.device != value.device:
        raise ValueError("Query, key, and value must be on the same device.")

    if query.dtype != key.dtype or query.dtype != value.dtype:
        raise ValueError("Query, key, and value must have the same dtype.")

    # 1. Calculate query-key similarity scores.
    #    [..., T, D] @ [..., D, T] -> [..., T, T]
    scores = torch.matmul(query, key.transpose(-2, -1))

    # 2. Scale scores to control their magnitude.
    scores = scores / math.sqrt(query.size(-1))

    # 3. Create a causal mask.
    sequence_length = query.size(-2)

    causal_mask = torch.tril(
        torch.ones(
            sequence_length,
            sequence_length,
            dtype=torch.bool,
            device=query.device,
        )
    )

    # 4. Prevent each position from attending to future positions.
    scores = scores.masked_fill(~causal_mask, float("-inf"))

    # 5. Convert scores into attention probabilities.
    attention_weights = torch.softmax(scores, dim=-1)

    # 6. Combine value vectors using the attention probabilities.
    attention_output = torch.matmul(attention_weights, value)

    return attention_output, attention_weights
