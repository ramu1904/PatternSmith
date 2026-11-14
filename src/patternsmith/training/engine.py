import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from patternsmith.training.gradient_clipping import clip_gradients


@dataclass
class EpochMetrics:
    loss: float
    perplexity: float
    supervised_tokens: int
    steps: int
    learning_rate: float
    mean_gradient_norm: float = 0.0


def calculate_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    """
    Calculate token-level cross-entropy loss.

    Labels with value -100 are ignored.
    """

    if logits.ndim != 3:
        raise ValueError("logits must have shape [batch, sequence, vocab].")

    if labels.ndim != 2:
        raise ValueError("labels must have shape [batch, sequence].")

    if logits.shape[:2] != labels.shape:
        raise ValueError("Logits and labels batch/sequence dimensions must match.")

    supervised_tokens = int((labels != -100).sum().item())

    if supervised_tokens == 0:
        raise ValueError("Batch contains no supervised target tokens.")

    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=-100,
    )

    return loss, supervised_tokens


def _calculate_perplexity(loss: float) -> float:
    if loss > 709:
        return float("inf")

    return math.exp(loss)


def train_one_epoch(
    model: nn.Module,
    dataloader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scheduler=None,
    max_grad_norm: float = 1.0,
) -> EpochMetrics:
    """Run one complete training epoch."""

    model.train()

    total_loss = 0.0
    total_tokens = 0
    total_steps = 0
    total_gradient_norm = 0.0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad(set_to_none=True)

        logits = model(input_ids)
        loss, supervised_tokens = calculate_loss(logits, labels)

        loss.backward()

        gradient_norm = clip_gradients(
            model,
            max_norm=max_grad_norm,
        )

        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item() * supervised_tokens
        total_tokens += supervised_tokens
        total_steps += 1
        total_gradient_norm += float(gradient_norm.item())

    if total_tokens == 0:
        raise ValueError("Training dataloader produced no supervised tokens.")

    average_loss = total_loss / total_tokens

    return EpochMetrics(
        loss=average_loss,
        perplexity=_calculate_perplexity(average_loss),
        supervised_tokens=total_tokens,
        steps=total_steps,
        learning_rate=optimizer.param_groups[0]["lr"],
        mean_gradient_norm=total_gradient_norm / total_steps,
    )


@torch.inference_mode()
def validate_one_epoch(
    model: nn.Module,
    dataloader,
    device: torch.device,
) -> EpochMetrics:
    """Evaluate the model without updating its parameters."""

    model.eval()

    total_loss = 0.0
    total_tokens = 0
    total_steps = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids)
        loss, supervised_tokens = calculate_loss(logits, labels)

        total_loss += loss.item() * supervised_tokens
        total_tokens += supervised_tokens
        total_steps += 1

    if total_tokens == 0:
        raise ValueError("Validation dataloader produced no supervised tokens.")

    average_loss = total_loss / total_tokens

    return EpochMetrics(
        loss=average_loss,
        perplexity=_calculate_perplexity(average_loss),
        supervised_tokens=total_tokens,
        steps=total_steps,
        learning_rate=0.0,
    )
