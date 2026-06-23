"""Evaluation helpers for small experiments."""

from __future__ import annotations

import torch
from torch import nn

from .utils import accuracy_from_logits


def evaluate_classifier(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    criterion: nn.Module | None = None,
) -> tuple[float, float | None]:
    """Evaluate classification accuracy and optional loss."""
    model.eval()
    with torch.no_grad():
        logits = model(x)
        accuracy = accuracy_from_logits(logits, y)
        loss = criterion(logits, y).item() if criterion is not None else None
    return accuracy, loss


def evaluate_regressor(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    criterion: nn.Module | None = None,
) -> float | None:
    """Evaluate optional regression loss."""
    model.eval()
    with torch.no_grad():
        predictions = model(x)
        return criterion(predictions, y).item() if criterion is not None else None
