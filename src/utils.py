"""Small utilities shared by examples and training scripts."""

from __future__ import annotations

import random
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set common random seeds for reproducible small experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Return CUDA when available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def plot_loss_curve(losses: Iterable[float], title: str = "Training Loss") -> None:
    """Plot a simple loss curve with Matplotlib."""
    plt.figure(figsize=(6, 4))
    plt.plot(list(losses), label="loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute classification accuracy from unnormalized logits."""
    predictions = logits.argmax(dim=1)
    return (predictions == labels).float().mean().item()
