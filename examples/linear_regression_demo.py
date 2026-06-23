"""Run a simple linear regression experiment."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.train import train_linear


if __name__ == "__main__":
    train_linear(epochs=80, lr=0.05)
