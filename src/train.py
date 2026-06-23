"""Command-line training entry points for small PyTorch experiments."""

from __future__ import annotations

import argparse

import torch
from torch import nn

from .datasets.synthetic import make_linear_regression_data, make_moon_classification_data
from .evaluate import evaluate_classifier
from .models.basic import LinearRegressionModel, MLPClassifier
from .utils import accuracy_from_logits, set_seed


def train_linear(epochs: int = 80, lr: float = 0.05) -> None:
    set_seed(42)
    x, y = make_linear_regression_data()
    model = LinearRegressionModel()
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        predictions = model(x)
        loss = criterion(predictions, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            print(f"epoch={epoch + 1:03d} loss={loss.item():.4f}")

    weight = model.linear.weight.item()
    bias = model.linear.bias.item()
    print(f"learned weight={weight:.3f}, bias={bias:.3f}")


def train_mlp(epochs: int = 80, lr: float = 0.01) -> None:
    set_seed(42)
    x_train, y_train, x_test, y_test = make_moon_classification_data()
    model = MLPClassifier()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        logits = model(x_train)
        loss = criterion(logits, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            train_acc = accuracy_from_logits(logits, y_train)
            print(f"epoch={epoch + 1:03d} loss={loss.item():.4f} train_acc={train_acc:.3f}")

    test_acc, test_loss = evaluate_classifier(model, x_test, y_test, criterion)
    print(f"test_loss={test_loss:.4f} test_acc={test_acc:.3f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a small PyTorch training experiment.")
    parser.add_argument("--task", choices=["linear", "mlp"], default="linear")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.task == "linear":
        train_linear(epochs=args.epochs, lr=args.lr if args.lr is not None else 0.05)
    else:
        train_mlp(epochs=args.epochs, lr=args.lr if args.lr is not None else 0.01)


if __name__ == "__main__":
    main()
