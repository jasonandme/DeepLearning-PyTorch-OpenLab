"""A minimal PyTorch training loop on synthetic data."""

import torch
from torch import nn


torch.manual_seed(42)

x = torch.randn(64, 1)
y = 2.0 * x - 1.0 + 0.1 * torch.randn(64, 1)

model = nn.Linear(1, 1)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

for epoch in range(50):
    predictions = model(x)
    loss = criterion(predictions, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(f"final loss: {loss.item():.4f}")
print(f"weight: {model.weight.item():.3f}, bias: {model.bias.item():.3f}")
