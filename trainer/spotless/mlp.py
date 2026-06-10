from __future__ import annotations

import torch
import torch.nn as nn


class SpotLessModule(nn.Module):
    """Tiny mask MLP used by the SLS-mlp path."""

    def __init__(self, num_features: int, num_classes: int = 1, hidden_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_classes),
            nn.Sigmoid(),
        )

    def get_regularizer(self) -> torch.Tensor:
        first = self.net[0].weight
        last = self.net[2].weight
        return torch.max(first.abs()) * torch.max(last.abs())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

