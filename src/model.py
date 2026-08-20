"""Baseline temporal models for PersonaTwin."""

from __future__ import annotations

import torch
from torch import nn


class PopulationGRU(nn.Module):
    """Shared GRU baseline for next-day target prediction."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
        output_size: int = 1,
    ) -> None:
        super().__init__()
        recurrent_dropout = dropout if num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=recurrent_dropout,
        )
        self.head = nn.Linear(hidden_size, output_size)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return the final shared temporal representation."""
        _, hidden = self.gru(x)
        return hidden[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict the next-day target from a sequence window."""
        return self.head(self.encode(x))
