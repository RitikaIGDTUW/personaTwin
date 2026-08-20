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


class PersonalizedGRU(PopulationGRU):
    """GRU baseline with a learned participant-specific embedding."""

    def __init__(
        self,
        input_size: int,
        num_participants: int,
        hidden_size: int = 64,
        embedding_size: int = 16,
        num_layers: int = 1,
        dropout: float = 0.0,
        output_size: int = 1,
    ) -> None:
        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            output_size=output_size,
        )
        self.participant_embedding = nn.Embedding(
            num_participants,
            embedding_size,
        )
        self.head = nn.Linear(hidden_size + embedding_size, output_size)

    def forward(self, x: torch.Tensor, participant: torch.Tensor) -> torch.Tensor:
        """Predict using the shared sequence state and participant embedding."""
        temporal_state = self.encode(x)
        personal_state = self.participant_embedding(participant)
        return self.head(torch.cat([temporal_state, personal_state], dim=-1))
