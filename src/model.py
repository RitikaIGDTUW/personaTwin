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
        projection_size: int | None = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        output_size: int = 1,
    ) -> None:
        super().__init__()
        recurrent_input_size = projection_size or input_size
        self.input_projection = (
            nn.Linear(input_size, projection_size)
            if projection_size is not None
            else nn.Identity()
        )
        recurrent_dropout = dropout if num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=recurrent_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=recurrent_dropout,
        )
        self.input_dropout = nn.Dropout(dropout)
        self.head_dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, output_size)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return the final shared temporal representation."""
        projected = self.input_projection(self.input_dropout(x))
        _, hidden = self.gru(projected)
        return hidden[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict the next-day target from a sequence window."""
        return self.head(self.head_dropout(self.encode(x)))


class PersonalizedGRU(PopulationGRU):
    """GRU baseline with a learned participant-specific embedding."""

    def __init__(
        self,
        input_size: int,
        num_participants: int,
        hidden_size: int = 64,
        projection_size: int | None = None,
        embedding_size: int = 16,
        num_layers: int = 1,
        dropout: float = 0.0,
        output_size: int = 1,
    ) -> None:
        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            projection_size=projection_size,
            num_layers=num_layers,
            dropout=dropout,
            output_size=output_size,
        )
        self.participant_embedding = nn.Embedding(
            num_participants,
            embedding_size,
        )
        self.personal_dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size + embedding_size, output_size)

    def forward(self, x: torch.Tensor, participant: torch.Tensor) -> torch.Tensor:
        """Predict using the shared sequence state and participant embedding."""
        temporal_state = self.encode(x)
        personal_state = self.participant_embedding(participant)
        combined = torch.cat([temporal_state, personal_state], dim=-1)
        return self.head(self.personal_dropout(combined))
