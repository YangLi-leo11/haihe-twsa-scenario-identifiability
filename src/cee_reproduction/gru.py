"""GRU architecture and strict recursive state update.

The official contained validation series spans November 2020 through October 2025
(n=60). October 2020 observed TWSA initializes the recursion once; observations do
not update state inside the horizon. Metrics are verified from the bundled strict-validation
prediction table by the release regression tests.
"""

from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class GRUConfig:
    training_start: str = "2005-02-01"
    training_end: str = "2020-10-01"
    validation_start: str = "2020-11-01"
    validation_end: str = "2025-10-01"
    sequence_rows: int = 13
    hidden_units: int = 24
    gru_layers: int = 1
    learning_rate: float = 0.006
    weight_decay: float = 0.02
    epochs: int = 90
    random_seed: int = 119


class StateAnchoredGRU(nn.Module):
    """Single-layer GRU and Linear(24,24)-ReLU-Linear(24,1) head."""

    def __init__(self, input_features: int, config: GRUConfig = GRUConfig()):
        super().__init__()
        self.config = config
        self.gru = nn.GRU(input_features, config.hidden_units, config.gru_layers, batch_first=True)
        self.head = nn.Sequential(nn.Linear(config.hidden_units, config.hidden_units), nn.ReLU(), nn.Linear(config.hidden_units, 1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        encoded, _ = self.gru(features)
        return self.head(encoded[:, -1, :]).squeeze(-1)


def seed_everything(seed: int = 119) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


def optimizer_and_loss(model: nn.Module, config: GRUConfig = GRUConfig()):
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    return optimizer, nn.HuberLoss()


def recursive_accumulation(initial_observed_twsa: float, predicted_monthly_changes) -> np.ndarray:
    """Use the October 2020 state once, then update only with predicted changes."""
    return initial_observed_twsa + np.cumsum(np.asarray(predicted_monthly_changes, dtype=float))
