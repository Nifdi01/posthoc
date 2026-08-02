from __future__ import annotations

import logging
from dataclasses import dataclass, field

import torch
from torch import nn

logger = logging.getLogger(__name__)


@dataclass
class MLPConfig:
    hidden_dims: list[int] = field(default_factory=lambda: [128, 64])
    dropout: float = 0.2
    batch_norm: bool = True
    activation: str = "relu"


_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "gelu": nn.GELU,
    "lrelu": nn.LeakyReLU,
}


class MLP(nn.Module):
    def __init__(self, input_dim: int, config: MLPConfig | None = None):
        super().__init__()
        self.config = config if config is not None else MLPConfig()
        self.input_dim = input_dim

        if self.config.activation not in _ACTIVATIONS:
            raise ValueError(
                f"Unknown activation: {self.config.activation}. "
                f"Expected one of {list(_ACTIVATIONS)}."
            )
        act_cls = _ACTIVATIONS[self.config.activation]()

        layers: list[nn.Module] = []
        prev_dim = input_dim

        for hidden_dim in self.config.hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if self.config.batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(act_cls)
            if self.config.dropout > 0:
                layers.append(nn.Dropout(self.config.dropout))
            prev_dim = hidden_dim

        self.hidden = nn.Sequential(*layers)
        self.head = nn.Linear(prev_dim, 1)

        logger.debug(
            "Built MLP: input_dim=%d hidden_dims=%s dropout=%.2f batch_norm=%s",
            input_dim,
            self.config.hidden_dims,
            self.config.dropout,
            self.config.batch_norm,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.hidden(x))


def build_mlp(
    n_snps: int, n_covariates: int = 0, config: MLPConfig | None = None
) -> MLP:
    return MLP(input_dim=n_snps + n_covariates, config=config)
