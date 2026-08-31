from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal, Protocol

import numpy as np
import torch
from torch import nn

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    """
    Config for the training loop (optimizer, schedule, splits).

    Deliberately model-agnostic. Model-specific knobs (layer sizes, kernel
    sizes, etc.) belong in that model's own config dataclass, e.g. MLPConfig.
    """

    task: Literal["logistic", "linear"] = "logistic"
    val_fraction: float = 0.2
    batch_size: int = 64
    max_epochs: int = 200
    patience: int = 15
    lr: float = 1e-4
    weight_decay: float = 1e-4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 0


@dataclass
class TrainResult:
    model: nn.Module
    best_val_loss: float
    best_auc_score: float
    train_losses: list[float]
    val_losses: list[float]
    stopped_epoch: int
    train_idx: np.ndarray
    val_idx: np.ndarray


@dataclass
class MLPConfig:
    hidden_dims: list[int] = field(default_factory=lambda: [128, 64])
    dropout: float = 0.2
    data_dropout: bool = False
    batch_norm: bool = True
    activation: str = "relu"


_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "gelu": nn.GELU,
    "lrelu": nn.LeakyReLU,
}


class TrainStrategy(Protocol):
    """
    Encapsulates how a single batch is turned into predictions and a loss.

    Trainer composes with a TrainStrategy instead of assuming a model's
    forward signature or loss. Most models can use a shared default
    strategy (see strategy.py); a model with unusual training needs
    (multi-input forward, auxiliary losses, custom metrics) supplies its
    own implementation of this protocol instead of forking the training
    loop in Trainer.
    """

    def compute_loss(
        self, model: nn.Module, x: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    def metric(self, y_true: np.ndarray, y_pred_raw: np.ndarray) -> float: ...


class ModelFactory(Protocol):
    """
    A model family (MLP, CNN, ...) implements this to build itself from
    its own config. Nothing needs to subclass a common base model.
    """

    def build(self, n_snps: int, n_covariates: int = 0) -> nn.Module: ...
