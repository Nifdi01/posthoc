from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class GenotypeDataset(Dataset):
    MISSING_CODE = -9

    def __init__(
        self,
        genotypes: np.ndarray,
        phenotype: np.ndarray,
        covariates: np.ndarray | None = None,
    ):
        geno = genotypes.astype(np.float32).copy()
        missing_mask = geno == self.MISSING_CODE

        if missing_mask.any():
            variant_means = np.where(missing_mask, np.nan, geno)
            variant_means = np.nanmean(variant_means, axis=0)
            fill = np.take(variant_means, np.where(missing_mask)[1])
            geno[missing_mask] = fill
            logger.info(
                "Mean-imputed %d missing genotype calls (%.3f%% of matrix)",
                missing_mask.sum(),
                100 * missing_mask.mean(),
            )

        self.genotypes = torch.from_numpy(geno)
        self.phenotype = torch.from_numpy(phenotype.astype(np.float32))
        self.covariates = (
            torch.from_numpy(covariates.astype(np.float32))
            if covariates is not None
            else None
        )

    def __len__(self) -> int:
        return self.genotypes.shape[0]

    def __getitem__(self, idx: int):
        x = self.genotypes[idx]
        if self.covariates is not None:
            x = torch.cat([x, self.covariates[idx]], dim=0)
        y = self.phenotype[idx]
        return x, y


@dataclass
class TrainConfig:
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
