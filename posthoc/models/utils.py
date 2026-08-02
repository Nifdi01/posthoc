import logging
from typing import cast

import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from posthoc.models.base import GenotypeDataset, TrainConfig, TrainResult

logger = logging.getLogger(__name__)


def make_split(
    n_samples: int,
    phenotype: np.ndarray,
    val_fraction: float,
    task: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:

    idx = np.arange(n_samples)
    stratify = phenotype if task == "logistic" else None

    train_idx, val_idx = train_test_split(
        idx, test_size=val_fraction, stratify=stratify, random_state=seed
    )
    return cast(np.ndarray, train_idx), cast(np.ndarray, val_idx)


def get_loss_fn(task: str) -> nn.Module:
    if task == "logistic":
        return nn.BCEWithLogitsLoss()
    elif task == "linear":
        return nn.MSELoss()
    else:
        raise ValueError(f"Unknown task: {task}. Expected 'logistic' or 'linear'.")


def train_model(
    model: nn.Module,
    genotypes: np.ndarray,
    phenotype: np.ndarray,
    covariates: np.ndarray | None,
    config: TrainConfig,
) -> TrainResult:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    dataset = GenotypeDataset(genotypes, phenotype, covariates)
    train_idx, val_idx = make_split(
        len(dataset), phenotype, config.val_fraction, config.task, config.seed
    )
    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx), batch_size=config.batch_size, shuffle=False
    )

    model = model.to(config.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    loss_fn = get_loss_fn(config.task)

    best_val_loss = float("inf")
    best_state = None
    epoch_without_improvement = 0
    train_losses, val_losses = list(), list()
    stopped_epoch = config.max_epochs

    for epoch in range(config.max_epochs):
        model.train()
        n_train = 0
        epoch_train_loss = 0
        for x, y in train_loader:
            x, y = x.to(config.device), y.to(config.device)
            pred = model(x).squeeze(-1)
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item() * x.size(0)
            n_train += x.size(0)
        epoch_train_loss /= n_train

        model.eval()
        epoch_val_loss = 0
        n_val = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(config.device), y.to(config.device)
                pred = model(x).squeeze(-1)
                loss = loss_fn(pred, y)
                epoch_val_loss += loss.item() * x.size(0)
                n_val += x.size(0)
        epoch_val_loss /= n_val

        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)

        logger.debug(
            "Epoch %d: train_loss=%.4f val_loss=%.4f",
            epoch,
            epoch_train_loss,
            epoch_val_loss,
        )

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epoch_without_improvement = 0
        else:
            epoch_without_improvement += 1
            if epoch_without_improvement >= config.patience:
                stopped_epoch = epoch
                logger.info(
                    "Early stopping at epoch %d (best val_loss=%.4f)",
                    epoch,
                    best_val_loss,
                )
                break
    if best_state is not None:
        model.load_state_dict(best_state)

    return TrainResult(
        model=model,
        best_val_loss=best_val_loss,
        train_losses=train_losses,
        val_losses=val_losses,
        stopped_epoch=stopped_epoch,
        train_idx=train_idx,
        val_idx=val_idx,
    )
