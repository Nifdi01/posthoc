from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Callable


import numpy as np
from torch import nn, seed

from posthoc.models.base import TrainConfig, TrainResult
from posthoc.models.utils import train_model
from posthoc.attribution.integrated_gradients import (
    IntegratedGradientsConfig,
    integrated_gradients_importance,
)


logger = logging.getLogger(__name__)


@dataclass
class SeedResult:
    seed: int
    mas: np.ndarray
    val_idx: np.ndarray
    best_val_loss: float


def run_multiseed_ig(
    model_factory: Callable[[], nn.Module],
    genotypes: np.ndarray,
    phenotype: np.ndarray,
    covariates: np.ndarray | None,
    train_config_base: TrainConfig,
    ig_config_base: IntegratedGradientsConfig,
    seeds: list[int],
) -> list[SeedResult]:

    if len(seeds) < 2:
        raise ValueError(
            f"run_multiseed_ig needs multiple seeds to aggregate over; got {len(seeds)}"
        )
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"seeds must be unique, got duplicates in {seeds}")

    results: list[SeedResult] = list()

    for i, seed in enumerate(seeds):
        logger.info("Multi-seed IG: model %d/%d (seed=%d)", i + 1, len(seeds), seed)

        train_config = replace(train_config_base, seed=seed)
        model = model_factory()

        train_result: TrainResult = train_model(
            model, genotypes, phenotype, covariates, train_config
        )
        logger.info(
            "  seed=%d: stopped at epoch %d, best val loss %.4f",
            seed,
            train_result.stopped_epoch,
            train_result.best_val_loss,
        )

        ig_config = replace(ig_config_base, seed=seed)

        val_idx = train_result.val_idx
        genotypes_val = genotypes[val_idx]
        phenotype_val = phenotype[val_idx]
        covariates_val = covariates[val_idx] if covariates is not None else None

        attribution_result = integrated_gradients_importance(
            train_result.model, genotypes_val, phenotype_val, covariates_val, ig_config
        )

        results.append(
            SeedResult(
                seed=seed,
                mas=attribution_result.importances,
                val_idx=val_idx,
                best_val_loss=train_result.best_val_loss,
            )
        )
    return results
