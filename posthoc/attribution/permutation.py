from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
from scipy import stats
from torch import nn

from posthoc.models.base import GenotypeDataset

logger = logging.getLogger(__name__)


@dataclass
class PermutationConfig:
    n_repeats: int = 20
    batch_size: int = 256
    device: str = "cpu"
    seed: int = 0
    task: str = "logistic"


@dataclass
class AttributionResult:
    importances: np.ndarray
    p_values: np.ndarray
    n_repeats: int
    method: str = "permutation"


def _per_sample_load(pred: torch.Tensor, y: torch.Tensor, task: str) -> torch.Tensor:
    if task == "logistic":
        return nn.functional.binary_cross_entropy_with_logits(pred, y, reduction="none")
    elif task == "linear":
        return nn.functional.mse_loss(pred, y, reduction="none")
    else:
        raise ValueError(f"Unknown task: {task}. Expected 'logistic' or 'linear'.")


@torch.no_grad()
def _score_dataset(
    model: nn.Module,
    genotypes: torch.Tensor,
    phenotype: torch.Tensor,
    covariates: torch.Tensor | None,
    task: str,
    device: str,
    batch_size: int,
) -> np.ndarray:

    model.eval()
    n = genotypes.shape[0]
    losses = np.empty(n, dtype=np.float64)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        x = genotypes[start:end].to(device)
        if covariates is not None:
            x = torch.cat([x, covariates[start:end].to(device)], dim=1)
        y = phenotype[start:end].to(device)
        pred = model(x).squeeze(-1)
        loss = _per_sample_load(pred, y, task)
        losses[start:end] = loss.cpu().numpy()
    return losses


def permutation_importance(
    model: nn.Module,
    genotypes: np.ndarray,
    phenotype: np.ndarray,
    covariates: np.ndarray | None,
    config: PermutationConfig,
) -> AttributionResult:

    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)

    dataset = GenotypeDataset(genotypes, phenotype, covariates)
    geno_t = dataset.genotypes.clone()
    pheno_t = dataset.phenotype
    covar_t = dataset.covariates
    n_samples, n_snps = geno_t.shape

    baseline_losses = _score_dataset(
        model, geno_t, pheno_t, covar_t, config.task, config.device, config.batch_size
    )

    baseline_mean = baseline_losses.mean()
    logger.info("Baseline mean loss: %.4f", baseline_mean)

    importances = np.zeros(n_snps, dtype=np.float64)
    p_values = np.ones(n_snps, dtype=np.float64)

    for j in range(n_snps):
        deltas = np.empty(config.n_repeats, dtype=np.float64)
        for r in range(config.n_repeats):
            perm = rng.permutation(n_samples)
            geno_perm = geno_t.clone()
            geno_perm[:, j] = geno_t[perm, j]

            permutted_losses = _score_dataset(
                model,
                geno_perm,
                pheno_t,
                covar_t,
                config.task,
                config.device,
                config.batch_size,
            )
            deltas[r] = permutted_losses.mean() - baseline_mean

        importances[j] = deltas.mean()

        if np.allclose(deltas, deltas[0]):
            p_values[j] = 0.0 if deltas[0] > 0 else 1.0
        else:
            t_stat, p_two_sided = stats.ttest_1samp(deltas, popmean=0.0)
            p_values[j] = p_two_sided / 2 if t_stat > 0 else 1 - p_two_sided / 2

        if n_snps >= 10 and (j + 1) % (n_snps // 10) == 0:
            logger.debug("Permutation importance: %d/%d SNPs done", j + 1, n_snps)
    return AttributionResult(
        importances=importances, p_values=p_values, n_repeats=config.n_repeats
    )
