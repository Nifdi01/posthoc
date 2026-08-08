from __future__ import annotations
import logging
from dataclasses import dataclass

import numpy as np
import torch
from captum.attr import IntegratedGradients
from scipy import stats
from statsmodels.stats.multitest import multipletests
from torch import nn

from posthoc.models.base import GenotypeDataset

logger = logging.getLogger(__name__)


@dataclass
class IntegratedGradientsConfig:
    n_steps: int = 50
    batch_size: int = 256
    device: str = "cpu"
    seed: int = 0
    task: str = "logistic"
    baseline: str = "zero"


@dataclass
class AttributionResult:
    importances: np.ndarray
    p_values: np.ndarray
    p_corrected: np.ndarray
    n_repeats: int
    method: str = "integrated_gradients"


class _ModelLogitWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        if out.dim() > 1:
            out = out.squeeze(-1)
        return out


def _make_baseline(
    x: torch.Tensor, mode: str, feature_means: torch.Tensor | None
) -> torch.Tensor:
    if mode == "zero":
        return torch.zeros_like(x)
    elif mode == "mean":
        if feature_means is None:
            raise ValueError("feature_means required for mode='mean'")
        return feature_means.expand_as(x).clone()
    else:
        raise ValueError(f"Unknown baseline mode: {mode}. Expected 'zero' or 'mean'.")


def integrated_gradients_importance(
    model: nn.Module,
    genotypes: np.ndarray,
    phenotype: np.ndarray,
    covariates: np.ndarray | None,
    config: IntegratedGradientsConfig,
) -> AttributionResult:

    torch.manual_seed(config.seed)
    n_snps = genotypes.shape[1]

    dataset = GenotypeDataset(genotypes, phenotype, covariates)
    geno_t = dataset.genotypes.to(config.device)
    covar_t = (
        dataset.covariates.to(config.device) if dataset.covariates is not None else None
    )

    if covar_t is not None:
        x_full = torch.cat([geno_t, covar_t], dim=1)
    else:
        x_full = geno_t

    model = model.to(config.device)
    model.eval()
    wrapped_model = _ModelLogitWrapper(model)
    ig = IntegratedGradients(wrapped_model)

    feature_means = (
        x_full.mean(dim=0, keepdim=True) if config.baseline == "mean" else None
    )
    baseline = _make_baseline(x_full, config.baseline, feature_means)

    n_samples = x_full.shape[0]
    n_features = x_full.shape[1]
    attributions = np.empty((n_samples, n_features), dtype=np.float64)
    logger.info(
        "Running Integrated Gradients (%d steps, baseline=%s) on %d samples x %d features",
        config.n_steps,
        config.baseline,
        n_samples,
        n_features,
    )

    for start in range(0, n_samples, config.batch_size):
        end = min(start + config.batch_size, n_samples)
        x_batch = x_full[start:end].requires_grad_(True)
        baseline_batch = baseline[start:end]

        attr_batch = ig.attribute(
            x_batch,
            baselines=baseline_batch,
            n_steps=config.n_steps,
        )

        attributions[start:end] = attr_batch.detach().cpu().numpy()

        if n_samples >= 10 and (end % (n_samples // 10) < config.batch_size):
            logger.debug("Integrated Gradients: %d/%d samples done", end, n_samples)

    snp_attributions = attributions[:, :n_snps]
    importances = np.abs(snp_attributions).mean(axis=0)

    p_values = np.ones(n_snps, dtype=np.float64)

    for j in range(n_snps):
        col = snp_attributions[:, j]
        if np.allclose(col, col[0]):
            p_values[j] = 0.0 if abs(col[0]) > 0 else 1.0
        else:
            _, p_two_sided = stats.ttest_1samp(col, popmean=0.0)
            p_values[j] = p_two_sided

    _, p_bonf, _, _ = multipletests(p_values, method="bonferroni")

    logger.info(f"Integrated Gradients done: {n_snps} attributed")

    return AttributionResult(
        importances=importances,
        p_values=p_values,
        p_corrected=p_bonf,
        n_repeats=config.n_steps,
    )
