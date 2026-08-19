from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PALConfig:
    theta_percentile: float = 99.99
    ld_window: int = 20
    ld_r2_threshold: float = 0.5
    min_model_occurence: int | None = None


@dataclass
class PALResult:
    mu: np.ndarray
    amas: np.ndarray
    theta: float
    per_model_thetas: np.ndarray
    pal_amas: np.ndarray
    pal_common: np.ndarray
    element_counts: dict[int, int]
    n_models: int


def stack_mas(mas_list: list[np.ndarray]) -> np.ndarray:
    A = np.asarray(mas_list, dtype=np.float64)
    if A.ndim != 2:
        raise ValueError(
            f"Expected mas_list to stack into 2D (m, L) array, got shape {A.shape}"
        )
    return A


def _ld_clump(
    detected_per_model: list[np.ndarray], genotypes: np.ndarray, config: PALConfig
) -> dict[int, int]:
    n_samples, n_snps = genotypes.shape
    W = config.ld_window
    threshold = config.ld_r2_threshold

    G = genotypes.astype(np.float64)
    G_mean = G.mean(axis=0)
    G_std = G.std(axis=0)
    G_std[G_std == 0] = 1.0  # To avoid division by zero
    Z = (G - G_mean) / G_std

    clumped = [np.array(d, dtype=int).copy() for d in detected_per_model]

    for i, detected in enumerate(detected_per_model):
        if len(detected) == 0:
            clumped[i] = np.unique(clumped[i])
            continue

        extra = []

        for pos in detected:
            lo = max(0, pos - W)
            hi = min(n_snps, pos + W + 1)

            r = (Z[:, pos] @ Z[:, lo:hi]) / n_samples
            neighbors = np.arange(lo, hi)
            mask = (abs(r) > threshold) & (neighbors != pos)
            extra.append(neighbors[mask])

        if extra:
            clumped[i] = np.unique(np.concatenate([clumped[i]] + extra))
        else:
            clumped[i] = np.unique(clumped[i])

    flattened = [int(p) for sub in clumped for p in sub]
    element_counts = {}
    for p in flattened:
        element_counts[p] = element_counts.get(p, 0) + 1

    return element_counts


def compute_pal(
    mas_list: list[np.ndarray], genotypes: np.ndarray, config: PALConfig | None = None
) -> PALResult:
    config = config or PALConfig()
    A = stack_mas(mas_list)
    n_models, n_snps = A.shape

    if genotypes.shape[1] != n_snps:
        raise ValueError(
            f"genotypes has {genotypes.shape[1]} SNPs but MAS vectors have {n_snps}"
        )

    per_model_thetas = np.array(
        [np.percentile(A[a], config.theta_percentile) for a in range(n_models)]
    )

    mu = A.mean(axis=0)

    detected_per_model = [
        np.nonzero(A[a] > per_model_thetas[a])[0] for a in range(n_models)
    ]

    element_counts = _ld_clump(detected_per_model, genotypes, config)

    global_theta = np.percentile(mu, config.theta_percentile)
    amas = mu.copy()
    above = np.nonzero(mu > global_theta)[0]

    for j in above:
        occ = element_counts.get(int(j), 0)
        amas[j] = amas[j] * (occ / n_models)

    pal_amas = np.nonzero(amas > global_theta)[0]

    if n_models and config.min_model_occurence is None:
        required = n_models
    elif n_models:
        required = config.min_model_occurence
    else:
        required = 0

    pal_common = np.array(
        sorted(pos for pos, cnt in element_counts.items() if cnt >= required), dtype=int
    )

    logger.info(
        "PAL: %d models, theta_percentile=%.2f -> PAL_Common=%d, PAL_AMAS=%d SNPs",
        n_models,
        config.theta_percentile,
        len(pal_common),
        len(pal_amas),
    )

    return PALResult(
        mu=mu,
        amas=amas,
        theta=global_theta,
        per_model_thetas=per_model_thetas,
        pal_amas=pal_amas,
        pal_common=pal_common,
        element_counts=element_counts,
        n_models=n_models,
    )
