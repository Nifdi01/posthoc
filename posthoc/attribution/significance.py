from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.stats import halfnorm

from posthoc.attribution.pal import PALConfig, PALResult, compute_pal

logger = logging.getLogger(__name__)


@dataclass
class SignificanceConfig:
    n_bootstrap: int = 100
    seed: int = 0


@dataclass
class SignificanceResult:
    p_values: np.ndarray
    halfnorm_scale: float
    positions: np.ndarray


def fit_null_halfnorm(null_mas_list: list[np.ndarray]) -> float:
    null_mas = np.concatenate([np.asarray(m).ravel() for m in null_mas_list])
    _, scale = halfnorm.fit(null_mas)
    logger.info("Fitted half-normal null: scale=%.6g (n=%d)", scale, null_mas.size)

    return scale


def _sample_and_rank_null(
    scale: float, shape: tuple[int, int], observed: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    sampled = halfnorm.rvs(scale=scale, size=shape, random_state=rng)
    ranked = np.zeros_like(sampled)
    for i in range(shape[0]):
        desc_idx = np.argsort(-observed[i])
        sampled_desc = np.sort(sampled[i])[::-1]
        ranked[i, desc_idx] = sampled_desc

    return ranked


def compute_pal_pvalues(
    observed_mas_list: list[np.ndarray],
    null_mas_list: list[np.ndarray],
    genotypes: np.ndarray,
    pal_result: PALResult,
    pal_config: PALConfig,
    sig_config: SignificanceConfig | None = None,
) -> SignificanceResult:

    sig_config = sig_config or SignificanceConfig()
    rng = np.random.default_rng(sig_config.seed)

    observed = np.asarray(observed_mas_list, dtype=np.float64)
    n_models, n_snps = observed.shape

    scale = fit_null_halfnorm(null_mas_list)

    pal_positions = pal_result.pal_amas
    if pal_positions.size == 0:
        logger.warning("PAL_AMAS is empty; no positions to compute P-values for.")
        return SignificanceResult(
            p_values=np.array([]), halfnorm_scale=scale, positions=pal_positions
        )

    exceed_counts = np.zeros(len(pal_positions), dtype=np.int64)

    for b in range(sig_config.n_bootstrap):
        null_matrix = _sample_and_rank_null(scale, (n_models, n_snps), observed, rng)

        boot_pal = compute_pal(
            mas_list=[null_matrix[a] for a in range(n_models)],
            genotypes=genotypes,
            config=pal_config,
        )

        for idx, pos in enumerate(pal_positions):
            exceed_counts[idx] += int(np.sum(boot_pal.amas > pal_result.amas[pos]))

        if (b + 1) % max(1, sig_config.n_bootstrap // 10) == 0:
            logger.debug(
                "Significance bootstrap %d/%d done", b + 1, sig_config.n_bootstrap
            )

    p_values = exceed_counts / (sig_config.n_bootstrap * n_snps)

    return SignificanceResult(
        p_values=p_values,
        halfnorm_scale=scale,
        positions=pal_positions,
    )
