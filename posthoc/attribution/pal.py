from __future__ import annotations
import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PALConfig:
    theta_percentile: float = 0.99
    ld_window: int = 20
    ld_r2_threshold: float = 0.5
    min_model_occurence: int | None = None


@dataclass
class PALResult:
    mu: np.ndarray
    weights: np.ndarray
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
) -> tuple[list[np.ndarray], dict[int, int]]:
    n_snps = genotypes.shape[1]
    clumped = [np.array(d, dtype=int).copy() for d in detected_per_model]

    for i, detected in enumerate(detected_per_model):
        extra = []
        for pos in detected:
            for offset in range(-config.ld_window, config.ld_window + 1):
                if offset == 0:
                    continue
                neighbor = pos + offset
                if neighbor < 0 or neighbor >= n_snps:
                    continue
                r = np.corrcoef(genotypes[:, pos], genotypes[:, neighbor])[0, 1]
                if abs(r) > config.ld_r2_threshold:
                    extra.append(neighbor)

        if extra:
            clumped[i] = np.unique(
                np.concatenate([clumped[i], np.array(extra, dtype=int)])
            )
        else:
            clumped[i] = np.unique(clumped[i])

    flattened = [int(p) for sub in clumped for p in sub]
    element_counts = {}
    for p in flattened:
        element_counts[p] = element_counts.get(p, 0) + 1

    return clumped, element_counts


def compute_pal(
    mas_list: list[np.ndarray], genotypes: np.ndarray, config: PALConfig | None = None
) -> PALResult:
    pass
