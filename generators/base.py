from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class GenomeGenerationResult:
    genotypes: np.ndarray
    metadata: dict = field(default_factory=dict)


class BaseGenomeGenerator(ABC):
    def __init__(self, genome_size: int, random_state: Optional[int] = None):
        self.genome_size = genome_size
        self.rng = np.random.default_rng(random_state)

    @abstractmethod
    def _generate(self, n_samples: int) -> np.ndarray:
        """Generate a genotype matrix of shape (n_samples, genome_size)."""
        raise NotImplementedError

    def generate(
        self, n_samples: int, shuffle_snp_order: bool = False
    ) -> GenomeGenerationResult:
        genotypes = self._generate(n_samples)

        self._validate_shape(genotypes, n_samples)

        if shuffle_snp_order:
            genotypes = self._shuffle_columns(genotypes)

        return GenomeGenerationResult(
            genotypes=genotypes,
            metadata=self._build_metadata(),
        )

    def _validate_shape(self, arr: np.ndarray, n_samples: int) -> None:
        expected = (n_samples, self.genome_size)

        if arr.shape != expected:
            raise ValueError(
                f"{type(self).__name__} produced shape {arr.shape}, expected {expected}"
            )

    def _shuffle_columns(self, genotypes: np.ndarray) -> np.ndarray:
        perm = self.rng.permutation(self.genome_size)
        return genotypes[:, perm]

    def _build_metadata(self) -> dict:
        return {}

    def __repr__(self) -> str:
        return f"{type(self).__name__}(genome_size={self.genome_size})"
