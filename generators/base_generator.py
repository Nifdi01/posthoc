from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class GenomeGenerationResult:
    genotypes: np.ndarray
    status: np.ndarray
    causal_indices: np.ndarray
    metadata: dict = field(default_factory=dict)


class BaseGenomeGenerator(ABC):
    def __init__(
        self, genome_size: int, causal_pos: int, random_state: Optional[int] = None
    ):
        if causal_pos > genome_size:
            raise ValueError("causal_pos cannot exceed genome_size")

        self.genome_size = genome_size
        self.causal_pos = causal_pos
        self.rng = np.random.default_rng(random_state)

        self.causal_indices = self._assign_causal_indices()

    @abstractmethod
    def _generate_controls(self, n_samples: int) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def _generate_cases(self, n_smaples: int) -> np.ndarray:
        raise NotImplementedError

    def _assign_causal_indices(self) -> np.ndarray:
        return np.arange(self.causal_pos)

    def generate(
        self, n_control: int, n_case: int, shuffle_snp_order: bool = False
    ) -> GenomeGenerationResult:
        controls = self._generate_controls(n_control)
        cases = self._generate_cases(n_case)

        self._validate_shape(controls, n_control)
        self._validate_shape(cases, n_case)

        genotypes = np.vstack((controls, cases))
        status = np.concatenate((np.zeros(n_control), np.ones(n_case)))
        causal_indices = self.causal_indices.copy()

        if shuffle_snp_order:
            genotypes, causal_indices = self._shuffle_columns(genotypes, causal_indices)

        return GenomeGenerationResult(
            genotypes=genotypes,
            status=status,
            causal_indices=causal_indices,
            metadata=self._build_metadata(),
        )

    def _validate_shape(self, arr: np.ndarray, n_samples: int) -> None:
        expected = (n_samples, self.genome_size)

        if arr.shape != expected:
            raise ValueError(
                f"{type(self).__name__} produced shape {arr.shape}, expected {expected}"
            )

    def _shuffle_columns(self, genotypes: np.ndarray, causal_indices: np.ndarray):
        perm = self.rng.permutation(self.genome_size)

        new_pos = np.empty_like(perm)
        new_pos[perm] = np.arange(self.genome_size)
        return genotypes[:, perm], new_pos[causal_indices]

    def _build_metadata(self) -> dict:
        return {}

    def __repr__(self) -> str:
        return f"{type(self).__name__}(genome_size={self.genome_size}, causal_pos={self.causal_pos})"
