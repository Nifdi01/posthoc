from typing import Optional
import numpy as np

from generators.base import BaseGenomeGenerator


class IndependentSNPGenerator(BaseGenomeGenerator):
    def __init__(
        self,
        genome_size: int,
        causal_pos: int,
        control_distribution=(0.10, 0.45, 0.45),
        case_distribution=(0.90, 0.05, 0.05),
        random_state: Optional[int] = None,
    ):
        self.control_distribution = np.asarray(control_distribution)
        self.case_distribution = np.asarray(case_distribution)
        self._validate_distribution(self.control_distribution)
        self._validate_distribution(self.case_distribution)
        super().__init__(genome_size, causal_pos, random_state)

    @staticmethod
    def _validate_distribution(dist: np.ndarray) -> None:
        if not np.isclose(dist.sum(), 1.0):
            raise ValueError(f"Distribution must sum to 1, got {dist.sum()}")

    def _generate_controls(self, n_samples: int) -> np.ndarray:
        return self.rng.choice(
            [-1, 0, 1], size=(n_samples, self.genome_size), p=self.control_distribution
        )

    def _generate_cases(self, n_samples: int) -> np.ndarray:
        causal = self.rng.choice(
            [-1, 0, 1], size=(n_samples, self.causal_pos), p=self.case_distribution
        )
        neutral = self.rng.choice(
            [-1, 0, 1],
            size=(n_samples, self.genome_size - self.causal_pos),
            p=self.control_distribution,
        )

        return np.hstack((causal, neutral))
