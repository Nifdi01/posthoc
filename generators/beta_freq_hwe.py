from typing import Optional
import numpy as np
from generators.base_generator import BaseGenomeGenerator


class BetaFrequencyHWEGenerator(BaseGenomeGenerator):
    def __init__(
        self,
        genome_size: int,
        causal_pos: int,
        beta_a: float = 0.5,
        beta_b: float = 0.5,
        case_freq_shift: float = 0.35,
        random_state: Optional[int] = None,
    ):
        self.beta_a = beta_a
        self.beta_b = beta_b
        self.case_freq_shift = case_freq_shift
        super().__init__(genome_size, causal_pos, random_state)

        self.maf = self.rng.beta(self.beta_a, self.beta_b, size=self.genome_size)

        self.case_maf = self.maf.copy()
        self.case_maf[self.causal_indices] = np.clip(
            self.case_maf[self.causal_indices] + self.case_freq_shift, 0.01, 0.99
        )

    def _hwe_genotypes(self, n_samples: int, freqs: np.ndarray) -> np.ndarray:
        q = freqs
        p = 1 - q
        probs = np.stack([p**2, 2 * p * q, q**2], axis=1)

        cum_probs = np.cumsum(probs, axis=1)
        u = self.rng.random(size=(n_samples, self.genome_size))
        genotype_idx = (u[:, :, None] > cum_probs[None, :, :]).sum(axis=2)

        return genotype_idx - 1  # to map to -1, 0, 1

    def _generate_controls(self, n_samples: int) -> np.ndarray:
        return self._hwe_genotypes(n_samples, self.maf)

    def _generate_cases(self, n_samples: int) -> np.ndarray:
        return self._hwe_genotypes(n_samples, self.case_maf)

    def _build_metadata(self) -> dict:
        return {"maf": self.maf, "case_maf": self.case_maf}
