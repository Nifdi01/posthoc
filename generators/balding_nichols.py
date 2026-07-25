from typing import Optional

import numpy as np

from generators.base_generator import BaseGenomeGenerator


class BaldingNicholsGenerator(BaseGenomeGenerator):
    def __init__(
        self,
        genome_size: int,
        causal_pos: int,
        n_populations: int = 2,
        fst: float = 0.05,
        case_freq_shit: float = 0.30,
        random_state: Optional[int] = None,
    ):
        self.n_populations = n_populations
        self.fst = fst
        self.case_freq_shit = case_freq_shit
        super().__init__(genome_size, causal_pos, random_state)

        self.p_ancestral = self.rng.uniform(0.05, 0.95, size=self.genome_size)

        alpha = self.p_ancestral * (1 - self.fst) / self.fst
        beta = (1 - self.p_ancestral) * (1 - self.fst) / self.fst

        self.pop_freqs = np.stack(
            [self.rng.beta(alpha, beta) for _ in range(self.n_populations)]
        )

        self.case_pop_freqs = self.pop_freqs.copy()
        self.case_pop_freqs[:, self.causal_indices] = np.clip(
            self.case_pop_freqs[:, self.causal_indices] + self.case_freq_shit,
            0.01,
            0.99,
        )

    def _hwe_genotypes_for_pop(
        self, n_samples: int, freqs_row: np.ndarray
    ) -> np.ndarray:
        q = freqs_row
        p = 1 - q
        probs = np.stack([p**2, 2 * p * q, q**2], axis=1)
        cum_probs = np.cumsum(probs, axis=1)
        u = self.rng.random(size=(n_samples, self.genome_size))
        genotype_idx = (u[:, :, np.newaxis] > cum_probs[np.newaxis, :, :]).sum(axis=2)
        return genotype_idx - 1

    def _assign_populations(self, n_samples: int) -> np.ndarray:
        return self.rng.integers(0, self.n_populations, size=n_samples)

    def _generate_stratified(
        self, n_samples: int, freq_matrix: np.ndarray
    ) -> np.ndarray:

        pop_assignment = self._assign_populations(n_samples)
        out = np.empty((n_samples, self.genome_size), dtype=int)
        for pop in range(self.n_populations):
            mask = pop_assignment == pop
            n_in_pop = mask.sum()
            if n_in_pop > 0:
                out[mask] = self._hwe_genotypes_for_pop(n_in_pop, freq_matrix[pop])
        self._last_pop_assignment = pop_assignment
        return out

    def _generate_controls(self, n_samples: int) -> np.ndarray:
        genos = self._generate_stratified(n_samples, self.pop_freqs)
        self._control_pops = self._last_pop_assignment
        return genos

    def _generate_cases(self, n_samples: int) -> np.ndarray:
        genos = self._generate_stratified(n_samples, self.case_pop_freqs)
        self._case_pops = self._last_pop_assignment
        return genos

    def _build_metadata(self) -> dict:
        meta = {"p_ancestral": self.p_ancestral, "fst": self.fst}
        if hasattr(self, "_control_pops") and hasattr(self, "_case_pops"):
            meta["population_labels"] = np.concatenate(
                (self._control_pops, self._case_pops)
            )
        return meta
