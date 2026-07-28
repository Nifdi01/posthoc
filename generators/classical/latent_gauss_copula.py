from typing import Optional
import numpy as np
from generators.base import BaseGenomeGenerator


class LatentGaussianCopulaGenerator(BaseGenomeGenerator):
    def __init__(
        self,
        genome_size: int,
        causal_pos: int,
        ld_decay: float = 0.85,
        ld_window: int = 10,
        control_maf: float = 0.30,
        case_freq_shift: float = 0.30,
        random_state: Optional[int] = None,
    ):
        self.ld_decay = ld_decay
        self.ld_window = ld_window
        self.control_maf = control_maf
        self.case_freq_shift = case_freq_shift
        super().__init__(genome_size, causal_pos, random_state)

        self._kernel = self.ld_decay ** np.arange(self.ld_window)

    def _correlated_latent(self, n_samples: int) -> np.ndarray:
        pad = self.ld_window - 1
        noise = self.rng.standard_normal(size=(n_samples, self.genome_size + pad))
        latent = np.apply_along_axis(
            lambda row: np.convolve(row, self._kernel, mode="valid"), axis=1, arr=noise
        )
        latent = (latent - latent.mean(0)) / (latent.std(axis=0) + 1e-8)
        return latent

    def _threshold_to_genotypes(self, latent: np.ndarray, maf: float) -> np.ndarray:
        from scipy.stats import norm

        q = maf
        p = 1 - q
        cdf_thresh_1 = norm.ppf(p**2)
        cdf_thresh_2 = norm.ppf(p**2 + 2 * p * q)

        genotypes = np.full(latent.shape, 1, dtype=int)
        genotypes = np.where(latent < cdf_thresh_1, 0, genotypes)
        genotypes = np.where(latent >= cdf_thresh_2, 2, genotypes)
        genotypes = np.where(
            (latent >= cdf_thresh_1) & (latent < cdf_thresh_2), 1, genotypes
        )
        return genotypes - 1

    def _generate_controls(self, n_samples: int) -> np.ndarray:
        latent = self._correlated_latent(n_samples)
        return self._threshold_to_genotypes(latent, self.control_maf)

    def _generate_cases(self, n_samples: int) -> np.ndarray:
        latent = self._correlated_latent(n_samples)
        genos = self._threshold_to_genotypes(latent, self.control_maf)

        causal_maf = min(self.control_maf + self.case_freq_shift, 0.99)
        causal_latent = latent[:, self.causal_indices]
        genos[:, self.causal_indices] = self._threshold_to_genotypes(
            causal_latent, causal_maf
        )
        return genos

    def _build_metadata(self) -> dict:
        return {"ld_decay": self.ld_decay, "ld_window": self.ld_window}
