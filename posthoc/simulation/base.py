from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from posthoc.io.genotype_reader import GenotypeData


@dataclass
class PhenotypeModel:
    """Additive + dominant + recessive + interaction liability model,
    following Yelmen et al. (2026)."""

    additive_indices: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=int)
    )
    additive_effects: np.ndarray = field(default_factory=lambda: np.array([]))

    dominant_indices: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=int)
    )
    dominant_effects: np.ndarray = field(default_factory=lambda: np.array([]))

    recessive_indices: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=int)
    )
    recessive_effects: np.ndarray = field(default_factory=lambda: np.array([]))

    interaction_pairs: list[tuple[int, int]] = field(default_factory=list)
    interaction_effects: np.ndarray = field(default_factory=lambda: np.array([]))

    interaction_triples: list[tuple[int, int, int]] = field(default_factory=list)
    interaction_triple_effects: np.ndarray = field(default_factory=lambda: np.array([]))

    heritability: float = 0.5
    task: str = "linear"  # "linear" | "logistic"
    prevalence: float = 0.1
    seed: int | None = None

    def __post_init__(self) -> None:
        pairs = [
            (self.additive_indices, self.additive_effects),
            (self.dominant_indices, self.dominant_effects),
            (self.recessive_indices, self.recessive_effects),
        ]
        for idx, eff in pairs:
            if len(idx) != len(eff):
                raise ValueError("effect index/value arrays must be same length")
        if len(self.interaction_pairs) != len(self.interaction_effects):
            raise ValueError(
                "interaction_pairs and interaction_effects must be same length"
            )
        if len(self.interaction_triples) != len(self.interaction_triple_effects):
            raise ValueError(
                "interaction_triples and interaction_triple_effects must be same length"
            )
        if not 0.0 < self.heritability <= 1.0:
            raise ValueError("heritability must be in (0, 1]")

    def all_causal_indices(self) -> np.ndarray:
        """Union of every SNP index involved in any effect term."""
        idx = set(self.additive_indices.tolist())
        idx |= set(self.dominant_indices.tolist())
        idx |= set(self.recessive_indices.tolist())
        for i, j in self.interaction_pairs:
            idx |= {i, j}
        for i, j, k in self.interaction_triples:
            idx |= {i, j, k}
        return np.array(sorted(idx), dtype=int)


@dataclass
class SimulationResult:
    """Real genotypes + simulated phenotype, with ground truth for scoring."""

    genotype_data: GenotypeData
    phenotype: np.ndarray
    causal_indices: np.ndarray
    liability: np.ndarray | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def sample_ids(self) -> list[str]:
        return self.genotype_data.sample_ids

    @property
    def variant_df(self) -> pd.DataFrame:
        return self.genotype_data.variant_ids

    def as_pheno_df(self, pheno_name: str = "PHENO1") -> pd.DataFrame:
        return pd.DataFrame({"#IID": self.sample_ids, pheno_name: self.phenotype})

    def causal_variant_ids(self) -> list[str]:
        return self.variant_df["ID"].to_numpy()[self.causal_indices].tolist()


def _fill_missing(geno_cols: np.ndarray) -> np.ndarray:
    """Mean-impute -9 (missing) using column means, in place on a copy."""
    geno_cols = geno_cols.astype(np.float64)
    missing = geno_cols == -9
    if missing.any():
        means = np.where(missing, np.nan, geno_cols)
        means = np.nanmean(means, axis=0)
        fill = np.take(means, np.where(missing)[1])
        geno_cols[missing] = fill
    return geno_cols


def simulate_phenotype(
    genotypes: np.ndarray,
    model: PhenotypeModel,
    covariates: np.ndarray | None = None,
    covariate_effects: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    rng = np.random.default_rng(model.seed)
    n_samples = genotypes.shape[0]
    component = np.zeros(n_samples)

    if len(model.additive_indices):
        g = _fill_missing(genotypes[:, model.additive_indices])
        component += g @ model.additive_effects

    if len(model.dominant_indices):
        g = _fill_missing(genotypes[:, model.dominant_indices])
        # dominant: contributes only where genotype == -1 (het/rare-carrier encoding)
        mask = g == -1
        component += (np.where(mask, g, 0) * model.dominant_effects).sum(axis=1)

    if len(model.recessive_indices):
        g = _fill_missing(genotypes[:, model.recessive_indices])
        mask = g == 1
        component += (np.where(mask, g, 0) * model.recessive_effects).sum(axis=1)

    if model.interaction_pairs:
        for (i, j), beta in zip(model.interaction_pairs, model.interaction_effects):
            gi = _fill_missing(genotypes[:, [i]])[:, 0]
            gj = _fill_missing(genotypes[:, [j]])[:, 0]
            component += beta * gi * gj

    if model.interaction_triples:
        for (i, j, k), beta in zip(
            model.interaction_triples, model.interaction_triple_effects
        ):
            gi = _fill_missing(genotypes[:, [i]])[:, 0]
            gj = _fill_missing(genotypes[:, [j]])[:, 0]
            gk = _fill_missing(genotypes[:, [k]])[:, 0]
            component += beta * gi * gj * gk

    cov_component = np.zeros(n_samples)
    if covariates is not None and covariate_effects is not None:
        cov_component = covariates @ covariate_effects

    genetic_var = np.var(component)
    if genetic_var == 0:
        raise ValueError(
            "Genetic component has zero variance — check causal effect terms"
        )

    noise_var = genetic_var * (1.0 - model.heritability) / model.heritability
    noise = rng.normal(0.0, np.sqrt(noise_var), size=n_samples)
    liability = component + cov_component + noise

    if model.task == "linear":
        return liability, None
    if model.task == "logistic":
        threshold = np.quantile(liability, 1.0 - model.prevalence)
        phenotype = (liability > threshold).astype(np.float64)
        return phenotype, liability
    raise ValueError(f"Unknown task: {model.task}")


def simulate(
    genotype_data: GenotypeData,
    phenotype_model: PhenotypeModel,
    covariates: np.ndarray | None = None,
    covariate_effects: np.ndarray | None = None,
) -> SimulationResult:
    """Simulate a phenotype on top of real genotype data (e.g. from read_pgen)."""
    phenotype, liability = simulate_phenotype(
        genotype_data.genotypes, phenotype_model, covariates, covariate_effects
    )
    return SimulationResult(
        genotype_data=genotype_data,
        phenotype=phenotype,
        causal_indices=phenotype_model.all_causal_indices(),
        liability=liability,
        metadata={
            "n_samples": genotype_data.n_samples,
            "n_variants": genotype_data.n_variants,
            "task": phenotype_model.task,
            "heritability": phenotype_model.heritability,
        },
    )
