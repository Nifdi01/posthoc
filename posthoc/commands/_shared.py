from __future__ import annotations

import logging
import numpy as np
import click

from posthoc.io.genotype_reader import GenotypeData, read_pgen
from dataclasses import replace, dataclass

from posthoc.io.pheno_covar_reader import align_samples, load_covar, load_pheno
from posthoc.qc.filters import run_qc


logger = logging.getLogger(__name__)


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
    )


def subset_samples(
    data: GenotypeData, keep_mask: np.ndarray, ordered_ids: list[str]
) -> GenotypeData:
    return replace(data, genotypes=data.genotypes[keep_mask, :], sample_ids=ordered_ids)


@dataclass
class LoadedData:
    data: GenotypeData

    phenotype: np.ndarray
    covariates: np.ndarray | None


def load_and_prepare(
    pfile: str,
    pheno: str,
    pheno_name: str | None,
    covar: str | None,
    task: str,
    min_maf: float,
    max_missing: float,
    indep_pairwise: tuple[int, int, float] | None,
) -> LoadedData:

    logger.info(f"Loading genotypes from {pfile}")
    data = read_pgen(pfile)
    logger.info(f"Loaded {data.n_samples} samples x {data.n_variants} variants")

    pheno_series = load_pheno(pheno, pheno_name=pheno_name)

    if task == "logistic":
        unique_vals = set(pheno_series.dropna().unique())
        if unique_vals <= {1.0, 2.0}:
            logger.info("Recoding PLINK-style phenotype (1=control/2=case) to (0/1)")
            pheno_series = pheno_series.map({1.0: 0.0, 2.0: 1.0})
        elif not unique_vals <= {0.0, 1.0}:
            raise click.UsageError(
                f"--logistic expects phenotype coded as 0/1 or PLINK-style 1/2; "
                f"got values {sorted(unique_vals)}"
            )

    covar_df = load_covar(covar) if covar is not None else None

    keep_mask, align_pheno, align_covar = align_samples(
        data.sample_ids, pheno_series, covar_df
    )

    ordered_ids = [sid for sid, keep in zip(data.sample_ids, keep_mask) if keep]
    data = subset_samples(data, keep_mask, ordered_ids)
    n_dropped_samples = int((~keep_mask).sum())

    logger.info(
        "Sample alignment: dropped %d / %d samples (missing pheno/covar); %d remain",
        n_dropped_samples,
        len(keep_mask),
        data.n_samples,
    )

    qc_results = run_qc(
        data,
        min_maf=min_maf,
        max_missing=max_missing,
        indep_pairwise_params=indep_pairwise,
        pfile_prefix=pfile if indep_pairwise is not None else None,
    )

    data = qc_results.data
    logger.info(qc_results.summary())

    return LoadedData(data, align_pheno, align_covar)
