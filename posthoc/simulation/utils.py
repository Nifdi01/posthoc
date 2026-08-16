from pathlib import Path

import click
import numpy as np
import pandas as pd

import logging


def heritability_to_k(heritability: float) -> float:
    """Convert heritability to the paper's noise scaling factor k
    (noise_var = k * genetic_var), for logging/parity with Yelmen et al."""
    return (1.0 - heritability) / heritability


def resolve_snp_pool(
    variant_df: pd.DataFrame,
    chromosome: str,
    snp_pool_path: str | None,
) -> np.ndarray:
    """Return an array of eligible variant *positional indices* to sample
    causal SNPs from, honoring --chromosome and/or --snp-pool. Defaults to
    the whole genome (all variants) when neither is given, matching the
    paper's main SCZ simulation scenarios."""

    pool_mask = np.ones(len(variant_df), dtype=bool)

    if chromosome is not None:
        if "CHROM" not in variant_df.columns:
            raise click.UsageError(
                "--chromosome requires CHROM column in the variant metadata."
            )
        pool_mask &= (variant_df["CHROM"].astype(str) == str(chromosome)).to_numpy()

    if snp_pool_path is not None:
        wanted_ids = pd.Series(set(Path(snp_pool_path).read_text().split()))
        if "ID" not in variant_df.columns:
            raise click.UsageError(
                "--snp-pool requires an ID column in the variant metadata."
            )
        pool_mask &= variant_df["ID"].isin(wanted_ids).to_numpy()

    pool = np.where(pool_mask)[0]
    if len(pool) == 0:
        raise click.UsageError(
            "No variants remain after applying --chromosome / --snp-pool filters."
        )
    return pool
