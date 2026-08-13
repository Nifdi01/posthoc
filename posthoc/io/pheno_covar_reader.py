from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PLINK_MISSING = {"-9", "NA", "NaN", "nan", ""}


def _read_plink_table(path: str | Path) -> pd.DataFrame:
    """
    Read a whitespace-delimited PLINK-format table into a DataFrame.

    Standardizes the sample identifier column to ``IID`` by detecting
    common PLINK header conventions.

    Parameters
    ----------
    path : str or Path
        Path to the whitespace-delimited PLINK table file.

    Returns
    -------
    pd.DataFrame
        DataFrame with all values read as strings, missing values (as
        defined by `PLINK_MISSING`) converted to NaN, and the sample
        identifier column renamed to ``IID``.
    """
    df = pd.read_csv(path, sep=r"\s+", dtype=str, na_values=list(PLINK_MISSING))
    if "IID" in df.columns:
        id_col = "IID"
    elif "#IID" in df.columns:
        id_col = "#IID"
    else:
        id_col = df.columns[1]
    df = df.rename(columns={id_col: "IID"})
    return df


def load_pheno(pheno_path: str | Path, pheno_name: str | None = None) -> pd.Series:
    """
    Load a phenotype value for each sample from a PLINK-format phenotype file.

    Parameters
    ----------
    pheno_path : str or Path
        Path to the whitespace-delimited phenotype file.
    pheno_name : str, optional
        Name of the phenotype column to load. If None, the first
        non-``FID``/``IID`` column in the file is used.

    Returns
    -------
    pd.Series
        Series of phenotype values indexed by sample ``IID``, cast to float.
    """

    df = _read_plink_table(pheno_path)
    value_cols = [c for c in df.columns if c not in ("FID", "IID")]
    col = pheno_name or value_cols[0]
    series = df.set_index("IID")[col].astype(float)
    return series


def load_covar(covar_path: str | Path) -> pd.DataFrame:
    """
    Load covariate values for each sample from a PLINK-format covariate file.

    Parameters
    ----------
    covar_path : str or Path
        Path to the whitespace-delimited covariate file.

    Returns
    -------
    pd.DataFrame
        DataFrame of covariate values indexed by sample ``IID``, with all
        covariate columns (excluding ``FID`` and ``IID``) cast to float.
    """
    df = _read_plink_table(covar_path)
    value_cols = [c for c in df.columns if c not in ("FID", "IID")]
    covar_df = df.set_index("IID")[value_cols].astype(float)
    return covar_df


def align_samples(
    sample_ids: list[str], pheno: pd.Series, covar: pd.DataFrame | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """
    Align genotype sample order with phenotype and (optionally) covariate data.

    Determines the set of samples present in `sample_ids` that also have
    non-missing values in `pheno` and, if provided, `covar`, then returns
    a boolean mask and correspondingly ordered phenotype and covariate arrays.

    Parameters
    ----------
    sample_ids : list of str
        Sample identifiers in genotype order.
    pheno : pd.Series
        Phenotype values indexed by sample identifier.
    covar : pd.DataFrame, optional
        Covariate values indexed by sample identifier. If None, only
        `pheno` is used for alignment.

    Returns
    -------
    keep_mask : np.ndarray
        Boolean array of length ``len(sample_ids)`` indicating which
        samples (in original order) are retained.
    aligned_pheno : np.ndarray
        Phenotype values for the retained samples, in `sample_ids` order.
    aligned_covar : np.ndarray or None
        Covariate values for the retained samples, in `sample_ids` order,
        or None if `covar` was not provided.

    Raises
    ------
    ValueError
        If there are no overlapping, non-missing samples across the
        provided genotype, phenotype, and covariate data.
    """
    common = pd.Index(sample_ids).intersection(pheno.dropna().index)
    if covar is not None:
        common = common.intersection(covar.dropna().index)

    if len(common) == 0:
        raise ValueError(
            "No overlapping, non-missing samples between genotypes, "
            "pheno, and covar files."
        )

    keep_mask = np.array([sid in common for sid in sample_ids])
    aligned_pheno = pheno.loc[np.array(sample_ids)[keep_mask]].to_numpy()
    aligned_covar = (
        covar.loc[np.array(sample_ids)[keep_mask]].to_numpy()
        if covar is not None
        else None
    )

    return keep_mask, aligned_pheno, aligned_covar
