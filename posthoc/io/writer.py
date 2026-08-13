from __future__ import annotations

from pathlib import Path

import pandas as pd
import numpy as np


GLM_COLUMNS = [
    "#CHROM",
    "POS",
    "ID",
    "REF",
    "ALT",
    "A1",
    "TEST",
    "IMPORTANCE",
    "P_PERM",
    "P_CORRECTED",
    "N",
]


def write_glm(
    variant_ids: pd.DataFrame,
    importances: np.ndarray,
    p_values: np.ndarray,
    p_corrected: np.ndarray,
    test_name: str,
    n_samples: int,
    out_path: str | Path,
) -> None:
    """
    Write GLM (generalized linear model) association results to a tab-separated file.

    Combines variant metadata with importance scores, permutation p-values,
    and corrected p-values into a single output table.

    Parameters
    ----------
    variant_ids : pd.DataFrame
        DataFrame containing variant metadata with columns ``CHROM``, ``POS``,
        ``ID``, ``REF``, and ``ALT``. Must have the same number of rows as
        `importances`.
    importances : np.ndarray
        Array of importance/effect-size scores, one per variant.
    p_values : np.ndarray
        Array of permutation p-values, one per variant.
    p_corrected : np.ndarray
        Array of multiple-testing-corrected p-values, one per variant.
    test_name : str
        Name of the statistical test used, written to the ``TEST`` column.
    n_samples : int
        Number of samples used in the analysis, written to the ``N`` column.
    out_path : str or Path
        Destination file path for the tab-separated output.

    Returns
    -------
    None
        Writes the result directly to `out_path`.

    Raises
    ------
    ValueError
        If the length of `importances` does not match the number of rows
        in `variant_ids`.
    """
    if len(importances) != len(variant_ids):
        raise ValueError("importances length must match number of variants")

    out_df = variant_ids.copy()
    out_df = out_df.rename(columns={"CHROM": "#CHROM"})
    out_df["A1"] = out_df["ALT"]
    out_df["TEST"] = test_name
    out_df["IMPORTANCE"] = importances
    out_df["P_PERM"] = p_values
    out_df["P_CORRECTED"] = p_corrected
    out_df["N"] = n_samples
    out_df = out_df[GLM_COLUMNS]

    out_df.to_csv(out_path, sep="\t", index=False)


PAL_COLUMNS = [
    "#CHROM",
    "POS",
    "ID",
    "REF",
    "ALT",
    "MU",
    "AMAS",
    "IN_PAL_COMMON",
    "IN_PAL_AMAS",
    "P_VALUE",
    "N_MODELS",
]


def write_pal(
    variant_ids: pd.DataFrame,
    mu: np.ndarray,
    amas: np.ndarray,
    pal_common_idx: np.ndarray,
    pal_amas_idx: np.ndarray,
    p_values: np.ndarray,
    n_models: int,
    out_path: str | Path,
) -> None:
    """
    Write PAL (polygenic adaptation / allele-sharing) results to a tab-separated file.

    Combines variant metadata with mu and AMAS statistics, boolean flags
    indicating membership in common and AMAS-specific PAL sets, and
    associated p-values.

    Parameters
    ----------
    variant_ids : pd.DataFrame
        DataFrame containing variant metadata with columns ``CHROM``, ``POS``,
        ``ID``, ``REF``, and ``ALT``. Must have the same number of rows as `mu`.
    mu : np.ndarray
        Array of mu statistics, one per variant.
    amas : np.ndarray
        Array of AMAS statistics, one per variant.
    pal_common_idx : np.ndarray
        Integer indices (into the variant array) of variants belonging to
        the common PAL set.
    pal_amas_idx : np.ndarray
        Integer indices (into the variant array) of variants belonging to
        the AMAS-specific PAL set. Also used to index `p_values`.
    p_values : np.ndarray
        P-values corresponding to the variants indexed by `pal_amas_idx`.
        Variants not in `pal_amas_idx` are assigned NaN.
    n_models : int
        Number of models used in the analysis, written to the ``N_MODELS`` column.
    out_path : str or Path
        Destination file path for the tab-separated output.

    Returns
    -------
    None
        Writes the result directly to `out_path`.

    Raises
    ------
    ValueError
        If the length of `mu` does not match the number of rows in `variant_ids`.
    """
    if len(mu) != len(variant_ids):
        raise ValueError("mu length must match number of variants")

    out_df = variant_ids.copy()
    out_df = out_df.rename(columns={"CHROM": "#CHROM"})
    out_df["MU"] = mu
    out_df["AMAS"] = amas

    in_common = np.zeros(len(mu), dtype=bool)
    in_common[pal_common_idx] = True
    out_df["IN_PAL_COMMON"] = in_common

    in_amas = np.zeros(len(mu), dtype=bool)
    in_amas[pal_amas_idx] = True
    out_df["IN_PAL_AMAS"] = in_amas

    pval_col = np.full(len(mu), np.nan)
    pval_col[pal_amas_idx] = p_values
    out_df["P_VALUE"] = pval_col

    out_df["N_MODELS"] = n_models
    out_df = out_df[PAL_COLUMNS]

    out_df.to_csv(out_path, sep="\t", index=False)
