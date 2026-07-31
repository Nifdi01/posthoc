from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PLINK_MISSING = {"-9", "NA", "NaN", "nan", ""}


def _read_plink_table(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", dtype=str, na_values=list(PLINK_MISSING))
    id_col = "IID" if "IID" in df.columns else df.columns[1]
    df = df.rename(columns={id_col: "IID"})
    return df


def load_pheno(pheno_path: str | Path, pheno_name: str | None = None) -> pd.Series:
    df = _read_plink_table(pheno_path)
    value_cols = [c for c in df.columns if c not in ("FID", "IID")]
    col = pheno_name or value_cols[0]
    series = df.set_index("IID")[col].astype(float)
    return series


def load_covar(covar_path: str | Path) -> pd.DataFrame:
    df = _read_plink_table(covar_path)
    value_cols = [c for c in df.columns if c not in ("FID", "IID")]
    covar_df = df.set_index("IID")[value_cols].astype(float)
    return covar_df


def align_samples(
    sample_ids: list[str], pheno: pd.Series, covar: pd.DataFrame | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
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
