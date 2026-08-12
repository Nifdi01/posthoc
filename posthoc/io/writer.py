from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

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
