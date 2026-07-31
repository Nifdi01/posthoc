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
    "N",
]


def write_glm(
    variant_ids: pd.DataFrame,
    importances: np.ndarray,
    p_values: np.ndarray,
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
    out_df["N"] = n_samples
    out_df = out_df[GLM_COLUMNS]

    out_df.to_csv(out_path, sep="\t", index=False)
