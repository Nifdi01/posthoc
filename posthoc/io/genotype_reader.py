from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import pgenlib as pg
from io import StringIO


@dataclass
class GenotypeData:
    """Container for a genotype matrix and variant/sample metadata"""

    genotypes: np.ndarray
    variant_ids: pd.DataFrame
    sample_ids: list[str]

    @property
    def n_samples(self) -> int:
        return self.genotypes.shape[0]

    @property
    def n_variants(self) -> int:
        return self.genotypes.shape[1]


def _read_psam(psam_path: Path) -> list[str]:
    df = pd.read_csv(psam_path, sep=r"\s+", dtype=str)
    id_col = "#IID" if "#IID" in df.columns else "IID"
    return df[id_col].tolist()


def _read_pvar(pvar_path: Path) -> pd.DataFrame:
    with open(pvar_path) as f:
        lines = [ln for ln in f if not ln.startswith("##")]

    df = pd.read_csv(StringIO("".join(lines)), sep=r"\s+", dtype=str)
    df = df.rename(columns={"#CHROM": "CHROM"})
    return df[["CHROM", "POS", "ID", "REF", "ALT"]]


def read_pgen(pfile_prefix: str | Path) -> GenotypeData:
    prefix = Path(pfile_prefix)
    pgen_path = prefix.with_suffix(".pgen")
    pvar_path = prefix.with_suffix(".pvar")
    psam_path = prefix.with_suffix(".psam")

    for p in (pgen_path, pvar_path, psam_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing expected file: {p}")

    sample_ids = _read_psam(psam_path)
    variant_df = _read_pvar(pvar_path)
    n_samples = len(sample_ids)
    n_variants = len(variant_df)

    with pg.PgenReader(str(pgen_path).encode("utf-8")) as reader:
        geno = np.empty((n_variants, n_samples), dtype=np.int32)
        buf = np.empty(n_samples, dtype=np.int32)
        for variant_idx in range(n_variants):
            reader.read(variant_idx, buf)
            geno[variant_idx] = buf

    genotypes = geno.T.astype(np.int8)

    return GenotypeData(
        genotypes, variant_df.reset_index(drop=True), sample_ids=sample_ids
    )
