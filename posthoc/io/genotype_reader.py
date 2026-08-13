from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import pgenlib as pg


@dataclass
class GenotypeData:
    """
    Container for a genotype matrix and associated variant/sample metadata.

    Attributes
    ----------
    genotypes : np.ndarray
        Genotype matrix of shape (n_samples, n_variants).
    variant_ids : pd.DataFrame
        DataFrame of variant metadata (e.g. ``CHROM``, ``POS``, ``ID``,
        ``REF``, ``ALT``), one row per variant.
    sample_ids : list of str
        Sample identifiers corresponding to the rows of `genotypes`.
    """

    genotypes: np.ndarray
    variant_ids: pd.DataFrame
    sample_ids: list[str]

    @property
    def n_samples(self) -> int:
        """
        Number of samples in the genotype matrix.

        Returns
        -------
        int
            Number of rows in `genotypes`.
        """
        return self.genotypes.shape[0]

    @property
    def n_variants(self) -> int:
        """
        Number of variants in the genotype matrix.

        Returns
        -------
        int
            Number of columns in `genotypes`.
        """
        return self.genotypes.shape[1]


def _read_psam(psam_path: Path) -> list[str]:
    """
    Read sample identifiers from a PLINK2 .psam file.

    Parameters
    ----------
    psam_path : Path
        Path to the .psam file.

    Returns
    -------
    list of str
        Sample identifiers in file order, taken from the ``#IID`` or
        ``IID`` column.
    """
    df = pd.read_csv(psam_path, sep=r"\s+", dtype=str)
    id_col = "#IID" if "#IID" in df.columns else "IID"
    return df[id_col].tolist()


def _read_pvar(pvar_path: Path) -> pd.DataFrame:
    """
    Read variant metadata from a PLINK2 .pvar file.

    Skips header lines beginning with ``##`` and retains only the core
    variant identification columns.

    Parameters
    ----------
    pvar_path : Path
        Path to the .pvar file.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``CHROM``, ``POS``, ``ID``, ``REF``, and ``ALT``.
    """
    with open(pvar_path) as f:
        lines = [ln for ln in f if not ln.startswith("##")]

    df = pd.read_csv(StringIO("".join(lines)), sep=r"\s+", dtype=str)
    df = df.rename(columns={"#CHROM": "CHROM"})
    return df[["CHROM", "POS", "ID", "REF", "ALT"]]


def read_pgen(pfile_prefix: str | Path) -> GenotypeData:
    """
    Read a PLINK2 .pgen/.pvar/.psam fileset into a GenotypeData container.

    Parameters
    ----------
    pfile_prefix : str or Path
        Path prefix shared by the ``.pgen``, ``.pvar``, and ``.psam`` files
        (i.e. the path without the file extension).

    Returns
    -------
    GenotypeData
        Container holding the genotype matrix (samples x variants),
        variant metadata, and sample identifiers.

    Raises
    ------
    FileNotFoundError
        If any of the expected ``.pgen``, ``.pvar``, or ``.psam`` files
        does not exist at the given prefix.
    """
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
