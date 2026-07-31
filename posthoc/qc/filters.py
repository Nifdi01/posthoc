from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from dataclasses import MISSING, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from posthoc.io.genotype_reader import GenotypeData


logger = logging.getLogger(__name__)

MISSING_CODE = -9  # Default missing code for PgenReader


@dataclass
class QCResult:
    data: GenotypeData
    n_dropped_maf: int = 0
    n_dropped_geno: int = 0
    n_dropped_ld: int = 0

    def summary(self) -> str:
        return (
            f"QC: dropped {self.n_dropped_maf} (MAF), "
            f"{self.n_dropped_geno} (missingness), "
            f"{self.n_dropped_ld} (LD pruning); "
            f"{self.data.n_variants} variants remain."
        )


def compute_maf(genotypes: np.ndarray) -> np.ndarray:
    valid = genotypes != MISSING_CODE
    with np.errstate(invalid="ignore", divide="ignore"):
        allele_sum = np.where(valid, genotypes, 0).sum(axis=0)
        called = valid.sum(axis=0) * 2
        allele_freq = np.divide(
            allele_sum,
            called,
            out=np.zeros_like(allele_sum, dtype=float),
            where=called > 0,
        )
    return np.minimum(allele_freq, 1 - allele_freq)


def compute_missing_rate(genotypes: np.ndarray) -> np.ndarray:
    return (genotypes == MISSING_CODE).mean(axis=0)


def _subset(data: GenotypeData, keep_mask: np.ndarray) -> GenotypeData:
    return replace(
        data,
        genotypes=data.genotypes[:, keep_mask],
        variant_ids=data.variant_ids.loc[keep_mask].reset_index(drop=True),
    )


def filter_maf(data: GenotypeData, min_maf: float) -> tuple[GenotypeData, int]:
    if min_maf <= 0:
        return data, 0
    maf = compute_maf(data.genotypes)
    keep = maf >= min_maf
    n_dropped = int((~keep).sum())
    logger.info(
        "MAF filter (>= %.4f): dropping %d / %d variants",
        min_maf,
        n_dropped,
        data.n_variants,
    )
    return _subset(data, keep), n_dropped


def filter_geno(data: GenotypeData, max_missing: float) -> tuple[GenotypeData, int]:
    if max_missing >= 1:
        return data, 0

    miss = compute_missing_rate(data.genotypes)
    keep = miss <= max_missing
    n_dropped = int((~keep).sum())
    logger.info(
        "--geno filter (<= %.4f missing): dropping %d / %d variants",
        max_missing,
        n_dropped,
        data.n_variants,
    )
    return _subset(data, keep), n_dropped


def _find_plink2() -> str:
    path = shutil.which("plink2")
    if path is None:
        raise RuntimeError(
            "--indep-pairwise requires a plink2 binary on PATH, but none was found. "
            "Install plink2 (https://www.cog-genomics.org/plink/2.0/) or omit --indep-pairwise."
        )
    return path


def indep_pairwise(
    pfile_prefix: str | Path,
    window_size: int,
    step: int,
    r2_threshold: float,
    plink2_path: str | None = None,
) -> list[str]:
    plink2 = plink2_path or _find_plink2()
    with tempfile.TemporaryDirectory() as tempdir:
        out_prefix = str(Path(tempdir) / "prune")
        cmd = [
            plink2,
            "--pfile",
            str(pfile_prefix),
            "--indep-pairwise",
            str(window_size),
            str(step),
            str(r2_threshold),
            "--out",
            out_prefix,
        ]

        logger.info("Running: %s", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"plink2 --indep-pairwise failed:\n{result.stderr}")

        prune_in = Path(f"{out_prefix}.prune.in")
        if not prune_in.exists():
            raise RuntimeError(f"Expected plink2 output not found: {prune_in}")
        return prune_in.read_text().split()


def apply_ld_pruning(
    data: GenotypeData, keep_ids: list[str]
) -> tuple[GenotypeData, int]:
    keep_set = set(keep_ids)
    keep = data.variant_ids["ID"].isin(list(keep_set)).to_numpy()
    n_dropped = int((~keep).sum())
    logger.info("LD pruning: dropping %d / %d variants", n_dropped, data.n_variants)
    return _subset(data, keep), n_dropped


def run_qc(
    data: GenotypeData,
    *,
    min_maf: float = 0.0,
    max_missing: float = 1.0,
    indep_pairwise_params: tuple[int, int, float] | None = None,
    pfile_prefix: str | Path | None = None,
) -> QCResult:

    data, n_geno = filter_geno(data, max_missing)
    data, n_maf = filter_maf(data, min_maf)

    n_ld = 0

    if indep_pairwise_params is not None:
        if pfile_prefix is None:
            raise ValueError(
                "indep_pairwise_params given but no pfile_prefix to run plink2 against."
            )
        window, step, r2 = indep_pairwise_params
        keep_ids = indep_pairwise(pfile_prefix, window, step, r2)
        data, n_ld = apply_ld_pruning(data, keep_ids)

    return QCResult(data, n_dropped_maf=n_maf, n_dropped_geno=n_geno, n_dropped_ld=n_ld)
