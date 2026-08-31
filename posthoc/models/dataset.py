from __future__ import annotations

import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)


from torch.utils.data import Dataset


class GenotypeDataset(Dataset):
    MISSING_CODE = -9

    def __init__(
        self,
        genotypes: np.ndarray,
        phenotype: np.ndarray,
        covariates: np.ndarray | None = None,
    ):
        geno = genotypes.astype(np.float32).copy()
        missing_mask = geno == self.MISSING_CODE

        if missing_mask.any():
            variant_means = np.where(missing_mask, np.nan, geno)
            variant_means = np.nanmean(variant_means, axis=0)
            fill = np.take(variant_means, np.where(missing_mask)[1])
            geno[missing_mask] = fill
            logger.info(
                "Mean-imputed %d missing genotype calls (%.3f%% of matrix)",
                missing_mask.sum(),
                100 * missing_mask.mean(),
            )

        self.genotypes = torch.from_numpy(geno)
        self.phenotype = torch.from_numpy(phenotype.astype(np.float32))
        self.covariates = (
            torch.from_numpy(covariates.astype(np.float32))
            if covariates is not None
            else None
        )

    def __len__(self) -> int:
        return self.genotypes.shape[0]

    def __getitem__(self, idx: int):
        x = self.genotypes[idx]
        if self.covariates is not None:
            x = torch.cat([x, self.covariates[idx]], dim=0)
        y = self.phenotype[idx]
        return x, y
