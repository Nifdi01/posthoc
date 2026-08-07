from __future__ import annotations

import logging
import numpy as np


from posthoc.io.genotype_reader import GenotypeData
from dataclasses import replace


logger = logging.getLogger(__name__)


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
    )


def subset_samples(
    data: GenotypeData, keep_mask: np.ndarray, ordered_ids: list[str]
) -> GenotypeData:
    return replace(data, genotypes=data.genotypes[keep_mask, :], sample_ids=ordered_ids)
