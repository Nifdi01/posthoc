from dataclasses import dataclass, field
import numpy as np


@dataclass
class GenotypeMetadata:
    sample_ids: np.ndarray
    populations: np.ndarray
    variant_ids: np.ndarray
    chrom: np.ndarray
    pos: np.ndarray
    ref: np.ndarray
    alt: np.ndarray
    maf: np.ndarray | None = None
    pca_components: np.ndarray | None = None
    extra: dict = field(default_factory=dict)

    def subset_samples(self, mask: np.ndarray) -> "GenotypeMetadata":
        return GenotypeMetadata(
            sample_ids=self.sample_ids[mask],
            populations=self.populations[mask],
            variant_ids=self.variant_ids,
            chrom=self.chrom,
            pos=self.pos,
            ref=self.ref,
            alt=self.alt,
            maf=self.maf,
            pca_components=self.pca_components[mask]
            if self.pca_components is not None
            else None,
            extra=self.extra,
        )
