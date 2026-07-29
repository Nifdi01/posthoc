from pathlib import Path
import numpy as np
import pandas as pd
import pgenlib

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from metadata import GenotypeMetadata


class GenotypeDataset:
    def __init__(self, pfile_prefix: str | Path):
        self.prefix = Path(pfile_prefix)
        self.pgen_path = self.prefix.with_suffix(".pgen")
        self.pvar_path = self.prefix.with_suffix(".pvar")
        self.psam_path = self.prefix.with_suffix(".psam")

        for p in (self.pgen_path, self.pvar_path, self.psam_path):
            if not p.exists():
                raise FileNotFoundError(f"Expected p file component not found: {p}")

        self._X: np.ndarray | None = None
        self.metadata: GenotypeMetadata | None = None

    def _load_psam(self) -> pd.DataFrame:
        df = pd.read_csv(self.psam_path, sep=r"\s+", dtype=str)
        df.columns = [c.lstrip("#") for c in df.columns]
        if "IID" not in df.columns:
            raise ValueError(
                f"No IID column found in {self.psam_path}: {df.columns.tolist()}"
            )

        population_column = next(
            (
                c
                for c in df.columns
                if c.lower() in ("superpop", "superpopulation", "population", "pop")
            ),
            None,
        )

        if population_column is None:
            raise ValueError(
                f"No population column found in {self.psam_path}. "
                f"Available columns: {df.columns.tolist()}"
            )
        return df.rename(columns={population_column: "SuperPop"})

    def _load_pvar(self) -> pd.DataFrame:
        with open(self.pvar_path) as f:
            skip = 0
            for line in f:
                if line.startswith("##"):
                    skip += 1
                    continue
                break
        df = pd.read_csv(self.pvar_path, sep="\t", skiprows=skip)
        df.columns = [c.lstrip("#") for c in df.columns]
        expected = {"CHROM", "POS", "ID", "REF", "ALT"}
        missing = expected - set(df.columns)
        if missing:
            raise ValueError(
                f"pvar missing expected columns {missing}. Got: {df.columns.tolist()}"
            )
        return df

    def load(self) -> "GenotypeDataset":
        psam = self._load_psam()
        pvar = self._load_pvar()

        n_samples = len(psam)
        n_variants = len(pvar)

        reader = pgenlib.PgenReader(bytes(str(self.pgen_path), "utf-8"))
        if reader.get_raw_sample_ct() != n_samples:
            raise ValueError(
                f"Sample count mismatch: psam has {n_samples}"
                f"pgen has  {reader.get_raw_sample_ct()}"
            )
        if reader.get_variant_ct() != n_variants:
            raise ValueError(
                f"Variant count mismatch: pvar has {n_variants}"
                f"pgen has {reader.get_variant_ct()}"
            )

        geno = np.empty((n_variants, n_samples), dtype=np.int8)
        reader.read_range(0, n_variants, geno)
        reader.close()

        X = geno.T.astype(np.float32)

        self._X = X

        self.metadata = GenotypeMetadata(
            sample_ids=psam["IID"].to_numpy(),
            populations=psam["SuperPop"].to_numpy(),
            variant_ids=pvar["ID"].to_numpy(),
            chrom=pvar["CHROM"].to_numpy(),
            pos=pvar["POS"].to_numpy(),
            ref=pvar["REF"].to_numpy(),
            alt=pvar["ALT"].to_numpy(),
            maf=np.minimum(X.mean(axis=0) / 2.0, 1 - X.mean(axis=0) / 2.0),
        )
        return self

    def compute_pca(
        self, n_components: int = 10, standardize: bool = True
    ) -> "GenotypeDataset":
        if self._X is None:
            raise RuntimeError("Call load() before compute_pca().")

        X = self._X
        if standardize:
            mean = X.mean(axis=0)
            std = X.std(axis=0)
            std[std == 0] = 1.0
            X_std = (X - mean) / std
        else:
            X_std = X

        pca = PCA(n_components=n_components, random_state=0)
        components = pca.fit_transform(X_std)
        self.metadata.pca_components = components
        self.metadata.extra["pca_explained_variance_ratio"] = (
            pca.explained_variance_ratio_
        )
        return self

    def split(
        self,
        test_size: float = 0.2,
        stratify_by_population: bool = True,
        random_state: int = 0,
    ) -> tuple[np.ndarray, np.ndarray, GenotypeMetadata, GenotypeMetadata]:

        if self._X is None:
            raise RuntimeError("Call load() before split()")

        n = self._X.shape[0]
        idx = np.arange(n)
        stratify = self.metadata.populations if stratify_by_population else None
        idx_train, idx_test = train_test_split(
            idx, test_size=test_size, stratify=stratify, random_state=random_state
        )
        mask_train = np.zeros(n, dtype=bool)
        mask_train[idx_train] = True
        mask_test = ~mask_train

        X_train = self._X[mask_train]
        X_test = self._X[mask_test]
        meta_train = self.metadata.subset_samples(mask_train)
        meta_test = self.metadata.subset_samples(mask_test)

        return X_train, X_test, meta_train, meta_test


def build_dataset(
    pfile_prefix: str | Path,
    n_pcs: int = 10,
    test_size: float = 0.2,
    random_state: int = 0,
):
    ds = GenotypeDataset(pfile_prefix).load().compute_pca(n_pcs)
    X_train, X_test, meta_train, meta_test = ds.split(
        test_size=test_size, random_state=random_state
    )
    return X_train, X_test, {"train": meta_train, "test": meta_test}
