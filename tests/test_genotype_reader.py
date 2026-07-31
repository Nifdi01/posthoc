import numpy as np
import pytest
from posthoc.io.genotype_reader import read_pgen


TOY_PREFIX = "datasets/data/processed/chr22_subset"


def test_read_pgen_shapes():
    data = read_pgen(TOY_PREFIX)
    assert data.n_samples == len(data.sample_ids)
    assert data.n_variants == len(data.variant_ids)
    assert data.genotypes.shape == (data.n_samples, data.n_variants)


def test_genotype_values_in_range():
    data = read_pgen(TOY_PREFIX)
    unique_vals = np.unique(data.genotypes)
    # allowed: 0, 1, 2 (dosage) and -9 (missing)
    assert set(unique_vals.tolist()).issubset({-9, 0, 1, 2})


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        read_pgen("examples/toy_data/does_not_exist")


def test_missing_genotype_sentinel(toy_missing_pfile):
    """Regression test: confirms pgenlib's missing-call sentinel is -9
    and survives the int32 -> int8 cast in read_pgen unchanged."""
    data = read_pgen(toy_missing_pfile)

    # S2 (index 1) at rs2 (index 1) was deliberately set to ./. in the source VCF
    assert data.genotypes[1, 1] == -9
    assert set(np.unique(data.genotypes)) == {-9, 0, 1, 2}
