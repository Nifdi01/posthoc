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
