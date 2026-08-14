from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from posthoc.io.pheno_covar_reader import (
    _read_plink_table,
    align_samples,
    load_covar,
    load_pheno,
)

# ---------- fixtures ----------


@pytest.fixture
def pheno_file(tmp_path):
    path = tmp_path / "toy.pheno"
    path.write_text(
        "FID\tIID\tPHENO1\n"
        "F1\tS1\t1\n"
        "F2\tS2\t0\n"
        "F3\tS3\t-9\n"  # PLINK missing code
        "F4\tS4\t1\n"
    )
    return path


@pytest.fixture
def pheno_file_named(tmp_path):
    path = tmp_path / "toy_named.pheno"
    path.write_text(
        "FID\tIID\tHEIGHT\tBMI\nF1\tS1\t170\t22.1\nF2\tS2\t165\tNA\nF3\tS3\t180\t24.5\n"
    )
    return path


@pytest.fixture
def covar_file(tmp_path):
    path = tmp_path / "toy.covar"
    path.write_text(
        "FID\tIID\tAGE\tSEX\tPC1\n"
        "F1\tS1\t45\t1\t0.01\n"
        "F2\tS2\t50\t0\tNA\n"  # missing PC1 for S2
        "F3\tS3\t60\t1\t-0.02\n"
        "F4\tS4\t55\t0\t0.03\n"
    )
    return path


# ---------- load_pheno ----------


def test_load_pheno_default_column_uses_first_value_col(pheno_file):
    pheno = load_pheno(pheno_file)
    assert list(pheno.index) == ["S1", "S2", "S3", "S4"]
    assert pheno.loc["S1"] == 1.0
    assert pheno.loc["S2"] == 0.0


def test_load_pheno_missing_code_becomes_nan(pheno_file):
    pheno = load_pheno(pheno_file)
    assert np.isnan(pheno.loc["S3"])  # was "-9" in file


def test_load_pheno_named_column(pheno_file_named):
    bmi = load_pheno(pheno_file_named, pheno_name="BMI")
    assert bmi.loc["S1"] == 22.1
    assert np.isnan(bmi.loc["S2"])  # "NA" in file
    height = load_pheno(pheno_file_named, pheno_name="HEIGHT")
    assert height.loc["S3"] == 180.0


def test_load_pheno_dtype_is_float(pheno_file):
    pheno = load_pheno(pheno_file)
    assert pheno.dtype == np.float64


# ---------- load_covar ----------


def test_load_covar_returns_all_value_columns(covar_file):
    covar = load_covar(covar_file)
    assert list(covar.columns) == ["AGE", "SEX", "PC1"]
    assert list(covar.index) == ["S1", "S2", "S3", "S4"]


def test_load_covar_missing_value_becomes_nan(covar_file):
    covar = load_covar(covar_file)
    assert np.isnan(covar.loc["S2", "PC1"])
    assert covar.loc["S1", "PC1"] == 0.01


# ---------- _read_plink_table ----------


def test_read_plink_table_detects_iid_column(pheno_file):
    df = _read_plink_table(pheno_file)
    assert "IID" in df.columns
    assert df["IID"].tolist() == ["S1", "S2", "S3", "S4"]


def test_read_plink_table_falls_back_to_second_column(tmp_path):
    # No explicit "IID" header -> should fall back to column index 1
    path = tmp_path / "no_iid_header.txt"
    path.write_text("FAM\tSAMPLE\tVAL\nF1\tS1\t10\nF2\tS2\t20\n")
    df = _read_plink_table(path)
    assert "IID" in df.columns
    assert df["IID"].tolist() == ["S1", "S2"]


# ---------- align_samples ----------


def test_align_samples_basic_intersection(pheno_file, covar_file):
    pheno = load_pheno(pheno_file)
    covar = load_covar(covar_file)
    sample_ids = ["S1", "S2", "S3", "S4"]

    keep_mask, aligned_pheno, aligned_covar = align_samples(sample_ids, pheno, covar)

    # S2 is dropped: has NaN PC1 in covar.
    # S3 is dropped: has NaN pheno (-9 missing code).
    assert keep_mask.tolist() == [True, False, False, True]
    assert aligned_pheno.tolist() == [1.0, 1.0]  # S1, S4
    assert aligned_covar.shape == (2, 3)


def test_align_samples_preserves_sample_ids_order_not_file_order(pheno_file):
    """
    sample_ids order should win, even if it differs from the order samples
    appear in the pheno file. This matters because align_samples is used to
    subset the genotype tensor, which is ordered by sample_ids.
    """
    pheno = load_pheno(pheno_file)  # file order: S1, S2, S3, S4
    sample_ids = ["S4", "S1", "S2", "S3"]  # deliberately reordered

    keep_mask, aligned_pheno, _ = align_samples(sample_ids, pheno)

    # S3 excluded (missing pheno); remaining order should follow sample_ids,
    # i.e. S4, S1, S2 -- not the pheno file's S1, S2, S4 order.
    assert keep_mask.tolist() == [True, True, True, False]
    expected = pheno.loc[["S4", "S1", "S2"]].to_numpy()
    np.testing.assert_array_equal(aligned_pheno, expected)


def test_align_samples_excludes_samples_missing_from_genotype_ids(pheno_file):
    pheno = load_pheno(pheno_file)
    sample_ids = ["S1", "S2", "S_UNKNOWN"]  # S_UNKNOWN not in pheno file at all

    keep_mask, aligned_pheno, _ = align_samples(sample_ids, pheno)

    assert keep_mask.tolist() == [True, True, False]
    assert len(aligned_pheno) == 2


def test_align_samples_without_covar(pheno_file):
    pheno = load_pheno(pheno_file)
    sample_ids = ["S1", "S2", "S3", "S4"]

    keep_mask, _aligned_pheno, aligned_covar = align_samples(sample_ids, pheno)

    assert aligned_covar is None
    # Only S3 excluded (missing pheno); no covar to exclude on.
    assert keep_mask.tolist() == [True, True, False, True]


def test_align_samples_raises_on_no_overlap():
    pheno = pd.Series([1.0, 0.0], index=["A", "B"])
    sample_ids = ["X", "Y", "Z"]

    with pytest.raises(ValueError, match="No overlapping"):
        align_samples(sample_ids, pheno)


def test_align_samples_all_covar_rows_present_but_some_nan(covar_file, pheno_file):
    """Sanity check: a sample present in covar but with any NaN covariate
    value is dropped entirely (dropna on full row), not just NaN-imputed."""
    pheno = load_pheno(pheno_file)
    covar = load_covar(covar_file)
    sample_ids = ["S1", "S2", "S3", "S4"]

    keep_mask, _, aligned_covar = align_samples(sample_ids, pheno, covar)

    # S2 has NaN in PC1 -> entire sample dropped, not partially included
    assert "S2" not in np.array(sample_ids)[keep_mask]
    assert not np.isnan(aligned_covar).any()
