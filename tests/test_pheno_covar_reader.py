from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from posthoc.io.pheno_covar_reader import (
    _read_plink_table,
    load_pheno,
    load_covar,
    align_samples,
)


def _write_table(tmp_path, name, header, rows):
    path = tmp_path / name
    lines = ["\t".join(header)] + ["\t".join(r) for r in rows]
    path.write_text("\n".join(lines) + "\n")
    return path


class TestReadPlinkTable:
    def test_standard_iid_column(self, tmp_path):
        path = _write_table(
            tmp_path, "t.txt", ["FID", "IID", "PHENO"], [["F1", "S1", "1.5"]]
        )
        df = _read_plink_table(path)
        assert "IID" in df.columns
        assert df["IID"].tolist() == ["S1"]

    def test_hash_iid_renamed(self, tmp_path):
        path = _write_table(tmp_path, "t.txt", ["#IID", "PHENO"], [["S1", "1.5"]])
        df = _read_plink_table(path)
        assert "IID" in df.columns
        assert "#IID" not in df.columns

    def test_fallback_to_second_column(self, tmp_path):
        # Neither IID nor #IID present -> falls back to df.columns[1]
        path = _write_table(
            tmp_path, "t.txt", ["FAM", "SAMPLE", "PHENO"], [["F1", "S1", "1.5"]]
        )
        df = _read_plink_table(path)
        assert "IID" in df.columns
        assert df["IID"].tolist() == ["S1"]

    def test_missing_values_converted_to_nan(self, tmp_path):
        path = _write_table(
            tmp_path,
            "t.txt",
            ["FID", "IID", "PHENO"],
            [["F1", "S1", "-9"], ["F1", "S2", "NA"], ["F1", "S3", "1.0"]],
        )
        df = _read_plink_table(path)
        assert pd.isna(df["PHENO"].iloc[0])
        assert pd.isna(df["PHENO"].iloc[1])
        assert df["PHENO"].iloc[2] == "1.0"

    def test_all_values_read_as_strings(self, tmp_path):
        path = _write_table(
            tmp_path, "t.txt", ["FID", "IID", "PHENO"], [["F1", "S1", "1.5"]]
        )
        df = _read_plink_table(path)
        assert df["PHENO"].iloc[0] == "1.5"
        assert isinstance(df["PHENO"].iloc[0], str)


class TestLoadPheno:
    def test_explicit_pheno_name(self, tmp_path):
        path = _write_table(
            tmp_path,
            "pheno.txt",
            ["FID", "IID", "HEIGHT", "WEIGHT"],
            [["F1", "S1", "170.5", "70.0"], ["F1", "S2", "180.0", "80.0"]],
        )
        series = load_pheno(path, pheno_name="WEIGHT")
        assert series.loc["S1"] == 70.0
        assert series.loc["S2"] == 80.0

    def test_default_first_value_column(self, tmp_path):
        path = _write_table(
            tmp_path,
            "pheno.txt",
            ["FID", "IID", "HEIGHT", "WEIGHT"],
            [["F1", "S1", "170.5", "70.0"]],
        )
        series = load_pheno(path)
        assert series.loc["S1"] == 170.5

    def test_cast_to_float(self, tmp_path):
        path = _write_table(
            tmp_path, "pheno.txt", ["FID", "IID", "HEIGHT"], [["F1", "S1", "170"]]
        )
        series = load_pheno(path)
        assert series.dtype == float

    def test_missing_phenotype_becomes_nan(self, tmp_path):
        path = _write_table(
            tmp_path, "pheno.txt", ["FID", "IID", "HEIGHT"], [["F1", "S1", "-9"]]
        )
        series = load_pheno(path)
        assert np.isnan(series.loc["S1"])

    def test_invalid_pheno_name_raises_keyerror(self, tmp_path):
        path = _write_table(
            tmp_path, "pheno.txt", ["FID", "IID", "HEIGHT"], [["F1", "S1", "170"]]
        )
        with pytest.raises(KeyError):
            load_pheno(path, pheno_name="NONEXISTENT")

    def test_indexed_by_iid(self, tmp_path):
        path = _write_table(
            tmp_path,
            "pheno.txt",
            ["FID", "IID", "HEIGHT"],
            [["F1", "S1", "170"], ["F2", "S2", "180"]],
        )
        series = load_pheno(path)
        assert set(series.index) == {"S1", "S2"}


class TestLoadCovar:
    def test_excludes_fid_iid(self, tmp_path):
        path = _write_table(
            tmp_path,
            "covar.txt",
            ["FID", "IID", "AGE", "SEX"],
            [["F1", "S1", "30", "1"]],
        )
        df = load_covar(path)
        assert list(df.columns) == ["AGE", "SEX"]

    def test_cast_to_float(self, tmp_path):
        path = _write_table(
            tmp_path,
            "covar.txt",
            ["FID", "IID", "AGE"],
            [["F1", "S1", "30"]],
        )
        df = load_covar(path)
        assert df["AGE"].dtype == float

    def test_indexed_by_iid(self, tmp_path):
        path = _write_table(
            tmp_path,
            "covar.txt",
            ["FID", "IID", "AGE"],
            [["F1", "S1", "30"], ["F2", "S2", "40"]],
        )
        df = load_covar(path)
        assert list(df.index) == ["S1", "S2"]

    def test_single_covariate_column(self, tmp_path):
        path = _write_table(
            tmp_path, "covar.txt", ["FID", "IID", "PC1"], [["F1", "S1", "0.1"]]
        )
        df = load_covar(path)
        assert df.shape == (1, 1)

    def test_missing_covar_value_is_nan(self, tmp_path):
        path = _write_table(
            tmp_path, "covar.txt", ["FID", "IID", "PC1"], [["F1", "S1", "NA"]]
        )
        df = load_covar(path)
        assert np.isnan(df["PC1"].iloc[0])


class TestAlignSamples:
    def test_basic_alignment_pheno_only(self):
        sample_ids = ["S1", "S2", "S3"]
        pheno = pd.Series({"S1": 1.0, "S2": 2.0, "S3": 3.0})
        keep, aligned_pheno, aligned_covar = align_samples(sample_ids, pheno)
        assert keep.tolist() == [True, True, True]
        assert aligned_pheno.tolist() == [1.0, 2.0, 3.0]
        assert aligned_covar is None

    def test_pheno_missing_sample_excluded(self):
        sample_ids = ["S1", "S2", "S3"]
        pheno = pd.Series({"S1": 1.0, "S2": np.nan, "S3": 3.0})
        keep, aligned_pheno, _ = align_samples(sample_ids, pheno)
        assert keep.tolist() == [True, False, True]
        assert aligned_pheno.tolist() == [1.0, 3.0]

    def test_sample_not_in_pheno_excluded(self):
        sample_ids = ["S1", "S2", "S3"]
        pheno = pd.Series({"S1": 1.0, "S3": 3.0})  # S2 absent entirely
        keep, aligned_pheno, _ = align_samples(sample_ids, pheno)
        assert keep.tolist() == [True, False, True]

    def test_with_covar_intersection(self):
        sample_ids = ["S1", "S2", "S3"]
        pheno = pd.Series({"S1": 1.0, "S2": 2.0, "S3": 3.0})
        covar = pd.DataFrame({"AGE": [30.0, np.nan, 50.0]}, index=["S1", "S2", "S3"])
        keep, aligned_pheno, aligned_covar = align_samples(sample_ids, pheno, covar)
        assert keep.tolist() == [True, False, True]
        assert aligned_pheno.tolist() == [1.0, 3.0]
        assert aligned_covar.tolist() == [[30.0], [50.0]]

    def test_order_preserved_as_sample_ids(self):
        sample_ids = ["S3", "S1", "S2"]
        pheno = pd.Series({"S1": 1.0, "S2": 2.0, "S3": 3.0})
        _, aligned_pheno, _ = align_samples(sample_ids, pheno)
        assert aligned_pheno.tolist() == [3.0, 1.0, 2.0]

    def test_no_overlap_raises(self):
        sample_ids = ["S1", "S2"]
        pheno = pd.Series({"S3": 1.0, "S4": 2.0})
        with pytest.raises(ValueError, match="No overlapping"):
            align_samples(sample_ids, pheno)

    def test_all_pheno_missing_raises(self):
        sample_ids = ["S1", "S2"]
        pheno = pd.Series({"S1": np.nan, "S2": np.nan})
        with pytest.raises(ValueError, match="No overlapping"):
            align_samples(sample_ids, pheno)

    def test_empty_sample_ids_raises(self):
        pheno = pd.Series({"S1": 1.0})
        with pytest.raises(ValueError, match="No overlapping"):
            align_samples([], pheno)

    def test_duplicate_sample_ids(self):
        # Duplicate id in sample_ids - both positions should be kept if present in common
        sample_ids = ["S1", "S1", "S2"]
        pheno = pd.Series({"S1": 1.0, "S2": 2.0})
        keep, aligned_pheno, _ = align_samples(sample_ids, pheno)
        assert keep.tolist() == [True, True, True]
        assert aligned_pheno.tolist() == [1.0, 1.0, 2.0]

    def test_covar_multi_column(self):
        sample_ids = ["S1", "S2"]
        pheno = pd.Series({"S1": 1.0, "S2": 2.0})
        covar = pd.DataFrame(
            {"AGE": [30.0, 40.0], "PC1": [0.1, 0.2]}, index=["S1", "S2"]
        )
        _, _, aligned_covar = align_samples(sample_ids, pheno, covar)
        assert aligned_covar.shape == (2, 2)

