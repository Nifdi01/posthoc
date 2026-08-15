from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from posthoc.io.writer import GLM_COLUMNS, PAL_COLUMNS, write_glm, write_pal


@pytest.fixture
def variant_ids():
    return pd.DataFrame(
        {
            "CHROM": ["1", "1", "2"],
            "POS": [100, 200, 300],
            "ID": ["rs1", "rs2", "rs3"],
            "REF": ["A", "C", "G"],
            "ALT": ["T", "G", "A"],
        }
    )


class TestWriteGlm:
    def test_basic_output_columns_and_order(self, tmp_path, variant_ids):
        out_path = tmp_path / "out.tsv"
        importances = np.array([0.1, 0.2, 0.3])
        p_values = np.array([0.01, 0.02, 0.03])
        p_corrected = np.array([0.05, 0.06, 0.07])

        write_glm(
            variant_ids,
            importances,
            p_values,
            p_corrected,
            test_name="linear",
            n_samples=100,
            out_path=out_path,
        )

        result = pd.read_csv(out_path, sep="\t")
        assert list(result.columns) == GLM_COLUMNS

    def test_values_written_correctly(self, tmp_path, variant_ids):
        out_path = tmp_path / "out.tsv"
        importances = np.array([0.1, 0.2, 0.3])
        p_values = np.array([0.01, 0.02, 0.03])
        p_corrected = np.array([0.05, 0.06, 0.07])

        write_glm(
            variant_ids,
            importances,
            p_values,
            p_corrected,
            test_name="linear",
            n_samples=100,
            out_path=out_path,
        )

        result = pd.read_csv(out_path, sep="\t")
        assert (result["IMPORTANCE"] == importances).all()
        assert (result["P_PERM"] == p_values).all()
        assert (result["P_CORRECTED"] == p_corrected).all()
        assert (result["TEST"] == "linear").all()
        assert (result["N"] == 100).all()
        # A1 mirrors ALT
        assert (result["A1"] == result["ALT"]).all()

    def test_chrom_renamed_to_hash_chrom(self, tmp_path, variant_ids):
        out_path = tmp_path / "out.tsv"
        write_glm(
            variant_ids,
            np.array([0.1, 0.2, 0.3]),
            np.array([0.01, 0.02, 0.03]),
            np.array([0.05, 0.06, 0.07]),
            test_name="linear",
            n_samples=100,
            out_path=out_path,
        )
        header = out_path.read_text().splitlines()[0]
        assert "#CHROM" in header.split("\t")
        assert "CHROM" not in [c for c in header.split("\t") if c != "#CHROM"]

    def test_mismatched_length_raises(self, variant_ids, tmp_path):
        with pytest.raises(ValueError, match="importances length must match"):
            write_glm(
                variant_ids,
                np.array([0.1, 0.2]),  # length 2, variant_ids has 3 rows
                np.array([0.01, 0.02]),
                np.array([0.05, 0.06]),
                test_name="linear",
                n_samples=100,
                out_path=tmp_path / "out.tsv",
            )

    def test_empty_variant_ids(self, tmp_path):
        empty = pd.DataFrame(columns=["CHROM", "POS", "ID", "REF", "ALT"])
        out_path = tmp_path / "out.tsv"
        write_glm(
            empty,
            np.array([]),
            np.array([]),
            np.array([]),
            test_name="linear",
            n_samples=0,
            out_path=out_path,
        )
        result = pd.read_csv(out_path, sep="\t")
        assert len(result) == 0
        assert list(result.columns) == GLM_COLUMNS

    def test_does_not_mutate_input_dataframe(self, tmp_path, variant_ids):
        original = variant_ids.copy()
        write_glm(
            variant_ids,
            np.array([0.1, 0.2, 0.3]),
            np.array([0.01, 0.02, 0.03]),
            np.array([0.05, 0.06, 0.07]),
            test_name="linear",
            n_samples=100,
            out_path=tmp_path / "out.tsv",
        )
        pd.testing.assert_frame_equal(variant_ids, original)

    def test_out_path_accepts_str_and_path(self, tmp_path, variant_ids):
        str_path = str(tmp_path / "out_str.tsv")
        write_glm(
            variant_ids,
            np.array([0.1, 0.2, 0.3]),
            np.array([0.01, 0.02, 0.03]),
            np.array([0.05, 0.06, 0.07]),
            test_name="linear",
            n_samples=100,
            out_path=str_path,
        )
        assert pd.read_csv(str_path, sep="\t").shape[0] == 3

    def test_nan_p_values_preserved(self, tmp_path, variant_ids):
        out_path = tmp_path / "out.tsv"
        p_values = np.array([0.01, np.nan, 0.03])
        write_glm(
            variant_ids,
            np.array([0.1, 0.2, 0.3]),
            p_values,
            np.array([0.05, 0.06, 0.07]),
            test_name="linear",
            n_samples=100,
            out_path=out_path,
        )
        result = pd.read_csv(out_path, sep="\t")
        assert np.isnan(result["P_PERM"].iloc[1])

    def test_zero_length_but_matching_arrays_not_error(self, tmp_path):
        # Explicitly cover len(importances) == len(variant_ids) == 0 boundary
        empty = pd.DataFrame(columns=["CHROM", "POS", "ID", "REF", "ALT"])
        try:
            write_glm(
                empty,
                np.array([]),
                np.array([]),
                np.array([]),
                test_name="t",
                n_samples=0,
                out_path=tmp_path / "o.tsv",
            )
        except ValueError:
            pytest.fail("Should not raise for matching zero-length inputs")


class TestWritePal:
    def test_basic_output_columns(self, tmp_path, variant_ids):
        out_path = tmp_path / "pal.tsv"
        mu = np.array([1.0, 2.0, 3.0])
        amas = np.array([0.5, 0.6, 0.7])
        write_pal(
            variant_ids,
            mu,
            amas,
            pal_common_idx=np.array([0, 1]),
            pal_amas_idx=np.array([1]),
            p_values=np.array([0.02]),
            n_models=10,
            out_path=out_path,
        )
        result = pd.read_csv(out_path, sep="\t")
        assert list(result.columns) == PAL_COLUMNS

    def test_boolean_flags_correct(self, tmp_path, variant_ids):
        out_path = tmp_path / "pal.tsv"
        mu = np.array([1.0, 2.0, 3.0])
        amas = np.array([0.5, 0.6, 0.7])
        write_pal(
            variant_ids,
            mu,
            amas,
            pal_common_idx=np.array([0, 2]),
            pal_amas_idx=np.array([1]),
            p_values=np.array([0.02]),
            n_models=10,
            out_path=out_path,
        )
        result = pd.read_csv(out_path, sep="\t")
        assert result["IN_PAL_COMMON"].tolist() == [True, False, True]
        assert result["IN_PAL_AMAS"].tolist() == [False, True, False]

    def test_p_value_only_set_for_amas_idx(self, tmp_path, variant_ids):
        out_path = tmp_path / "pal.tsv"
        mu = np.array([1.0, 2.0, 3.0])
        amas = np.array([0.5, 0.6, 0.7])
        write_pal(
            variant_ids,
            mu,
            amas,
            pal_common_idx=np.array([0]),
            pal_amas_idx=np.array([0, 2]),
            p_values=np.array([0.01, 0.03]),
            n_models=10,
            out_path=out_path,
        )
        result = pd.read_csv(out_path, sep="\t")
        assert result["P_VALUE"].iloc[0] == pytest.approx(0.01)
        assert np.isnan(result["P_VALUE"].iloc[1])
        assert result["P_VALUE"].iloc[2] == pytest.approx(0.03)

    def test_empty_idx_arrays(self, tmp_path, variant_ids):
        out_path = tmp_path / "pal.tsv"
        mu = np.array([1.0, 2.0, 3.0])
        amas = np.array([0.5, 0.6, 0.7])
        write_pal(
            variant_ids,
            mu,
            amas,
            pal_common_idx=np.array([], dtype=int),
            pal_amas_idx=np.array([], dtype=int),
            p_values=np.array([]),
            n_models=10,
            out_path=out_path,
        )
        result = pd.read_csv(out_path, sep="\t")
        assert not result["IN_PAL_COMMON"].any()
        assert not result["IN_PAL_AMAS"].any()
        assert result["P_VALUE"].isna().all()

    def test_mismatched_mu_length_raises(self, variant_ids, tmp_path):
        with pytest.raises(ValueError, match="mu length must match"):
            write_pal(
                variant_ids,
                np.array([1.0, 2.0]),  # too short
                np.array([0.5, 0.6, 0.7]),
                pal_common_idx=np.array([0]),
                pal_amas_idx=np.array([0]),
                p_values=np.array([0.01]),
                n_models=10,
                out_path=tmp_path / "pal.tsv",
            )

    def test_duplicate_indices_in_common_idx(self, tmp_path, variant_ids):
        # Duplicate index shouldn't error; final boolean stays True
        out_path = tmp_path / "pal.tsv"
        write_pal(
            variant_ids,
            np.array([1.0, 2.0, 3.0]),
            np.array([0.5, 0.6, 0.7]),
            pal_common_idx=np.array([0, 0, 1], dtype=int),
            pal_amas_idx=np.array([], dtype=int),
            p_values=np.array([]),
            n_models=10,
            out_path=out_path,
        )
        result = pd.read_csv(out_path, sep="\t")
        assert result["IN_PAL_COMMON"].tolist() == [True, True, False]

    def test_out_of_bounds_idx_raises_index_error(self, tmp_path, variant_ids):
        with pytest.raises(IndexError):
            write_pal(
                variant_ids,
                np.array([1.0, 2.0, 3.0]),
                np.array([0.5, 0.6, 0.7]),
                pal_common_idx=np.array([10]),  # out of bounds
                pal_amas_idx=np.array([]),
                p_values=np.array([]),
                n_models=10,
                out_path=tmp_path / "pal.tsv",
            )

    def test_n_models_broadcast(self, tmp_path, variant_ids):
        out_path = tmp_path / "pal.tsv"
        write_pal(
            variant_ids,
            np.array([1.0, 2.0, 3.0]),
            np.array([0.5, 0.6, 0.7]),
            pal_common_idx=np.array([], dtype=int),
            pal_amas_idx=np.array([], dtype=int),
            p_values=np.array([]),
            n_models=42,
            out_path=out_path,
        )
        result = pd.read_csv(out_path, sep="\t")
        assert (result["N_MODELS"] == 42).all()
