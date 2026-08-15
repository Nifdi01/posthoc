from __future__ import annotations

from unittest import mock

import numpy as np
import pandas as pd
import pytest

from posthoc.io.genotype_reader import GenotypeData, _read_psam, _read_pvar, read_pgen


class TestGenotypeData:
    def test_n_samples_n_variants(self):
        geno = np.zeros((5, 10), dtype=np.int8)  # 5 samples, 10 variants
        variant_ids = pd.DataFrame({"ID": [f"v{i}" for i in range(10)]})
        gd = GenotypeData(geno, variant_ids, sample_ids=[f"s{i}" for i in range(5)])
        assert gd.n_samples == 5
        assert gd.n_variants == 10

    def test_empty_genotype_matrix(self):
        geno = np.zeros((0, 0), dtype=np.int8)
        variant_ids = pd.DataFrame(columns=["ID"])
        gd = GenotypeData(geno, variant_ids, sample_ids=[])
        assert gd.n_samples == 0
        assert gd.n_variants == 0


class TestReadPsam:
    def test_hash_iid_column(self, tmp_path):
        path = tmp_path / "t.psam"
        path.write_text("#IID\tSEX\nS1\t1\nS2\t2\n")
        ids = _read_psam(path)
        assert ids == ["S1", "S2"]

    def test_plain_iid_column(self, tmp_path):
        path = tmp_path / "t.psam"
        path.write_text("FID\tIID\tSEX\nF1\tS1\t1\n")
        ids = _read_psam(path)
        assert ids == ["S1"]

    def test_preserves_order(self, tmp_path):
        path = tmp_path / "t.psam"
        path.write_text("#IID\nS3\nS1\nS2\n")
        ids = _read_psam(path)
        assert ids == ["S3", "S1", "S2"]

    def test_single_sample(self, tmp_path):
        path = tmp_path / "t.psam"
        path.write_text("#IID\nS1\n")
        assert _read_psam(path) == ["S1"]


class TestReadPvar:
    def test_skips_double_hash_headers(self, tmp_path):
        path = tmp_path / "t.pvar"
        path.write_text(
            "##fileformat=VCFv4.2\n"
            "##contig=<ID=1>\n"
            "#CHROM\tPOS\tID\tREF\tALT\n"
            "1\t100\trs1\tA\tT\n"
        )
        df = _read_pvar(path)
        assert list(df.columns) == ["CHROM", "POS", "ID", "REF", "ALT"]
        assert df.iloc[0]["ID"] == "rs1"

    def test_chrom_renamed(self, tmp_path):
        path = tmp_path / "t.pvar"
        path.write_text("#CHROM\tPOS\tID\tREF\tALT\n1\t100\trs1\tA\tT\n")
        df = _read_pvar(path)
        assert "CHROM" in df.columns
        assert "#CHROM" not in df.columns

    def test_extra_columns_dropped(self, tmp_path):
        path = tmp_path / "t.pvar"
        path.write_text(
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
            "1\t100\trs1\tA\tT\t.\tPASS\t.\n"
        )
        df = _read_pvar(path)
        assert list(df.columns) == ["CHROM", "POS", "ID", "REF", "ALT"]

    def test_multiple_variants(self, tmp_path):
        path = tmp_path / "t.pvar"
        path.write_text(
            "#CHROM\tPOS\tID\tREF\tALT\n"
            "1\t100\trs1\tA\tT\n"
            "1\t200\trs2\tC\tG\n"
            "2\t50\trs3\tG\tA\n"
        )
        df = _read_pvar(path)
        assert len(df) == 3
        assert df["CHROM"].tolist() == ["1", "1", "2"]

    def test_no_meta_headers_present(self, tmp_path):
        path = tmp_path / "t.pvar"
        path.write_text("#CHROM\tPOS\tID\tREF\tALT\n1\t100\trs1\tA\tT\n")
        df = _read_pvar(path)
        assert len(df) == 1

    def test_missing_required_column_raises_keyerror(self, tmp_path):
        path = tmp_path / "t.pvar"
        # No ALT column
        path.write_text("#CHROM\tPOS\tID\tREF\n1\t100\trs1\tA\n")
        with pytest.raises(KeyError):
            _read_pvar(path)


class TestReadPgen:
    def _make_fileset(self, tmp_path, n_samples=3, n_variants=2, prefix="data"):
        pgen_path = tmp_path / f"{prefix}.pgen"
        pvar_path = tmp_path / f"{prefix}.pvar"
        psam_path = tmp_path / f"{prefix}.psam"

        pgen_path.write_bytes(b"")  # placeholder; PgenReader mocked
        psam_path.write_text(
            "#IID\n" + "\n".join(f"S{i}" for i in range(n_samples)) + "\n"
        )
        pvar_lines = ["#CHROM\tPOS\tID\tREF\tALT"]
        for i in range(n_variants):
            pvar_lines.append(f"1\t{100 + i}\trs{i}\tA\tT")
        pvar_path.write_text("\n".join(pvar_lines) + "\n")

        return tmp_path / prefix

    def test_missing_pgen_raises_file_not_found(self, tmp_path):
        prefix = self._make_fileset(tmp_path)
        (tmp_path / "data.pgen").unlink()
        with pytest.raises(FileNotFoundError, match=r"\.pgen"):
            read_pgen(prefix)

    def test_missing_pvar_raises_file_not_found(self, tmp_path):
        prefix = self._make_fileset(tmp_path)
        (tmp_path / "data.pvar").unlink()
        with pytest.raises(FileNotFoundError, match=r"\.pvar"):
            read_pgen(prefix)

    def test_missing_psam_raises_file_not_found(self, tmp_path):
        prefix = self._make_fileset(tmp_path)
        (tmp_path / "data.psam").unlink()
        with pytest.raises(FileNotFoundError, match=r"\.psam"):
            read_pgen(prefix)

    def test_reads_genotype_matrix_shape_and_transpose(self, tmp_path):
        n_samples, n_variants = 3, 2
        prefix = self._make_fileset(
            tmp_path, n_samples=n_samples, n_variants=n_variants
        )

        # variant-major fake data: variant 0 -> [0,1,2], variant 1 -> [2,1,0]
        fake_variant_rows = [
            np.array([0, 1, 2], dtype=np.int32),
            np.array([2, 1, 0], dtype=np.int32),
        ]

        class FakeReader:
            def __init__(self, path):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self, variant_idx, buf):
                buf[:] = fake_variant_rows[variant_idx]

        with mock.patch("posthoc.io.genotype_reader.pg.PgenReader", FakeReader):
            gd = read_pgen(prefix)

        assert gd.genotypes.shape == (n_samples, n_variants)
        assert gd.genotypes.dtype == np.int8
        # Column 0 (variant 0) should equal [0, 1, 2] after transpose
        assert gd.genotypes[:, 0].tolist() == [0, 1, 2]
        assert gd.genotypes[:, 1].tolist() == [2, 1, 0]
        assert gd.sample_ids == ["S0", "S1", "S2"]
        assert len(gd.variant_ids) == n_variants

    def test_variant_ids_index_reset(self, tmp_path):
        prefix = self._make_fileset(tmp_path, n_samples=2, n_variants=2)

        class FakeReader:
            def __init__(self, path):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self, variant_idx, buf):
                buf[:] = np.zeros(2, dtype=np.int32)

        with mock.patch("posthoc.io.genotype_reader.pg.PgenReader", FakeReader):
            gd = read_pgen(prefix)

        assert gd.variant_ids.index.tolist() == [0, 1]

    def test_zero_variants(self, tmp_path):
        prefix = self._make_fileset(tmp_path, n_samples=3, n_variants=0)
        # overwrite pvar with header only
        (tmp_path / "data.pvar").write_text("#CHROM\tPOS\tID\tREF\tALT\n")

        class FakeReader:
            def __init__(self, path):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self, variant_idx, buf):
                pass  # never called

        with mock.patch("posthoc.io.genotype_reader.pg.PgenReader", FakeReader):
            gd = read_pgen(prefix)

        assert gd.n_variants == 0
        assert gd.n_samples == 3

    def test_pfile_prefix_accepts_string(self, tmp_path):
        prefix = self._make_fileset(tmp_path, n_samples=1, n_variants=1)

        class FakeReader:
            def __init__(self, path):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self, variant_idx, buf):
                buf[:] = np.array([1], dtype=np.int32)

        with mock.patch("posthoc.io.genotype_reader.pg.PgenReader", FakeReader):
            gd = read_pgen(str(prefix))

        assert gd.n_samples == 1

