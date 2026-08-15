# tests/conftest.py
import shutil
import subprocess

import pytest

VCF_CONTENT = """##fileformat=VCFv4.2
##contig=<ID=21>
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\tS2\tS3
21\t100\trs1\tA\tG\t.\t.\t.\tGT\t0/0\t0/1\t1/1
21\t200\trs2\tC\tT\t.\t.\t.\tGT\t0/0\t./.\t1/1
"""


@pytest.fixture(scope="session")
def toy_missing_pfile(tmp_path_factory):
    """A tiny pgen fixture with one known-missing genotype (S2 @ rs2)."""
    if shutil.which("plink2") is None:
        pytest.skip("plink2 not on PATH")

    tmpdir = tmp_path_factory.mktemp("toy_missing")
    vcf_path = tmpdir / "toy_missing.vcf"
    vcf_path.write_text(VCF_CONTENT)

    prefix = str(tmpdir / "toy_missing")
    result = subprocess.run(
        ["plink2", "--vcf", str(vcf_path), "--make-pgen", "--out", prefix],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return prefix
