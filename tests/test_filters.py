from posthoc.qc.filters import MISSING_CODE, compute_missing_rate
from posthoc.io.genotype_reader import read_pgen


def test_missing_code_matches_reader_output(toy_missing_pfile):
    data = read_pgen(toy_missing_pfile)
    assert MISSING_CODE == -9
    # rs2 (column 1) has 1 missing call out of 3 samples
    miss_rate = compute_missing_rate(data.genotypes)
    assert miss_rate[1] == 1 / 3
    assert miss_rate[0] == 0.0
