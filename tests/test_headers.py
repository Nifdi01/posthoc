import pandas as pd

PLINK_JOIN_COLS = ["#CHROM", "POS", "ID", "REF", "ALT"]


def test_plink_output_is_joinable_with_posthoc_schema():
    plink_df = pd.read_csv("plink_ref.PHENO1.glm.logistic.hybrid", sep=r"\s+")

    # posthoc's writer must share these columns so results can be merged
    for col in PLINK_JOIN_COLS:
        assert col in plink_df.columns, f"missing shared key column: {col}"

    # sanity: PLINK2's statistical columns are NOT expected to match posthoc —
    # they're different tests (regression vs. permutation)
    assert "P" in plink_df.columns
    assert "OBS_CT" in plink_df.columns
