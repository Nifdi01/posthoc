import pandas as pd

from posthoc.io.writer import GLM_COLUMNS, write_glm


def test_write_glm_format(tmp_path):
    variant_ids = pd.DataFrame(
        {
            "CHROM": ["21", "21"],
            "POS": [100, 200],
            "ID": ["rs1", "rs2"],
            "REF": ["A", "C"],
            "ALT": ["G", "T"],
        }
    )
    out_path = tmp_path / "test.glm"
    write_glm(
        variant_ids,
        importances=[0.1, 0.9],
        p_values=[0.5, 0.01],
        p_corrected=[0.8, 0.05],
        test_name="IG",
        n_samples=100,
        out_path=out_path,
    )

    df = pd.read_csv(out_path, sep="\t")
    assert list(df.columns) == GLM_COLUMNS
    assert df.iloc[1]["IMPORTANCE"] == 0.9
