from __future__ import annotations

import logging

import click

from posthoc.io.genotype_reader import read_pgen
from posthoc.io.writer import write_filtered_pgen
from posthoc.qc.filters import find_plink2, run_qc

logger = logging.getLogger(__name__)


@click.command(name="qc")
@click.option(
    "--pfile", required=True, help="Prefix of PLINK2 .pgen/.pvar/.psam fileset."
)
@click.option(
    "--maf", "min_maf", type=float, default=0.0, help="Minimum minor allele frequency."
)
@click.option(
    "--geno",
    "max_missing",
    type=float,
    default=1.0,
    help="Max per-variant missingness.",
)
@click.option(
    "--indep-pairwise",
    type=(int, int, float),
    default=None,
    help="LD pruning: WINDOW_SIZE STEP R2_THRESHOLD (requires plink2 on PATH).",
)
@click.option(
    "--out",
    "out_prefix",
    required=True,
    type=click.Path(),
    help="Output pfile prefix for the filtered dataset.",
)
def qc(
    pfile: str,
    min_maf: float,
    max_missing: float,
    indep_pairwise: tuple[int, int, float] | None,
    out_prefix: str,
) -> None:
    logger.info("Loading genotypes from %s", pfile)
    data = read_pgen(pfile)
    logger.info("Loaded %d samples x %d variants", data.n_samples, data.n_variants)

    qc_result = run_qc(
        data,
        min_maf=min_maf,
        max_missing=max_missing,
        indep_pairwise_params=indep_pairwise,
        pfile_prefix=pfile if indep_pairwise is not None else None,
    )
    logger.info(qc_result.summary())

    plink2_path = find_plink2()
    keep_ids = qc_result.data.variant_ids["ID"].tolist()
    write_filtered_pgen(pfile, keep_ids, out_prefix, plink2_path)
    click.echo(
        f"Wrote {qc_result.data.n_variants} variants to {out_prefix} "
        f"(dropped {qc_result.n_dropped_maf} MAF, {qc_result.n_dropped_geno} missingness, "
        f"{qc_result.n_dropped_ld} LD)"
    )
