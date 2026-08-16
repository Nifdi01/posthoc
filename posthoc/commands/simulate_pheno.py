from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path

import click
import numpy as np

from posthoc.io.genotype_reader import read_pgen
from posthoc.simulation import PhenotypeModel, simulate
from posthoc.simulation.utils import (
    allocate_causal_positions,
    heritability_to_k,
    log_ratio_percentages,
    resolve_snp_pool,
    validate_ratios,
)

logger = logging.getLogger(__name__)


@click.command(name="simulate-pheno")
@click.option(
    "--pfile", required=True, help="Prefix of PLINK2 .pgen/.pvar/.psam fileset."
)
@click.option(
    "--n-causal",
    type=int,
    required=True,
    help="Total number of causal SNPs to allocate across dominant/recessive/"
    "pair/triple effect types (Yelmen et al. used 100 and 1000).",
)
@click.option(
    "--ratio-dominant",
    type=float,
    default=5.0,
    show_default=True,
    help="Relative allocation of causal SNPs to dominant effects.",
)
@click.option(
    "--ratio-recessive",
    type=float,
    default=5.0,
    show_default=True,
    help="Relative allocation of causal SNPs to recessive effects.",
)
@click.option(
    "--ratio-pair",
    type=float,
    default=4.0,
    show_default=True,
    help="Relative allocation of causal SNPs to two-way interaction terms "
    "(SNP count, i.e. 2 SNPs per term).",
)
@click.option(
    "--ratio-triple",
    type=float,
    default=6.0,
    show_default=True,
    help="Relative allocation of causal SNPs to three-way interaction terms "
    "(SNP count, i.e. 3 SNPs per term).",
)
@click.option(
    "--chromosome",
    default=None,
    help="Restrict causal-position sampling to this chromosome (e.g. '6'). "
    "Omit for whole-genome sampling (paper's main SCZ simulation scenarios).",
)
@click.option(
    "--snp-pool",
    "snp_pool_path",
    default=None,
    type=click.Path(exists=True),
    help="Path to a text file of variant IDs (one per line) to restrict "
    "causal-position sampling to. Combinable with --chromosome.",
)
@click.option(
    "--logistic", "task_logistic", is_flag=True, help="Binary phenotype (case/control)."
)
@click.option("--linear", "task_linear", is_flag=True, help="Continuous phenotype.")
@click.option(
    "--heritability",
    type=float,
    default=0.5,
    show_default=True,
    help="Fraction of liability variance from genotype. Internally converted "
    "to the paper's noise scaling factor k = (1 - heritability) / heritability.",
)
@click.option(
    "--prevalence",
    type=float,
    default=0.1,
    help="Target case prevalence (logistic task only).",
)
@click.option(
    "--recode-centered",
    is_flag=True,
    help="Recode genotypes from 0/1/2 to -1/0/1 before simulating (Yelmen et al.).",
)
@click.option(
    "--pheno-name", default="PHENO1", help="Column name for the output phenotype."
)
@click.option("--seed", type=int, default=0)
@click.option(
    "--out",
    "out_path",
    required=True,
    type=click.Path(),
    help="Output phenotype file path.",
)
def simulate_pheno(
    pfile: str,
    n_causal: int,
    ratio_dominant: float,
    ratio_recessive: float,
    ratio_pair: float,
    ratio_triple: float,
    chromosome: str | None,
    snp_pool_path: str | None,
    task_logistic: bool,
    task_linear: bool,
    heritability: float,
    prevalence: float,
    recode_centered: bool,
    pheno_name: str,
    seed: int,
    out_path: str,
) -> None:
    if task_logistic == task_linear:
        raise click.UsageError("Specify exactly one of --logistic or --linear.")
    task = "logistic" if task_logistic else "linear"

    total_ratio = validate_ratios(
        ratio_dominant, ratio_recessive, ratio_pair, ratio_triple
    )

    logger.info("Loading genotypes from %s", pfile)
    data = read_pgen(pfile)
    logger.info("Loaded %d samples x %d variants", data.n_samples, data.n_variants)

    if recode_centered:
        logger.info("Recoding genotypes 0/1/2 -> -1/0/1")
        data = replace(data, genotypes=data.genotypes - 1)

    pool = resolve_snp_pool(data.variant_ids, chromosome, snp_pool_path)
    logger.info(
        "Causal-position pool: %d variants (chromosome=%s, snp_pool=%s)",
        len(pool),
        chromosome,
        snp_pool_path,
    )

    log_ratio_percentages(
        ratio_dominant,
        ratio_recessive,
        ratio_pair,
        ratio_triple,
        total_ratio,
        n_causal,
    )

    rng = np.random.default_rng(seed)
    dominant_idx, recessive_idx, interaction_pairs, interaction_triples = (
        allocate_causal_positions(
            n_causal,
            pool,
            ratio_dominant,
            ratio_recessive,
            ratio_pair,
            ratio_triple,
            total_ratio,
            rng,
        )
    )

    # beta ~ N(0, 1) per Yelmen et al.
    dominant_eff = rng.normal(0.0, 1.0, size=len(dominant_idx))
    recessive_eff = rng.normal(0.0, 1.0, size=len(recessive_idx))
    interaction_effects = rng.normal(0.0, 1.0, size=len(interaction_pairs))
    interaction_triple_effects = rng.normal(0.0, 1.0, size=len(interaction_triples))

    model = PhenotypeModel(
        dominant_indices=dominant_idx,
        dominant_effects=dominant_eff,
        recessive_indices=recessive_idx,
        recessive_effects=recessive_eff,
        interaction_pairs=interaction_pairs,
        interaction_effects=interaction_effects,
        interaction_triples=interaction_triples,
        interaction_triple_effects=interaction_triple_effects,
        heritability=heritability,
        task=task,
        prevalence=prevalence,
        seed=seed,
    )

    k = heritability_to_k(heritability)
    logger.info(
        "Simulating phenotype: %d causal SNPs (%d dominant, %d recessive, "
        "%d pairs, %d triples), heritability=%.3f (k=%.3f), task=%s",
        len(model.all_causal_indices()),
        len(dominant_idx),
        len(recessive_idx),
        len(interaction_pairs),
        len(interaction_triples),
        heritability,
        k,
        task,
    )
    result = simulate(data, model)

    pheno_df = result.as_pheno_df(pheno_name=pheno_name)
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    pheno_df.to_csv(out_file, sep="\t", index=False)

    causal_ids = result.causal_variant_ids()
    truth_path = out_file.with_suffix(".causal.txt")
    with open(truth_path, "w") as f:
        f.write("\n".join(causal_ids) + "\n")

    click.echo(
        f"Wrote {len(pheno_df)} samples to {out_path} "
        f"({len(causal_ids)} causal variants; ground truth: {truth_path})"
    )
