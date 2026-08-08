from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path

import click
import numpy as np

from posthoc.io.genotype_reader import read_pgen
from posthoc.simulation import PhenotypeModel, simulate

logger = logging.getLogger(__name__)


@click.command(name="simulate-pheno")
@click.option(
    "--pfile", required=True, help="Prefix of PLINK2 .pgen/.pvar/.psam fileset."
)
@click.option(
    "--additive",
    "additive_terms",
    type=(int, float),
    multiple=True,
    help="Additive causal SNP: INDEX EFFECT_SIZE. Repeatable.",
)
@click.option(
    "--dominant",
    "dominant_terms",
    type=(int, float),
    multiple=True,
    help="Dominant causal SNP: INDEX EFFECT_SIZE. Repeatable.",
)
@click.option(
    "--recessive",
    "recessive_terms",
    type=(int, float),
    multiple=True,
    help="Recessive causal SNP: INDEX EFFECT_SIZE. Repeatable.",
)
@click.option(
    "--interaction2",
    "interaction2_terms",
    type=(int, int, float),
    multiple=True,
    help="Two-way interaction: INDEX_I INDEX_J EFFECT_SIZE. Repeatable.",
)
@click.option(
    "--interaction3",
    "interaction3_terms",
    type=(int, int, int, float),
    multiple=True,
    help="Three-way interaction: INDEX_I INDEX_J INDEX_K EFFECT_SIZE. Repeatable.",
)
@click.option(
    "--logistic", "task_logistic", is_flag=True, help="Binary phenotype (case/control)."
)
@click.option("--linear", "task_linear", is_flag=True, help="Continuous phenotype.")
@click.option(
    "--heritability",
    type=float,
    default=0.5,
    help="Fraction of liability variance from genotype.",
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
    additive_terms: tuple[tuple[int, float], ...],
    dominant_terms: tuple[tuple[int, float], ...],
    recessive_terms: tuple[tuple[int, float], ...],
    interaction2_terms: tuple[tuple[int, int, float], ...],
    interaction3_terms: tuple[tuple[int, int, int, float], ...],
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

    if not any(
        [
            additive_terms,
            dominant_terms,
            recessive_terms,
            interaction2_terms,
            interaction3_terms,
        ]
    ):
        raise click.UsageError(
            "Specify at least one causal term via --additive/--dominant/"
            "--recessive/--interaction2/--interaction3."
        )

    logger.info("Loading genotypes from %s", pfile)
    data = read_pgen(pfile)
    logger.info("Loaded %d samples x %d variants", data.n_samples, data.n_variants)

    if recode_centered:
        logger.info("Recoding genotypes 0/1/2 -> -1/0/1")
        data = replace(data, genotypes=data.genotypes - 1)

    def split(terms: tuple, n_idx: int) -> tuple[np.ndarray, np.ndarray]:
        if not terms:
            return np.array([], dtype=int), np.array([])

        arr = np.array(terms)
        if n_idx == 1:
            return arr[:, 0].astype(int), arr[:, 1]
        idx = arr[:, :n_idx].astype(int)
        eff = arr[:, n_idx]
        return idx, eff

    additive_idx, additive_eff = split(additive_terms, 1)
    dominant_idx, dominant_eff = split(dominant_terms, 1)
    recessive_idx, recessive_eff = split(recessive_terms, 1)

    interaction_pairs = [(i, j) for i, j, _ in interaction2_terms]
    interaction_effects = np.array([e for _, _, e in interaction2_terms])
    interaction_triples = [(i, j, k) for i, j, k, _ in interaction3_terms]
    interaction_triple_effects = np.array([e for _, _, _, e in interaction3_terms])

    model = PhenotypeModel(
        additive_indices=additive_idx,
        additive_effects=additive_eff,
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

    logger.info(
        "Simulating phenotype: %d causal SNPs, heritability=%.2f, task=%s",
        len(model.all_causal_indices()),
        heritability,
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
