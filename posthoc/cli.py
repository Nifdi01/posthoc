from __future__ import annotations

import logging
from pathlib import Path

import click
import numpy as np
from dataclasses import replace

from numpy.random import sample

from posthoc import attribution
from posthoc.attribution import permutation
from posthoc.io.genotype_reader import GenotypeData, read_pgen
from posthoc.io.pheno_covar_reader import align_samples, load_covar, load_pheno
from posthoc.io.writer import write_glm
from posthoc.qc.filters import run_qc
from posthoc.models.base import TrainResult, TrainConfig
from posthoc.models.mlp import MLPConfig, build_mlp
from posthoc.models.utils import train_model
from posthoc.attribution.permutation import PermutationConfig, permutation_importance


logger = logging.getLogger(__name__)


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _subset_samples(
    data: GenotypeData, keep_mask: np.ndarray, ordered_ids: list[str]
) -> GenotypeData:
    return replace(data, genotypes=data.genotypes[keep_mask, :], sample_ids=ordered_ids)


@click.group()
@click.option("--verbose", is_flag=True, help="Enable debug logging.")
def main(verbose: bool) -> None:
    _setup_logging(verbose)


@main.command()
@click.option(
    "--pfile", required=True, help="Prefix of PLINK2 .pgen/.pvar/.psam fileset."
)
@click.option(
    "--pheno", required=True, type=click.Path(exists=True), help="Phenotype file."
)
@click.option(
    "--pheno-name",
    default=None,
    help="Phenotype column name (default: first value column).",
)
@click.option(
    "--covar", default=None, type=click.Path(exists=True), help="Covariate file."
)
@click.option(
    "--logistic", "task_logistic", is_flag=True, help="Binary phenotype (BCE loss)."
)
@click.option(
    "--linear", "task_linear", is_flag=True, help="Continuous phenotype (MSE loss)."
)
@click.option("--model", "model_name", type=click.Choice(["mlp"]), default="mlp")
@click.option(
    "--attribution",
    "attribution_name",
    type=click.Choice(["permutation"]),
    default="permutation",
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
    "--hidden-dims", default="256,64", help="Comma-separated MLP hidden layer sizes."
)
@click.option("--epochs", "max_epochs", type=int, default=200)
@click.option("--patience", type=int, default=15)
@click.option("--lr", type=float, default=1e-4)
@click.option("--val-fraction", type=float, default=0.2)
@click.option("--n-repeats", type=int, default=20, help="Permutation repeats per SNP.")
@click.option("--seed", type=int, default=0)
@click.option("--device", default="cpu")
@click.option(
    "--out", "out_path", required=True, type=click.Path(), help="Output .glm path."
)
def run(
    pfile: str,
    pheno: str,
    pheno_name: str | None,
    covar: str | None,
    task_logistic: bool,
    task_linear: bool,
    model_name: str,
    attribution_name: str,
    min_maf: float,
    max_missing: float,
    indep_pairwise: tuple[int, int, float] | None,
    hidden_dims: str,
    max_epochs: int,
    patience: int,
    lr: float,
    val_fraction: float,
    n_repeats: int,
    seed: int,
    device: str,
    out_path: str,
) -> None:
    if task_logistic == task_linear:
        raise click.UsageError("Specify exactly one of --logistic or --linear.")
    task = "logistic" if task_logistic else "linear"

    # I/O Stage
    logger.info("Loading genotypes from %s", pfile)
    data = read_pgen(pfile)
    logger.info("Loaded %d samples x %d variants", data.n_samples, data.n_variants)

    pheno_series = load_pheno(pheno, pheno_name=pheno_name)
    covar_df = load_covar(covar) if covar is not None else None

    keep_mask, align_pheno, align_covar = align_samples(
        data.sample_ids, pheno_series, covar_df
    )
    ordered_ids = [sid for sid, keep in zip(data.sample_ids, keep_mask) if keep]
    data = _subset_samples(data, keep_mask, ordered_ids)
    n_dropped_samples = int((~keep_mask).sum())
    logger.info(
        "Sample alignment: dropped %d / %d samples (missing pheno/covar); %d remain",
        n_dropped_samples,
        len(keep_mask),
        data.n_samples,
    )

    # QC Stage
    qc_results = run_qc(
        data,
        min_maf=min_maf,
        max_missing=max_missing,
        indep_pairwise_params=indep_pairwise,
        pfile_prefix=pfile if indep_pairwise is not None else None,
    )
    data = qc_results.data
    logger.info(qc_results.summary())

    genotypes = data.genotypes
    phenotype = align_pheno
    covariates = align_covar

    # Modeling Stage
    # This is temporary. Need to add other models in the future
    if model_name != "mlp":
        raise click.UsageError(f"Unknown model: {model_name}")

    hidden_dims_list = [int(h) for h in hidden_dims.split(",") if h]
    n_covariates = covariates.shape[1] if covariates is not None else 0
    model = build_mlp(
        n_snps=data.n_variants,
        n_covariates=n_covariates,
        config=MLPConfig(hidden_dims=hidden_dims_list),
    )
    train_config = TrainConfig(
        task=task,
        val_fraction=val_fraction,
        max_epochs=max_epochs,
        patience=patience,
        lr=lr,
        device=device,
        seed=seed,
    )

    logger.info(
        "Training %s (%s task) on %d SNPs, %d samples",
        model_name,
        task,
        data.n_variants,
        data.n_samples,
    )
    train_result = train_model(model, genotypes, phenotype, covariates, train_config)
    logger.info(
        "Training done: stopped at epoch %d, best val loss %.4f",
        train_result.stopped_epoch,
        train_result.best_val_loss,
    )

    # Attribution Stage
    if attribution_name != "permutation":
        raise click.UsageError(f"Unknown attribution method: {attribution_name}")

    attribution_config = PermutationConfig(
        n_repeats=n_repeats, device=device, seed=seed, task=task
    )
    logger.info("Running permutation importance (%d repeats/SNP)", n_repeats)

    attribution_result = permutation_importance(
        train_result.model, genotypes, phenotype, covariates, attribution_config
    )

    write_glm(
        data.variant_ids,
        importances=attribution_result.importances,
        p_values=attribution_result.p_values,
        test_name=f"{model_name.upper()}_{attribution_result.method}",
        n_samples=data.n_samples,
        out_path=Path(out_path),
    )

    click.echo(f"Wrote results for {data.n_variants} variants to {out_path}")


if __name__ == "__main__":
    main()
