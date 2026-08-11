from __future__ import annotations

import logging

import click
import numpy as np

from posthoc.commands._shared import load_and_prepare
from posthoc.io.writer import write_pal
from posthoc.models.base import TrainConfig
from posthoc.models.mlp import MLPConfig, build_mlp
from posthoc.models.utils import train_model
from posthoc.attribution.integrated_gradients import (
    IntegratedGradientsConfig,
    integrated_gradients_importance,
)
from posthoc.attribution.pal import PALConfig, compute_pal
from posthoc.attribution.significance import SignificanceConfig, compute_pal_pvalues


logger = logging.getLogger(__name__)


def _case_mas(
    model,
    genotypes: np.ndarray,
    phenotype: np.ndarray,
    covariates: np.ndarray | None,
    ig_config: IntegratedGradientsConfig,
) -> np.ndarray:
    case_mask = phenotype == 1
    if not case_mask.any():
        raise click.UsageError(
            "No case samples (phenotype == 1) found for MAS computation."
        )

    case_genotypes = genotypes[case_mask]
    case_phenotype = phenotype[case_mask]
    case_covariates = covariates[case_mask] if covariates is not None else None

    result = integrated_gradients_importance(
        model, case_genotypes, case_phenotype, case_covariates, ig_config
    )

    return result.importances


@click.command()
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
@click.option("--dropout", default=0.2, help="Dropout parameter of neural network")
@click.option("--epochs", "max_epochs", type=int, default=200)
@click.option("--patience", type=int, default=15)
@click.option("--lr", type=float, default=1e-4)
@click.option("--val-fraction", type=float, default=0.2)
@click.option(
    "--n-steps", type=int, default=50, help="Integrated Gradients interpolation steps."
)
@click.option(
    "--ig-baseline",
    type=click.Choice(["zero", "mean"]),
    default="zero",
    help="Baseline reference input for Integrated Gradients.",
)
@click.option(
    "--n-models",
    type=int,
    default=10,
    help="Number of models trained with different seeds (real labels).",
)
@click.option(
    "--n-null-models",
    type=int,
    default=10,
    help="Number of models trained on permuted labels for the null distribution.",
)
@click.option(
    "--theta-percentile",
    type=float,
    default=99.99,
    help="Percentile threshold for PAL detection (99.99=strict, 99.95=relaxed in the paper).",
)
@click.option(
    "--ld-window", type=int, default=20, help="+/- SNP window checked for LD clumping."
)
@click.option(
    "--ld-r-threshold",
    type=float,
    default=0.5,
    help="Absolute Pearson r threshold for LD clumping.",
)
@click.option(
    "--n-bootstrap",
    type=int,
    default=100,
    help="Bootstrap iterations for PAL_AMAS P-values.",
)
@click.option("--seed", type=int, default=0, help="Base seed; model i uses seed+i.")
@click.option("--device", default="cpu")
@click.option(
    "--out", "out_path", required=True, type=click.Path(), help="Output PAL .tsv path."
)
def pal(
    pfile: str,
    pheno: str,
    pheno_name: str | None,
    covar: str | None,
    task_logistic: bool,
    task_linear: bool,
    model_name: str,
    min_maf: float,
    max_missing: float,
    indep_pairwise: tuple[int, int, float] | None,
    hidden_dims: str,
    max_epochs: int,
    patience: int,
    lr: float,
    dropout: float,
    val_fraction: float,
    n_steps: int,
    ig_baseline: str,
    n_models: int,
    n_null_models: int,
    theta_percentile: float,
    ld_window: int,
    ld_r_threshold: float,
    n_bootstrap: int,
    seed: int,
    device: str,
    out_path: str,
) -> None:
    if task_logistic == task_linear:
        raise click.UsageError("Specify explicity one of --logistic or --linear")
    task = "logistic" if task_logistic else "linear"

    if model_name != "mlp":
        raise click.UsageError(f"Unkown model: {model_name}")

    loaded = load_and_prepare(
        pfile, pheno, pheno_name, covar, task, min_maf, max_missing, indep_pairwise
    )

    data = loaded.data
    genotypes = data.genotypes
    phenotype = loaded.phenotype
    covariates = loaded.covariates

    hidden_dims_list = [int(h) for h in hidden_dims.split(",") if h]
    n_covariates = covariates.shape[1] if covariates is not None else 0

    ig_config = IntegratedGradientsConfig(
        n_steps=n_steps, device=device, seed=seed, task=task, baseline=ig_baseline
    )

    def _train_one(seed_i: int, labels: np.ndarray):
        model = build_mlp(
            n_snps=data.n_variants,
            n_covariates=n_covariates,
            config=MLPConfig(hidden_dims_list, dropout=dropout),
        )
        train_config = TrainConfig(
            task=task,
            val_fraction=val_fraction,
            max_epochs=max_epochs,
            patience=patience,
            lr=lr,
            device=device,
            seed=seed_i,
        )
        result = train_model(model, genotypes, labels, covariates, train_config)
        return result

    observed_mas_list: list[np.ndarray] = []
    for i in range(n_models):
        seed_i = seed + i
        logger.info(
            "Training real-label model %d/%d (seed=%d)", i + 1, n_models, seed_i
        )
        model = _train_one(seed_i, phenotype)
        mas = _case_mas(model.model, genotypes, phenotype, covariates, ig_config)
        observed_mas_list.append(mas)

    null_mas_list: list[np.ndarray] = []
    rng = np.random.default_rng(seed)
    for i in range(n_null_models):
        seed_i = seed + n_models + i
        logger.info(
            "Training null-label model %d/%d (seed=%d)", i + 1, n_null_models, seed_i
        )
        shuffled = phenotype.copy()
        rng.shuffle(shuffled)
        model = _train_one(seed_i, shuffled)

        mas = _case_mas(model.model, genotypes, shuffled, covariates, ig_config)
        null_mas_list.append(mas)

    pal_config = PALConfig(
        theta_percentile=theta_percentile,
        ld_window=ld_window,
        ld_r2_threshold=ld_r_threshold,
    )

    logger.info("Computing PAL / AMAS over %d models", n_models)
    pal_result = compute_pal(observed_mas_list, genotypes, pal_config)

    logger.info(
        "Fitting null distribution and computing PAL_AMAS P-values (%d bootstraps)",
        n_bootstrap,
    )
    sig_result = compute_pal_pvalues(
        observed_mas_list=observed_mas_list,
        null_mas_list=null_mas_list,
        genotypes=genotypes,
        pal_result=pal_result,
        pal_config=pal_config,
        sig_config=SignificanceConfig(n_bootstrap=n_bootstrap, seed=seed),
    )

    write_pal(
        variant_ids=data.variant_ids,
        mu=pal_result.mu,
        amas=pal_result.amas,
        pal_common_idx=pal_result.pal_common,
        pal_amas_idx=pal_result.pal_amas,
        p_values=sig_result.p_values,
        n_models=n_models,
        out_path=out_path,
    )

    click.echo(
        f"Wrote PAL results for {data.n_variants} variants "
        f"(PAL_Common={len(pal_result.pal_common)}, PAL_AMAS={len(pal_result.pal_amas)}) to {out_path}"
    )
