from __future__ import annotations

import logging
from pathlib import Path

import click

from posthoc.attribution.integrated_gradients import (
    IntegratedGradientsConfig,
    integrated_gradients_importance,
)
from posthoc.commands._shared import load_and_prepare
from posthoc.io.writer import write_glm
from posthoc.models.base import _ACTIVATIONS, TrainConfig
from posthoc.models.mlp import MLPConfig, build_mlp
from posthoc.models.utils import train_model

logger = logging.getLogger(__name__)


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
    "--attribution",
    "attribution_name",
    type=click.Choice(["integrated_gradients"]),
    default="integrated_gradients",
)
@click.option(
    "--hidden-dims", default="256,64", help="Comma-separated MLP hidden layer sizes."
)
@click.option("--dropout", default=0.2, help="Dropout parameter of neural network")
@click.option("--epochs", "max_epochs", type=int, default=200)
@click.option("--patience", type=int, default=15)
@click.option("--lr", type=float, default=1e-4)
@click.option("--activation", type=click.Choice(_ACTIVATIONS.keys()), default="relu")
@click.option("--weight-decay", "weight_decay", type=float, default=1e-4)
@click.option("--data-dropout", "data_dropout", is_flag=True, default=False)
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
@click.option("--seed", type=int, default=0)
@click.option("--device", default="cpu")
@click.option(
    "--out", "out_path", required=True, type=click.Path(), help="Output .glm path."
)
def attribute(
    pfile: str,
    pheno: str,
    pheno_name: str | None,
    covar: str | None,
    task_logistic: bool,
    task_linear: bool,
    model_name: str,
    attribution_name: str,
    hidden_dims: str,
    max_epochs: int,
    patience: int,
    lr: float,
    dropout: float,
    data_dropout: bool,
    activation: str,
    weight_decay: float,
    val_fraction: float,
    n_steps: int,
    ig_baseline: str,
    seed: int,
    device: str,
    out_path: str,
) -> None:

    if task_logistic == task_linear:
        raise click.UsageError("Specify explicitly one of --logistic or --linear.")
    task = "logistic" if task_logistic else "linear"

    loaded = load_and_prepare(
        pfile,
        pheno,
        pheno_name,
        covar,
        task,
    )

    data = loaded.data
    genotypes = data.genotypes
    phenotype = loaded.phenotype
    covariates = loaded.covariates

    if model_name != "mlp":
        raise click.UsageError(f"Unknown model: {model_name}")

    hidden_dims_list = [int(h) for h in hidden_dims.split(",") if h]
    n_covariates = covariates.shape[1] if covariates is not None else 0
    model = build_mlp(
        n_snps=data.n_variants,
        n_covariates=n_covariates,
        config=MLPConfig(
            hidden_dims_list,
            dropout=dropout,
            data_dropout=data_dropout,
            activation=activation,
        ),
    )

    train_config = TrainConfig(
        task=task,
        val_fraction=val_fraction,
        max_epochs=max_epochs,
        patience=patience,
        weight_decay=weight_decay,
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

    if attribution_name != "integrated_gradients":
        raise click.UsageError(f"Unknown attribution name: {attribution_name}")

    attribution_config = IntegratedGradientsConfig(
        n_steps=n_steps, device=device, seed=seed, task=task, baseline=ig_baseline
    )
    logger.info(
        "Running Integrated Gradients (%d steps, baseline=%s)", n_steps, ig_baseline
    )

    val_idx = train_result.val_idx
    genotypes_val = genotypes[val_idx]
    phenotype_val = phenotype[val_idx]
    covariates_val = covariates[val_idx] if covariates is not None else None

    attribution_result = integrated_gradients_importance(
        train_result.model,
        genotypes_val,
        phenotype_val,
        covariates_val,
        attribution_config,
    )

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    write_glm(
        data.variant_ids,
        importances=attribution_result.importances,
        p_values=attribution_result.p_values,
        p_corrected=attribution_result.p_corrected,
        test_name=f"{model_name.upper()}_{attribution_result.method}",
        n_samples=len(val_idx),
        out_path=out_file,
    )

    click.echo(f"Wrote results for {data.n_variants} variants to {out_file}")
