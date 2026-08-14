from __future__ import annotations

import logging

import click
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

from posthoc.commands._shared import load_and_prepare
from posthoc.models.utils import make_split

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
@click.option("--maf", "min_maf", type=float, default=0.0)
@click.option("--geno", "max_missing", type=float, default=1.0)
@click.option("--indep-pairwise", type=(int, int, float), default=None)
@click.option("--val-fraction", type=float, default=0.2)
@click.option("--seed", type=int, default=0)
def baseline(
    pfile: str,
    pheno: str,
    pheno_name: str | None,
    covar: str | None,
    min_maf: float,
    max_missing: float,
    indep_pairwise: tuple[int, int, float] | None,
    val_fraction: float,
    seed: int,
) -> None:

    loaded = load_and_prepare(
        pfile,
        pheno,
        pheno_name,
        covar,
        "logistic",
        min_maf,
        max_missing,
        indep_pairwise,
    )

    data = loaded.data
    genotypes = data.genotypes
    phenotype = loaded.phenotype
    covariates = loaded.covariates

    train_idx, val_idx = make_split(
        len(genotypes), phenotype, val_fraction, "logistic", seed
    )

    missing_mask = genotypes == -9

    if missing_mask.any():
        train_means = np.where(missing_mask[train_idx], np.nan, genotypes[train_idx])
        train_means = np.nanmean(train_means, axis=0)
        fill = np.take(train_means, np.where(missing_mask)[1])
        genotypes[missing_mask] = fill

    X = genotypes

    if covariates is not None:
        X = np.concatenate([X, covariates], axis=1)

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = phenotype[train_idx], phenotype[val_idx]

    click.echo(f"n_train={len(train_idx)} n_val={len(val_idx)} n_features={X.shape[1]}")
    click.echo(
        f"{'C':>8} {'train_loss':>12} {'val_loss':>10} {'val_auc':>8} {'n_nonzero':>10}"
    )

    for C in (0.001, 0.01, 0.1, 1.0):
        clf = LogisticRegression(
            l1_ratio=1, C=C, solver="liblinear", max_iter=2000, random_state=seed
        )
        clf.fit(X_train, y_train)

        train_loss = log_loss(y_train, clf.predict_proba(X_train)[:, 1])
        val_pred = clf.predict_proba(X_val)[:, 1]
        val_loss = log_loss(y_val, val_pred)
        val_auc = roc_auc_score(y_val, val_pred)
        n_nonzero = int((clf.coef_ != 0).sum())

        click.echo(
            f"{C:>8} {train_loss:>12.4f} {val_loss:>10.4f} {val_auc:>8.3f} {n_nonzero:>10}"
        )
