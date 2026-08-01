from __future__ import annotations

import numpy as np
import pytest
import torch

from posthoc.models.base import GenotypeDataset, TrainConfig
from posthoc.models.utils import make_split, train_model

from posthoc.models.mlp import MLP, MLPConfig, build_mlp


# ---------- fixtures ----------


@pytest.fixture
def toy_genotypes():
    rng = np.random.default_rng(0)
    return rng.integers(0, 3, size=(40, 10)).astype(np.float32)


@pytest.fixture
def toy_genotypes_with_missing(toy_genotypes):
    geno = toy_genotypes.copy()
    geno[0, 0] = -9
    geno[5, 3] = -9
    return geno


@pytest.fixture
def toy_pheno_binary():
    # Balanced-ish binary phenotype, deterministic
    rng = np.random.default_rng(1)
    return rng.integers(0, 2, size=40).astype(np.float32)


@pytest.fixture
def toy_pheno_continuous():
    rng = np.random.default_rng(2)
    return rng.normal(size=40).astype(np.float32)


@pytest.fixture
def toy_covariates():
    rng = np.random.default_rng(3)
    return rng.normal(size=(40, 2)).astype(np.float32)


def make_separable_dataset(n=200, n_snps=5, seed=0):
    """Genotype matrix + binary phenotype that's linearly separable on
    SNP 0, so a trained MLP should drive train loss down substantially."""
    rng = np.random.default_rng(seed)
    geno = rng.integers(0, 3, size=(n, n_snps)).astype(np.float32)
    logit = 3.0 * (geno[:, 0] - 1.0)  # strong signal on SNP 0 only
    prob = 1 / (1 + np.exp(-logit))
    pheno = (rng.random(n) < prob).astype(np.float32)
    return geno, pheno


# ---------- GenotypeDataset ----------


def test_genotype_dataset_length_matches_samples(toy_genotypes, toy_pheno_binary):
    ds = GenotypeDataset(toy_genotypes, toy_pheno_binary)
    assert len(ds) == toy_genotypes.shape[0]


def test_genotype_dataset_getitem_shapes(toy_genotypes, toy_pheno_binary):
    ds = GenotypeDataset(toy_genotypes, toy_pheno_binary)
    x, y = ds[0]
    assert x.shape == (toy_genotypes.shape[1],)
    assert y.shape == ()


def test_genotype_dataset_concatenates_covariates(
    toy_genotypes, toy_pheno_binary, toy_covariates
):
    ds = GenotypeDataset(toy_genotypes, toy_pheno_binary, toy_covariates)
    x, _ = ds[0]
    assert x.shape == (toy_genotypes.shape[1] + toy_covariates.shape[1],)


def test_genotype_dataset_mean_imputes_missing(
    toy_genotypes_with_missing, toy_pheno_binary
):
    ds = GenotypeDataset(toy_genotypes_with_missing, toy_pheno_binary)
    # no -9 sentinel should survive imputation
    assert not torch.any(ds.genotypes == -9)
    # imputed value at (0, 0) should equal the mean of the non-missing
    # values in column 0
    col0 = toy_genotypes_with_missing[:, 0]
    expected = col0[col0 != -9].mean()
    assert ds.genotypes[0, 0].item() == pytest.approx(expected, abs=1e-4)


def test_genotype_dataset_no_missing_is_noop(toy_genotypes, toy_pheno_binary):
    ds = GenotypeDataset(toy_genotypes, toy_pheno_binary)
    np.testing.assert_array_equal(
        ds.genotypes.numpy(), toy_genotypes.astype(np.float32)
    )


# ---------- make_split ----------


def test_make_split_sizes(toy_pheno_binary):
    train_idx, val_idx = make_split(
        n_samples=40,
        phenotype=toy_pheno_binary,
        val_fraction=0.25,
        task="logistic",
        seed=0,
    )
    assert len(train_idx) == 30
    assert len(val_idx) == 10


def test_make_split_disjoint_and_covers_all(toy_pheno_binary):
    train_idx, val_idx = make_split(
        n_samples=40,
        phenotype=toy_pheno_binary,
        val_fraction=0.2,
        task="logistic",
        seed=0,
    )
    assert set(train_idx).isdisjoint(set(val_idx))
    assert set(train_idx) | set(val_idx) == set(range(40))


def test_make_split_deterministic_given_seed(toy_pheno_binary):
    a_train, a_val = make_split(40, toy_pheno_binary, 0.2, "logistic", seed=7)
    b_train, b_val = make_split(40, toy_pheno_binary, 0.2, "logistic", seed=7)
    np.testing.assert_array_equal(a_train, b_train)
    np.testing.assert_array_equal(a_val, b_val)


def test_make_split_linear_task_ignores_stratify(toy_pheno_continuous):
    # Should not raise even though a continuous target can't be stratified
    train_idx, val_idx = make_split(
        n_samples=40,
        phenotype=toy_pheno_continuous,
        val_fraction=0.2,
        task="linear",
        seed=0,
    )
    assert len(train_idx) + len(val_idx) == 40


# ---------- train_model (integration, on MLP) ----------


def test_train_model_runs_and_returns_expected_fields(toy_genotypes, toy_pheno_binary):
    model = build_mlp(n_snps=toy_genotypes.shape[1], config=MLPConfig(hidden_dims=[8]))
    config = TrainConfig(
        task="logistic", max_epochs=3, batch_size=8, device="cpu", seed=0
    )
    result = train_model(model, toy_genotypes, toy_pheno_binary, None, config)

    assert isinstance(result.model, MLP)
    assert len(result.train_losses) == len(result.val_losses) == result.stopped_epoch
    assert result.best_val_loss == pytest.approx(min(result.val_losses))


def test_train_model_early_stopping_triggers(toy_genotypes, toy_pheno_binary):
    model = build_mlp(n_snps=toy_genotypes.shape[1], config=MLPConfig(hidden_dims=[8]))
    config = TrainConfig(
        task="logistic",
        max_epochs=100,
        batch_size=8,
        patience=2,
        device="cpu",
        seed=0,
    )
    result = train_model(model, toy_genotypes, toy_pheno_binary, None, config)
    # with patience=2 on a tiny noisy dataset, should stop well short of max_epochs
    assert result.stopped_epoch < config.max_epochs


def test_train_model_reduces_loss_on_separable_data():
    geno, pheno = make_separable_dataset(n=200, n_snps=5, seed=0)
    model = build_mlp(n_snps=5, config=MLPConfig(hidden_dims=[16], dropout=0.0))
    config = TrainConfig(
        task="logistic", max_epochs=60, batch_size=16, lr=1e-2, device="cpu", seed=0
    )
    result = train_model(model, geno, pheno, None, config)
    # first epoch's train loss vs. best val loss achieved: should improve a lot
    assert result.best_val_loss < result.train_losses[0] * 0.8


def test_train_model_with_covariates(toy_genotypes, toy_pheno_binary, toy_covariates):
    model = build_mlp(
        n_snps=toy_genotypes.shape[1],
        n_covariates=toy_covariates.shape[1],
        config=MLPConfig(hidden_dims=[8]),
    )
    config = TrainConfig(
        task="logistic", max_epochs=3, batch_size=8, device="cpu", seed=0
    )
    result = train_model(model, toy_genotypes, toy_pheno_binary, toy_covariates, config)
    assert result.stopped_epoch >= 1


def test_train_model_linear_task_uses_mse(toy_genotypes, toy_pheno_continuous):
    model = build_mlp(n_snps=toy_genotypes.shape[1], config=MLPConfig(hidden_dims=[8]))
    config = TrainConfig(
        task="linear", max_epochs=3, batch_size=8, device="cpu", seed=0
    )
    result = train_model(model, toy_genotypes, toy_pheno_continuous, None, config)
    assert all(loss >= 0 for loss in result.train_losses)


def test_train_model_invalid_task_raises(toy_genotypes, toy_pheno_binary):
    model = build_mlp(n_snps=toy_genotypes.shape[1])
    config = TrainConfig(task="bogus", max_epochs=1, batch_size=8, device="cpu")
    with pytest.raises(ValueError, match="Unknown task"):
        train_model(model, toy_genotypes, toy_pheno_binary, None, config)


def test_genotype_dataset_getitem_covariate_values_correct(
    toy_genotypes, toy_pheno_binary, toy_covariates
):
    ds = GenotypeDataset(toy_genotypes, toy_pheno_binary, toy_covariates)
    x, _ = ds[3]
    np.testing.assert_allclose(x[: toy_genotypes.shape[1]].numpy(), toy_genotypes[3])
    np.testing.assert_allclose(x[toy_genotypes.shape[1] :].numpy(), toy_covariates[3])


# ---------- MLP ----------


def test_mlp_output_shape(toy_genotypes):
    model = MLP(input_dim=toy_genotypes.shape[1], config=MLPConfig(hidden_dims=[8, 4]))
    x = torch.from_numpy(toy_genotypes)
    out = model(x)
    assert out.shape == (toy_genotypes.shape[0], 1)


def test_mlp_invalid_activation_raises():
    with pytest.raises(ValueError, match="Unknown activation"):
        MLP(input_dim=5, config=MLPConfig(activation="swish"))


def test_build_mlp_infers_input_dim():
    model = build_mlp(n_snps=100, n_covariates=3)
    first_linear = model.hidden[0]
    assert isinstance(first_linear, torch.nn.Linear)
    assert first_linear.in_features == 103
