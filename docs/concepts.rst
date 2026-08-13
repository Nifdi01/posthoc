Concepts and methodology
==========================

This page explains the statistical/ML machinery behind PostHoc's four
commands, at a level intended to help you interpret results and choose
sensible ``--`` options — not to fully reproduce every equation from the
underlying papers. See :ref:`relationship-to-prior-work` for citations.

The overall approach
----------------------

Traditional GWAS tools (PLINK2, REGENIE) fit a predefined per-variant linear
or logistic model and report an association statistic per SNP. PostHoc
takes a different route:

1. **Train** a flexible model (currently a multilayer perceptron, MLP) to
   predict phenotype from the full genotype matrix at once, rather than one
   variant at a time.
2. **Attribute** the trained model's predictions back to individual input
   SNPs using Integrated Gradients, producing a per-SNP importance score.
3. **Aggregate** those importance scores across repeated model fits (PAL)
   to separate SNPs that are robustly important from those that are
   important only by chance in a single fit.
4. **Test significance** of the aggregated scores against a null
   distribution built from models trained on permuted phenotypes.

This can surface non-linear or interaction-driven signal that a strictly
additive linear/logistic model would miss, at the cost of needing many
more model fits and offering weaker formal guarantees than a calibrated
per-variant test.

The MLP model
----------------

:func:`~posthoc.models.mlp.build_mlp` builds a plain feed-forward network:
genotypes (plus any covariates) go in as raw allele counts, hidden layers
apply ``Linear -> [BatchNorm] -> activation -> [Dropout]`` per
:class:`~posthoc.models.base.MLPConfig`, and a single linear output head
produces one logit/scalar prediction. Missing genotype calls (``-9``) are
mean-imputed per-variant at dataset construction time
(:class:`~posthoc.models.base.GenotypeDataset`).

Training (:func:`~posthoc.models.utils.train_model`) uses:

* an ``80/20`` (configurable via ``--val-fraction``) stratified
  train/validation split for logistic tasks, plain random split for linear
  tasks;
* ``AdamW`` optimization with configurable learning rate and weight decay;
* binary cross-entropy loss for ``--logistic``, MSE for ``--linear``;
* early stopping on validation loss with a configurable ``--patience``, and
  the best-validation-loss weights are restored at the end of training.

Integrated Gradients attribution
------------------------------------

:func:`~posthoc.attribution.integrated_gradients.integrated_gradients_importance`
computes `Integrated Gradients <https://arxiv.org/abs/1703.01365>`_ (via
`Captum <https://captum.ai/>`_) for every held-out validation sample,
against either a zero baseline or a per-feature mean baseline
(``--ig-baseline``). For each SNP this produces one signed attribution
value per sample; PostHoc reduces that to:

* ``IMPORTANCE`` — mean absolute attribution across samples;
* a per-SNP p-value from a one-sample t-test of the signed attributions
  against zero, then Bonferroni-corrected across all variants.

This per-model, per-SNP importance vector is what feeds into PAL.

PAL and AMAS
--------------

PAL (**Post-hoc Attribution Loci**) and AMAS (its aggregated variant) are
implemented in :mod:`posthoc.attribution.pal`, following the
methodology of Yelmen et al. The goal is to turn "important according to
one model fit" into "important because it shows up robustly across many
independently trained models."

Given ``n_models`` independently trained models (different random seeds,
``--n-models``), each with its own Integrated Gradients importance vector
(restricted to case samples — see
:func:`posthoc.commands.pal._case_mas`, the "MAS" — Model Attribution
Score):

1. **Per-model threshold.** For each model, compute the
   ``--theta-percentile`` percentile of its own MAS distribution
   (default ``99.99``, i.e. only the extreme upper tail). SNPs above their
   model's threshold are that model's "detected" set.
2. **LD clumping.** For each model's detected set, expand it to include
   neighboring SNPs (within ``--ld-window`` positions) that are in high LD
   (``|r| > --ld-r-threshold``) with a detected SNP. This avoids penalizing
   a truly-causal SNP just because a slightly different but tightly-linked
   SNP was the one that happened to cross the threshold in a given fit.
3. **PAL_Common.** The intersection of detected sets across *all* models —
   SNPs independently flagged in every single fit.
4. **MU and AMAS.** ``MU`` is the SNP's MAS averaged across all models.
   ``AMAS`` starts as ``MU``, then for any SNP above the *global* threshold
   (mean of the per-model thresholds), it is down-weighted by
   ``occurrence_count / n_models`` — how often that SNP (or an LD neighbor)
   appeared in a model's detected set after clumping. A SNP that is
   important in every model keeps its full ``MU``; one that spikes in only
   one model out of ten gets scaled down by ``~0.1``.
5. **PAL_AMAS.** SNPs whose (possibly down-weighted) ``AMAS`` still exceeds
   the global threshold. This is the primary discovery set reported in the
   ``pal`` output.

Null model and significance testing
---------------------------------------

To assess whether a PAL_AMAS hit is more extreme than chance,
:mod:`posthoc.attribution.significance` builds a null distribution:

1. Train ``--n-null-models`` additional models, identical in every way
   except the phenotype labels are randomly permuted per model.
2. Compute each null model's MAS the same way as for real models, and pool
   every null MAS value across all null models and all SNPs.
3. Fit a half-normal distribution to this pooled null MAS pool
   (:func:`~posthoc.attribution.significance.fit_null_halfnorm`) — MAS
   values are non-negative and concentrated near zero under the null, which
   the half-normal approximates.
4. For ``--n-bootstrap`` iterations, sample a synthetic null MAS matrix from
   the fitted half-normal (matching the real matrix's shape and rank
   structure), recompute PAL/AMAS on that synthetic matrix, and count how
   often the synthetic AMAS at each PAL_AMAS SNP exceeds the real observed
   AMAS value there.
5. The empirical p-value is that exceedance count divided by
   ``n_bootstrap * n_snps``.

This gives one p-value per PAL_AMAS SNP (``P_VALUE`` in the output), left
as ``NaN`` for SNPs not in PAL_AMAS.

Practical implications for choosing options
----------------------------------------------

* ``--n-models`` / ``--n-null-models`` directly trade off compute time
  against how well PAL_Common/PAL_AMAS and the null distribution are
  estimated. The examples in :doc:`quickstart` use ``10``/``10`` as a
  reasonable starting point for exploration; published-quality analyses
  will likely want more.
* ``--theta-percentile`` controls how strict the per-model detection
  threshold is. ``99.99`` (default) is strict; the underlying paper also
  discusses ``99.95`` as a more relaxed alternative that will yield a
  larger PAL_AMAS set at the cost of more false positives.
* ``--ld-window`` / ``--ld-r-threshold`` should be set with your marker
  density and population LD structure in mind — a window that's too narrow
  will fail to clump truly-linked variants together, inflating apparent
  inconsistency across models.

.. _relationship-to-prior-work:

Relationship to prior work
-----------------------------

PostHoc's PAL/AMAS methodology builds on:

   Yelmen, B., Alver, M., Estonian Biobank Research Team, Jay, F., & Milani,
   L. (2024). *Interpreting artificial neural networks to detect
   genome-wide association signals for complex traits*. arXiv:2407.18811.

PostHoc is an independent, modular software implementation built around
this methodology — adding PLINK2-native genotype I/O, phenotype
simulation, a command-line interface, and GWAS-style tabular outputs — and
is not affiliated with the original authors.
