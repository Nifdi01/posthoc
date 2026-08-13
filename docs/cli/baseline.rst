baseline
=========

Run an L1-regularized logistic regression baseline (with QC filtering and
optional covariates) across several regularization strengths, and print
train/validation metrics for each. Useful as a fast sanity check before
investing in MLP training via :doc:`attribute` or :doc:`pal`.

.. click:: posthoc.commands.baseline:baseline
   :prog: posthoc baseline
   :nested: full

Behavior notes
----------------

* The phenotype is always treated as binary/logistic — there is no
  ``--linear`` mode for ``baseline``.
* Missing genotype calls are mean-imputed per-variant using **training-split
  means only**, then applied to both train and validation folds.
* If ``--covar`` is given, covariates are concatenated to the genotype
  matrix as additional predictor columns.
* For each ``C`` in ``(0.001, 0.01, 0.1, 1.0)``, a
  ``sklearn.linear_model.LogisticRegression`` with ``l1_ratio=1``,
  ``solver="liblinear"`` is fit, and the script prints train loss,
  validation loss, validation AUC, and the number of non-zero
  coefficients.
* This command does not write an output file — results are printed to
  stdout only.

Example output
-----------------

.. code-block:: text

   n_train=403 n_val=101 n_features=6112
          C   train_loss   val_loss  val_auc  n_nonzero
      0.001       0.6512     0.6603    0.612          3
       0.01       0.5981     0.6120    0.671         41
        0.1       0.4872     0.5734    0.703        387
        1.0       0.3105     0.6488    0.681       1822
