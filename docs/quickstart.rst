Quickstart
===========

This walks through a full PostHoc analysis on a simulated phenotype, from
data preparation to locus discovery. It mirrors the workflow used in
PostHoc's own testing.

Prerequisites
---------------

* PostHoc installed with the ``genotype`` extra (see :doc:`installation`).
* A PLINK2 fileset to work with. If you don't have one handy, the
  repository ships ``scripts/prepare_data.sh``, which downloads a public
  1000 Genomes-derived chromosome 22 region and QC-filters it with
  ``plink2`` (requires ``plink2`` on ``PATH`` and network access):

  .. code-block:: bash

     bash scripts/prepare_data.sh
     # produces datasets/data/processed/chr22_subset.{pgen,pvar,psam}

The examples below assume that resulting prefix,
``datasets/data/processed/chr22_subset``. Substitute your own ``--pfile``
prefix throughout.

1. Simulate a phenotype with known causal SNPs
-------------------------------------------------

Rather than needing real phenotypes with known ground truth, PostHoc can
simulate one on top of real genotypes so you can validate discovery
accuracy directly. Here we make SNP indices ``10`` and ``25`` additively
causal and add a two-way interaction between them:

.. code-block:: bash

   posthoc simulate-pheno \
     --pfile datasets/data/processed/chr22_subset \
     --additive 10 0.6 \
     --additive 25 -0.4 \
     --interaction2 10 25 0.8 \
     --logistic \
     --heritability 0.5 \
     --prevalence 0.1 \
     --out outputs/simulated.pheno

This writes ``outputs/simulated.pheno`` (the phenotype table) and
``outputs/simulated.causal.txt`` (ground-truth causal variant IDs) — see
:doc:`cli/simulate_pheno` for every option.

2. Sanity-check with a linear baseline
------------------------------------------

Before training neural models, it's worth checking what a standard L1
logistic regression baseline can already recover, across a few
regularization strengths:

.. code-block:: bash

   posthoc baseline \
     --pfile datasets/data/processed/chr22_subset \
     --pheno outputs/simulated.pheno \
     --pheno-name PHENO1 \
     --maf 0.01 \
     --geno 0.05 \
     --val-fraction 0.2 \
     --seed 42

This prints train/validation loss, validation AUC, and the number of
non-zero coefficients for ``C in (0.001, 0.01, 0.1, 1.0)`` — no output file
is written. See :doc:`cli/baseline`.

3. Train an MLP and compute SNP attributions
------------------------------------------------

.. code-block:: bash

   posthoc attribute \
     --pfile datasets/data/processed/chr22_subset \
     --pheno outputs/simulated.pheno \
     --pheno-name PHENO1 \
     --logistic \
     --hidden-dims 256,64 \
     --epochs 200 \
     --patience 15 \
     --n-steps 50 \
     --ig-baseline mean \
     --out outputs/ig_results.glm

This trains one MLP, runs Integrated Gradients on the held-out validation
split, and writes a PLINK-``.glm``-style TSV to ``outputs/ig_results.glm``
(see :doc:`outputs`). See :doc:`cli/attribute` for the full option list,
and :doc:`concepts` for what the model and attribution step are actually
doing.

4. Run PAL for robust, multi-model locus discovery
--------------------------------------------------------

A single model's attribution can be noisy. ``pal`` trains many models on
real labels and many more on permuted (null) labels, then aggregates:

.. code-block:: bash

   posthoc pal \
     --pfile datasets/data/processed/chr22_subset \
     --pheno outputs/simulated.pheno \
     --pheno-name PHENO1 \
     --logistic \
     --n-models 10 \
     --n-null-models 10 \
     --theta-percentile 99.99 \
     --n-bootstrap 100 \
     --out outputs/pal.tsv

This is the most compute-intensive command — it trains
``n_models + n_null_models`` MLPs end to end. Expect this step to take
substantially longer than ``attribute``. See :doc:`cli/pal` and
:doc:`concepts` for what ``PAL_Common``/``PAL_AMAS``/``P_VALUE`` mean.

5. Compare against ground truth
-----------------------------------

Since the phenotype was simulated, you can check how well PAL recovered
the causal SNPs:

.. code-block:: bash

   # causal variant IDs from step 1
   cat outputs/simulated.causal.txt

   # PAL_AMAS discoveries
   awk -F'\t' '$9=="True" {print $3}' outputs/pal.tsv

(Column 9 is ``IN_PAL_AMAS`` — see :doc:`outputs` for the full column
layout, and adjust the field index if you've changed the output columns.)

Next steps
------------

* :doc:`cli/index` — full CLI reference for all four commands.
* :doc:`concepts` — the statistical/ML machinery behind attribution, PAL,
  and AMAS.
* :doc:`data_formats` — details on genotype/phenotype/covariate file
  requirements and QC filters.
* :doc:`api/index` — Python API reference, if you want to script against
  PostHoc's internals directly rather than through the CLI.
