pal
====

Run repeated-model PAL (Post-hoc Attribution Loci) analysis: train many
models on real labels and many on permuted labels, derive ``PAL_Common``
and ``PAL_AMAS`` discovery sets, and estimate ``PAL_AMAS`` p-values via
null-distribution bootstrapping. This is PostHoc's primary locus-discovery
command — see :doc:`../concepts` for the full methodology.

.. click:: posthoc.commands.pal:pal
   :prog: posthoc pal
   :nested: full

Behavior notes
----------------

* This command trains ``--n-models + --n-null-models`` MLPs sequentially —
  expect runtime roughly that multiple of a single :doc:`attribute` run.
  Consider ``--device cuda`` if available (see :doc:`../installation`).
* Model attribution scores (MAS) for both real and null models are
  computed **only on case samples** (``phenotype == 1``); a
  ``click.UsageError`` is raised if a training split has no case samples at
  all. This means ``pal`` is only meaningful for ``--logistic`` in
  practice, even though ``--linear`` is accepted at the training-config
  level.
* ``--seed`` is the base seed: real-label model ``i`` uses ``seed + i``,
  and null-label model ``i`` uses ``seed + n_models + i``, so every model
  in the run is trained with a distinct seed.
* ``pal`` exposes additional MLP training knobs not present on
  ``attribute``: ``--activation``, ``--weight-decay``, ``--data-dropout``.
  These default to the same values ``attribute`` uses implicitly
  (``relu``, ``1e-4``, off), but are surfaced here because tuning them can
  matter more when training many models for aggregation.
* ``--ld-window`` and ``--ld-r-threshold`` control the LD-clumping step
  used when computing ``PAL_AMAS`` occurrence counts — see :doc:`../concepts`.

Example
---------

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
