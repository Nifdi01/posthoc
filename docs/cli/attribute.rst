attribute
==========

Train a single MLP and compute per-SNP Integrated Gradients attribution,
writing a PLINK-``.glm``-style TSV. See :doc:`../concepts` for what the
model and attribution step are doing, and :doc:`../outputs` for the output
column layout.

.. click:: posthoc.commands.attribute:attribute
   :prog: posthoc attribute
   :nested: full

Behavior notes
----------------

* Exactly one of ``--logistic``/``--linear`` must be given.
* ``--model`` currently only accepts ``mlp``; ``--attribution`` currently
  only accepts ``integrated_gradients``. Both are exposed as explicit
  choices to keep the CLI stable if/when additional models or attribution
  methods are added.
* ``--hidden-dims`` is a comma-separated list of integers, e.g.
  ``256,64`` for a two-hidden-layer MLP with 256 then 64 units.
* Attribution is computed only on the **held-out validation split** used
  during training (``train_result.val_idx``), not the full dataset — so
  ``N`` in the output reflects the validation split size, and
  ``--val-fraction`` indirectly controls how many samples the attribution
  statistics are based on.
* ``--ig-baseline mean`` uses the per-feature mean of the validation
  design matrix as the Integrated Gradients reference input, instead of an
  all-zeros baseline. This is generally more appropriate for genotype data,
  where ``0`` is a meaningful allele count rather than a neutral reference.

Example
---------

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
