simulate-pheno
================

Simulate a phenotype from real genotypes using additive, dominant,
recessive, and interaction effects, following the liability-threshold model
described in :doc:`../concepts`.

.. click:: posthoc.commands.simulate_pheno:simulate_pheno
   :prog: posthoc simulate-pheno
   :nested: full

Effect term encoding
-----------------------

* ``--additive INDEX EFFECT`` — contributes ``genotype * EFFECT`` to the
  liability, where ``genotype`` is the ``0``/``1``/``2`` allele count (or
  ``-1``/``0``/``1`` if ``--recode-centered`` is set).
* ``--dominant INDEX EFFECT`` — contributes ``EFFECT`` only where the
  (centered) genotype equals ``-1``; intended for use with
  ``--recode-centered``.
* ``--recessive INDEX EFFECT`` — contributes ``EFFECT`` only where the
  (centered) genotype equals ``1``.
* ``--interaction2 INDEX_I INDEX_J EFFECT`` — contributes
  ``EFFECT * genotype_i * genotype_j``.
* ``--interaction3 INDEX_I INDEX_J INDEX_K EFFECT`` — contributes
  ``EFFECT * genotype_i * genotype_j * genotype_k``.

All five options are repeatable — pass ``--additive`` (or any other) more
than once to add multiple independent terms. ``INDEX`` values are
zero-based positions into the variant order of the ``--pfile``, not
variant IDs. At least one causal term must be given across all five
options combined.

Exactly one of ``--logistic``/``--linear`` must be given. For
``--logistic``, the liability is thresholded at the ``--prevalence``
quantile to produce a binary phenotype; for ``--linear``, the liability
itself (with noise scaled to hit the target ``--heritability``) is the
phenotype.

Outputs
---------

See :doc:`../outputs` for the two files this command writes.
