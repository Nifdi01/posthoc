Output formats
===============

``simulate-pheno`` outputs
----------------------------

Two files are written:

``<out>``
   A whitespace/tab-delimited phenotype table (``#IID`` + the phenotype
   column, named per ``--pheno-name``), directly usable as the ``--pheno``
   argument to ``baseline``, ``attribute``, or ``pal``.

``<out with .causal.txt suffix>``
   One variant ID per line, the union of all SNPs referenced by any
   ``--additive``/``--dominant``/``--recessive``/``--interaction2``/``--interaction3``
   term. This is the ground truth used to score locus-discovery accuracy in
   the :doc:`quickstart`.

For example, ``--out outputs/simulated.pheno`` produces
``outputs/simulated.pheno`` and ``outputs/simulated.causal.txt``.

``attribute`` output (``.glm``-style TSV)
--------------------------------------------

A PLINK-``.glm``-flavored tab-separated table, one row per variant:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Column
     - Meaning
   * - ``#CHROM``, ``POS``, ``ID``, ``REF``, ``ALT``
     - Copied from the input ``.pvar``. Kept identical to PLINK2's column
       names/order so the file can be joined against real GWAS summary
       stats on these five columns.
   * - ``A1``
     - The tested allele; currently always equal to ``ALT``.
   * - ``TEST``
     - Model/attribution identifier, e.g. ``MLP_integrated_gradients``.
   * - ``IMPORTANCE``
     - Mean absolute Integrated Gradients attribution for that SNP across
       held-out validation samples.
   * - ``P_PERM``
     - Two-sided p-value from a one-sample t-test of the per-sample signed
       attribution values against zero (not a permutation test despite the
       column name — see :func:`~posthoc.attribution.integrated_gradients.integrated_gradients_importance`).
   * - ``P_CORRECTED``
     - ``P_PERM`` after Bonferroni correction across all tested variants.
   * - ``N``
     - Number of samples the attribution was computed on (the held-out
       validation split).

``pal`` output (TSV)
----------------------

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Column
     - Meaning
   * - ``#CHROM``, ``POS``, ``ID``, ``REF``, ``ALT``
     - As above.
   * - ``MU``
     - Mean Model Attribution Score (MAS) across all real-label models
       (``mu`` in :class:`~posthoc.attribution.pal.PALResult`).
   * - ``AMAS``
     - Aggregated MAS: ``MU`` down-weighted by how consistently the variant
       (or an LD-linked neighbor) was independently flagged across models.
       See :doc:`concepts` for the full definition.
   * - ``IN_PAL_COMMON``
     - ``True`` if the variant exceeded its per-model threshold in
       **every** real-label model.
   * - ``IN_PAL_AMAS``
     - ``True`` if the variant's ``AMAS`` score exceeds the global
       threshold — the primary PAL discovery set.
   * - ``P_VALUE``
     - Bootstrap-estimated p-value against the fitted null distribution,
       populated only for ``IN_PAL_AMAS`` variants (``NaN`` elsewhere).
   * - ``N_MODELS``
     - Number of real-label models trained (``--n-models``).

Both writers validate that the per-variant array lengths match the number
of rows in the ``.pvar``-derived variant table before writing, and raise
``ValueError`` on mismatch.
