Input data formats
===================

Every PostHoc command that touches real data (``baseline``, ``attribute``,
``pal``, and the genotype-loading half of ``simulate-pheno``) shares the same
three input types. This page documents what each one must look like and how
PostHoc aligns them.

Genotype input (``--pfile``)
------------------------------

PostHoc reads genotypes from a `PLINK2 <https://www.cog-genomics.org/plink/2.0/>`_
fileset, given as a path *prefix* (no extension). The prefix must resolve to
three existing files:

.. code-block:: text

   <prefix>.pgen   # genotype calls
   <prefix>.pvar   # variant metadata (CHROM, POS, ID, REF, ALT)
   <prefix>.psam   # sample metadata (sample IDs)

For example ``--pfile datasets/data/processed/chr22_subset`` expects
``chr22_subset.pgen``, ``chr22_subset.pvar``, and ``chr22_subset.psam`` in
that directory.

Genotypes are loaded into memory as a dense ``(n_samples, n_variants)``
integer matrix with allele-count coding ``0``/``1``/``2`` and ``-9`` for
missing calls (PLINK2's convention). Missing calls are mean-imputed
per-variant immediately before model training or MAF/simulation
computations — see :class:`~posthoc.models.base.GenotypeDataset` and
:func:`~posthoc.qc.filters.compute_maf`.

The ``.psam`` sample-ID column may be named either ``IID`` or ``#IID``; the
``.pvar`` is parsed with any leading ``##`` header lines stripped, and only
the ``CHROM``, ``POS``, ``ID``, ``REF``, ``ALT`` columns are kept.

.. note::

   Because the whole genotype matrix is loaded into memory as ``int32``
   during read and cast down to ``int8`` afterward, very large filesets
   (genome-wide, biobank-scale) are not the target use case out of the box.
   Subset to a region or a variant panel first — see
   ``scripts/prepare_data.sh`` in the repository for a worked example that
   subsets to a chromosome 22 region.

Phenotype input (``--pheno``)
-------------------------------

A whitespace-delimited, PLINK-style table with one row per sample:

.. code-block:: text

   #IID     PHENO1
   HG00096  0
   HG00097  1
   HG00099  1

Rules:

* The sample ID column must be named ``IID`` or ``#IID``.
* The phenotype column defaults to the first non-ID column unless
  ``--pheno-name`` is given explicitly.
* For logistic tasks (``--logistic``), values must be coded either as
  ``0``/``1`` or PLINK-style ``1``/``2`` (control/case). ``1``/``2`` coding
  is detected automatically and recoded to ``0``/``1``; any other coding
  raises an error.
* Missing values (``-9``, ``NA``, ``NaN``, empty) are recognized and treated
  as missing during sample alignment (see below).

Covariate input (``--covar``)
--------------------------------

An optional whitespace-delimited table, same ID conventions as the
phenotype file, with one or more numeric covariate columns:

.. code-block:: text

   #IID     AGE  SEX  PC1     PC2
   HG00096  54   1    0.0123  -0.0041
   HG00097  61   0    0.0098  0.0012

All non-``FID``/``IID`` columns are treated as covariates and are
concatenated to the genotype matrix as extra input features for
model-training commands, or as additional predictors for the ``baseline``
logistic regression.

Sample alignment
-----------------

Before any modeling happens, PostHoc aligns samples across the genotype,
phenotype, and (if given) covariate inputs:

1. Intersect sample IDs across all provided sources.
2. Drop any sample with a missing phenotype value.
3. Drop any sample with a missing covariate value, if covariates are used.
4. Reorder the genotype matrix to match the aligned, filtered sample order.

If the intersection is empty, PostHoc raises an error rather than silently
proceeding with zero samples. The number of samples dropped at this step is
logged at ``INFO`` level, e.g.:

.. code-block:: text

   Sample alignment: dropped 12 / 504 samples (missing pheno/covar); 492 remain

Quality control filters
-------------------------

Three QC filters are available on every command that reads genotypes
(``baseline``, ``attribute``, ``pal``), applied in this order:

``--geno FLOAT`` (default ``1.0``, i.e. off)
   Drop variants with per-variant missingness above this threshold.

``--maf FLOAT`` (default ``0.0``, i.e. off)
   Drop variants with minor allele frequency below this threshold.

``--indep-pairwise WINDOW STEP R2`` (default: not applied)
   LD-prune variants by shelling out to ``plink2 --indep-pairwise WINDOW STEP
   R2`` against the original ``--pfile`` and keeping only variants in the
   resulting ``.prune.in`` list. Requires a ``plink2`` binary on ``PATH`` —
   see :doc:`installation`. Because this re-runs against the file on disk,
   it uses the *original* ``--pfile`` variant set, independent of any
   ``--maf``/``--geno`` filtering already applied in memory.

Each filter logs how many variants it dropped and how many remain; the QC
summary line looks like:

.. code-block:: text

   QC: dropped 1204 (MAF), 88 (missingness), 340 (LD pruning); 6112 variants remain

Output formats
---------------

See :doc:`outputs` for the column layouts produced by ``attribute`` and
``pal``.
