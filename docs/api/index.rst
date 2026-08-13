Python API reference
======================

PostHoc's CLI commands are thin wrappers around a regular importable Python
package. Every function and class documented here is available under the
``posthoc`` namespace and can be used directly in scripts or notebooks —
for example, to run a custom analysis loop around :func:`posthoc.attribution
.pal.compute_pal` without going through the command-line interface.

.. toctree::
   :maxdepth: 1

   io
   qc
   models
   attribution
   simulation

Package overview
-------------------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Module
     - Contents
   * - :mod:`posthoc.io`
     - Genotype (``.pgen``/``.pvar``/``.psam``), phenotype, and covariate
       readers; ``.glm``/PAL TSV writers.
   * - :mod:`posthoc.qc`
     - MAF, missingness, and LD-pruning filters.
   * - :mod:`posthoc.models`
     - The MLP model, its training loop, and shared training utilities.
   * - :mod:`posthoc.attribution`
     - Integrated Gradients attribution, PAL/AMAS aggregation, and
       null-model significance testing.
   * - :mod:`posthoc.simulation`
     - The additive/dominant/recessive/interaction phenotype simulation
       model.
   * - :mod:`posthoc.commands`
     - The Click command implementations documented in
       :doc:`../cli/index` — thin orchestration over the modules above.
