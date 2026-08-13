Development
============

Setup
-------

.. code-block:: bash

   git clone https://github.com/Nifdi01/posthoc.git
   cd posthoc
   pip install -e ".[dev,genotype]"

This installs ``pytest``, ``pytest-cov``, ``black``, and ``ruff`` alongside
the core and genotype-I/O dependencies.

Running tests
---------------

.. code-block:: bash

   python -m pytest

Tests live under ``tests/`` and cover genotype/phenotype/covariate readers,
QC filters, model training, and output writers. ``tests/conftest.py``
defines shared fixtures; ``tests/plink_ref.PHENO1.glm.logistic.hybrid`` is a
reference PLINK2 output used to cross-check PostHoc's output format.

Project layout
-----------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Path
     - Contents
   * - ``posthoc/commands/``
     - CLI entrypoints (one module per subcommand), thin orchestration over
       the modules below.
   * - ``posthoc/io/``
     - Genotype/phenotype/covariate readers and output writers.
   * - ``posthoc/models/``
     - The MLP model and training utilities.
   * - ``posthoc/attribution/``
     - Integrated Gradients, PAL, and significance-testing logic.
   * - ``posthoc/simulation/``
     - Phenotype simulation framework.
   * - ``posthoc/qc/``
     - MAF / missingness / LD-pruning filters.
   * - ``tests/``
     - Unit tests.
   * - ``scripts/prepare_data.sh``
     - Example script that downloads and QC-filters a chromosome 22 region
       for use in the :doc:`quickstart`.

Building the documentation locally
--------------------------------------

This documentation is built with `Sphinx <https://www.sphinx-doc.org/>`_
and hosted on `Read the Docs <https://readthedocs.org/>`_, configured via
``.readthedocs.yaml`` at the repository root. To build it locally:

.. code-block:: bash

   pip install -e ".[genotype]"
   pip install -r docs/requirements.txt
   sphinx-build -b html docs docs/_build/html

The Python API reference pages under :doc:`api/index` use
``sphinx.ext.autodoc`` and therefore import ``posthoc`` at build time — the
``genotype`` extra must be installed (or at least ``pgenlib``) for
``posthoc.io.genotype_reader`` to import successfully.

Coding conventions
---------------------

* Formatting: ``black``.
* Linting: ``ruff``.
* Type hints are used throughout; most dataclasses (e.g.
  :class:`~posthoc.models.base.TrainConfig`,
  :class:`~posthoc.attribution.pal.PALConfig`) act as the de facto
  configuration schema for their corresponding function.

Contributing
--------------

Issues and pull requests are welcome on the
`GitHub repository <https://github.com/Nifdi01/posthoc>`_. If you're adding
a new model or attribution method, note that the CLI currently gates
``--model``/``--attribution`` behind ``click.Choice`` allow-lists
(see :doc:`cli/attribute`) — extending those is the natural entry point.
