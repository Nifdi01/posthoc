Command-line reference
========================

After installation (see :doc:`../installation`), all functionality is
available through the ``posthoc`` command:

.. code-block:: bash

   posthoc --help
   posthoc <command> --help

.. click:: posthoc.cli:main
   :prog: posthoc
   :nested: none

The top-level ``--verbose`` flag enables debug-level logging for any
subcommand; place it before the subcommand name, e.g.
``posthoc --verbose attribute ...``.

Every subcommand that reads real genotype data (``baseline``, ``attribute``,
``pal``) shares the same ``--pfile``/``--pheno``/``--pheno-name``/``--covar``
and QC options (``--maf``, ``--geno``, ``--indep-pairwise``) — see
:doc:`../data_formats` for what those expect and how samples are aligned.

.. toctree::
   :maxdepth: 1
   :caption: Commands

   simulate_pheno
   baseline
   attribute
   pal
