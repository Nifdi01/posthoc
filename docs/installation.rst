Installation
============

Requirements
------------

* Python 3.10 or later
* ``pip``
* `PLINK2 <https://www.cog-genomics.org/plink/2.0/>`_ on your ``PATH`` — only
  required if you plan to use ``--indep-pairwise`` LD pruning

PostHoc's genotype I/O layer reads PLINK2 ``.pgen``/``.pvar``/``.psam``
filesets directly via `pgenlib <https://github.com/chrchang/plink-ng>`_, so
you do not need PLINK2 installed just to run commands — only for the optional
LD-pruning QC step, which shells out to the ``plink2`` binary.

Install from source
--------------------

PostHoc is not yet published on PyPI. Install it directly from GitHub:

.. code-block:: bash

   git clone https://github.com/Nifdi01/posthoc.git
   cd posthoc
   pip install -e .

This installs the core dependencies (PyTorch, NumPy, pandas, scikit-learn,
Captum, statsmodels, Click) and exposes the ``posthoc`` command on your
``PATH``.

Optional dependency sets
-------------------------

PostHoc defines two extras in ``pyproject.toml``:

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Extra
     - Installs
     - When you need it
   * - ``genotype``
     - ``pgenlib``, ``pandas-plink``
     - Always, in practice — ``pgenlib`` is what actually reads ``.pgen``
       files. Install this unless you already have it.
   * - ``dev``
     - ``pytest``, ``pytest-cov``, ``black``, ``ruff``
     - Contributing to PostHoc or running the test suite.

Install either (or both) alongside the editable install:

.. code-block:: bash

   # genotype I/O (recommended for any real usage)
   pip install -e .[genotype]

   # development tools
   pip install -e .[dev]

   # both
   pip install -e ".[genotype,dev]"

Verifying the install
----------------------

.. code-block:: bash

   posthoc --help

should list the four subcommands: :doc:`cli/simulate_pheno`,
:doc:`cli/baseline`, :doc:`cli/attribute`, and :doc:`cli/pal`.

GPU support
-----------

All model-training commands (``attribute``, ``pal``) accept a ``--device``
option (default ``cpu``). If you have a CUDA-capable GPU and a
CUDA-enabled PyTorch build installed, pass ``--device cuda`` to train faster.
PostHoc does not manage the CUDA/PyTorch install for you — follow the
`official PyTorch install instructions <https://pytorch.org/get-started/locally/>`_
for your platform first.

Next steps
----------

Continue to :doc:`quickstart` for an end-to-end example, or
:doc:`data_formats` for details on the expected genotype/phenotype/covariate
file formats.
