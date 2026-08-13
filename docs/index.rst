PostHoc documentation
=====================

**PostHoc** is a Python toolkit for post-hoc variant attribution on
GWAS-scale genotype data. It trains neural models on PLINK2 genotype
matrices and produces PLINK-like output tables that can be compared or
merged with standard GWAS pipelines.

PostHoc builds upon the neural-network attribution framework introduced by
Yelmen et al. for identifying genome-wide association signals from
artificial neural networks. In particular, PostHoc implements and extends
the PAL (Post-hoc Attribution Loci) analysis described in that work within
a modular, command-line framework designed for reproducible analysis of
genotype data. See :doc:`concepts` for details and citation.

The project currently includes:

* phenotype simulation on real genotype matrices
* baseline sparse logistic regression benchmarking
* Integrated Gradients SNP attribution
* PAL (Post-hoc Attribution Loci) discovery with null-model significance testing
* PLINK2-compatible genotype input and GWAS-style outputs
* repeated-model analysis for robust locus discovery

Get started with :doc:`installation` and :doc:`quickstart`.

.. toctree::
   :maxdepth: 2
   :caption: User guide

   installation
   quickstart
   data_formats
   outputs
   concepts

.. toctree::
   :maxdepth: 2
   :caption: Reference

   cli/index
   api/index

.. toctree::
   :maxdepth: 1
   :caption: Project

   development

Useful links
------------

- `Project repository <https://github.com/Nifdi01/posthoc>`_
- `Issue tracker <https://github.com/Nifdi01/posthoc/issues>`_
- `README <https://github.com/Nifdi01/posthoc/blob/main/README.md>`_

License
--------

PostHoc is distributed under the Apache License — see
`LICENSE <https://github.com/Nifdi01/posthoc/blob/main/LICENSE>`_ in the
repository.
