# PostHoc

PostHoc is a Python toolkit for post-hoc variant attribution on GWAS-scale genotype data. It trains neural models on PLINK2 genotype matrices and produces PLINK-like output tables that can be compared or merged with standard GWAS pipelines.

PostHoc builds upon the neural-network attribution framework introduced by Yelmen et al. [1] for identifying genome-wide association signals from artificial neural networks. In particular, PostHoc implements and extends the PAL (Post-hoc Attribution Loci) analysis described in that work within a modular, command-line framework designed for reproducible analysis of genotype data.

The project currently includes:

- phenotype simulation on real genotype matrices
- baseline sparse logistic regression benchmarking
- integrated gradients SNP attribution
- PAL (Post-hoc Attribution Loci) discovery with null-model significance testing
- PLINK2-compatible genotype input and GWAS-style outputs
- repeated-model analysis for robust locus discovery

## Table of contents

- [On PostHoc](#on-posthoc)
- [Relationship to prior work](#relationship-to-prior-work)
- [Installation](#installation)
- [Input data expectations](#input-data-expectations)
- [Command-line interface](#command-line-interface)
  - [`simulate-pheno`](#simulate-pheno)
  - [`baseline`](#baseline)
  - [`attribute`](#attribute)
  - [`pal`](#pal)
- [Outputs](#outputs)
- [Typical workflow](#typical-workflow)
- [Development](#development)
- [Project layout](#project-layout)
- [License](#license)
- [Citation](#citation)

## On PostHoc

Traditional GWAS tools report association statistics from predefined linear or logistic models. PostHoc explores a complementary approach:

1. train flexible predictive models (currently MLP)
2. compute feature attribution scores for SNPs
3. convert those attributions into GWAS-friendly tabular outputs
4. identify robust loci across repeated model fits (PAL)

This is useful for exploring non-linear or interaction-heavy genetic architectures, especially with simulated phenotypes where ground truth is available.

## Relationship to prior work

PostHoc builds upon the attribution-based framework introduced by Yelmen et al. [1]:

> Yelmen, B., Alver, M., Estonian Biobank Research Team, Jay, F., & Milani, L. (2024). *Interpreting artificial neural networks to detect genome-wide association signals for complex traits*. arXiv:2407.18811.

The PAL/AMAS methodology implemented in PostHoc is based on this prior work. PostHoc is intended as a modular and extensible software framework around this methodology, with PLINK2-native genotype input, phenotype simulation, configurable neural models, attribution analysis, GWAS-style outputs, and command-line workflows.

## Installation

### Requirements

- Python 3.10+
- `pip`
- PLINK2 on `PATH` if you use `--indep-pairwise` QC pruning

### Install from source

```bash
git clone https://github.com/Nifdi01/posthoc.git
cd posthoc
pip install -e .
````

### Optional dependency sets

```bash
# development tools
pip install -e .[dev]

# genotype ecosystem extras (pgenlib/pandas-plink)
pip install -e .[genotype]
```

## Input data expectations

### Genotype input (`--pfile`)

Commands that read genotype data expect a PLINK2 prefix with these files:

* `<prefix>.pgen`
* `<prefix>.pvar`
* `<prefix>.psam`

### Phenotype input (`--pheno`)

A whitespace-delimited PLINK-style phenotype table with sample IDs and at least one phenotype column.

* accepted sample ID column names: `IID` or `#IID`
* phenotype defaults to first non-ID column unless `--pheno-name` is provided
* for logistic tasks, labels must be either `0/1` or PLINK-style `1/2` (automatically recoded to `0/1`)

### Covariate input (`--covar`)

Optional whitespace-delimited table with `IID`/`#IID` plus one or more numeric covariate columns.

Samples are aligned by ID across genotype/phenotype/covariates, and samples with missing required values are dropped.

## Command-line interface

After installation, use:

```bash
posthoc --help
posthoc <command> --help
```

Available commands:

* `simulate-pheno`
* `baseline`
* `attribute`
* `pal`

---

## `simulate-pheno`

Simulate a phenotype from real genotypes using additive/dominant/recessive and interaction effects.

Example:

```bash
posthoc simulate-pheno \
  --pfile datasets/data/processed/chr22_subset \
  --additive 10 0.6 \
  --additive 25 -0.4 \
  --interaction2 10 25 0.8 \
  --logistic \
  --heritability 0.5 \
  --prevalence 0.1 \
  --out outputs/simulated.pheno
```

This writes:

* phenotype file at `--out`
* causal variant IDs at `<out>.causal.txt`

---

## `baseline`

Run an L1-style logistic regression baseline (with QC and optional covariates) and print train/validation metrics for several regularization values.

Example:

```bash
posthoc baseline \
  --pfile datasets/data/processed/chr22_subset \
  --pheno outputs/simulated.pheno \
  --pheno-name PHENO1 \
  --maf 0.01 \
  --geno 0.05 \
  --val-fraction 0.2 \
  --seed 42
```

---

## `attribute`

Train an MLP and compute per-SNP integrated gradients attribution.

Example:

```bash
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
```

---

## `pal`

Run repeated-model PAL analysis:

* train `n_models` on real labels
* train `n_null_models` on permuted labels
* derive PAL_Common and PAL_AMAS loci
* estimate PAL_AMAS p-values from null distribution bootstrapping

Example:

```bash
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
```

## Outputs

### `attribute` output (`.glm`-style TSV)

Columns:

* `#CHROM`, `POS`, `ID`, `REF`, `ALT`
* `A1`
* `TEST`
* `IMPORTANCE`
* `P_PERM`
* `P_CORRECTED`
* `N`

The first five columns are intentionally PLINK-join friendly.

### `pal` output (TSV)

Columns:

* `#CHROM`, `POS`, `ID`, `REF`, `ALT`
* `MU`
* `AMAS`
* `IN_PAL_COMMON`
* `IN_PAL_AMAS`
* `P_VALUE`
* `N_MODELS`

## Typical workflow

1. Prepare or subset PLINK2 genotype data (see `scripts/prepare_data.sh` for one example).
2. Simulate phenotype with known causal structure (`simulate-pheno`).
3. Train/inspect baseline model performance (`baseline`).
4. Run integrated gradients attribution (`attribute`).
5. Run robust multi-model locus detection (`pal`).
6. Compare discovered loci against `<simulated>.causal.txt` ground truth.

## Development

Install dev dependencies:

```bash
pip install -e .[dev]
```

Run tests:

```bash
python -m pytest
```

## Project layout

* `posthoc/commands/` — CLI entrypoints
* `posthoc/io/` — genotype/phenotype/covariate readers and output writers
* `posthoc/models/` — MLP model and training utilities
* `posthoc/attribution/` — integrated gradients, PAL, significance logic
* `posthoc/simulation/` — phenotype simulation framework
* `posthoc/qc/` — MAF/missingness/LD-pruning filters
* `tests/` — unit tests

## License

Apache (see [`LICENSE`](LICENSE)).

## Citation

If you use PostHoc in your research, please cite the PostHoc software as well as the methodological work on which its PAL/AMAS analysis builds.

### PostHoc

```bibtex
@software{guliyev2026posthoc,
  author  = {Guliyev, Nifdi},
  title   = {PostHoc: Post-hoc variant attribution for neural networks in GWAS},
  year    = {2026},
  url     = {https://github.com/Nifdi01/posthoc}
}
```

### Methodological foundation

```bibtex
@misc{yelmen_interpreting_2024,
  title         = {Interpreting artificial neural networks to detect genome-wide association signals for complex traits},
  author        = {Yelmen, Burak and Alver, Maris and Team, Estonian Biobank Research and Jay, Flora and Milani, Lili},
  year          = {2024},
  month         = jul,
  publisher     = {arXiv},
  eprint        = {2407.18811},
  archivePrefix = {arXiv},
  primaryClass  = {cs, q-bio},
  url           = {https://arxiv.org/abs/2407.18811}
}
```
