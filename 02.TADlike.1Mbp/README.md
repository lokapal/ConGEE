# Synthetic Genomic Datasets Generator

`generate_datasets.py` generates paired gene-expression and genomic-domain BED datasets for controlled concordance modelling. It supports reproducible simulation of within-domain expression similarity across multiple independent replicates.

**Domain geometry.** Domain sizes and inter-domain gaps are sampled independently from normal distributions. Their standard deviations are fixed at 2.5% and 5% of their respective means. Negative draws are bounded below by 1 bp, and domains extending beyond chromosome boundaries are discarded.

**Gene assignment.** Genes are assigned to domains according to following modes:

- <u>proportional</u>: gene expression is distributed proportionally across all domains intersecting the gene. A gene spanning multiple domains contributes to each of them according to the fraction of its length contained within the respective domain;

- <u>start</u>: gene expression is determined exclusively by the domain containing the transcription start site;

- <u>midpoint</u>: gene expression is midpoint-assigned, meaning that each gene is associated with the domain containing its genomic midpoint;

- <u>end</u>: gene expression is determined exclusively by the domain containing the transcription termination site.

Genes outside domain boundaries retain independently sampled background expression.

**Expression model.** Two parameters control within-domain expression:

* **Concordance (C)**: probability that a populated domain receives a shared base expression level, with (C in [0,1.0]).

* **Noise (H)**: half-width of the uniform relative perturbation applied to genes in active domains. The maximum relative expression spread within an active domain is (2H). Background expression follows a log-normal distribution.

**Requirements:** Python 3.10 or later, NumPy, pandas

Gene annotations can be supplied locally in BED5, BED6, or supported Ensembl-style BED format. Alternatively, annotations and chromosome sizes are retrieved from the UCSC Genome Browser REST API for the specified genome assembly.

**Usage:** Generate 1000 TAD-like (mean domain size=1Mbp, mean inter-domain gap size=40kbp) domain replicates with target expression concordance (C=0.8) and noise half-width (H=0.25):

```
        python3 generate_datasets.py \
            --genes hg38.HEK293.bed \
            --mean-domain-size 1000000 \
            --mean-gap 40000 \
            --concordance 0.8 \
            --noise 0.25 \
            --n-files 1000 \
            --cores 32 \
            --outdir "datasets.025_08"

```

## Main parameters

| Argument             | Default                  | Description                                          |
| -------------------- | ------------------------ | ---------------------------------------------------- |
| `--genes`            | UCSC annotation          | Local gene annotation file                           |
| `--genome`           | `hg38`                   | Genome assembly                                      |
| `--concordance`      | `0.5`                    | Probability of activating each populated domain      |
| `--noise`            | `0.05`                   | Intra-domain noise half-width; alias: `--noise-half` |
| `--mean-domain-size` | `115000`                 | Mean domain size in bp                               |
| `--mean-gap`         | `1000`                   | Mean inter-domain gap in bp                          |
| `--n-files`          | `10`                     | Number of replicate pairs                            |
| `--outdir`           | `./datasets`             | Output directory                                     |
| `--tag`              | Derived from (C) and (H) | Output filename label                                |
| `--seed`             | Randomly initialized     | By default defined by os.urandom()                   |
| `--cores`            | CPU count minus one      | Number of worker processes                           |
| `--mode`             | `middle`                 | Gene assignment: `proportional, start, middle, end`  |
| `--bg-mu`            | `2.5`                    | Background log-normal mean                           |
| `--bg-sigma`         | `1.5`                    | Background log-normal standard deviation             |

**Output:** Each replicate comprises of two files:

* `domains_<tag>_<replicate>.bed` — BED3 domain coordinates (`chrom`, `start`, `end`).

* `genes_<tag>_<replicate>.bed` — BED5 gene coordinates, synthetic expression scores, and strand (`chrom`, `start`, `end`, `expression`, `strand`).

The generator also writes `summary_<tag>.tsv`, containing simulation parameters, domain statistics, gene coverage, and observed concordance-related measures.

**Reproducibility:** Each replicate uses an independent random-number generator initialized with `seed + i`, where (i) is the zero-based replicate index. 

## 
