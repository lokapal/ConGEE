**Random Datasets Generator for FDR evaluation:**`generate_random.py` generates randomized gene annotations and genomic domain boundaries for empirical false discovery rate (FDR) estimation in gene-expression concordance analysis. It implements a null model without an explicit concordance-generating mechanism. The generator preserves strand-specific gene counts and prevents overlaps within each strand, but genes on opposite strands may overlap. 

**Gene coordinates:** Gene annotations are read from a BED5 file. For each chromosome, the generator preserves the observed gene count on each strand and derives the coordinate range, gene-length range, and expression-score range from the input annotation.

Gene lengths are sampled uniformly from the observed range. Gene positions are generated independently for the positive and negative strands using sorted uniform random positions, ensuring non-overlapping genes within each strand.

**Expression values.** Two expression models are available:

- `uniform`: expression scores are sampled independently from a uniform distribution over the observed chromosome-specific range.

- `original`: expression scores are randomly sampled from the observed scores, without replacement when possible and with replacement otherwise.

**Domain geometry:** Domain sizes follow a normal distribution with a specified mean and a standard deviation equal to 2.5% of the mean. Inter-domain gaps follow one of two distributions:

- `normal`: normally distributed gaps with a standard deviation equal to 5% of the specified mean.

- `uniform`: uniformly distributed gaps between 1 bp and the specified maximum.

Domains extending beyond chromosome boundaries are discarded rather than truncated. Mitochondrial and other non-canonical chromosomes are excluded.

**Requirements:** Python 3.9 or later, NumPy, pandas

**Usage:** The input must be a tab-separated BED5 gene annotation with the following columns: Chromosome, Start coordinate, End Coordinate, Expression score, Strand (+ or -). Usage example to generate 1000 datasets with mammalian TAD-like parameters:

```
python3 generate_random.py \
   --mean-domain-size 1000000 --mean-gap 40000 \
   --gap-mode normal --expression uniform \
   --n-files 10000 --cores 16 --outdir random.TAD.1Mbp
```

To use uniformly distributed inter-domain gaps, change parameters appropriately:`--gap-mode uniform --mean-gap 5000`.

**Command-line arguments**

| Argument               | Default                 | Description                                                   |
| ---------------------- | ----------------------- | ------------------------------------------------------------- |
| `--genes`              | Required                | Input gene annotation in BED5 format                          |
| `--expression`         | `uniform`               | Expression model: `uniform` or `original`                     |
| `--mean-domain-size`   | `1000000`               | Mean domain size in base pairs                                |
| `--mean-gap`           | `40000`                 | Mean gap (`normal`) or maximum gap (`uniform`), in base pairs |
| `--gap-mode`           | `normal`                | Gap distribution: `normal` or `uniform`                       |
| `--n-files`            | `100`                   | Number of replicate pairs                                     |
| `--outdir`             | `./random_datasets`     | Output directory                                              |
| `--tag`                | Automatically generated | Filename label                                                |
| `--seed`               | ``                      | Base random seed, by default defined by urandom               |
| `--cores`              | CPU count minus one     | Number of worker processes                                    |

**Output:** Each replicate produces a pair of BED files:

- `genes_<tag>_<replicate>.bed` — randomized gene coordinates, expression scores, and strand annotations in BED5 format.

- `domains_<tag>_<replicate>.bed` — randomized domain boundaries in BED3 format.

Replicate indices are zero-padded to four digits, starting at `0001`. A summary file, `summary_<tag>.tsv`, records the number of genes and domains, observed domain-size statistics, parameter settings, and output file paths for each replicate.

**Reproducibility:** Each replicate's thread is initialized with seed value = `seed + i` (where i - thread number). Initial seed value can be set via command-line or taken from OS urandom() interface (default behaviour)
