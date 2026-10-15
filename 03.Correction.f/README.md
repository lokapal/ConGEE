# `Script to recalibrate expression dispersion detection`

`Calibrate.Dispersion.Table.py` estimates the expression dispersion correction factor *f* for a given genomic domain structure by MC simulations. *f* corrects the median relative expression range of concordant domains for the number of genes per domain, enabling the noise estimator in `ConGEE` to recover the dispersion amplitude *H*.
The script derives a structure-specific correction factor by simulating concordant domains using the supplied gene annotations and domain boundaries. Genes are assigned to domains by their midpoints, employing the genes-to-domains mapping  function from `ConGEE`. Domains containing fewer than two genes are excluded.
For each expression dispersion amplitude *H*, gene expression values are sampled from the uniform distribution U(1-*H*,1+*H*). The relative expression range is calculated for each multi-gene domain, and its median across domains is divided by *H* to obtain a replicate-specific correction factor. The simulation is repeated over a predefined grid of dispersions (from 0.05 to 0.5 with step 0.05).
The script reports the median correction factor and its [5%, 95%] interval at each *H*. A pooled factor is calculated as the median of the factors obtained across the simulated noise levels. The variation of *H* is also assessed to evaluate deviation from the assumption that a single correction factor represents the entire expression dispersion range.

**Dependencies:** numpy, pandas

**Input:**
- Gene annotations: BED5 file containing chromosome, start, end, score, and strand.
- Genomic domains: BED3 file containing chromosome, start, and end.
Both files must follow the formats used by `ConGEE` and be in the same chromosome notation (you cannot mix, *e.g.*, 2L and chr2L or chr11 and NC_000011.10) 

**Usage example**
```
python Calibrate.Dispersion.Table.py \
    --genes dm6_genes.bed --domains dm6_TADs.bed \
    --replicates 1000 \
    --output f_table_dm6.tsv
```

**Command-line options**
| Option           | Description                                                     | Default              |
| ---------------- | --------------------------------------------------------------- | -------------------- |
| `--genes`        | Path to the BED5 gene annotation file                           | Required             |
| `--domains`      | Path to the BED3 domain file                                    | Required             |
| `--h-grid`       | Noise-amplitude grid, specified as `START:STOP:STEP`            | `0.05:0.50:0.05`     |
| `--replicates`   | Number of independent simulation replicates per noise amplitude | `10000`              |
| `--label`        | Label identifying the domain structure                          | Domain filename stem |
| `--output`, `-o` | Output TSV file                                                 | `f_table.tsv`        |
| `--seed`         | Seed can be set for reproducibility                             | `os.urandom()`       |

**Output:** The output is a tab-separated table containing the simulated correction factors for individual noise amplitudes and a pooled calibration estimate. Its columns are:

- `label`: Domain structure identifier.
- `H`: Simulated noise amplitude or `pooled` for the aggregate estimate.
- `median_n`: Median number of genes per multi-gene domain.
- `n_domains`: Number of multi-gene domains included in the simulation.
- `replicates`: Number of simulation replicates per noise amplitude.
- `f_median`: Median correction factor.
- `f_low`, `f_high`: 5th and 95th percentiles for individual noise amplitudes; minimum and maximum simulated median factors, respectively, for the pooled row.

The script also reports the domain gene-count distribution, the pooled correction factor, its variation across dispersion amplitudes, and its difference from the correction factor in the existing calibration table. A warning is issued when the relative variation across dispersion amplitudes exceeds 3%, indicating that the assumption of a dispersion-independent correction factor may introduce bias.
The resulting correction factor *f* for each number of genes per domain, *N*, should be stored in the `CALIB_FAC` array in `ConGEE` or loaded from an external calibration file (see the `ConGEE` documentation).


