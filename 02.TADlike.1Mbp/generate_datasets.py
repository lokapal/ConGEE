#!/usr/bin/env python3.10
"""
generate_datasets.py
--------------------
Generate paired (genes BED + domains BED) datasets for concordance modelling.

For each replicate:
  1. Draw a new random domain layout across the genome using Normal distributions
     for both domain sizes and inter-domain gaps.
  2. Assign synthetic expression values to genes with a controlled level of
     within-domain concordance.

Concordance definition
  A domain is 'concordant' when all genes inside share a base expression value
  with at most +/-noise_half relative deviation:
      (max_expr - min_expr) / base_expr  <=  2 * noise_half
  --concordance C is the fraction of populated domains that are concordant.
  C=0 -> purely random expression; C=1 -> all domains fully concordant.

Parallelism and reproducibility
  Replicates are generated in parallel with multiprocessing.Pool (--cores N).
  Each replicate i uses an independent RNG seeded with (base_seed + i), so
  output files are bit-for-bit identical regardless of how many cores are used
  or in what order the workers finish. Setting --cores 1 runs serially and
  produces the same files as any multi-core run with the same --seed.

Output per replicate i (written to --outdir):
  domains_<tag>_<i>.bed  -- BED3: chrom, start, end
  genes_<tag>_<i>.bed    -- BED5: chrom, start, end, expression, strand
                            col-4 = expression, compatible with optimized.py

Usage examples
  # Use a local gene annotation file:
  python generate_datasets.py --genes hg38_genes.bed --concordance 0.9 --n-files 10

  # Parallel generation (4 cores):
  python generate_datasets.py --genes hg38_genes.bed --concordance 0.9 --n-files 20 --cores 4

  # Switch to D. melanogaster:
  python generate_datasets.py --genome dm6 --concordance 0.75 --n-files 5

  # Sweep concordance levels, 5 replicates each:
  for C in 0.0 0.25 0.5 0.75 1.0; do
      python generate_datasets.py --genes hg38_genes.bed \
          --concordance $C --n-files 5 --outdir data_C${C}
  done
"""

import argparse
import sys
import json
import time
import os
import urllib.request
from pathlib import Path
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd


# ==============================================================================
# TUNABLE CONSTANTS
# Change these at the top of the script to adjust global behaviour without
# touching the CLI or function signatures.
# ==============================================================================

# --- Domain size distribution (Normal) ---
# SD = DOMAIN_SIZE_SD_FRACTION x mean_domain_size.
# E.g. 0.025 -> SD = 2.5% of mean -> 68% of domains within +/-2.5%, 95% within +/-5%.
DOMAIN_SIZE_SD_FRACTION: float = 0.025

# --- Inter-domain gap distribution (Normal) ---
# SD = GAP_SD_FRACTION x mean_gap.
# E.g. 0.05 -> SD = 5% of mean -> 68% of gaps within +/-5%, 95% within +/-10%.
GAP_SD_FRACTION: float = 0.05

# --- Intra-domain expression noise ---
# noise_half (H) sets the half-width of the Uniform noise applied to every
# gene inside a concordant domain:  expr = base * Uniform(1-H, 1+H).
# The full relative range within the domain is at most 2*H.
# H = 0.05  ->  tight concordance  (max 10% spread) -- highly concordant
# H = 0.30  ->  moderate noise     (max 60% spread) -- borderline
# H = 0.50  ->  fully discordant   (max 100% spread)-- indistinguishable from random
# Valid range: [0.0, 1.0)  (H >= 1 produces negative expression values)
NOISE_HALF_DEFAULT: float = 0.05

# --- Background expression distribution (log-normal, ln-space parameters) ---
BG_LOGNORM_MU: float    = 2.5             # ln-space mean  (~12 in linear space)
BG_LOGNORM_SIGMA: float = 1.5             # ln-space SD    (wide realistic range)

# --- UCSC REST API URL templates ({genome} is replaced at runtime) ---
UCSC_GENES_URL: str = (
    "https://api.genome.ucsc.edu/getData/track?"
    "genome={genome}&track=ncbiRefSeq&maxItemsOutput=500000"
)
UCSC_CHROMSIZES_URL: str = (
    "https://api.genome.ucsc.edu/getData/track?"
    "genome={genome}&track=chromInfo&maxItemsOutput=10000"
)

# --- Chromosomes to exclude (substring matching) ---
# Any chrom whose name contains one of these strings is dropped.
EXCLUDED_CHROM_SUBSTRINGS: tuple = ('chrY', 'chrM', 'chrUn', '_random', '_alt', 'chrEBV')

# --- hg38 chromosome sizes -- offline fallback when network is unavailable ---
HG38_CHROM_SIZES_FALLBACK: dict = {
    'chr1':  248_956_422, 'chr2':  242_193_529, 'chr3':  198_295_559,
    'chr4':  190_214_555, 'chr5':  181_538_259, 'chr6':  170_805_979,
    'chr7':  159_345_973, 'chr8':  145_138_636, 'chr9':  138_394_717,
    'chr10': 133_797_422, 'chr11': 135_086_622, 'chr12': 133_275_309,
    'chr13': 114_364_328, 'chr14': 107_043_718, 'chr15': 101_991_189,
    'chr16':  90_338_345, 'chr17':  83_257_441, 'chr18':  80_373_285,
    'chr19':  58_617_616, 'chr20':  64_444_167, 'chr21':  46_709_983,
    'chr22':  50_818_468, 'chrX':  156_040_895,
}

# ==============================================================================


# ------------------------------------------------------------------------------
# 1.  Chromosome size loading
# ------------------------------------------------------------------------------

def _is_excluded(chrom: str) -> bool:
    """Return True if the chromosome name matches any exclusion pattern."""
    return any(pat in chrom for pat in EXCLUDED_CHROM_SUBSTRINGS)


def fetch_chrom_sizes(genome: str) -> dict[str, int]:
    """
    Fetch chromosome sizes from UCSC for the given genome assembly.
    Returns dict {chrom_name: length}, excluding unwanted contigs.
    Falls back to the built-in hg38 table if genome=='hg38' and fetch fails.
    """
    url = UCSC_CHROMSIZES_URL.format(genome=genome)
    print(f"  Fetching chrom sizes for {genome} from UCSC ... ", end="", flush=True)
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            data = json.loads(resp.read())
        rows = data.get("chromInfo", [])
        if not rows:
            raise RuntimeError("empty response")
        sizes = {
            r['chrom']: int(r['size'])
            for r in rows
            if not _is_excluded(r['chrom'])
        }
        print(f"got {len(sizes)} canonical chroms.")
        return sizes
    except Exception as exc:
        if genome == 'hg38' and HG38_CHROM_SIZES_FALLBACK:
            print(f"failed ({exc}). Using built-in hg38 fallback.")
            return {c: s for c, s in HG38_CHROM_SIZES_FALLBACK.items()
                    if not _is_excluded(c)}
        raise RuntimeError(
            f"Could not fetch chrom sizes for {genome!r}: {exc}\n"
            "Provide a --genes file whose chromosomes define the genome, "
            "or check network connectivity."
        ) from exc


# ------------------------------------------------------------------------------
# 2.  Gene annotation fetching / loading
# ------------------------------------------------------------------------------

def fetch_genes_ucsc(genome: str) -> pd.DataFrame:
    """
    Download RefSeq gene coordinates for `genome` from the UCSC REST API.
    Returns a DataFrame with columns: chrom, start, end, name, strand.
    """
    url = UCSC_GENES_URL.format(genome=genome)
    print(f"  Fetching {genome} RefSeq genes from UCSC ... ", end="", flush=True)
    try:
        with urllib.request.urlopen(url, timeout=120) as resp:
            data = json.loads(resp.read())
    except Exception as e:
        raise RuntimeError(
            f"UCSC gene fetch failed for genome {genome!r}: {e}\n"
            "Use --genes to supply a local annotation file."
        ) from e

    rows = data.get("ncbiRefSeq", [])
    if not rows:
        raise RuntimeError(f"UCSC returned no records for genome={genome!r}.")

    df = pd.DataFrame(rows)[['chrom', 'txStart', 'txEnd', 'name', 'strand']]
    df.columns = ['chrom', 'start', 'end', 'name', 'strand']
    df['start'] = df['start'].astype(int)
    df['end']   = df['end'].astype(int)
    print(f"got {len(df):,} records.")
    return df


def load_genes_bed(path: str) -> pd.DataFrame:
    """
    Load a gene annotation BED-like file. Auto-detects column layout:

    Layout A -- BED6 (standard):      chrom  start  end  name  score  strand
    Layout B -- Ensembl-style:         chrom  start  end  strand  ensembl  name
    Layout C -- BED5 (our own output): chrom  start  end  expression  strand

    Detection: col-3 is '+'/'-'  -> Layout B
               ncols >= 6        -> Layout A
               ncols == 5        -> Layout C

    Always returns DataFrame with columns: chrom, start, end, name, strand.
    """
    with open(path) as fh:
        for raw in fh:
            if raw.startswith('#') or not raw.strip():
                continue
            fields = raw.rstrip('\n\r').split('\t')
            ncols  = len(fields)
            col3   = fields[3].strip() if ncols > 3 else ''
            break
        else:
            raise RuntimeError(f"Gene file {path!r} appears empty.")

    if col3 in ('+', '-'):
        cols = ['chrom', 'start', 'end', 'strand', 'ensembl', 'name']
        df = pd.read_csv(path, sep='\t', comment='#', header=None,
                         names=cols, usecols=[0, 1, 2, 3, 4, 5],
                         dtype={'chrom': str, 'start': int, 'end': int, 'strand': str})
    elif ncols >= 6:
        cols = ['chrom', 'start', 'end', 'name', 'score', 'strand']
        df = pd.read_csv(path, sep='\t', comment='#', header=None,
                         names=cols, usecols=[0, 1, 2, 3, 4, 5],
                         dtype={'chrom': str, 'start': int, 'end': int, 'strand': str})
    elif ncols == 5:
        cols = ['chrom', 'start', 'end', 'score', 'strand']
        df = pd.read_csv(path, sep='\t', comment='#', header=None,
                         names=cols, usecols=[0, 1, 2, 3, 4],
                         dtype={'chrom': str, 'start': int, 'end': int, 'strand': str})
        df['name'] = '.'
    else:
        raise RuntimeError(f"Gene file {path!r} has {ncols} columns; expected 5 or 6.")

    return df[['chrom', 'start', 'end', 'name', 'strand']]


def prepare_genes(genes_path: str | None,
                  genome: str,
                  valid_chroms: set[str]) -> pd.DataFrame:
    """
    Load or fetch gene annotation, filter to valid chromosomes,
    and deduplicate (keeping longest isoform per locus).
    """
    if genes_path:
        df = load_genes_bed(genes_path)
    else:
        df = fetch_genes_ucsc(genome)

    df = df[df['chrom'].isin(valid_chroms)].copy()

    df['length'] = df['end'] - df['start']
    df = (df.sort_values('length', ascending=False)
            .drop_duplicates(subset=['chrom', 'start'])
            .drop(columns='length')
            .sort_values(['chrom', 'start'])
            .reset_index(drop=True))

    if df.empty:
        raise RuntimeError(
            "No genes remain after filtering to valid chromosomes. "
            "Check that the gene file uses the same chromosome naming as the genome."
        )

    print(f"  Using {len(df):,} genes across {df['chrom'].nunique()} chromosomes.")
    return df


# ------------------------------------------------------------------------------
# 3.  Random domain generation -- Normal distributions
# ------------------------------------------------------------------------------

def generate_domains(rng: np.random.Generator,
                     chrom_sizes: dict[str, int],
                     mean_domain_size: int,
                     mean_gap: int,
                     domain_size_sd_frac: float = DOMAIN_SIZE_SD_FRACTION,
                     gap_sd_frac: float = GAP_SD_FRACTION) -> pd.DataFrame:
    """
    Generate non-overlapping random domains tiling the genome.

    Domain sizes are drawn from a Normal distribution:
        size ~ N(mean_domain_size,  (domain_size_sd_frac * mean_domain_size)^2)
        SD = domain_size_sd_frac x mean  (default 2.5% of mean).

    Inter-domain gaps are drawn from a Normal distribution:
        gap  ~ N(mean_gap,  (gap_sd_frac * mean_gap)^2)
        SD = gap_sd_frac x mean  (default 5% of mean).

    Negative draws are clipped to 1 bp.
    Domains that would extend past a chromosome boundary are discarded entirely
    (not clipped) to preserve the Normal size distribution.

    Returns a BED3 DataFrame: chrom, start, end.
    """
    domain_size_sd = domain_size_sd_frac * mean_domain_size
    gap_sd         = gap_sd_frac         * mean_gap

    all_rows = []

    for chrom, chrom_len in sorted(chrom_sizes.items()):
        pos = 0
        while True:
            gap = max(1, int(round(rng.normal(mean_gap, gap_sd))))
            pos += gap
            if pos >= chrom_len:
                break

            size = max(1, int(round(rng.normal(mean_domain_size, domain_size_sd))))
            end  = pos + size

            # Discard domains that would extend past the chromosome end --
            # clipping creates severely undersized stub domains that violate
            # the Normal size distribution (skewness, excess kurtosis).
            if end > chrom_len:
                break

            all_rows.append((chrom, pos, end))
            pos = end

    return pd.DataFrame(all_rows, columns=['chrom', 'start', 'end'])


# ------------------------------------------------------------------------------
# 4.  Gene-to-domain mapping (binary search, vectorised per chromosome)
# ------------------------------------------------------------------------------

def assign_genes_to_domains(genes: pd.DataFrame,
                             domains: pd.DataFrame,
                             mode: str = 'middle') -> dict[int, list[int]]:
    """
    Map gene row-indices to domain row-indices.

    mode='middle' : gene assigned to domain containing its midpoint.
    mode='start'  : gene assigned to domain containing its TSS.

    Returns domain_to_genes: dict { domain_row_idx -> [gene_row_idx, ...] }
    """
    domain_to_genes: dict[int, list[int]] = {}
    domain_chroms = set(domains['chrom'].unique())

    for chrom in genes['chrom'].unique():
        if chrom not in domain_chroms:
            continue

        g = genes[genes['chrom'] == chrom]
        d = domains[domains['chrom'] == chrom].sort_values('start')

        d_starts = d['start'].values
        d_ends   = d['end'].values
        d_idx    = d.index.values

        for gi, grow in g.iterrows():
            point = (
                (int(grow['start']) + int(grow['end'])) // 2
                if mode == 'middle'
                else int(grow['start'])
            )
            pos = np.searchsorted(d_starts, point, side='right') - 1
            if 0 <= pos < len(d_starts) and d_starts[pos] <= point < d_ends[pos]:
                domain_to_genes.setdefault(d_idx[pos], []).append(gi)

    return domain_to_genes


# ------------------------------------------------------------------------------
# 5.  Expression synthesis with controlled concordance
# ------------------------------------------------------------------------------

def synthesize_expression(genes: pd.DataFrame,
                           domain_to_genes: dict[int, list[int]],
                           concordance: float,
                           rng: np.random.Generator,
                           noise_half: float = NOISE_HALF_DEFAULT,
                           bg_lognorm_mu: float = BG_LOGNORM_MU,
                           bg_lognorm_sigma: float = BG_LOGNORM_SIGMA) -> np.ndarray:
    """
    Assign a synthetic expression value to every gene.

    Two independent axes control the output:

    concordance (C) -- fraction of populated domains that receive shared
      expression. With prob C a domain is "active": all its genes draw
      from base * Uniform(1-H, 1+H). With prob (1-C) genes keep
      independent background draws.

    noise_half (H) -- half-width of the intra-domain Uniform noise.
      Sets how similar genes within an active domain actually are:
        H = 0.05  ->  max 10% spread   (tight, clearly concordant)
        H = 0.30  ->  max 60% spread   (moderate, borderline)
        H = 0.50  ->  max 100% spread  (fully discordant-looking)
      Valid range: [0.0, 1.0). H >= 1 produces negative expression.

    Combined usage:
      C=1.0, H=0.05  -> all domains tight, clearly concordant
      C=1.0, H=0.50  -> all domains involved but maximally noisy
      C=0.5, H=0.05  -> half domains tight, half random
      C=0.5, H=0.50  -> half domains noisy, half random (similar appearance)
    """
    n    = len(genes)
    expr = rng.lognormal(mean=bg_lognorm_mu, sigma=bg_lognorm_sigma, size=n)

    for gene_indices in domain_to_genes.values():
        if not gene_indices:
            continue
        if rng.random() < concordance:
            base = rng.lognormal(mean=bg_lognorm_mu, sigma=bg_lognorm_sigma)
            expr[gene_indices] = base * rng.uniform(
                1.0 - noise_half, 1.0 + noise_half, size=len(gene_indices)
            )

    return expr


# ------------------------------------------------------------------------------
# 6.  I/O helpers
# ------------------------------------------------------------------------------

def write_domains_bed(domains: pd.DataFrame, path: Path) -> None:
    """Write BED3: chrom  start  end."""
    domains[['chrom', 'start', 'end']].to_csv(
        path, sep='\t', header=False, index=False
    )


def write_genes_bed(genes: pd.DataFrame, expr: np.ndarray, path: Path) -> None:
    """
    Write BED5: chrom  start  end  expression  strand
    col-4 = expression (6 dp float) -- compatible with optimized.py.
    """
    out           = genes[['chrom', 'start', 'end']].copy()
    out['score']  = np.round(expr, 6)
    out['strand'] = genes['strand'].values
    out.to_csv(path, sep='\t', header=False, index=False, float_format='%.6f')


# ------------------------------------------------------------------------------
# 7.  Per-replicate worker
# ------------------------------------------------------------------------------

def _run_replicate(task: dict) -> dict:
    """
    Generate one (domains, genes) replicate and write BED files.
    Called by both the serial loop and multiprocessing.Pool workers.

    Reproducibility guarantee
    -------------------------
    Replicate i creates its own RNG seeded with (base_seed + i).
    This seed drives the entire replicate: first domain layout draws,
    then expression synthesis draws -- in that fixed order.
    Because each replicate's RNG is fully independent of all others:

      * Output files are bit-for-bit identical regardless of core count.
      * --cores 1 (serial) produces the same files as --cores 8 (or any N).
      * Any single replicate can be regenerated in isolation by rerunning
        with the same --seed and selecting that replicate index.

    Returns a stats dict; private keys prefixed '_' are stripped before
    the summary TSV is written.
    """
    i                = task['i']
    base_seed        = task['base_seed']
    genes            = task['genes']
    chrom_sizes      = task['chrom_sizes']
    outdir           = task['outdir']
    tag              = task['tag']
    concordance      = task['concordance']
    noise_half       = task['noise_half']
    bg_mu            = task['bg_mu']
    bg_sigma         = task['bg_sigma']
    mean_domain_size = task['mean_domain_size']
    mean_gap         = task['mean_gap']
    mode             = task['mode']
    domain_sd        = task['domain_sd']
    gap_sd           = task['gap_sd']
    genome           = task['genome']

    t0 = time.time()

    # Each replicate gets its own independent, deterministic RNG.
    # seed = base_seed + i ensures replicates never share state.
    rng = np.random.default_rng(base_seed + i)

    # Phase A: domain layout (consumes RNG draws for sizes and gaps)
    domains = generate_domains(rng, chrom_sizes, mean_domain_size, mean_gap)

    # Phase B: gene mapping (deterministic, no RNG used)
    domain_to_genes = assign_genes_to_domains(genes, domains, mode=mode)

    # Phase C: expression synthesis (consumes further RNG draws)
    # The number of draws in Phase A is fully determined by chrom_sizes and
    # the Normal parameters -- identical across serial and parallel runs.
    expr = synthesize_expression(
        genes, domain_to_genes, concordance, rng,
        noise_half=noise_half, bg_lognorm_mu=bg_mu, bg_lognorm_sigma=bg_sigma,
    )

    dom_path  = outdir / f"domains_{tag}_{i+1:04d}.bed"
    gene_path = outdir / f"genes_{tag}_{i+1:04d}.bed"
    write_domains_bed(domains, dom_path)
    write_genes_bed(genes, expr, gene_path)

    dom_sizes     = (domains['end'] - domains['start']).values
    genes_in_doms = sum(len(v) for v in domain_to_genes.values())

    rel_ranges = []
    for gene_indices in domain_to_genes.values():
        if len(gene_indices) < 2:
            continue
        vals = expr[gene_indices]
        rel_ranges.append((vals.max() - vals.min()) / (vals.mean() + 1e-12))

    threshold            = 2 * noise_half
    frac_concordant_doms = (
        float(np.mean(np.array(rel_ranges) <= threshold))
        if rel_ranges else float('nan')
    )

    return {
        # Public summary columns (written to TSV)
        'replicate'              : i + 1,
        'tag'                    : tag,
        'genome'                 : genome,
        'concordance_target'     : concordance,
        'domain_size_mean_bp'    : mean_domain_size,
        'domain_size_sd_bp'      : round(domain_sd),
        'gap_mean_bp'            : mean_gap,
        'gap_sd_bp'              : round(gap_sd),
        'n_domains'              : len(domains),
        'obs_domain_mean_bp'     : float(dom_sizes.mean()),
        'obs_domain_sd_bp'       : float(dom_sizes.std()),
        'obs_domain_median_bp'   : float(np.median(dom_sizes)),
        'n_genes_total'          : len(genes),
        'n_genes_in_domains'     : genes_in_doms,
        'n_genes_out_domains'    : len(genes) - genes_in_doms,
        'frac_genes_in_domains'  : genes_in_doms / len(genes),
        'frac_concordant_domains': frac_concordant_doms,
        'noise_half'             : noise_half,
        'obs_mean_rel_range'     : float(np.mean(rel_ranges)) if rel_ranges else float('nan'),
        'genes_file'             : str(gene_path),
        'domains_file'           : str(dom_path),
        # Private fields: used by caller, stripped before TSV
        '_i'                     : i,
        '_elapsed'               : time.time() - t0,
    }


def _print_progress(result: dict, n_files: int) -> None:
    """Print one progress line immediately when a replicate finishes."""
    print(
        f"  [{result['replicate']:3d}/{n_files}]  "
        f"domains={result['n_domains']:,}  "
        f"genes_in={result['n_genes_in_domains']:,}  "
        f"active={result['frac_concordant_domains']:.1%}  "
        f"noise={result['noise_half']:.2f}  "
        f"mean_spread={result['obs_mean_rel_range']:.1%}  "
        f"({result['_elapsed']:.1f}s)",
        flush=True
    )


# ------------------------------------------------------------------------------
# 8.  Main
# ------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate paired (genes, domains) BED datasets for concordance modelling.\n"
            "Domain sizes and inter-domain gaps are drawn from Normal distributions\n"
            "with user-specified means and fixed relative SDs (see top-of-file constants).\n"
            "Supports parallel generation (--cores N) with guaranteed reproducibility:\n"
            "output is bit-for-bit identical for any core count given the same --seed."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── Input / genome ─────────────────────────────────────────────────────────
    input_grp = parser.add_argument_group("Input / genome")
    input_grp.add_argument(
        '--genes', metavar='FILE',
        help="Gene annotation file (BED5, BED6, or Ensembl-style BED). "
             "If omitted, genes are fetched from UCSC for --genome."
    )
    input_grp.add_argument(
        '--genome', metavar='ASSEMBLY', default='hg38',
        help="UCSC genome assembly name (e.g. hg38, mm39, dm6). Default: hg38."
    )

    # ── Concordance ────────────────────────────────────────────────────────────
    conc_grp = parser.add_argument_group(
        "Concordance",
        "Two orthogonal axes independently control expression similarity:\n"
        "  --concordance C   fraction of domains that are 'active' (share a base level)\n"
        "  --noise       H   half-width of intra-domain noise (0=identical, ~0.5=chaotic)\n"
        "Examples:\n"
        "  C=1.0 H=0.05  tight concordance in every domain\n"
        "  C=1.0 H=0.50  every domain active but maximally noisy (discordant-looking)\n"
        "  C=0.5 H=0.05  half domains tight concordant, half purely random\n"
        "  C=0.0 H=any   pure null -- all genes independently random"
    )
    conc_grp.add_argument(
        '--concordance', type=float, default=0.5, metavar='C',
        help="Fraction of populated domains that share a common base expression level "
             "[0.0-1.0]. C=1.0 means every domain is active. Default: 0.5."
    )
    conc_grp.add_argument(
        '--noise', '--noise-half', dest='noise_half',
        type=float, default=NOISE_HALF_DEFAULT, metavar='H',
        help="Half-width of the Uniform intra-domain noise [0.0, 1.0). "
             "Genes in an active domain receive base*Uniform(1-H, 1+H), "
             "giving a maximum within-domain relative spread of 2*H: "
             "H=0.05 -> max 10%%%% spread (concordant); "
             "H=0.30 -> max 60%%%% spread (moderate); "
             "H=0.50 -> max 100%%%% spread (discordant-looking). "
             f"Default: {NOISE_HALF_DEFAULT}."
    )

    # ── Domain geometry ────────────────────────────────────────────────────────
    dom_grp = parser.add_argument_group(
        "Domain geometry",
        "Sizes ~ Normal(mean, SD) and gaps ~ Normal(mean, SD).\n"
        "SD fractions are set by DOMAIN_SIZE_SD_FRACTION / GAP_SD_FRACTION "
        "at the top of the script."
    )
    dom_grp.add_argument(
        '--mean-domain-size', type=int, default=115_000, metavar='BP',
        help=f"Mean domain size in bp. SD = {DOMAIN_SIZE_SD_FRACTION*100:.1f}%% of mean. Default: 115000."
    )
    dom_grp.add_argument(
        '--mean-gap', type=int, default=1_000, metavar='BP',
        help=f"Mean inter-domain gap in bp. SD = {GAP_SD_FRACTION*100:.1f}%% of mean. Default: 1000."
    )

    # ── Output ─────────────────────────────────────────────────────────────────
    out_grp = parser.add_argument_group("Output")
    out_grp.add_argument(
        '--n-files', type=int, default=10, metavar='N',
        help="Number of replicate dataset pairs to generate. Default: 10."
    )
    out_grp.add_argument(
        '--outdir', type=str, default='./datasets', metavar='DIR',
        help="Output directory (created if absent). Default: ./datasets."
    )
    out_grp.add_argument(
        '--tag', type=str, default=None, metavar='TAG',
        help="Label embedded in filenames. Default: C<concordance>_N<noise> e.g. C0.90_N0.05."
    )
    defseed = int.from_bytes(os.urandom(4), byteorder='little', signed=False)
    out_grp.add_argument(
        '--seed', type=int, default=defseed, metavar='INT',
        help="Base random seed. Replicate i uses seed+i. Default: urandom from OS."
    )

    # ── Parallelism ────────────────────────────────────────────────────────────
    par_grp = parser.add_argument_group("Parallelism")
    par_grp.add_argument(
        '--cores', type=int,
        default=max(1, cpu_count() - 1),
        metavar='N',
        help="Worker processes for parallel generation. "
             "1 = serial (identical output to any parallel run). "
             f"Default: {max(1, cpu_count()-1)} (all CPUs minus one)."
    )

    # ── Advanced ───────────────────────────────────────────────────────────────
    adv_grp = parser.add_argument_group("Advanced")
    adv_grp.add_argument(
        '--mode', choices=['middle', 'start'], default='middle',
        help="Gene-to-domain assignment rule. Default: middle."
    )
    adv_grp.add_argument(
        '--bg-mu', type=float, default=BG_LOGNORM_MU,
        help=f"Log-normal mu for background expression (ln scale). Default: {BG_LOGNORM_MU}."
    )
    adv_grp.add_argument(
        '--bg-sigma', type=float, default=BG_LOGNORM_SIGMA,
        help=f"Log-normal sigma for background expression (ln scale). Default: {BG_LOGNORM_SIGMA}."
    )

    args = parser.parse_args()

    # ── Validate ───────────────────────────────────────────────────────────────
    if not (0.0 <= args.concordance <= 1.0):
        parser.error("--concordance must be between 0.0 and 1.0")
    if not (0.0 <= args.noise_half < 1.0):
        parser.error("--noise must be in [0.0, 1.0) -- "
                     "values >= 1.0 produce zero or negative expression")
    if args.n_files < 1:
        parser.error("--n-files must be >= 1")
    if args.mean_domain_size < 1:
        parser.error("--mean-domain-size must be >= 1")
    if args.mean_gap < 1:
        parser.error("--mean-gap must be >= 1")
    if args.cores < 1:
        parser.error("--cores must be >= 1")

    tag    = args.tag if args.tag else f"C{args.concordance:.2f}_N{args.noise_half:.2f}"
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    domain_sd = DOMAIN_SIZE_SD_FRACTION * args.mean_domain_size
    gap_sd    = GAP_SD_FRACTION         * args.mean_gap

    # ── Load chromosome sizes and genes (once, in the main process) ────────────
    print("=" * 65)
    print(f"Genome assembly : {args.genome}")
    chrom_sizes  = fetch_chrom_sizes(args.genome)
    valid_chroms = set(chrom_sizes.keys())

    print("Loading gene annotation ...")
    genes = prepare_genes(args.genes, args.genome, valid_chroms)

    # Cap workers to number of jobs -- no benefit spawning more
    n_cores_actual = min(args.cores, args.n_files)

    print(f"  Concordance      : {args.concordance:.2f}  "
          f"(fraction of active domains)")
    print(f"  Noise half-width : {args.noise_half:.2f}  "
          f"(max within-domain spread {args.noise_half*2*100:.0f}%)")
    print(f"  Domain size      : N({args.mean_domain_size:,} bp, "
          f"SD={domain_sd:,.0f} bp = {DOMAIN_SIZE_SD_FRACTION*100:.1f}% of mean)")
    print(f"  Inter-domain gap : N({args.mean_gap:,} bp, "
          f"SD={gap_sd:,.0f} bp = {GAP_SD_FRACTION*100:.1f}% of mean)")
    print(f"  Replicates       : {args.n_files}")
    print(f"  Cores            : {n_cores_actual}"
          f"{'  (serial)' if n_cores_actual == 1 else '  (parallel)'}")
    print(f"  Seed             : {args.seed}  (replicate i -> seed {args.seed}+i)")
    print(f"  Output dir       : {outdir.resolve()}")
    print(f"  Tag              : {tag}")
    print("=" * 65)

    # ── Build task list ────────────────────────────────────────────────────────
    # Each task dict is self-contained: workers receive everything they need.
    # The genes DataFrame is shared via copy-on-write fork on Linux (zero cost),
    # or pickled once per worker on macOS/Windows (~5 MB, acceptable).
    tasks = [
        dict(
            i=i, base_seed=args.seed,
            genes=genes, chrom_sizes=chrom_sizes,
            outdir=outdir, tag=tag, genome=args.genome,
            concordance=args.concordance, noise_half=args.noise_half,
            bg_mu=args.bg_mu, bg_sigma=args.bg_sigma,
            mean_domain_size=args.mean_domain_size, mean_gap=args.mean_gap,
            mode=args.mode, domain_sd=domain_sd, gap_sd=gap_sd,
        )
        for i in range(args.n_files)
    ]

    # ── Execute: serial (cores=1) or parallel (cores>1) ───────────────────────
    t_total    = time.time()
    stats_rows = [None] * args.n_files    # pre-allocate; filled by replicate index

    if n_cores_actual == 1:
        # Serial path -- walk tasks in order, print immediately
        for task in tasks:
            result = _run_replicate(task)
            stats_rows[result['_i']] = result
            _print_progress(result, args.n_files)
    else:
        # Parallel path: imap_unordered dispatches all tasks, yields results as
        # each worker finishes (arbitrary order). We store by '_i' to restore
        # replicate order in the summary TSV.
        with Pool(processes=n_cores_actual) as pool:
            for result in pool.imap_unordered(_run_replicate, tasks):
                stats_rows[result['_i']] = result
                _print_progress(result, args.n_files)

    # ── Write summary TSV (strip private _ keys, replicate order preserved) ────
    public_rows = [
        {k: v for k, v in r.items() if not k.startswith('_')}
        for r in stats_rows
    ]
    stats_df   = pd.DataFrame(public_rows)
    stats_path = outdir / f"summary_{tag}.tsv"
    stats_df.to_csv(stats_path, sep='\t', index=False, float_format='%.4f')

    print("=" * 65)
    print(f"Done. {args.n_files} replicates in {time.time()-t_total:.1f}s")
    print(f"Summary stats -> {stats_path}")
    print()
    print("Run concordance analysis:")
    print(f"  python optimized.py {outdir}/genes_{tag}_001.bed "
          f"{outdir}/domains_{tag}_001.bed --chromosomewide --perms 1000")


if __name__ == '__main__':
    main()
