#!/usr/bin/env python3
"""
calibrate_dispersion_f.py
--------------------
Derive the noise correction factor f for a specific domain structure by direct
simulation, without any permutation test and without generated datasets.

Background
----------
The noise estimator of optconn computes, for the concordant population,

        H_est = median(rel_range) / f(median_n)

where rel_range = (max - min) / mean of the expression values of the genes of
one domain and median_n is the median number of genes per multi-gene domain.
In a concordant domain every gene has expression  base * U(1 - H, 1 + H).
The base cancels in rel_range, so the distribution of rel_range depends only on
H and on the number of genes in each domain.  Therefore

        f = median(rel_range of concordant domains) / H_true

can be obtained by simulating the concordant domains of the real structure:

  1. Genes are mapped to domains by their midpoint, with exactly the functions
     and filters the estimator uses (optconn10.compute_domain_rel_ranges), so
     the gene count per domain n_d is the one the estimator will see.
  2. For each H of the grid and for each replicate, every multi-gene domain d
     gets n_d values drawn from U(1 - H, 1 + H); rel_range is computed and the
     median over domains is divided by H.
  3. The replicates give the median and the 5-95% spread of f for each H.  The
     per-H medians are pooled into a single f for the structure, and the
     dependence of f on H is reported (the estimator assumes f does not depend
     on H; the table of the 1 Mbp TADs held this to within about 3%).

The result is a one-row table  (median_n, f)  for the structure.  The estimator
interpolates linearly and clamps outside the tabulated range, so a one-row
table acts as a constant.

Usage
-----
    python calibrate_noise_f.py --genes dm6_genes.bed --domains dm6_tads.bed \\
        --label dm6_TAD143k --output f_table_dm6.tsv

Input files are in the same format as for optconn:
    genes   BED5  chrom start end score strand   (score is not used)
    domains BED3  chrom start end

Requires optconn10.py in the same directory.
"""

import argparse
import sys
import time
import os

from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Maximum spread of f across H that the estimator tolerates (the f table of the
# 1 Mbp TADs held this to within about 3%).
H_DEPENDENCE_WARN = 0.03
_CALIB_N   = np.array([ 2,     3,     5,     7,     10,    15,    20,    25,    30])
_CALIB_FAC = np.array([0.777, 0.987, 1.339, 1.522, 1.662, 1.774, 1.830, 1.864, 1.886])


def _correction_factor(median_n_genes):
    """
    Return the n-aware correction factor for the concordant_median -> H mapping.
    Interpolates linearly from the calibration table; clamps at boundaries.
    """
    return float(np.interp(median_n_genes, _CALIB_N, _CALIB_FAC,
                           left=_CALIB_FAC[0], right=_CALIB_FAC[-1]))

# ==========================================
# 3. Gene-to-Domain Mapping
# ==========================================

def map_genes_to_domains(genes_df, domains_df, mode):
    """
    Maps gene row-indices to domain row-indices based on spatial logic.

    Pure-Python implementation using pre-extracted numpy arrays.
    Avoids pandas iterrows() overhead (~5x faster than the iterrows version)
    while remaining fully portable with no compiled dependencies.

    Returns domain_map: list of lists, where domain_map[i] = [(gene_idx, weight), ...]
    Weights are 1.0 for point modes, fractional overlap for 'proportional'.

    Modes
    -----
    start        Transcription start site (strand-aware):
                 g_start for + strand genes, g_end for - strand genes.
    end          Transcription end site (strand-aware):
                 g_end for + strand genes, g_start for - strand genes.
    middle       Midpoint of the gene body.  Strand-agnostic.
    longer_part  Assign to the domain containing the longer gene-body portion.
    proportional Fractional assignment proportional to gene-body overlap per domain.
    """
    domain_map    = [[] for _ in range(len(domains_df))]
    domain_chroms = set(domains_df['chrom'].unique())

    for chrom in genes_df['chrom'].unique():
        if chrom not in domain_chroms:
            continue
        g_subset = genes_df[genes_df['chrom'] == chrom]
        d_subset = domains_df[domains_df['chrom'] == chrom].sort_values('start')
        if d_subset.empty or g_subset.empty:
            continue

        d_starts  = d_subset['start'].values
        d_ends    = d_subset['end'].values
        d_indices = d_subset.index.values

        # Extract arrays once per chromosome — avoids constructing a pandas
        # Series for every gene (which was the iterrows() bottleneck).
        g_starts_arr  = g_subset['start'].values
        g_ends_arr    = g_subset['end'].values
        g_strands_arr = g_subset['strand'].values
        g_idx_arr     = g_subset.index.values

        for k in range(len(g_subset)):
            g_start  = int(g_starts_arr[k])
            g_end    = int(g_ends_arr[k])
            g_strand = g_strands_arr[k]
            g_idx    = int(g_idx_arr[k])

            if mode in ('start', 'end', 'middle'):
                if   mode == 'start':  point = g_start if g_strand == '+' else g_end
                elif mode == 'end':    point = g_end   if g_strand == '+' else g_start
                else:                  point = (g_start + g_end) // 2
                pos = np.searchsorted(d_starts, point, side='right') - 1
                if 0 <= pos < len(d_starts) and d_starts[pos] <= point < d_ends[pos]:
                    domain_map[d_indices[pos]].append((g_idx, 1.0))
            else:
                lo = np.searchsorted(d_ends,   g_start, side='right')
                hi = np.searchsorted(d_starts, g_end,   side='left')
                if lo >= hi:
                    continue
                gene_len = max(g_end - g_start, 1)
                temp_map = []
                for loc in range(lo, hi):
                    inter_start = max(g_start, d_starts[loc])
                    inter_end   = min(g_end,   d_ends[loc])
                    inter_len   = max(0, inter_end - inter_start)
                    if inter_len > 0:
                        temp_map.append((d_indices[loc], inter_len, inter_len / gene_len))
                if mode == 'proportional':
                    for d_i, _, frac in temp_map:
                        domain_map[d_i].append((g_idx, frac))
                elif mode == 'longer_part' and temp_map:
                    best = max(temp_map, key=lambda x: x[1])
                    domain_map[best[0]].append((g_idx, 1.0))

    return domain_map


def compute_domain_rel_ranges(expression_array, domain_map,
                              genes_df=None, domains_df=None):
    """
    For every domain with >= 2 mapped genes, compute the relative range:
        rel_range = (max_expr - min_expr) / mean_expr

    Gene assignment for noise estimation always uses MIDPOINT mode, regardless
    of the --mode used for the actual concordance statistics. This is essential
    because proportional mode assigns boundary genes to multiple domains with
    fractional weights, causing genes from adjacent discordant domains to
    contaminate concordant-domain rel_ranges and destroy the bimodal signal.

    When genes_df and domains_df are provided, a fresh midpoint-based mapping
    is computed internally. Otherwise, domain_map is used directly (which should
    itself be a midpoint/start/end mapping, not proportional).

    Args:
        expression_array : float64 array (n_genes,)
        domain_map       : list of lists [(gene_idx, weight), ...] per domain
                           Used directly when genes_df is None.
        genes_df         : DataFrame with gene coordinates (optional).
                           If provided, a midpoint mapping is recomputed.
        domains_df       : DataFrame with domain coordinates (optional).

    Returns:
        rel_ranges         : float64 array, one value per multi-gene domain
        n_genes_per_domain : int32 array, gene count per multi-gene domain
    """
    # If raw DataFrames are supplied, recompute with midpoint mode to ensure
    # the rel_range distribution is not contaminated by proportional-overlap
    # boundary genes.
    if genes_df is not None and domains_df is not None:
        use_map = map_genes_to_domains(genes_df, domains_df, mode='middle')
    else:
        use_map = domain_map

    rel_ranges         = []
    n_genes_per_domain = []

    for entries in use_map:
        if len(entries) < 2:
            continue
        gene_indices = np.array([g for g, _ in entries], dtype=np.int64)
        vals         = expression_array[gene_indices]   # raw expression, weight=1 for midpoint
        mean_val     = vals.mean()
        if mean_val <= 0:
            continue
        rel_ranges.append((vals.max() - vals.min()) / mean_val)
        n_genes_per_domain.append(len(gene_indices))

    return np.array(rel_ranges), np.array(n_genes_per_domain, dtype=np.int32)





# ──────────────────────────────────────────────────────────────────────────────
# Structure: gene counts per domain, exactly as the estimator sees them
# ──────────────────────────────────────────────────────────────────────────────

def load_structure(genes_path: str, domains_path: str):
    """
    Read the BED files like optconn does and return
        n_gpd : gene counts of the multi-gene domains (estimator's view)
        n_all : gene counts of all domains (including 0 and 1 gene)
    """
    genes = pd.read_csv(genes_path, sep='\\s+', comment='#', header=None,
                        names=['chrom', 'start', 'end', 'score', 'strand'],
                        usecols=[0, 1, 2, 3, 4],
                        dtype={'chrom': str, 'score': float})
    domains = pd.read_csv(domains_path, sep='\\s+', comment='#', header=None,
                          names=['chrom', 'start', 'end'],
                          usecols=[0, 1, 2], dtype={'chrom': str})
    genes   = genes.sort_values(['chrom', 'start']).reset_index(drop=True)
    domains = domains.sort_values(['chrom', 'start']).reset_index(drop=True)

    # Same call as in the noise estimation of optconn; unit expression keeps
    # every domain (mean > 0) so only the >= 2 genes filter is applied.
    _, n_gpd = compute_domain_rel_ranges(
        np.ones(len(genes)), None, genes_df=genes, domains_df=domains)
    dm    = map_genes_to_domains(genes, domains, mode='middle')
    n_all = np.array([len(e) for e in dm], dtype=np.int32)
    return n_gpd, n_all, len(genes), len(domains)


# ──────────────────────────────────────────────────────────────────────────────
# Simulation
# ──────────────────────────────────────────────────────────────────────────────

def simulate_f(n_gpd: np.ndarray, H: float, replicates: int,
               rng: np.random.Generator) -> np.ndarray:
    """
    For one H return `replicates` values of  median(rel_range) / H  where the
    domains have the gene counts n_gpd and concordant noise U(1 - H, 1 + H).
    """
    n_vals, n_cnt = np.unique(n_gpd, return_counts=True)
    lo, hi = 1.0 - H, 1.0 + H
    f = np.empty(replicates)
    for r in range(replicates):
        parts = []
        for n, c in zip(n_vals, n_cnt):
            u = rng.uniform(lo, hi, size=(c, n))
            parts.append((u.max(axis=1) - u.min(axis=1)) / u.mean(axis=1))
        f[r] = np.median(np.concatenate(parts)) / H
    return f


def parse_grid(text: str) -> np.ndarray:
    try:
        a, b, s = (float(x) for x in text.split(':'))
    except ValueError:
        raise argparse.ArgumentTypeError("use START:STOP:STEP, e.g. 0.05:0.50:0.05")
    return np.round(np.arange(a, b + s / 2, s), 6)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Derive the noise correction factor f for a domain structure "
                    "by simulating concordant domains (no permutations needed).")
    ap.add_argument('--genes',   required=True, metavar='FILE',
                    help="Gene annotation BED5 (chrom start end score strand).")
    ap.add_argument('--domains', required=True, metavar='FILE',
                    help="Domain BED3 (chrom start end).")
    ap.add_argument('--h-grid', type=parse_grid, default=parse_grid('0.05:0.50:0.05'),
                    metavar='START:STOP:STEP',
                    help="Noise levels H to simulate (default 0.05:0.50:0.05).")
    ap.add_argument('--replicates', type=int, default=10000, metavar='N',
                    help="Independent noise draws per H (default 10000).")
    ap.add_argument('--label', default=None, metavar='NAME',
                    help="Label of the structure written to the table "
                         "(default: domains file name).")
    ap.add_argument('--output', '-o', default='f_table.tsv', metavar='FILE',
                    help="Output TSV (default f_table.tsv).")
    defseed = int.from_bytes(os.urandom(4), byteorder='little', signed=False)
    ap.add_argument('--seed', type=int, default=defseed, metavar='INT',
                    help="Random seed (default urandom).")
    args = ap.parse_args()

    if args.replicates < 2:
        ap.error("--replicates must be >= 2")
    label = args.label or Path(args.domains).stem

    # ── structure ─────────────────────────────────────────────────────────────
    n_gpd, n_all, n_genes, n_domains = load_structure(args.genes, args.domains)
    if len(n_gpd) == 0:
        sys.exit("No domain contains two or more genes; f cannot be derived.")

    median_n = float(np.median(n_gpd))
    print(f"Structure '{label}'")
    print(f"  genes: {n_genes:,}   domains: {n_domains:,}")
    print(f"  domains with 0 genes: {(n_all == 0).sum():,}   "
          f"with 1 gene: {(n_all == 1).sum():,}   "
          f"with >= 2 genes (used): {len(n_gpd):,}")
    q = np.percentile(n_gpd, [5, 25, 50, 75, 95])
    print(f"  genes per used domain: median {median_n:g}  mean {n_gpd.mean():.2f}  "
          f"p5/p25/p75/p95 = {q[0]:g}/{q[1]:g}/{q[3]:g}/{q[4]:g}  max {n_gpd.max()}")

    # ── simulation ────────────────────────────────────────────────────────────
    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    rows, medians = [], []
    print(f"\n  {'H':>6}  {'f median':>9}  {'f p5':>8}  {'f p95':>8}")
    for H in args.h_grid:
        f = simulate_f(n_gpd, float(H), args.replicates, rng)
        fm, f5, f95 = np.median(f), *np.percentile(f, [5, 95])
        medians.append(fm)
        rows.append((label, f"{H:g}", median_n, len(n_gpd), args.replicates,
                     fm, f5, f95))
        print(f"  {H:>6.2f}  {fm:>9.4f}  {f5:>8.4f}  {f95:>8.4f}")

    medians  = np.array(medians)
    f_pooled = float(np.median(medians))
    spread   = float((medians.max() - medians.min()) / f_pooled)
    rows.append((label, 'pooled', median_n, len(n_gpd), args.replicates,
                 f_pooled, medians.min(), medians.max()))

    print(f"\n  pooled f = {f_pooled:.4f}  at median n = {median_n:g}   "
          f"(simulation took {time.time() - t0:.1f} s)")
    print(f"  variation of f across H: {spread * 100:.1f}%  "
          f"(range {medians.min():.4f} - {medians.max():.4f})")
    if spread > H_DEPENDENCE_WARN:
        print(f"  WARNING: f varies with H by more than {H_DEPENDENCE_WARN*100:.0f}%; "
              f"a single f(n) will bias H_est at the extremes of the H range.")
    print(f"  f of the current table at n = {median_n:g}: "
          f"{_correction_factor(median_n):.4f}  "
          f"(difference {100 * (f_pooled / _correction_factor(median_n) - 1):+.1f}%)")

    # ── output ────────────────────────────────────────────────────────────────
    out = pd.DataFrame(rows, columns=['label', 'H', 'median_n', 'n_domains',
                                      'replicates', 'f_median', 'f_low', 'f_high'])
    with open(args.output, 'w') as fh:
        fh.write(f"# noise correction factor f = median(rel_range) / H for '{label}'\n")
        fh.write(f"# genes: {args.genes}   domains: {args.domains}\n")
        fh.write("# rows with numeric H: f_low/f_high = 5th/95th percentile over replicates\n")
        fh.write("# row 'pooled': f_median = median over H, f_low/f_high = min/max over H\n")
        out.to_csv(fh, sep='\t', index=False, float_format='%.5f')
    print(f"\nTable written to {args.output}")
    print("Calibration row for the estimator:  "
          f"median_n = {median_n:g}   f = {f_pooled:.4f}")


if __name__ == '__main__':
    main()
