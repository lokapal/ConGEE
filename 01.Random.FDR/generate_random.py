#!/usr/bin/env python3
"""
generate_random.py
------------------
Generate randomised (genes BED + domains BED) dataset pairs for FDR estimation.

Unlike generate_datasets.py there is NO concordance model.  Gene coordinates
are placed randomly per chromosome and strand using parameters derived from the
real annotation, and expression is either copied as-is or drawn uniformly from
the observed range.  These datasets serve as null-model replicates to calculate
empirical FDR for concordance analysis.

Gene placement model
--------------------
For each chromosome, + and − strands are handled independently:

  1. Draw n_genes lengths ~ Uniform(min_len, max_len) for that chromosome.
  2. Compute free_space = chrom_range − Σ lengths.
     If free_space < 0, lengths are scaled down proportionally to fit.
  3. Draw n_genes uniform positions in [0, free_space], sort them.
     Position u_i becomes the gap before gene i:
       start_i = chrom_min + u_i + Σ lengths[:i]
       end_i   = start_i + length_i
  4. Join both strands, sort by start.

This gives a uniform distribution over all valid non-overlapping placements on
each strand, preserving roughly the observed gene count per chromosome.

Expression model
----------------
  original  Copy the real expression values from the annotation as-is.
  uniform   Draw from Uniform(min_score, max_score) per chromosome.

Domain geometry
---------------
Identical to generate_datasets.py.  Domain sizes ~ Normal(mean, SD).
Two gap modes:
  normal   gap ~ Normal(mean_gap, SD)   -- SD = GAP_SD_FRACTION × mean_gap
  uniform  gap ~ Uniform(1, max_gap)   -- max_gap = --mean-gap value

Usage
-----
  python generate_random.py --genes hg38_genes.bed --n-files 100

  python generate_random.py \\
      --genes hg38_genes.bed \\
      --expression uniform \\
      --gap-mode uniform --mean-gap 80000 \\
      --mean-domain-size 1000000 \\
      --n-files 100 --cores 8 \\
      --outdir ./null_datasets
"""

import argparse
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

DOMAIN_SIZE_SD_FRACTION = 0.025   # SD = 2.5% of mean domain size
GAP_SD_FRACTION         = 0.05    # SD = 5%  of mean gap (normal mode only)

EXCLUDED_CHROMS = {'chrM', 'chrMT', 'MT', 'M', 'chrEBV'}


# ──────────────────────────────────────────────────────────────────────────────
# 1.  Gene annotation loading
# ──────────────────────────────────────────────────────────────────────────────

def load_genes_bed(path: str) -> pd.DataFrame:
    """
    Load a gene annotation BED5 file.

    Expected columns: chrom, start, end, score, strand
    Score is the expression value (column 4).
    """
    df = pd.read_csv(
        path, sep='\t', header=None, comment='#',
        names=['chrom', 'start', 'end', 'score', 'strand'],
        usecols=[0, 1, 2, 3, 4],
        dtype={'chrom': str, 'start': int, 'end': int,
               'score': float, 'strand': str},
    )
    df = df[~df['chrom'].isin(EXCLUDED_CHROMS)].reset_index(drop=True)
    df['length'] = (df['end'] - df['start']).abs()
    return df


def analyze_annotation(genes: pd.DataFrame) -> dict:
    """
    Derive per-chromosome parameters from the annotation:
        chrom_min     : minimum start coordinate
        chrom_max     : maximum end coordinate
        min_len       : minimum gene length
        max_len       : maximum gene length
        min_score     : minimum expression value
        max_score     : maximum expression value
        n_plus        : number of + strand genes
        n_minus       : number of − strand genes
        scores_plus   : array of + strand scores  (for 'original' mode)
        scores_minus  : array of − strand scores

    Returns a dict keyed by chromosome name.
    """
    params = {}
    for chrom, grp in genes.groupby('chrom'):
        plus  = grp[grp['strand'] == '+']
        minus = grp[grp['strand'] == '-']
        params[chrom] = {
            'chrom_min'    : int(grp['start'].min()),
            'chrom_max'    : int(grp['end'].max()),
            'min_len'      : int(grp['length'].min()),
            'max_len'      : int(grp['length'].max()),
            'min_score'    : float(grp['score'].min()),
            'max_score'    : float(grp['score'].max()),
            'n_plus'       : len(plus),
            'n_minus'      : len(minus),
            'scores_plus'  : plus['score'].values.copy(),
            'scores_minus' : minus['score'].values.copy(),
        }
    return params


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Random gene placement
# ──────────────────────────────────────────────────────────────────────────────

def _place_genes_on_strand(rng: np.random.Generator,
                            n: int,
                            chrom_min: int,
                            chrom_max: int,
                            min_len: int,
                            max_len: int,
                            strand: str) -> pd.DataFrame:
    """
    Place n non-overlapping genes uniformly on one strand of a chromosome.

    Lengths are drawn from Uniform(min_len, max_len).
    Positions are drawn using the order-statistics method:
        free_space = (chrom_max - chrom_min) - Σ lengths
        u_sorted   = sorted uniform draws in [0, free_space]
        start_i    = chrom_min + u_sorted[i] + Σ lengths[:i]

    If total length exceeds the chromosome range, lengths are scaled down
    proportionally so they fit exactly.

    Returns a DataFrame with columns: chrom (placeholder ''), start, end, strand.
    """
    if n == 0:
        return pd.DataFrame(columns=['start', 'end', 'strand'])

    lengths = rng.integers(min_len, max_len + 1, size=n)
    chrom_range = chrom_max - chrom_min

    # Scale down if genes don't fit
    total_len = int(lengths.sum())
    if total_len >= chrom_range:
        scale   = (chrom_range * 0.95) / total_len
        lengths = np.maximum(1, (lengths * scale).astype(int))
        total_len = int(lengths.sum())

    free_space = chrom_range - total_len

    # Order-statistics placement: n uniform draws in [0, free_space]
    u = np.sort(rng.uniform(0, free_space, size=n)).astype(int)
    cumlen = np.concatenate([[0], np.cumsum(lengths[:-1])])
    starts = chrom_min + u + cumlen
    ends   = starts + lengths

    return pd.DataFrame({'start': starts, 'end': ends,
                          'strand': strand})


def generate_random_genes(rng: np.random.Generator,
                           chrom: str,
                           params: dict,
                           expression_mode: str) -> pd.DataFrame:
    """
    Generate randomly placed genes for one chromosome.

    Plus and minus strands are handled independently, then merged and sorted
    by start coordinate.  Expression is assigned according to expression_mode:
        'original'  Shuffle the real scores from the annotation.
        'uniform'   Draw from Uniform(min_score, max_score).

    Returns a BED5 DataFrame: chrom, start, end, score, strand.
    """
    p = params[chrom]

    # Place genes strand by strand
    df_plus  = _place_genes_on_strand(
        rng, p['n_plus'],  p['chrom_min'], p['chrom_max'],
        p['min_len'], p['max_len'], '+')
    df_minus = _place_genes_on_strand(
        rng, p['n_minus'], p['chrom_min'], p['chrom_max'],
        p['min_len'], p['max_len'], '-')

    df = pd.concat([df_plus, df_minus], ignore_index=True)
    df.insert(0, 'chrom', chrom)
    df = df.sort_values('start').reset_index(drop=True)

    n = len(df)
    if n == 0:
        df['score'] = pd.Series(dtype=float)
        return df

    # Assign expression
    if expression_mode == 'original':
        # Pool + and − real scores, shuffle, assign in start order
        real_scores = np.concatenate([p['scores_plus'], p['scores_minus']])
        if len(real_scores) >= n:
            scores = rng.choice(real_scores, size=n, replace=False)
        else:
            scores = rng.choice(real_scores, size=n, replace=True)
    else:   # 'uniform'
        scores = rng.uniform(p['min_score'], p['max_score'], size=n)

    df['score'] = np.round(scores, 6)
    return df[['chrom', 'start', 'end', 'score', 'strand']]


# ──────────────────────────────────────────────────────────────────────────────
# 3.  Domain generation  (unchanged from generate_datasets.py + uniform option)
# ──────────────────────────────────────────────────────────────────────────────

def generate_domains(rng: np.random.Generator,
                     chrom_params: dict,
                     mean_domain_size: int,
                     mean_gap: int,
                     gap_mode: str = 'normal',
                     domain_size_sd_frac: float = DOMAIN_SIZE_SD_FRACTION,
                     gap_sd_frac: float = GAP_SD_FRACTION) -> pd.DataFrame:
    """
    Generate non-overlapping random domains tiling the genome.

    Domain sizes ~ Normal(mean_domain_size, (domain_size_sd_frac×mean)²).
    Domains that extend past the chromosome boundary are discarded entirely
    (not clipped) to preserve the size distribution.

    Gap modes
    ---------
    normal   gap ~ Normal(mean_gap, (gap_sd_frac × mean_gap)²)
             -- mean_gap is the mean gap
    uniform  gap ~ Uniform(1, mean_gap)
             -- mean_gap is the maximum allowed gap

    Returns a BED3 DataFrame: chrom, start, end.
    """
    domain_size_sd = domain_size_sd_frac * mean_domain_size
    gap_sd         = gap_sd_frac * mean_gap

    all_rows = []

    for chrom, p in sorted(chrom_params.items()):
        chrom_min = p['chrom_min']
        chrom_max = p['chrom_max']
        pos = chrom_min

        while True:
            if gap_mode == 'uniform':
                gap = int(rng.integers(1, mean_gap + 1))
            else:
                gap = max(1, int(round(rng.normal(mean_gap, gap_sd))))

            pos += gap
            if pos >= chrom_max:
                break

            size = max(1, int(round(rng.normal(mean_domain_size, domain_size_sd))))
            end  = pos + size

            if end > chrom_max:
                break

            all_rows.append((chrom, pos, end))
            pos = end

    return pd.DataFrame(all_rows, columns=['chrom', 'start', 'end'])


# ──────────────────────────────────────────────────────────────────────────────
# 4.  I/O helpers
# ──────────────────────────────────────────────────────────────────────────────

def write_domains_bed(domains: pd.DataFrame, path: Path) -> None:
    domains[['chrom', 'start', 'end']].to_csv(
        path, sep='\t', header=False, index=False)


def write_genes_bed(genes: pd.DataFrame, path: Path) -> None:
    genes[['chrom', 'start', 'end', 'score', 'strand']].to_csv(
        path, sep='\t', header=False, index=False, float_format='%.6f')


# ──────────────────────────────────────────────────────────────────────────────
# 5.  Per-replicate worker
# ──────────────────────────────────────────────────────────────────────────────

def _run_replicate(task: dict) -> dict:
    """
    Generate one (domains, genes) replicate and write BED files.

    Reproducibility guarantee: replicate i is seeded with base_seed + i.
    Output is bit-for-bit identical regardless of core count.
    """
    i                = task['i']
    base_seed        = task['base_seed']
    chrom_params     = task['chrom_params']
    outdir           = task['outdir']
    tag              = task['tag']
    mean_domain_size = task['mean_domain_size']
    mean_gap         = task['mean_gap']
    gap_mode         = task['gap_mode']
    expression_mode  = task['expression_mode']
    chroms           = task['chroms']

    t0  = time.time()
    rng = np.random.default_rng(base_seed + i)

    # Phase A: domain layout
    domains = generate_domains(
        rng, chrom_params, mean_domain_size, mean_gap, gap_mode=gap_mode)

    # Phase B: random gene placement + expression for every chromosome
    gene_frames = []
    for chrom in chroms:
        if chrom in chrom_params:
            gene_frames.append(
                generate_random_genes(rng, chrom, chrom_params, expression_mode))
    genes = pd.concat(gene_frames, ignore_index=True) if gene_frames else \
            pd.DataFrame(columns=['chrom', 'start', 'end', 'score', 'strand'])

    dom_path  = outdir / f"domains_{tag}_{i+1:04d}.bed"
    gene_path = outdir / f"genes_{tag}_{i+1:04d}.bed"
    write_domains_bed(domains, dom_path)
    write_genes_bed(genes, gene_path)

    dom_sizes = (domains['end'] - domains['start']).values

    return {
        'replicate'           : i + 1,
        'tag'                 : tag,
        'expression_mode'     : expression_mode,
        'gap_mode'            : gap_mode,
        'mean_domain_size_bp' : mean_domain_size,
        'mean_gap_bp'         : mean_gap,
        'n_domains'           : len(domains),
        'obs_domain_mean_bp'  : float(dom_sizes.mean()) if len(dom_sizes) else 0.0,
        'obs_domain_sd_bp'    : float(dom_sizes.std())  if len(dom_sizes) else 0.0,
        'n_genes'             : len(genes),
        'genes_file'          : str(gene_path),
        'domains_file'        : str(dom_path),
        '_i'                  : i,
        '_elapsed'            : time.time() - t0,
    }


def _print_progress(result: dict, n_files: int) -> None:
    print(
        f"  [{result['replicate']:4d}/{n_files}]  "
        f"domains={result['n_domains']:,}  "
        f"genes={result['n_genes']:,}  "
        f"({result['_elapsed']:.1f}s)",
        flush=True)


# ──────────────────────────────────────────────────────────────────────────────
# 6.  Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog='generate_random.py',
        description=(
            "Generate randomised (genes, domains) BED pairs for FDR estimation.\n"
            "Gene positions are placed randomly per chromosome and strand\n"
            "using parameters derived from the real annotation. No concordance\n"
            "model is applied — these are pure null-model replicates."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── Input ─────────────────────────────────────────────────────────────────
    inp = parser.add_argument_group("Input")
    inp.add_argument(
        '--genes', required=True, metavar='FILE',
        help="Gene annotation BED5: chrom start end score strand. "
             "Chromosome ranges, gene length ranges, and expression ranges "
             "are derived from this file per chromosome.")

    # ── Expression ────────────────────────────────────────────────────────────
    exp = parser.add_argument_group("Expression model")
    exp.add_argument(
        '--expression', choices=['original', 'uniform'],
        default='uniform', metavar='MODE',
        help="Expression assignment per gene.\n"
             "  original : randomly sample from real scores in the annotation\n"
             "             (each replicate draws without replacement, or with\n"
             "              replacement if n_random > n_real on that chromosome)\n"
             "  uniform  : draw from Uniform(min_score, max_score) per chromosome\n"
             "Default: uniform")

    # ── Domain geometry ───────────────────────────────────────────────────────
    dom = parser.add_argument_group(
        "Domain geometry",
        "Domain sizes ~ Normal(mean, SD), SD = {:.1f}%% of mean.\n"
        "Gap distribution selected by --gap-mode.".format(
            DOMAIN_SIZE_SD_FRACTION * 100))
    dom.add_argument(
        '--mean-domain-size', type=int, default=1_000_000, metavar='BP',
        help=f"Mean domain size in bp. "
             f"SD = {DOMAIN_SIZE_SD_FRACTION*100:.1f}%% of mean. Default: 1000000.")
    dom.add_argument(
        '--mean-gap', type=int, default=40_000, metavar='BP',
        help="Inter-domain gap parameter in bp.\n"
             "  normal mode  : mean of Normal gap distribution\n"
             "                 (SD = {:.1f}%% of mean)\n"
             "  uniform mode : maximum allowed gap (gap ~ Uniform(1, max_gap))\n"
             "Default: 40000.".format(GAP_SD_FRACTION * 100))
    dom.add_argument(
        '--gap-mode', choices=['normal', 'uniform'], default='normal',
        metavar='MODE',
        help="Gap size distribution.\n"
             "  normal  : gap ~ Normal(mean_gap, SD)  [default]\n"
             "  uniform : gap ~ Uniform(1, mean_gap)")

    # ── Output ────────────────────────────────────────────────────────────────
    out = parser.add_argument_group("Output")
    out.add_argument(
        '--n-files', type=int, default=100, metavar='N',
        help="Number of replicate pairs to generate. Default: 100.")
    out.add_argument(
        '--outdir', type=str, default='./random_datasets', metavar='DIR',
        help="Output directory (created if absent). Default: ./random_datasets.")
    out.add_argument(
        '--tag', type=str, default=None, metavar='TAG',
        help="Label embedded in filenames. "
             "Default: rand_<expression_mode>_<gap_mode>.")
    out.add_argument(
        '--seed', type=int, default=42, metavar='INT',
        help="Base random seed. Replicate i uses seed+i. Default: 42.")

    # ── Parallelism ───────────────────────────────────────────────────────────
    par = parser.add_argument_group("Parallelism")
    par.add_argument(
        '--cores', type=int, default=max(1, cpu_count() - 1), metavar='N',
        help="Worker processes. 1 = serial (identical output to any parallel run). "
             f"Default: {max(1, cpu_count()-1)}.")

    args = parser.parse_args()

    # ── Validate ───────────────────────────────────────────────────────────────
    if args.n_files < 1:
        parser.error("--n-files must be >= 1")
    if args.mean_domain_size < 1:
        parser.error("--mean-domain-size must be >= 1")
    if args.mean_gap < 1:
        parser.error("--mean-gap must be >= 1")
    if args.cores < 1:
        parser.error("--cores must be >= 1")

    tag = (args.tag if args.tag
           else f"rand_{args.expression}_{args.gap_mode}")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ── Load annotation and derive per-chromosome parameters ──────────────────
    print("=" * 65)
    print(f"Loading gene annotation: {args.genes}")
    genes_df     = load_genes_bed(args.genes)
    chrom_params = analyze_annotation(genes_df)
    chroms       = sorted(chrom_params.keys())

    total_genes = sum(p['n_plus'] + p['n_minus'] for p in chrom_params.values())
    print(f"  Chromosomes : {len(chroms)}")
    print(f"  Total genes : {total_genes:,}  "
          f"(avg {total_genes/len(chroms):.0f} per chromosome)")

    # Print per-chromosome summary
    print(f"\n  {'chrom':<10}  {'range (Mbp)':>12}  {'len (bp)':>14}  "
          f"{'expr':>14}  {'n+':>6}  {'n-':>6}")
    for chrom in chroms:
        p = chrom_params[chrom]
        span_mbp = (p['chrom_max'] - p['chrom_min']) / 1e6
        print(f"  {chrom:<10}  {span_mbp:>11.1f}  "
              f"  [{p['min_len']:,}–{p['max_len']:,}]"
              f"  [{p['min_score']:.1f}–{p['max_score']:.1f}]"
              f"  {p['n_plus']:>6}  {p['n_minus']:>6}")

    gap_desc = (f"Normal(mean={args.mean_gap:,} bp, "
                f"SD={GAP_SD_FRACTION*100:.1f}%% of mean)"
                if args.gap_mode == 'normal'
                else f"Uniform(1, {args.mean_gap:,} bp)")

    domain_sd = DOMAIN_SIZE_SD_FRACTION * args.mean_domain_size
    print(f"\n  Domain size  : Normal({args.mean_domain_size:,} bp, "
          f"SD={domain_sd:,.0f} bp = {DOMAIN_SIZE_SD_FRACTION*100:.1f}%% of mean)")
    print(f"  Domain gap   : {gap_desc}")
    print(f"  Expression   : {args.expression}")
    print(f"  Replicates   : {args.n_files}")
    n_cores_actual = min(args.cores, args.n_files)
    print(f"  Cores        : {n_cores_actual}"
          f"{'  (serial)' if n_cores_actual == 1 else '  (parallel)'}")
    print(f"  Seed         : {args.seed}")
    print(f"  Output dir   : {outdir.resolve()}")
    print(f"  Tag          : {tag}")
    print("=" * 65)

    # ── Build task list ────────────────────────────────────────────────────────
    tasks = [
        dict(
            i=i, base_seed=args.seed,
            chrom_params=chrom_params, chroms=chroms,
            outdir=outdir, tag=tag,
            mean_domain_size=args.mean_domain_size,
            mean_gap=args.mean_gap,
            gap_mode=args.gap_mode,
            expression_mode=args.expression,
        )
        for i in range(args.n_files)
    ]

    # ── Execute ────────────────────────────────────────────────────────────────
    t_total    = time.time()
    stats_rows = [None] * args.n_files

    if n_cores_actual == 1:
        for task in tasks:
            result = _run_replicate(task)
            stats_rows[result['_i']] = result
            _print_progress(result, args.n_files)
    else:
        with Pool(processes=n_cores_actual) as pool:
            for result in pool.imap_unordered(_run_replicate, tasks):
                stats_rows[result['_i']] = result
                _print_progress(result, args.n_files)

    # ── Write summary TSV ──────────────────────────────────────────────────────
    public_rows = [{k: v for k, v in r.items() if not k.startswith('_')}
                   for r in stats_rows]
    stats_df   = pd.DataFrame(public_rows)
    stats_path = outdir / f"summary_{tag}.tsv"
    stats_df.to_csv(stats_path, sep='\t', index=False, float_format='%.4f')

    print("=" * 65)
    print(f"Done. {args.n_files} replicates in {time.time()-t_total:.1f}s")
    print(f"Summary: {stats_path}")


if __name__ == '__main__':
    main()
