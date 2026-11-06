#!/usr/bin/env python3
import argparse
import sys
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count, shared_memory
import time
from pathlib import Path
from scipy.stats import gaussian_kde, norm

__version__ = '1.3'

_Z_SIGNIFICANCE = -1.96
_P_FLOOR = 1e-300

# Precomputed calibration table: correction_factor(n_genes) such that
#   H_est = concordant_median / correction_factor
# where concordant_median = median rel_range of domains below the KDE valley.
# Derived from simulation of Uniform(1-H, 1+H) noise across H in [0.05, 0.50]
# and Poisson(n_mean) gene counts; values are stable across H to ±3%.
_CALIB_N   = np.array([ 2,     3,     5,     7,     10,    15,    20,    25,    30])
_CALIB_FAC = np.array([0.777, 0.987, 1.339, 1.522, 1.662, 1.774, 1.830, 1.864, 1.886])


def _diagnostic(message, *, verbose=False):
    """Routine diagnostics are opt-in; input-repair warnings are emitted separately."""
    if verbose:
        print(message, file=sys.stderr)

"""
Gene Expression Concordance in Genomic Domains (Cyclic Shift Test)
ConGEE: HDF5-only splines (conc_spline_final.py / make_splines_h5.py),
          2D correction mesh, strand-aware gene mapping and analytic p-values
          for the z_mean and z_var statistics.

"""

# ── Static default for the concordance spline level ──────────────────────────
# Supported calibration quantiles include '2.5%', '5%', and '50%'. The final
# median-corrected C is a model-calibrated estimate, not a verified confidence
# bound. Calibration files must match the annotation and assignment policy.
# Override at runtime with --spline-level.
DEFAULT_CONCORDANCE_LEVEL = '5%'

# Default splines file loaded automatically when --splines is not given.
# Set to None to disable auto-loading.
DEFAULT_SPLINES_FILE = 'splines.h5'

# Global state for worker processes
_shm_expr        = None
_shm_indices     = None
_shm_weights     = None
_shm_domain_ptr  = None
_shm_chrom       = None

_expr_shape       = None
_expr_dtype       = None
_n_domains        = None
_indices_shape    = None
_weights_shape    = None
_domain_ptr_shape = None
_chrom_shape      = None
_chrom_dtype      = None


def worker_init(shm_names, shapes, dtypes, n_domains):
    global _shm_expr, _shm_indices, _shm_weights, _shm_domain_ptr, _shm_chrom
    global _expr_shape, _expr_dtype, _n_domains
    global _indices_shape, _weights_shape, _domain_ptr_shape, _chrom_shape, _chrom_dtype

    _n_domains = n_domains

    _shm_expr = shared_memory.SharedMemory(name=shm_names['expr'])
    _expr_shape = shapes['expr']
    _expr_dtype = dtypes['expr']

    _shm_indices = shared_memory.SharedMemory(name=shm_names['indices'])
    _indices_shape = shapes['indices']

    _shm_weights = shared_memory.SharedMemory(name=shm_names['weights'])
    _weights_shape = shapes['weights']

    _shm_domain_ptr = shared_memory.SharedMemory(name=shm_names['domain_ptr'])
    _domain_ptr_shape = shapes['domain_ptr']

    _shm_chrom = shared_memory.SharedMemory(name=shm_names['chrom'])
    _chrom_shape = shapes['chrom']
    _chrom_dtype = dtypes['chrom']

# 1. Statistics Calculation (Formulas 1-4)
def calculate_statistics_vec(domain_sums, domain_sum_sq, domain_counts):
    # Only domains with >= 2 genes are valid for variance estimation
    valid = domain_counts >= 2
    if not valid.any():
        return {'mean_Ie': float('nan'), 'mean_sigma2': float('nan')}

    n  = domain_counts[valid].astype(np.float64)
    s  = domain_sums[valid]
    sq = domain_sum_sq[valid]

    # Formula (1): mean expression inside domain r  ->  I_e,r = sum / N_int,r
    means = s / n

    # Formula (3): unbiased variance inside domain r
    #   sigma^2_e,r = (sum_sq - N * mean^2) / (N - 1)
    vars_ = (sq - n * means ** 2) / (n - 1.0)
    vars_ = np.maximum(vars_, 0.0)   # guard against floating-point negatives

    return {
        'mean_Ie':     float(np.mean(means)),   # Formula (2): <I_e,r>_r
        'mean_sigma2': float(np.mean(vars_)),   # Formula (4): <sigma^2_e,r>_r
    }


def _prepare_csr_statistics(domain_ptr):

    counts = np.diff(domain_ptr).astype(np.int64, copy=False)
    nonempty = np.flatnonzero(counts > 0)
    entry_domains = np.repeat(np.arange(len(counts), dtype=np.int64), counts)
    return counts, nonempty, domain_ptr[nonempty], entry_domains


def _statistics_for_values(weighted_values, prepared):
    counts, nonempty, starts, entry_domains = prepared
    valid = counts >= 2
    if not valid.any():
        return {'mean_Ie': float('nan'), 'mean_sigma2': float('nan')}
    means = np.zeros(len(counts), dtype=np.float64)
    with np.errstate(over='raise', invalid='raise'):
        means[nonempty] = np.add.reduceat(weighted_values, starts) / counts[nonempty]
        residuals = weighted_values - means[entry_domains]
        sums_sq = np.zeros(len(counts), dtype=np.float64)
        sums_sq[nonempty] = np.add.reduceat(residuals * residuals, starts)
        residual_sums = np.zeros(len(counts), dtype=np.float64)
        residual_sums[nonempty] = np.add.reduceat(residuals, starts)
        variances = (sums_sq[valid] - residual_sums[valid] ** 2 / counts[valid]) / (counts[valid] - 1)
        return {
            'mean_Ie': float(np.mean(means[valid])),
            'mean_sigma2': float(np.mean(np.maximum(variances, 0.0))),
        }


def _batch_statistics(expression_array, indices, weights, domain_ptr,
                      chrom_arr, scope, seeds):
    prepared = _prepare_csr_statistics(domain_ptr)
    if scope == 'chromosome':
        chrom_groups = [np.flatnonzero(chrom_arr == c) for c in np.unique(chrom_arr)]
    results = []
    rng = np.random.RandomState(0)
    for seed in seeds:
        rng.seed(int(seed))
        if scope == 'genome':
            shifted = np.roll(expression_array, rng.randint(0, len(expression_array)))
        else:
            shifted = expression_array.copy()
            for rows in chrom_groups:
                shifted[rows] = np.roll(expression_array[rows], rng.randint(0, len(rows)))
        results.append(_statistics_for_values(shifted[indices] * weights, prepared))
    return results


def batch_permutation_worker(args):
    seeds, scope, batch_size = args

    expr_array = np.ndarray(_expr_shape,       dtype=_expr_dtype,  buffer=_shm_expr.buf)
    indices    = np.ndarray(_indices_shape,    dtype=np.int32,     buffer=_shm_indices.buf)
    weights    = np.ndarray(_weights_shape,    dtype=np.float64,   buffer=_shm_weights.buf)
    domain_ptr = np.ndarray(_domain_ptr_shape, dtype=np.int64,     buffer=_shm_domain_ptr.buf)
    chrom_arr  = np.ndarray(_chrom_shape,      dtype=_chrom_dtype, buffer=_shm_chrom.buf)

    return _batch_statistics(expr_array, indices, weights, domain_ptr,
                             chrom_arr, scope, seeds[:batch_size])

def _interval_coordinates(frame, label, *, repair_reversed=False):
    required = {'chrom', 'start', 'end'}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{label}: missing columns {sorted(missing)}")
    if frame['chrom'].isna().any() or (frame['chrom'].astype(str).str.len() == 0).any():
        raise ValueError(f"{label}: chromosome names must be nonempty")
    values = []
    for column in ('start', 'end'):
        try:
            arr = np.asarray(pd.to_numeric(frame[column], errors='raise'))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label}: {column} coordinates must be integers") from exc
        if (arr.dtype.kind not in 'iuf' or not np.isfinite(arr).all()
                or (arr < 0).any() or (arr >= 2**63).any()
                or (arr.dtype.kind == 'f' and (arr != np.floor(arr)).any())):
            raise ValueError(
                f"{label}: {column} coordinates must be finite nonnegative "
                "integers smaller than 2**63")
        values.append(arr.astype(np.int64, copy=False))
    starts, ends = values
    invalid = np.flatnonzero(ends == starts if repair_reversed else ends <= starts)
    if len(invalid):
        raise ValueError(
            f"{label}: BED intervals require start < end "
            f"(invalid data row {int(invalid[0]) + 1})")
    if repair_reversed:
        reversed_rows = np.flatnonzero(starts > ends)
        if len(reversed_rows):
            examples = ', '.join(
                f'data row {int(row) + 1} ({frame.iloc[row]["chrom"]}:'
                f'{starts[row]}-{ends[row]} -> {ends[row]}-{starts[row]})'
                for row in reversed_rows[:5])
            if len(reversed_rows) > 5:
                examples += f'; {len(reversed_rows) - 5} additional interval(s)'
            print(f'Warning: {label}: non-standard BED3: swapped start/end in '
                  f'{len(reversed_rows)} interval(s) with start > end; {examples}',
                  file=sys.stderr)
            starts, ends = np.minimum(starts, ends), np.maximum(starts, ends)
    return starts, ends


class _IntervalIndex:

    __slots__ = ('center', 'by_start', 'starts', 'by_end', 'ends', 'left', 'right')

    def __init__(self, starts, ends, row_ids):
        self.center = int(np.partition(starts, len(starts) // 2)[len(starts) // 2])
        left = ends <= self.center
        right = starts > self.center
        middle = ~(left | right)
        middle_ids = row_ids[middle]
        middle_starts, middle_ends = starts[middle], ends[middle]
        order = np.lexsort((middle_ids, middle_starts))
        self.by_start, self.starts = middle_ids[order], middle_starts[order]
        order = np.lexsort((middle_ids, middle_ends))
        self.by_end, self.ends = middle_ids[order], middle_ends[order]
        self.left = _IntervalIndex(starts[left], ends[left], row_ids[left]) if left.any() else None
        self.right = _IntervalIndex(starts[right], ends[right], row_ids[right]) if right.any() else None

    def overlaps(self, start, end):
        """Yield each domain row overlapping [start, end), exactly once."""
        pending = [self]
        while pending:
            node = pending.pop()
            if end <= node.center:
                stop = np.searchsorted(node.starts, end, side='left')
                yield from node.by_start[:stop]
                if node.left is not None:
                    pending.append(node.left)
            elif start >= node.center:
                first = np.searchsorted(node.ends, start, side='right')
                yield from node.by_end[first:]
                if node.right is not None:
                    pending.append(node.right)
            else:
                yield from node.by_start
                if node.left is not None:
                    pending.append(node.left)
                if node.right is not None:
                    pending.append(node.right)


def map_genes_to_domains(genes_df, domains_df, mode):
    """
    Modes
    -----
    start        Transcription start site (strand-aware):
                 g_start for + strand genes, g_end - 1 for - strand genes.
    end          Transcription end site (strand-aware):
                 g_end - 1 for + strand genes, g_start for - strand genes.
    middle       Midpoint of the gene body.  Strand-agnostic.
    longer_part  Assign to the SINGLE domain with the largest gene-body overlap.
                 Ties: smallest domain start, then original domain row position.
    proportional Fractional assignment proportional to gene-body overlap per domain.
    """
    modes = {'start', 'end', 'middle', 'longer_part', 'proportional'}
    if mode not in modes:
        raise ValueError(f"Unknown mapping mode {mode!r}; choose from {sorted(modes)}")
    g_starts, g_ends = _interval_coordinates(genes_df, 'genes')
    d_starts, d_ends = _interval_coordinates(domains_df, 'domains', repair_reversed=True)
    if mode in ('start', 'end'):
        if 'strand' not in genes_df or not genes_df['strand'].isin(['+', '-']).all():
            raise ValueError(f"{mode} mapping requires '+' or '-' strand for every gene")
        strands = genes_df['strand'].to_numpy()

    domain_map = [[] for _ in range(len(domains_df))]
    gene_chroms = genes_df['chrom'].astype(str).to_numpy()
    domain_chroms = domains_df['chrom'].astype(str).to_numpy()
    for chrom in pd.unique(gene_chroms):
        gene_rows = np.flatnonzero(gene_chroms == chrom)
        domain_rows = np.flatnonzero(domain_chroms == chrom)
        if not len(domain_rows):
            continue
        tree = _IntervalIndex(d_starts[domain_rows], d_ends[domain_rows], domain_rows)
        for gene_row in gene_rows:
            gene_row = int(gene_row)
            start, end = int(g_starts[gene_row]), int(g_ends[gene_row])
            if mode in ('start', 'end', 'middle'):
                if mode == 'start':
                    point = start if strands[gene_row] == '+' else end - 1
                elif mode == 'end':
                    point = end - 1 if strands[gene_row] == '+' else start
                else:
                    point = start + (end - start) // 2
                for domain_row in tree.overlaps(point, point + 1):
                    domain_map[int(domain_row)].append((gene_row, 1.0))
                continue

            best = None
            for domain_row in tree.overlaps(start, end):
                domain_row = int(domain_row)
                overlap = min(end, int(d_ends[domain_row])) - max(start, int(d_starts[domain_row]))
                if mode == 'proportional':
                    domain_map[domain_row].append((gene_row, overlap / (end - start)))
                else:
                    candidate = (-overlap, int(d_starts[domain_row]), domain_row)
                    if best is None or candidate < best:
                        best = candidate
            if mode == 'longer_part' and best is not None:
                domain_map[best[2]].append((gene_row, 1.0))

    return domain_map


def domain_map_to_csr(domain_map):
    domain_ptr = np.zeros(len(domain_map) + 1, dtype=np.int64)
    for i, entries in enumerate(domain_map):
        domain_ptr[i + 1] = domain_ptr[i] + len(entries)

    total   = int(domain_ptr[-1])
    indices = np.empty(total, dtype=np.int32)
    weights = np.empty(total, dtype=np.float64)

    for i, entries in enumerate(domain_map):
        start = domain_ptr[i]
        for j, (g_idx, w) in enumerate(entries):
            indices[start + j] = g_idx
            weights[start + j] = w

    return indices, weights, domain_ptr

def compute_real_stats(expression_array, csr_indices, csr_weights, domain_ptr):
    """
    Compute Formulas 1-4 on the observed (non-permuted) expression data.
    Reuses the same vectorized path as the permutation worker.
    """
    return _statistics_for_values(
        expression_array[csr_indices] * csr_weights,
        _prepare_csr_statistics(domain_ptr))

def _integer_array(values, name, dtype):
    array = np.asarray(values)
    maximum = np.iinfo(dtype).max
    if array.ndim != 1:
        raise ValueError(f'{name} must be a one-dimensional integer array')
    if not len(array):
        return np.ascontiguousarray(array, dtype=dtype)
    if array.dtype.kind in 'iu':
        valid = not (array < 0).any() and not (array > maximum).any()
    elif array.dtype.kind == 'f':
        valid = (np.isfinite(array).all() and not (array < 0).any()
                 and not (array >= maximum + 1).any()
                 and not (array != np.floor(array)).any())
    else:
        valid = False
    if not valid:
        raise ValueError(f'{name} must contain finite nonnegative integers <= {maximum}')
    return np.ascontiguousarray(array, dtype=dtype)


def run_permutations(expression_array, csr_indices, csr_weights, domain_ptr,
                     chrom_int, scope, n_perms, n_cores, chunk_size=100, seed=None):
    if scope not in ('genome', 'chromosome'):
        raise ValueError("scope must be 'genome' or 'chromosome'")
    for name, value, minimum in (('n_perms', n_perms, 2), ('n_cores', n_cores, 1),
                                 ('chunk_size', chunk_size, 1)):
        if not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    expression_array = np.ascontiguousarray(expression_array, dtype=np.float64)
    csr_indices = _integer_array(csr_indices, 'csr_indices', np.int32)
    csr_weights = np.ascontiguousarray(csr_weights, dtype=np.float64)
    domain_ptr = _integer_array(domain_ptr, 'domain_ptr', np.int64)
    chrom_int = np.ascontiguousarray(chrom_int)
    if expression_array.ndim != 1 or not len(expression_array) or not np.isfinite(expression_array).all():
        raise ValueError('expression_array must be a nonempty finite one-dimensional array')
    if chrom_int.ndim != 1 or len(chrom_int) != len(expression_array) or chrom_int.dtype.hasobject:
        raise ValueError('chrom_int must be a non-object one-dimensional array aligned with expression_array')
    if (csr_indices.ndim != 1 or csr_weights.ndim != 1
            or len(csr_indices) != len(csr_weights) or not np.isfinite(csr_weights).all()
            or (csr_indices < 0).any() or (csr_indices >= len(expression_array)).any()):
        raise ValueError('invalid CSR gene indices or weights')
    if (domain_ptr.ndim != 1 or not len(domain_ptr) or domain_ptr[0] != 0
            or domain_ptr[-1] != len(csr_indices) or (np.diff(domain_ptr) < 0).any()):
        raise ValueError('invalid CSR domain pointers')

    n_cores = min(n_cores, n_perms)
    rng = np.random.default_rng(seed)
    all_seeds = rng.integers(0, 2**32, size=n_perms, dtype=np.uint32)
    if n_cores == 1:
        return _batch_statistics(expression_array, csr_indices, csr_weights,
                                 domain_ptr, chrom_int, scope, all_seeds)

    arrays = dict(expr=expression_array, indices=csr_indices, weights=csr_weights,
                  domain_ptr=domain_ptr, chrom=chrom_int)
    segments = []
    try:
        shm_names = {}
        for key, arr in arrays.items():
            shm = shared_memory.SharedMemory(create=True, size=max(arr.nbytes, 1))
            segments.append(shm)
            np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)[:] = arr
            shm_names[key] = shm.name
        shapes = {key: arr.shape for key, arr in arrays.items()}
        dtypes = dict(expr=expression_array.dtype, chrom=chrom_int.dtype)
        batch_size = max(1, min(chunk_size, n_perms // n_cores))
        worker_args = [(all_seeds[i:i + batch_size], scope,
                        len(all_seeds[i:i + batch_size]))
                       for i in range(0, n_perms, batch_size)]
        with Pool(processes=n_cores, initializer=worker_init,
                  initargs=(shm_names, shapes, dtypes, len(domain_ptr) - 1)) as pool:
            batch_results = pool.map(batch_permutation_worker, worker_args)
        return [result for batch in batch_results for result in batch]
    finally:
        for shm in segments:
            try:
                shm.close()
            finally:
                shm.unlink()

# 6. Z-score calculation (Formulas 5-8)
def zscores_from_results(real_stats, perm_results):
    """
    Computes z-scores comparing observed statistics to the permutation distribution.

    Formula (8): sigma^2(I_ec)             = variance of permuted mean_Ie values
    Formula (5): tilde_sigma^2(sigma_ec^2) = variance of permuted mean_sigma2 values
    Formula (7): Z_mean = (I_e - <I_ec>_c) / sqrt(sigma^2(I_ec))
    Formula (6): Z_var  = (sigma^2_e - <sigma^2_ec>_c) / sqrt(tilde_sigma^2(sigma_ec^2))
    """
    def standardize(key):
        observed = real_stats[key]
        values = np.asarray([result[key] for result in perm_results], dtype=np.float64)
        if len(values) < 2 or not np.isfinite(observed) or not np.isfinite(values).all():
            return float('nan')
        average = float(np.mean(values))
        centered = values - average
        scale = float(np.max(np.abs(centered)))
        resolution = 8 * np.finfo(np.float64).eps * float(np.max(np.abs(values)))
        if scale <= resolution:
            return float('nan')
        sd = scale * np.sqrt(np.sum((centered / scale) ** 2) / (len(values) - 1))
        if not np.isfinite(sd) or sd <= 0:
            return float('nan')
        return float((observed - average) / sd)

    return standardize('mean_Ie'), standardize('mean_sigma2')

# 7. Noise estimation from domain rel-ranges
def compute_domain_rel_ranges(expression_array, domain_map,
                              genes_df=None, domains_df=None):
    expression_array = np.asarray(expression_array, dtype=np.float64)
    if not np.isfinite(expression_array).all() or (expression_array < 0).any():
        raise ValueError('H estimation requires finite nonnegative expression values')
    if (genes_df is None) != (domains_df is None):
        raise ValueError('supply both genes_df and domains_df, or neither')
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
        vals         = expression_array[gene_indices]
        mean_val     = vals.mean()
        if mean_val <= 0:
            continue
        rel_ranges.append((vals.max() - vals.min()) / mean_val)
        n_genes_per_domain.append(len(gene_indices))

    return np.array(rel_ranges), np.array(n_genes_per_domain, dtype=np.int32)


def _find_kde_valley(rel_ranges, n_points=2000, min_valley_depth=0.15):
    rel_ranges = np.asarray(rel_ranges, dtype=np.float64)
    positive = rel_ranges[np.isfinite(rel_ranges) & (rel_ranges > 0)]
    if len(positive) < 20:
        return None, None, None

    lo_cut    = np.percentile(positive, 1)
    rr_trim   = positive[positive >= lo_cut]
    if len(rr_trim) < 10:
        return None, None, None
    log_rr = np.log(rr_trim)

    try:
        kde    = gaussian_kde(log_rr, bw_method='silverman')
        log_x  = np.linspace(log_rr.min(), log_rr.max(), n_points)
        log_y  = kde(log_x)
    except (ValueError, np.linalg.LinAlgError):
        return None, None, None

    dy   = np.diff(log_y)
    sign = np.sign(dy)
    for i in range(1, len(sign)):
        if sign[i] == 0:
            sign[i] = sign[i - 1]

    changes = np.diff(sign)
    maxima  = np.where(changes < 0)[0] + 1
    minima  = np.where(changes > 0)[0] + 1

    if len(maxima) == 0 or len(minima) == 0:
        return None, log_x, log_y

    first_peak_idx = maxima[0]
    first_peak_h   = log_y[first_peak_idx]

    valid_minima = minima[minima > first_peak_idx]
    if len(valid_minima) == 0:
        return None, log_x, log_y

    MIN_VALLEY_ORIG            = 0.10
    MAX_VALLEY_ORIG            = 2.0    # 2 * H_max(0.5) * f_max(1.886) ≈ 1.886, rounded up
    MIN_VALLEY_TO_MEDIAN_RATIO = 1.8
    for valley_idx in valid_minima:
        valley_h      = log_y[valley_idx]
        valley_x_orig = float(np.exp(log_x[valley_idx]))
        depth         = (first_peak_h - valley_h) / first_peak_h
        if depth < min_valley_depth:
            continue
        if valley_x_orig < MIN_VALLEY_ORIG or valley_x_orig > MAX_VALLEY_ORIG:
            continue
        below = rel_ranges[rel_ranges < valley_x_orig]
        if len(below) < 5:
            continue
        med_below = float(np.median(below))
        if med_below <= 0 or valley_x_orig / med_below < MIN_VALLEY_TO_MEDIAN_RATIO:
            continue
        return valley_x_orig, log_x, log_y

    return None, log_x, log_y

def _correction_factor(median_n_genes):
    return float(np.interp(median_n_genes, _CALIB_N, _CALIB_FAC,
                           left=_CALIB_FAC[0], right=_CALIB_FAC[-1]))


def _calibrated_p25(rel_ranges, median_n_genes):
    med    = float(np.median(rel_ranges))
    factor = _correction_factor(median_n_genes)
    return float(np.clip(med / factor, 0.01, 0.99))


def estimate_noise_from_rel_ranges(rel_ranges, n_genes_per_domain=None, verbose=False):
    rel_ranges = np.asarray(rel_ranges, dtype=np.float64)
    if rel_ranges.ndim != 1 or not np.isfinite(rel_ranges).all() or (rel_ranges < 0).any():
        raise ValueError('rel_ranges must be a finite nonnegative one-dimensional array')
    if n_genes_per_domain is not None:
        n_genes_per_domain = np.asarray(n_genes_per_domain)
        if (n_genes_per_domain.shape != rel_ranges.shape
                or not np.isfinite(n_genes_per_domain).all()
                or (n_genes_per_domain < 2).any()):
            raise ValueError('n_genes_per_domain must match rel_ranges and contain counts >= 2')
    if len(rel_ranges) == 0:
        return float('nan'), {'method': 'no_eligible_positive_mean_domains',
                             'n_domains': 0, 'valley_x': None, 'H_est': float('nan')}

    n = len(rel_ranges)
    overall_median = float(np.median(rel_ranges))
    p25            = float(np.percentile(rel_ranges, 25))
    p75            = float(np.percentile(rel_ranges, 75))

    if n_genes_per_domain is not None and len(n_genes_per_domain) > 0:
        median_n = float(np.median(n_genes_per_domain))
    else:
        median_n = 5.0   
    factor = _correction_factor(median_n)

    valley_x, log_kde_x, log_kde_y = _find_kde_valley(rel_ranges)

    if valley_x is not None:
        below_valley = rel_ranges[rel_ranges < valley_x]
        if len(below_valley) >= 5:
            if n_genes_per_domain is not None:
                med_n_below = float(np.median(
                    n_genes_per_domain[rel_ranges < valley_x]))
                factor = _correction_factor(med_n_below)
            concordant_median   = float(np.median(below_valley))
            DEBIAS              = 1.01   # empirical: mean overestimate factor, C=0.2-0.9, for TAD scale only!
            H_est               = float(np.clip(concordant_median / factor / DEBIAS, 0.01, 0.99))
            method              = 'kde_valley'
            frac_concordant_est = len(below_valley) / n
        else:
            valley_x = None   # too few points below — fall back

    if valley_x is None:
        H_raw  = overall_median / factor
        if H_raw > 0.55:
            H_est  = 0.50
            method = 'median_fallback_capped'
        else:
            H_est  = float(np.clip(H_raw, 0.01, 0.99))
            method = 'median_fallback'
        frac_concordant_est = float(np.mean(rel_ranges < 2 * H_est))
        concordant_median   = overall_median

    diag = {
        'method'              : method,
        'n_domains'           : n,
        'valley_x'            : valley_x,
        'overall_median'      : overall_median,
        'p25'                 : p25,
        'p75'                 : p75,
        'median_n_genes'      : median_n,
        'correction_factor'   : factor,
        'concordant_median'   : concordant_median,
        'frac_concordant_est' : frac_concordant_est,
        'H_est'               : H_est,
        'n_zero_ranges'       : int(np.count_nonzero(rel_ranges == 0)),
    }

    if verbose:
        print(f"  [noise]  n_domains={n}  method={method}  "
              f"median_n_genes={median_n:.1f}  correction_factor={factor:.3f}",
              file=sys.stderr)
        print(f"  [noise]  rel_range: p25={p25:.3f}  median={overall_median:.3f}"
              f"  p75={p75:.3f}",
              file=sys.stderr)
        if valley_x is not None:
            print(f"  [noise]  KDE valley at {valley_x:.3f}  "
                  f"concordant_fraction≈{frac_concordant_est:.1%}",
                  file=sys.stderr)
        print(f"  [noise]  concordant_median={concordant_median:.3f}"
              f"  =>  H_est={H_est:.3f}",
              file=sys.stderr)

    return H_est, diag


def zscore_pvalues(z_mean: float, z_var: float) -> tuple:
    p_mean = 2.0 * norm.sf(abs(z_mean))
    p_var  = norm.cdf(z_var)
    return max(p_mean, _P_FLOOR), max(p_var, _P_FLOOR)


def emit(outfile_handle, sname_genes, sname_domains, chrom,
         z_mean, z_var, H_est, C_est):
    p_mean, p_var = zscore_pvalues(z_mean, z_var)
    c_field = f"{C_est:.6f}" if isinstance(C_est, float) else str(C_est)
    stats = (f"{z_mean:.8f}\t{p_mean:.3e}\t{z_var:.8f}\t{p_var:.3e}\t"
             f"{H_est:.4f}\t{c_field}")
    if chrom is None:
        line = f"{sname_genes}\t{sname_domains}\t{stats}"
    else:
        line = f"{sname_genes}\t{sname_domains}\t{chrom}\t{stats}"
    print(line)
    print(line, file=outfile_handle)


def _validate_spline(spline, c_min, c_max, label):
    from scipy.interpolate import PPoly
    if not (0 <= c_min < c_max <= 1):
        raise ValueError(f'{label}: expected 0 <= C_min < C_max <= 1')
    base_lo, base_hi = float(spline.t[spline.k]), float(spline.t[-spline.k - 1])
    if c_min < base_lo - 1e-12 or c_max > base_hi + 1e-12:
        raise ValueError(f'{label}: fitted C range lies outside the spline knot domain')
    poly = PPoly.from_spline(spline)
    breaks = np.unique(np.r_[c_min, spline.t[(spline.t > c_min) & (spline.t < c_max)], c_max])
    values = poly(breaks)
    if not np.isfinite(values).all() or not values[-1] < values[0]:
        raise ValueError(f'{label}: fitted z(C) must be finite and decreasing')
    derivative = poly.derivative()
    stationary = derivative.derivative().roots(extrapolate=False)
    stationary = stationary[np.isfinite(stationary) & (stationary > c_min) & (stationary < c_max)]
    probes = np.r_[breaks, stationary, np.nextafter(breaks[1:], breaks[:-1])]
    slopes = derivative(probes)
    tolerance = 1e-10 * max(1.0, float(np.max(np.abs(values)))) / (c_max - c_min)
    if not np.isfinite(slopes).all() or (slopes > tolerance).any() or (np.diff(values) >= 0).any():
        raise ValueError(f'{label}: fitted z(C) is not monotonically decreasing')


def _load_splines(path: str) -> dict:
    from scipy.interpolate import BSpline, PchipInterpolator
    import h5py

    if Path(path).suffix.lower() not in ('.h5', '.hdf5'):
        raise ValueError(f'{path}: only .h5/.hdf5 calibration files are supported')
    data = {}
    with h5py.File(path, 'r') as handle:
        for key in handle.keys():
            group = handle[key]
            if not isinstance(group, h5py.Group) or 'level' not in group.attrs:
                raise ValueError(f'{path}: group {key!r} lacks a calibration level')
            level = group.attrs['level']
            if isinstance(level, (bytes, np.bytes_)):
                level = level.decode('utf-8')
            if not isinstance(level, str) or not level.strip():
                raise ValueError(f'{path}: calibration level must be a nonempty string')
            level = level.strip()
            label = f'{path}: level {level!r}'
            if level in data:
                raise ValueError(f'{label}: duplicate calibration level')
            required = ('H_vals', 'C_vals', 'C_mins', 'matrix', 'best_s', 'loo_mse', 'splines')
            missing = [name for name in required if name not in group]
            if missing:
                raise ValueError(f'{label}: missing datasets {missing}')
            if any(not isinstance(group[name], h5py.Dataset) for name in required if name != 'splines'):
                raise ValueError(f'{label}: calibration arrays must be HDF5 datasets')
            h_vals = np.asarray(group['H_vals'][:], dtype=np.float64)
            c_vals = np.asarray(group['C_vals'][:], dtype=np.float64)
            c_mins = np.asarray(group['C_mins'][:], dtype=np.float64)
            for name, axis, minimum_length in (('H_vals', h_vals, 1), ('C_vals', c_vals, 2)):
                if (axis.ndim != 1 or len(axis) < minimum_length or not np.isfinite(axis).all()
                        or (np.diff(axis) <= 0).any() or (axis < 0).any() or (axis > 1).any()):
                    raise ValueError(f'{label}: {name} must be a finite strictly increasing axis in [0,1]')
            if c_mins.shape != h_vals.shape or not np.isfinite(c_mins).all():
                raise ValueError(f'{label}: C_mins must have one finite entry per H')
            spline_group = group['splines']
            if not isinstance(spline_group, h5py.Group):
                raise ValueError(f'{label}: splines must be an HDF5 group')
            if set(spline_group.keys()) != {str(i) for i in range(len(h_vals))}:
                raise ValueError(f'{label}: expected one consecutively numbered spline per H')
            splines = []
            for i in range(len(h_vals)):
                source = spline_group[str(i)]
                if not isinstance(source, h5py.Group):
                    raise ValueError(f'{label}: spline {i} must be an HDF5 group')
                if 't' not in source or 'c' not in source or 'k' not in source.attrs:
                    raise ValueError(f'{label}: spline {i} lacks t, c, or k')
                if not isinstance(source['t'], h5py.Dataset) or not isinstance(source['c'], h5py.Dataset):
                    raise ValueError(f'{label}: spline {i} knots and coefficients must be datasets')
                knots = np.asarray(source['t'][:], dtype=np.float64)
                coefficients = np.asarray(source['c'][:], dtype=np.float64)
                if (np.ndim(knots) != 1 or np.ndim(coefficients) != 1
                        or not np.isfinite(knots).all() or not np.isfinite(coefficients).all()):
                    raise ValueError(f'{label}: spline {i} has invalid knots or coefficients')
                degree = np.asarray(source.attrs['k'])
                if (degree.ndim != 0 or degree.dtype.kind not in 'iuf'
                        or not np.isfinite(degree) or degree != np.floor(degree)
                        or degree < 1 or degree >= len(knots) - 1):
                    raise ValueError(f'{label}: spline {i} degree must be a finite positive integer compatible with its knots')
                spline = BSpline(knots, coefficients, int(degree))
                _validate_spline(spline, float(c_mins[i]), float(c_vals[-1]), f'{label}, spline {i}')
                splines.append(spline)
            has_x, has_y = 'correction_x' in group, 'correction_y' in group
            if has_x != has_y:
                raise ValueError(f'{label}: correction_x and correction_y must both be present or both absent')
            if has_x and (not isinstance(group['correction_x'], h5py.Dataset)
                          or not isinstance(group['correction_y'], h5py.Dataset)):
                raise ValueError(f'{label}: correction arrays must be HDF5 datasets')
            cx = np.asarray(group['correction_x'][:], dtype=np.float64) if has_x else None
            cy = np.asarray(group['correction_y'][:], dtype=np.float64) if has_y else None
            correction_fns = None
            if has_x:
                if (cy.ndim != 1 or len(cy) < 2 or cx.ndim != 2
                        or cx.shape != (len(h_vals), len(cy))):
                    raise ValueError(f'{label}: expected a 2D correction_x mesh with one row per H and one column per correction_y value')
                if (not np.isfinite(cx).all() or not np.isfinite(cy).all()
                        or (np.diff(cx, axis=1) <= 0).any() or (np.diff(cy) < 0).any()
                        or (cy < 0).any() or (cy > 1).any()):
                    raise ValueError(f'{label}: correction coordinates must be finite, x strictly increasing, and y nondecreasing in [0,1]')
                correction_fns = []
                for row in cx:
                    interpolator = PchipInterpolator(row, cy, extrapolate=False)
                    lo, hi = float(row[0]), float(row[-1])
                    correction_fns.append(
                        lambda c, _p=interpolator, _lo=lo, _hi=hi:
                        float(np.clip(_p(np.clip(c, _lo, _hi)), 0.0, 1.0)))
            metadata = {name: group[name][:] for name in ('matrix', 'best_s', 'loo_mse')}
            data[level] = {
                'H_vals': h_vals, 'C_vals': c_vals, 'C_mins': c_mins,
                **metadata, 'splines': splines, 'correction_x': cx,
                'correction_y': cy, 'correction_fns': correction_fns,
                'correction_source': group.attrs.get('correction_source'),
            }
    if not data:
        raise ValueError(f'{path}: no calibration levels found')
    return data


def _invert_spline(spl, z_obs: float, C_min: float, C_max: float):
    """Invert a decreasing fitted z(C), preserving the rightmost extrapolated root.

    Loaded BSplines use exact piecewise-polynomial roots below C_min. A coarse
    sign scan can miss both a tangential root and a close pair of roots. An
    out-of-calibration observation is distinguished from a nonsignificant z.
    """
    from scipy.interpolate import BSpline, PPoly
    from scipy.optimize import brentq

    if not np.isfinite(z_obs) or not np.isfinite(C_min) or not np.isfinite(C_max):
        return float('nan'), 'invalid_input'
    if not 0 <= C_min < C_max <= 1:
        raise ValueError('spline inversion requires 0 <= C_min < C_max <= 1')
    if z_obs > _Z_SIGNIFICANCE:
        return 0.0, 'not_significant'
    v_min, v_max = float(spl(C_min)), float(spl(C_max))
    if not np.isfinite([v_min, v_max]).all() or v_min <= v_max:
        raise ValueError('spline is not finite and decreasing over its fitted bounds')
    if z_obs < v_max:
        return float(C_max), 'above_range'
    if z_obs <= v_min:
        root = brentq(lambda c: float(spl(c)) - z_obs, C_min, C_max,
                      xtol=1e-10, maxiter=200)
        return float(root), 'ok'

    if isinstance(spl, BSpline):
        roots = PPoly.from_spline(spl).solve(y=z_obs, extrapolate=True)
        tolerance = 1e-10
        roots = roots[np.isfinite(roots) & (roots >= -tolerance) & (roots <= C_min + tolerance)]
        if len(roots):
            return float(np.clip(np.max(roots), 0.0, C_min)), 'extrapolated'
    elif C_min > 0:
        x = np.linspace(0.0, C_min, 500)
        residuals = np.array([float(spl(c)) - z_obs for c in x])
        if not np.isfinite(residuals).all():
            return float('nan'), 'invalid_input'
        exact = x[residuals == 0]
        brackets = np.flatnonzero(np.signbit(residuals[:-1]) != np.signbit(residuals[1:]))
        roots = list(exact)
        for i in brackets:
            roots.append(brentq(lambda c: float(spl(c)) - z_obs, x[i], x[i + 1], xtol=1e-10))
        if roots:
            return float(max(roots)), 'extrapolated'
    return float('nan'), 'below_calibration_range'


def _estimate_concordance(spline_data: dict, z_var_obs: float,
                          H_est: float, level: str, *, verbose=False):
    """Return corrected C, NS for the z gate, or NA for unavailable calibration.

    Interpolation is between corrected inverse estimates, as in optconn10.
    The reported H is not overwritten when calibration clips it to its H grid.
    """
    if level not in spline_data:
        _diagnostic(f"  [concordance] level {level!r} unavailable; C_est=NA", verbose=verbose)
        return 'NA'
    if not np.isfinite(z_var_obs):
        return 'NA'
    if z_var_obs > _Z_SIGNIFICANCE:
        return 'NS'
    if not np.isfinite(H_est):
        return 'NA'
    sd = spline_data[level]
    h_vals = np.asarray(sd['H_vals'], dtype=np.float64)
    h_clip = float(np.clip(H_est, h_vals[0], h_vals[-1]))
    if h_clip != H_est:
        _diagnostic(f'  [concordance] H_est={H_est:.6g} outside calibration; using H={h_clip:.6g}', verbose=verbose)
    exact = np.flatnonzero(h_vals == h_clip)
    if len(exact):
        rows = [(int(exact[0]), 1.0)]
    else:
        hi = int(np.searchsorted(h_vals, h_clip, side='right'))
        lo = hi - 1
        weight = float((h_clip - h_vals[lo]) / (h_vals[hi] - h_vals[lo]))
        rows = [(lo, 1.0 - weight), (hi, weight)]
    estimate = 0.0
    for row, weight in rows:
        raw_c, status = _invert_spline(sd['splines'][row], z_var_obs,
                                      float(sd['C_mins'][row]), float(sd['C_vals'][-1]))
        if not np.isfinite(raw_c):
            _diagnostic(f'  [concordance] {status} at H={h_vals[row]:.6g}; C_est=NA', verbose=verbose)
            return 'NA'
        if status in ('extrapolated', 'above_range'):
            _diagnostic(f'  [concordance] {status} at H={h_vals[row]:.6g}; raw C={raw_c:.6g}', verbose=verbose)
        correction = sd.get('correction_fns')
        if correction is not None:
            cx = sd.get('correction_x')
            if cx is not None and not cx[row, 0] <= raw_c <= cx[row, -1]:
                _diagnostic(f'  [concordance] correction input outside knots at H={h_vals[row]:.6g}; using endpoint saturation', verbose=verbose)
            raw_c = correction[row](raw_c)
        if not np.isfinite(raw_c):
            return 'NA'
        estimate += weight * raw_c
    return float(np.clip(estimate, 0.0, 1.0))

# Main
def _load_input_tables(genes_path, domains_path):
    gene_columns = ['chrom', 'start', 'end', 'score', 'strand']
    domain_columns = ['chrom', 'start', 'end']
    try:
        genes = pd.read_csv(genes_path, sep=r'\s+', comment='#', header=None,
                            names=gene_columns, usecols=[0, 1, 2, 3, 4],
                            dtype={'chrom': str, 'strand': str}, keep_default_na=False)
        domains = pd.read_csv(domains_path, sep=r'\s+', comment='#', header=None,
                              names=domain_columns, usecols=[0, 1, 2],
                              dtype={'chrom': str}, keep_default_na=False)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise ValueError(f'Cannot read input files: {exc}. Expected genes: chrom start end expression strand; domains: chrom start end') from exc
    if genes.empty or domains.empty:
        raise ValueError('Gene and domain files must both contain at least one interval')
    genes['start'], genes['end'] = _interval_coordinates(genes, 'genes')
    domains['start'], domains['end'] = _interval_coordinates(
        domains, f'domains ({domains_path})', repair_reversed=True)
    try:
        genes['score'] = pd.to_numeric(genes['score'], errors='raise').astype(np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError('Gene expression scores must be finite numbers') from exc
    if not np.isfinite(genes['score'].to_numpy()).all():
        raise ValueError('Gene expression scores must be finite numbers')
    if not genes['strand'].isin(['+', '-', '.']).all():
        raise ValueError("Gene strands must be '+', '-', or '.' (unknown); the input is a custom five-column expression format, not standard BED5/BED6")
    return genes, domains


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Gene Expression Concordance in Genomic Domains (Cyclic Shift Test). '
                    'optconn10 with nested-domain support and validated calibration.')
    parser.add_argument('genes_bed', help='Custom five columns: chrom start end expression strand (not standard UCSC BED5)')
    parser.add_argument('domains_bed', help='Domain BED3: chrom start end; zero-based, half-open coordinates; reversed endpoints are swapped with a warning')
    parser.add_argument('--mode', choices=['start', 'end', 'middle', 'longer_part', 'proportional'],
                        default='proportional',
                        help='Default: proportional. Point/proportional modes retain all qualifying nested domains. '
                             'longer_part selects one largest overlap. start/end are strand-aware TSS/TES, with the rightmost included base end-1.')
    parser.add_argument('--strand', choices=['both', 'plus', 'minus'], default='both', help='Gene strand filter (default: both)')
    parser.add_argument('--perms', type=int, default=1000, help='Number of sampled rotations, >=2; sampling with replacement includes zero (default: 1000)')
    parser.add_argument('--cores', type=int, default=max(1, cpu_count() - 1), help='Worker processes, >=1 (default: all-1; capped at permutation count)')
    parser.add_argument('--chunk-size', type=int, default=None, metavar='N', help='Rotations per worker batch, >=1 (default: automatic)')
    parser.add_argument('--seed', type=int, default=None, help='Nonnegative random seed; results are independent of core/chunk counts')
    parser.add_argument('--output', '-o', default='results.txt', metavar='FILE', help='Output path; append in both scopes by default (default: results.txt)')
    parser.add_argument('--overwrite', action='store_true', help='Replace the output file after successful analysis instead of appending')
    parser.add_argument('--verbose', action='store_true', help='Include routine and calibration diagnostics on stderr; errors and warnings about automatically corrected domain coordinates are always printed')
    parser.add_argument('--version', action='version', version=f'optconn10 {__version__}')
    scope_group = parser.add_mutually_exclusive_group()
    scope_group.add_argument('--scope', choices=['genome', 'chromosome'], default=None,
                             help='genome: one rotation of all sorted genes; chromosome: separate analyses and output rows per chromosome')
    scope_group.add_argument('--genomewide', '--gw', dest='scope_gw', action='store_true', help='Equivalent to --scope genome')
    scope_group.add_argument('--chromosomewide', '--chr', dest='scope_chr', action='store_true', help='Equivalent to --scope chromosome')
    calibration_group = parser.add_mutually_exclusive_group()
    calibration_group.add_argument('--splines', default=None, metavar='FILE',
                                   help='HDF5 calibration file; an explicit invalid/missing file is an error. '
                                        f'If omitted, try {DEFAULT_SPLINES_FILE!r}; if absent, C_est=NA.')
    calibration_group.add_argument('--no-splines', action='store_true', help='Disable calibration, including automatic splines.h5 loading; C_est=NA')
    parser.add_argument('--spline-level', default=DEFAULT_CONCORDANCE_LEVEL, metavar='LEVEL',
                        help='Calibration quantile, e.g. 2.5%%, 5%%, 50%%. The corrected C is a model-calibrated estimate, '
                             'not a verified confidence bound. Default: %(default)s')
    args = parser.parse_args(argv)
    if args.perms < 2:
        parser.error('--perms must be >= 2')
    if args.cores < 1:
        parser.error('--cores must be >= 1')
    if args.chunk_size is not None and args.chunk_size < 1:
        parser.error('--chunk-size must be >= 1')
    if args.seed is not None and args.seed < 0:
        parser.error('--seed must be nonnegative')
    args.cores = min(args.cores, args.perms)
    if args.chunk_size is None:
        args.chunk_size = max(1, (args.perms // args.cores // 10) * 10)
    args.scope = 'genome' if args.scope_gw else 'chromosome' if args.scope_chr else args.scope or 'genome'

    try:
        output_path = Path(args.output).expanduser().resolve()
        input_paths = {Path(args.genes_bed).expanduser().resolve(), Path(args.domains_bed).expanduser().resolve()}
        spline_path = None if args.no_splines else args.splines if args.splines is not None else DEFAULT_SPLINES_FILE
        if spline_path is not None:
            input_paths.add(Path(spline_path).expanduser().resolve())
        aliases_input = output_path in input_paths
        if output_path.exists():
            aliases_input = aliases_input or any(
                path.exists() and output_path.samefile(path) for path in input_paths)
        if aliases_input:
            raise ValueError('Output path must differ from all input and calibration files')
        if not output_path.parent.is_dir() or output_path.is_dir():
            raise ValueError(f'Invalid output file path: {output_path}')
        genes, domains = _load_input_tables(args.genes_bed, args.domains_bed)
        if args.strand != 'both':
            genes = genes[genes['strand'] == ('+' if args.strand == 'plus' else '-')].copy()
        if genes.empty:
            raise ValueError('No genes remain after strand filtering')
        if args.mode in ('start', 'end') and not genes['strand'].isin(['+', '-']).all():
            raise ValueError(f"--mode {args.mode} requires '+' or '-' for every analyzed gene")
        genes = genes.sort_values(['chrom', 'start'], kind='stable').reset_index(drop=True)
        domains = domains.sort_values(['chrom', 'start'], kind='stable').reset_index(drop=True)
        gene_chroms, domain_chroms = set(genes['chrom']), set(domains['chrom'])
        common_chroms = sorted(gene_chroms & domain_chroms)
        if not common_chroms:
            raise ValueError('No chromosomes in common between genes and domains; names are matched literally')
        unmatched_genes, unmatched_domains = sorted(gene_chroms - domain_chroms), sorted(domain_chroms - gene_chroms)
        if unmatched_genes or unmatched_domains:
            action = 'retained in the genome-wide rotation pool' if args.scope == 'genome' else 'excluded from separate chromosome analyses'
            _diagnostic(f'  [input] gene-only chromosomes {unmatched_genes}: {action}; domain-only chromosomes {unmatched_domains} have no genes', verbose=args.verbose)

        spline_data = None
        if spline_path is not None:
            try:
                spline_data = _load_splines(str(spline_path))
            except FileNotFoundError:
                if args.splines is not None:
                    raise ValueError(f'Explicit calibration file not found: {spline_path}')
            except (OSError, ValueError, KeyError, TypeError, ImportError) as exc:
                if isinstance(exc, ImportError) and args.splines is None and not Path(spline_path).exists():
                    spline_data = None
                else:
                    raise ValueError(f'Cannot load calibration {spline_path}: {exc}') from exc
            if spline_data is not None and args.spline_level not in spline_data:
                raise ValueError(f'Calibration level {args.spline_level!r} unavailable; choose from {sorted(spline_data)}')

        seed_sequence = np.random.SeedSequence(args.seed)
        rng = np.random.default_rng(seed_sequence)
        _diagnostic(f'  [run] seed={seed_sequence.entropy} perms={args.perms} cores={args.cores} scope={args.scope} mode={args.mode}', verbose=args.verbose)
        if args.verbose:
            _diagnostic(f'  [input] genes={len(genes)} domains={len(domains)}; proportional scores use overlap/gene_length, with gene-count denominators', verbose=args.verbose)

        def analyze(g, d, label):
            domain_map = map_genes_to_domains(g, d, args.mode)
            indices, weights, ptr = domain_map_to_csr(domain_map)
            eligible = int(np.count_nonzero(np.diff(ptr) >= 2))
            if args.verbose:
                _diagnostic(f'  [mapping] {label}: {eligible}/{len(d)} domains contain >=2 assigned genes; assignments={len(indices)}', verbose=args.verbose)
            if eligible == 0:
                _diagnostic(f'  [statistics] {label}: no domains with >=2 genes; z/p/H are nan and C_est=NA', verbose=args.verbose)
                return float('nan'), float('nan'), float('nan'), 'NA'
            expression = g['score'].to_numpy(dtype=np.float64)
            if (expression < 0).any():
                h_est = float('nan')
                _diagnostic(f'  [noise] {label}: negative expression scores; H and C calibration unavailable', verbose=args.verbose)
            else:
                ranges, gene_counts = compute_domain_rel_ranges(expression, domain_map, genes_df=g, domains_df=d)
                h_est, diagnostics = estimate_noise_from_rel_ranges(ranges, gene_counts, verbose=args.verbose)
                if diagnostics['method'] in ('median_fallback_capped', 'no_eligible_positive_mean_domains'):
                    _diagnostic(f"  [noise] {label}: {diagnostics['method']}; H_est={h_est:.6g}", verbose=args.verbose)
            observed = compute_real_stats(expression, indices, weights, ptr)
            codes = np.unique(g['chrom'].to_numpy(), return_inverse=True)[1].astype(np.int32)
            permutations = run_permutations(expression, indices, weights, ptr, codes,
                                            'genome', args.perms, args.cores, args.chunk_size, seed=rng)
            z_mean, z_var = zscores_from_results(observed, permutations)
            if not np.isfinite(z_mean) or not np.isfinite(z_var):
                _diagnostic(f'  [statistics] {label}: an observed/null statistic is undefined or the sampled null is constant to numerical precision; affected z/p values are nan', verbose=args.verbose)
            c_est = 'NA' if spline_data is None else _estimate_concordance(spline_data, z_var, h_est, args.spline_level, verbose=args.verbose)
            return z_mean, z_var, h_est, c_est

        rows = []
        if args.scope == 'chromosome':
            for chrom in common_chroms:
                g = genes[genes['chrom'] == chrom].reset_index(drop=True)
                d = domains[domains['chrom'] == chrom].reset_index(drop=True)
                rows.append((chrom, *analyze(g, d, chrom)))
        else:
            rows.append((None, *analyze(genes, domains, 'genome')))
        with output_path.open('w' if args.overwrite else 'a') as output:
            for chrom, z_mean, z_var, h_est, c_est in rows:
                emit(output, Path(args.genes_bed).stem, Path(args.domains_bed).stem,
                     chrom, z_mean, z_var, h_est, c_est)
        return 0
    except (OSError, ValueError, TypeError, KeyError, FloatingPointError, RuntimeError) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print('Interrupted; no new result rows were written.', file=sys.stderr)
        return 130


if __name__ == '__main__':
    raise SystemExit(main())
