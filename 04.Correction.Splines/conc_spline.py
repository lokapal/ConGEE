#!/usr/bin/env python3
"""
conc_spline_final.py
--------------------
Precompute concordance splines from TSV calibration files and embed a 2D
bias-correction mesh, writing a single HDF5 file compatible with optconn9.

Calibration TSV format  (one file per percentile level)
--------------------------------------------------------
  # level: 5%
  Noise level<TAB>C_0.1<TAB>...<TAB>C_1.0
  0.05<TAB>-2.03<TAB>...<TAB>-13.46

Correction matrix format  (same layout as C_by_splines.tsv)
------------------------------------------------------------
  # splines level: 5%
  Noiselevel<TAB>C_0.1<TAB>...<TAB>C_1.0
  0.05<TAB>0.1931<TAB>...<TAB>0.8252

Each row corresponds to one H level; cells contain the median raw C_est
returned by the uncorrected spline at each (H, C_true) combination.  Each
row becomes an independent PCHIP correction curve (2D mesh).

HDF5 output layout (splh5, compatible with optconn9)
-----------------------------------------------------
  /   attrs: version=2, format="splh5"
  /<level_key>/
      attrs:  level="5%"   correction_source="..."
      datasets: H_vals, C_vals, C_mins, matrix, best_s, loo_mse
                correction_x  shape (nH, nC+1)  per-H C_est knots
                correction_y  shape (nC+1,)      shared C_true values
      /splines/  /0/ t,c  attrs:k=3  ...

Workflow
--------
  python conc_spline_final.py precompute \\
      --tsv p025.tsv p05.tsv \\
      --correction 5%%:C_by_splines.tsv \\
      --output splines.h5

  python conc_spline_final.py estimate \\
      --splines splines.h5 --z-var -4.23 --noise 0.18

Python API:
  from conc_spline_final import load_splines, estimate_concordance
  sd = load_splines('splines.h5')
  r  = estimate_concordance(sd, z_var_obs=-4.23, H_est=0.18, level='5%%')
  print(r['C_lower'])
"""

import argparse
import re
import sys
from io import StringIO
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline, BSpline, PchipInterpolator
from scipy.optimize import brentq


# ──────────────────────────────────────────────────────────────────────────────
# 1.  Data loading  (TSV version)
# ──────────────────────────────────────────────────────────────────────────────

def _detect_level_from_comment(first_line: str):
    """
    Extract a level label from a comment line such as '# level: 50%'.
    Returns the label string (e.g. '50%') or None.
    """
    if not first_line.startswith('#'):
        return None
    m = re.search(r'level[:\s]+(\S+)', first_line, re.IGNORECASE)
    return m.group(1) if m else None


def _detect_level_from_filename(path: str):
    """
    Infer the percentile level from the file name.

    Rules (applied to the lowercase stem):
        contains '2.5', '2_5', or 'p025'  →  '2.5%'
        contains 'median' or 'p50'         →  '50%'
        isolated '50'                      →  '50%'
        isolated '5' (not part of 50)      →  '5%'

    Returns a label string or None if no pattern matches.
    """
    name = Path(path).stem.lower()
    if re.search(r'2[_\.]5|p025', name):
        return '2.5%'
    if re.search(r'median|p50|(?<!\d)50(?!\d)', name):
        return '50%'
    if re.search(r'p0?5(?:[_\-\.]|$)|(?<!\d)5(?!\d)', name):
        return '5%'
    return None


def _parse_percentile_tsv(path: str, hint_label: str = None):
    """
    Parse a single calibration TSV file.

    Parameters
    ----------
    path        : path to the TSV file
    hint_label  : label supplied via --levels; used only if auto-detection fails

    Returns
    -------
    label  : str   percentile level label, e.g. '2.5%', '50%'
    H_vals : float64 array  (n_H,)
    C_vals : float64 array  (n_C,)
    matrix : float64 array  (n_H, n_C)
    """
    lines = Path(path).read_text().splitlines()
    if not lines:
        raise ValueError(f"{path}: file is empty")

    # ── Detect level label ──
    label = None
    data_start = 0   # index of the header row

    if lines[0].startswith('#'):
        label      = _detect_level_from_comment(lines[0])
        data_start = 1   # skip the comment line
    if label is None:
        label = _detect_level_from_filename(path)
    if label is None:
        label = hint_label
    if label is None:
        raise ValueError(
            f"{path}: cannot determine level label.\n"
            "Add a '# level: X%' comment as the first line, name the file to\n"
            "contain '2.5', '5', or '50', or pass --levels explicitly.")

    # ── Read the data section ──
    # Skip any remaining comment lines after the first
    while data_start < len(lines) and lines[data_start].startswith('#'):
        data_start += 1

    if data_start >= len(lines):
        raise ValueError(f"{path}: no data lines after comments")

    # Use pandas to parse the header + data block
    from io import StringIO
    block = '\n'.join(lines[data_start:])
    df    = pd.read_csv(StringIO(block), sep='\t', header=0)

    # Column 0 is 'Noise level' (H), columns 1… are 'C_0.1', 'C_0.2', …
    H_vals = df.iloc[:, 0].astype(float).values
    C_vals = (pd.Series(df.columns[1:])
                .str.replace('C_', '', regex=False)
                .astype(float)
                .values)
    matrix = df.iloc[:, 1:].astype(float).values

    return label, H_vals, C_vals, matrix


def load_percentile_table(tsv_paths, hint_labels=None) -> dict:
    """
    Load calibration data from one or more TSV files.

    Parameters
    ----------
    tsv_paths   : list of file paths
    hint_labels : optional list of level labels (same order as tsv_paths);
                  used only when auto-detection from comment/filename fails

    Returns
    -------
    dict  {level_label: {'H_vals': …, 'C_vals': …, 'matrix': …}}
    """
    if hint_labels is None:
        hint_labels = [None] * len(tsv_paths)
    if len(hint_labels) != len(tsv_paths):
        raise ValueError(
            f"--levels must have the same number of entries as --tsv "
            f"({len(tsv_paths)} files, {len(hint_labels)} labels)")

    result = {}
    for path, hint in zip(tsv_paths, hint_labels):
        label, H, C, M = _parse_percentile_tsv(path, hint_label=hint)
        if label in result:
            raise ValueError(
                f"Duplicate level label '{label}' from {path}. "
                "Each TSV file must represent a distinct percentile level.")
        result[label] = {'H_vals': H, 'C_vals': C, 'matrix': M}

    if not result:
        raise ValueError("No TSV files were loaded.")
    return result


# ──────────────────────────────────────────────────────────────────────────────
# 2.  LOO cross-validation for the smoothing factor
# ──────────────────────────────────────────────────────────────────────────────

def _loo_cv_smoothing(C_vals: np.ndarray,
                      z_row: np.ndarray,
                      s_candidates: np.ndarray) -> float:
    """
    Select the UnivariateSpline smoothing parameter s that minimises
    leave-one-out mean squared prediction error for a single H row.

    With only 9–10 C grid points the LOO is exact: for each candidate s we fit
    n splines on n-1 points and evaluate on the held-out point.
    """
    n = len(C_vals)
    best_s   = s_candidates[0]
    best_mse = np.inf

    for s in s_candidates:
        sq_errors = []
        for j in range(n):
            mask = np.ones(n, dtype=bool)
            mask[j] = False
            try:
                spl  = UnivariateSpline(C_vals[mask], z_row[mask], k=3, s=s)
                pred = float(spl(C_vals[j]))
                sq_errors.append((pred - z_row[j]) ** 2)
            except Exception:
                sq_errors.append(1e9)
        mse = float(np.mean(sq_errors))
        if mse < best_mse:
            best_mse = mse
            best_s   = s

    return best_s


# ──────────────────────────────────────────────────────────────────────────────
# 3.  Spline fitting
# ──────────────────────────────────────────────────────────────────────────────

Z_SIGNIFICANCE = -1.96   # observations above this are not significantly concordant


def fit_splines(H_vals: np.ndarray,
                C_vals: np.ndarray,
                matrix: np.ndarray,
                n_s_candidates: int = 200,
                verbose: bool = True) -> dict:
    """
    For each H row, keep only the grid points where p(C, H) <= Z_SIGNIFICANCE
    (-1.96), run LOO CV to select the smoothing factor, and fit a
    UnivariateSpline mapping C → p(C, H) on those points.

    Returns
    -------
    dict with keys:
        'H_vals', 'C_vals', 'matrix', 'splines', 'C_mins', 'best_s', 'loo_mse'
    """
    s_candidates = np.logspace(-4, 2, n_s_candidates)
    splines      = []
    C_mins       = []
    best_s_list  = []
    loo_mse_list = []

    if verbose:
        print(f"  Fitting {len(H_vals)} splines using only p <= {Z_SIGNIFICANCE} "
              f"(LOO CV with {n_s_candidates} candidates)…", file=sys.stderr)

    for H, row in zip(H_vals, matrix):
        keep   = row <= Z_SIGNIFICANCE
        C_keep = C_vals[keep]
        z_keep = row[keep]

        if len(C_keep) < 4:
            raise ValueError(
                f"H={H}: only {len(C_keep)} grid points have p <= {Z_SIGNIFICANCE}; "
                "need at least 4 for a cubic spline.")

        best_s = _loo_cv_smoothing(C_keep, z_keep, s_candidates)

        n  = len(C_keep)
        sq = []
        for j in range(n):
            mask    = np.ones(n, dtype=bool); mask[j] = False
            spl_tmp = UnivariateSpline(C_keep[mask], z_keep[mask], k=3, s=best_s)
            sq.append((float(spl_tmp(C_keep[j])) - z_keep[j]) ** 2)
        loo_mse = float(np.mean(sq))

        spl_final = UnivariateSpline(C_keep, z_keep, k=3, s=best_s)
        splines.append(spl_final)
        C_mins.append(float(C_keep[0]))
        best_s_list.append(best_s)
        loo_mse_list.append(loo_mse)

        if verbose:
            fit_errs = [abs(float(spl_final(c)) - v) for c, v in zip(C_keep, z_keep)]
            n_excl   = int((~keep).sum())
            print(f"    H={H:.2f}  C_min={C_keep[0]:.1f}  n_pts={len(C_keep)}"
                  f"(excl={n_excl})  best_s={best_s:.5f}  "
                  f"LOO_MSE={loo_mse:.5f}  max_fit_err={max(fit_errs):.5f}",
                  file=sys.stderr)

    return {
        'H_vals'  : H_vals,
        'C_vals'  : C_vals,
        'matrix'  : matrix,
        'splines' : splines,
        'C_mins'  : np.array(C_mins),
        'best_s'  : np.array(best_s_list),
        'loo_mse' : np.array(loo_mse_list),
    }


# ──────────────────────────────────────────────────────────────────────────────
# 4.  Saving / loading
# ──────────────────────────────────────────────────────────────────────────────



def parse_correction_matrix(path: str) -> tuple:
    """
    Parse a 10x10 correction matrix TSV (same format as C_by_splines.tsv).

    Each row is one H level; cells contain median raw C_est per (H, C_true).

    Returns
    -------
    H_vals : float64 (nH,)
    C_vals : float64 (nC,)
    matrix : float64 (nH, nC)
    """
    lines = Path(path).read_text().splitlines()
    data_lines = [l for l in lines if not l.startswith('#') and l.strip()]
    if not data_lines:
        raise ValueError(f"{path}: no data lines found")
    df = pd.read_csv(StringIO('\n'.join(data_lines)), sep='\t', header=0)
    H_vals = df.iloc[:, 0].astype(float).values
    C_vals = (df.columns[1:]
               .str.replace('C_', '', regex=False)
               .astype(float)
               .values)
    matrix = df.iloc[:, 1:].astype(float).values
    return H_vals, C_vals, matrix


def _derive_correction_mesh(C_est_matrix: np.ndarray,
                             C_true: np.ndarray) -> tuple:
    """
    Build 2D correction arrays from a C-estimation matrix.

    Each row becomes one PCHIP correction curve; a (0, 0) anchor is prepended.

    Returns
    -------
    correction_x : float64 (nH, nC+1)  — per-H C_est knots, col 0 = 0.0
    correction_y : float64 (nC+1,)     — shared C_true values
    """
    if not np.all(np.isfinite(C_est_matrix)):
        raise ValueError("the matrix contains missing or non-numeric cells.")
    for i, row in enumerate(C_est_matrix):
        if not np.all(np.diff(row) > 0):
            bad = np.where(np.diff(row) <= 0)[0]
            raise ValueError(
                f"Correction matrix row {i} not strictly increasing "
                f"at C_true index {bad}.")
    nH = C_est_matrix.shape[0]
    correction_x = np.hstack([np.zeros((nH, 1)), C_est_matrix])
    correction_y = np.concatenate([[0.0], C_true])
    return correction_x, correction_y


def _build_correction_fns(correction_x: np.ndarray,
                           correction_y: np.ndarray) -> list:
    """
    Build a list of per-H PCHIP callables from stored 2D correction arrays.

    correction_x : (nH, nC+1)  — one row of C_est knots per H level
    correction_y : (nC+1,)     — shared C_true values

    Returns a list of nH callables: raw C_est -> corrected C_true, clipped to [0,1].
    """
    fns = []
    for i in range(correction_x.shape[0]):
        _p = PchipInterpolator(correction_x[i], correction_y)
        fns.append(lambda c, _ph=_p: float(np.clip(_ph(c), 0.0, 1.0)))
    return fns


# ──────────────────────────────────────────────────────────────────────────────
# 4.  HDF5 serialisation
# ──────────────────────────────────────────────────────────────────────────────

def _level_key(level: str) -> str:
    """Convert a level label to a valid HDF5 group name.
    '2.5%' -> '2p5pct',  '5%' -> '5pct',  'mode' -> 'mode'.
    """
    return level.replace('.', 'p').replace('%', 'pct').replace(' ', '_')


def save_splines(spline_data: dict, output_path: str) -> None:
    """
    Save precomputed splines to a version-independent HDF5 file (.h5).

    All data is stored as plain numpy arrays — no Python objects are serialised.
    UnivariateSpline objects are decomposed to raw (t, c, k) B-spline parameters
    and reconstructed on load via scipy.interpolate.BSpline(t, c, k), making
    the file independent of scipy and Python versions.

    Bias-correction callables are not stored; only the (correction_x,
    correction_y) arrays are written and the callable is rebuilt by load_splines().
    """
    with h5py.File(output_path, 'w') as f:
        f.attrs['version'] = 2
        f.attrs['format']  = 'splh5'

        for level, sd in sorted(spline_data.items()):
            key = _level_key(level)
            g   = f.create_group(key)
            g.attrs['level'] = level
            if sd.get('correction_source'):
                g.attrs['correction_source'] = sd['correction_source']

            # ── scalar / array metadata ───────────────────────────────────
            for name in ('H_vals', 'C_vals', 'C_mins', 'matrix',
                         'best_s', 'loo_mse'):
                g.create_dataset(name, data=sd[name], compression='gzip')

            if sd.get('correction_x') is not None:
                g.create_dataset('correction_x', data=sd['correction_x'],
                                 compression='gzip')
                g.create_dataset('correction_y', data=sd['correction_y'],
                                 compression='gzip')

            # ── splines: one sub-group per H row, keyed by integer index ──
            sg = g.create_group('splines')
            for i, spl in enumerate(sd['splines']):
                t, c, k = spl._eval_args
                # FITPACK pads c to len(t); only the first len(t) - k - 1 entries
                # are coefficients (the rest is uninitialised memory that differs
                # between platforms).  Store just the real ones.
                c = np.asarray(c)[:len(t) - k - 1]
                row = sg.create_group(str(i))
                row.create_dataset('t', data=t, compression='gzip')
                row.create_dataset('c', data=c, compression='gzip')
                row.attrs['k'] = int(k)

    size_kb = Path(output_path).stat().st_size // 1024
    print(f"  Saved splines to {output_path} ({size_kb} KB)", file=sys.stderr)


def load_splines(splines_path: str) -> dict:
    """
    Load precomputed splines from an HDF5 file written by save_splines().

    Spline rows are reconstructed as BSpline(t, c, k) callables.
    Correction callables are rebuilt from the stored (correction_x,
    correction_y) arrays using PCHIP interpolation.

    The returned dict structure is identical to the one produced by the
    same dict structure as conc_spline_final.py, so all downstream code
    (estimate_concordance, optconn7._estimate_concordance) is unaffected.
    """
    data = {}
    with h5py.File(splines_path, 'r') as f:
        fmt = f.attrs.get('format', '')
        if fmt != 'splh5':
            raise ValueError(
                f"{splines_path}: unrecognised format '{fmt}'. "
                "Expected a file written by conc_spline_final.py save_splines().")

        for key in f.keys():
            g     = f[key]
            level = g.attrs['level']

            # ── rebuild splines from (t, c, k) arrays ────────────────────
            sg = g['splines']
            n  = len(sg)
            splines = [
                BSpline(sg[str(i)]['t'][:],
                        sg[str(i)]['c'][:],
                        int(sg[str(i)].attrs['k']))
                for i in range(n)
            ]

            # ── optional 2D correction mesh ───────────────────────────────
            cx = g['correction_x'][:] if 'correction_x' in g else None
            cy = g['correction_y'][:] if 'correction_y' in g else None

            data[level] = {
                'H_vals'            : g['H_vals'][:],
                'C_vals'            : g['C_vals'][:],
                'C_mins'            : g['C_mins'][:],
                'matrix'            : g['matrix'][:],
                'best_s'            : g['best_s'][:],
                'loo_mse'           : g['loo_mse'][:],
                'splines'           : splines,
                'correction_x'      : cx,
                'correction_y'      : cy,
                'correction_source' : g.attrs.get('correction_source'),
                'correction_fns'    : (_build_correction_fns(cx, cy)
                                       if cx is not None else None),
            }
    return data


# ──────────────────────────────────────────────────────────────────────────────
# 5.  Concordance estimation
# ──────────────────────────────────────────────────────────────────────────────

def _invert_single_spline(spl: UnivariateSpline,
                          z_obs: float,
                          C_min: float,
                          C_max: float) -> tuple:
    """
    Find C* such that spl(C*) = z_obs by Brent's method.

    Significance gate: if z_obs > Z_SIGNIFICANCE (-1.96) → C*=0, 'not_significant'.
    Returns (C_star, status) — status: 'not_significant', 'ok', 'above_range'.

    The main fitted range [C_min, C_max] is always monotone (calibration guarantee).
    The extrapolation zone [0, C_min] uses the cubic spline extended beyond its
    fitted domain and may be non-monotone (wiggles), particularly when C_min is
    large (e.g. 0.4–0.6 for higher percentile levels).  A naive brentq([0, C_min])
    therefore fails with 'f(a) and f(b) must have different signs' when z_obs falls
    between the endpoint values but the spline dips below z_obs in between.

    Robust strategy: scan [0, C_min] on a coarse grid, find all sign-change
    brackets, and use the rightmost one — giving the largest C* consistent with
    the observation.  If no crossing exists, z_obs is outside the spline's range
    in [0, C_min] and we return not_significant.
    """
    if z_obs > Z_SIGNIFICANCE:
        return 0.0, 'not_significant'

    v_at_Cmax = float(spl(C_max))
    if z_obs <= v_at_Cmax:
        return float(C_max), 'above_range'

    v_at_Cmin = float(spl(C_min))
    if z_obs <= v_at_Cmin:
        # Normal case: monotone fitted range [C_min, C_max]
        C_star = brentq(lambda c: float(spl(c)) - z_obs,
                        C_min, C_max, xtol=1e-6, maxiter=200)
        return float(C_star), 'ok'

    # Extrapolation: scan [0, C_min] for sign-change brackets
    _N_SCAN = 500
    x_scan = np.linspace(0.0, C_min, _N_SCAN)
    f_scan = np.array([float(spl(xi)) - z_obs for xi in x_scan])
    sign_changes = np.where(np.diff(np.sign(f_scan)) != 0)[0]

    if len(sign_changes) > 0:
        k = sign_changes[-1]   # rightmost bracket
        C_star = brentq(lambda c: float(spl(c)) - z_obs,
                        float(x_scan[k]), float(x_scan[k + 1]),
                        xtol=1e-6, maxiter=200)
        return float(C_star), 'ok'

    # No crossing: z_obs outside the extrapolated spline range → not significant
    return 0.0, 'not_significant'


def estimate_concordance(spline_data: dict,
                         z_var_obs: float,
                         H_est: float,
                         level: str = '2.5%') -> dict:
    """
    Estimate the concordance lower/point bound for the given level.

    If the level's entry in spline_data contains a 'correction' callable
    (embedded during precompute via --correction), C_lower is passed through
    it to convert the raw spline estimate to a bias-corrected value.

    Parameters
    ----------
    spline_data : dict returned by load_splines() or precompute()
    z_var_obs   : observed z_var from optconn6 (typically negative)
    H_est       : estimated noise half-width from optconn6
    level       : key in spline_data, e.g. '2.5%', '5%', '50%'

    Returns
    -------
    dict with keys: C_lower, C_lower_raw (pre-correction), level, H_est,
                    H_clipped, z_var_obs, status, interp_weight,
                    C_at_H_lo, C_at_H_hi, corrected (bool)
    """
    if level not in spline_data:
        raise ValueError(
            f"Level '{level}' not found. Available: {sorted(spline_data.keys())}")

    sd      = spline_data[level]
    H_vals  = sd['H_vals']
    C_vals  = sd['C_vals']
    C_mins  = sd['C_mins']
    splines = sd['splines']
    C_max   = float(C_vals[-1])

    H_clipped = float(np.clip(H_est, H_vals[0], H_vals[-1]))
    idx       = int(np.clip(
        np.searchsorted(H_vals, H_clipped, side='right') - 1,
        0, len(H_vals) - 2))

    H_lo, H_hi = H_vals[idx], H_vals[idx + 1]
    w = float((H_clipped - H_lo) / (H_hi - H_lo)) if H_hi != H_lo else 0.0

    C_lo, status_lo = _invert_single_spline(
        splines[idx],     z_var_obs, C_mins[idx],     C_max)
    C_hi, status_hi = _invert_single_spline(
        splines[idx + 1], z_var_obs, C_mins[idx + 1], C_max)

    C_lower_raw  = float((1.0 - w) * C_lo + w * C_hi)
    final_status = status_lo if status_lo == status_hi else 'ok'

    # Apply 2D correction mesh: per-H PCHIP callables, same H interpolation as splines
    corr_fns = sd.get('correction_fns')
    if corr_fns is not None and final_status == 'ok':
        C_lower   = float((1.0 - w) * corr_fns[idx](C_lo)
                          + w       * corr_fns[idx + 1](C_hi))
        corrected = True
    else:
        C_lower   = C_lower_raw
        corrected = False

    return {
        'C_lower'       : C_lower,
        'C_lower_raw'   : C_lower_raw,
        'level'         : level,
        'H_est'         : float(H_est),
        'H_clipped'     : H_clipped,
        'z_var_obs'     : float(z_var_obs),
        'status'        : final_status,
        'interp_weight' : w,
        'C_at_H_lo'     : C_lo,
        'C_at_H_hi'     : C_hi,
        'corrected'     : corrected,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 6.  High-level commands
# ──────────────────────────────────────────────────────────────────────────────

def _prepare_corrections(correction_files: dict, tables: dict, verbose: bool) -> dict:
    """
    Validate every requested --correction and build its 2D mesh arrays.

    Any problem is an error.  A correction that was asked for is never dropped
    silently: unknown level label, missing or unreadable file, H/C grid that does
    not match the calibration table, missing cells, or a row that is not strictly
    increasing all raise an exception.  Called before the (slow) spline fit, so
    a mistake stops the run immediately and no output file is written.

    Returns {level_label: (correction_x, correction_y, source_path)}.
    """
    prepared = {}
    if not correction_files:
        return prepared

    unknown = [lv for lv in correction_files if lv not in tables]
    if unknown:
        raise ValueError(
            f"--correction level(s) {unknown} match no calibration table. "
            f"Levels of the --tsv files: {sorted(tables)}. The label before the "
            f"colon must be identical to the level label (e.g. '5%'). In a Windows "
            f".bat/.cmd file write '%' as '%%'.")

    for level_label, corr_path in correction_files.items():
        if verbose:
            print(f"  Loading correction matrix for {level_label}: {corr_path}",
                  file=sys.stderr)
        if not Path(corr_path).is_file():
            raise FileNotFoundError(
                f"--correction file for level '{level_label}' not found: {corr_path}")
        try:
            H_corr, C_corr, C_est_mat = parse_correction_matrix(corr_path)
        except Exception as exc:
            raise ValueError(
                f"cannot read correction matrix '{corr_path}' "
                f"(level {level_label}): {exc}") from exc

        table = tables[level_label]
        H_cal, C_cal = table['H_vals'], table['C_vals']
        if (H_corr.shape != H_cal.shape or C_corr.shape != C_cal.shape
                or not np.allclose(H_corr, H_cal) or not np.allclose(C_corr, C_cal)):
            raise ValueError(
                f"correction matrix '{corr_path}': its H/C grid does not match the "
                f"calibration table of level {level_label}.\n"
                f"  correction matrix: H={H_corr.tolist()}  C={C_corr.tolist()}\n"
                f"  calibration table: H={H_cal.tolist()}  C={C_cal.tolist()}")
        try:
            cx, cy = _derive_correction_mesh(C_est_mat, C_corr)
        except ValueError as exc:
            raise ValueError(
                f"correction matrix '{corr_path}' (level {level_label}): {exc}") from exc

        prepared[level_label] = (cx, cy, str(corr_path))
        if verbose:
            print(f"  2D correction mesh embedded: "
                  f"{cx.shape[0]} H rows x {cx.shape[1]-1} C knots",
                  file=sys.stderr)
            print(f"  {'C_est (raw)':>12}  {'C_true':>8}  {'bias':>8}"
                  f"   (row H={H_corr[0]:.2f})", file=sys.stderr)
            for x_k, y_k in zip(cx[0, 1:], cy[1:]):
                print(f"  {x_k:>12.6f}  {y_k:>8.4f}  {x_k-y_k:>+8.4f}",
                      file=sys.stderr)
    return prepared


def precompute(tsv_paths,
               hint_labels,
               output_path: str,
               n_s_candidates: int = 200,
               correction_files: dict = None,
               verbose: bool = True) -> dict:
    """
    Full precomputation pipeline:
        1. Load calibration data from TSV files.
        2. Fit splines for every percentile level found.
        3. Optionally load and embed bias-correction tables (one per level).
        4. Save to HDF5 file.

    Parameters
    ----------
    correction_files : dict {level_label: path} or None
        Maps a percentile level (e.g. '5%') to a 10x10 correction matrix TSV
        produced from validation experiments.  The correction is embedded in
        the HDF5 file so that estimate_concordance() and optconn apply it
        automatically.  Every entry is validated before the spline fit and any
        problem raises an exception (see _prepare_corrections).
    """
    if verbose:
        print(f"Loading calibration tables from {len(tsv_paths)} TSV file(s)…",
              file=sys.stderr)

    tables = load_percentile_table(tsv_paths, hint_labels)

    # Validate / build all requested corrections BEFORE the slow spline fit.
    corrections = _prepare_corrections(correction_files, tables, verbose)

    spline_data = {}
    for level_label, table in sorted(tables.items()):
        if verbose:
            print(f"\nFitting splines for {level_label} percentile…",
                  file=sys.stderr)
        sd = fit_splines(
            table['H_vals'], table['C_vals'], table['matrix'],
            n_s_candidates=n_s_candidates, verbose=verbose)

        sd['correction_x']      = None
        sd['correction_y']      = None
        sd['correction_fns']    = None
        sd['correction_source'] = None
        if level_label in corrections:
            cx, cy, src = corrections[level_label]
            sd['correction_x']      = cx
            sd['correction_y']      = cy
            sd['correction_fns']    = _build_correction_fns(cx, cy)
            sd['correction_source'] = src

        spline_data[level_label] = sd

    save_splines(spline_data, output_path)
    return spline_data


def print_summary(spline_data: dict) -> None:
    """Print a human-readable summary of precomputed splines to stdout."""
    print("=== Concordance Spline Summary ===")
    for level, sd in sorted(spline_data.items()):
        print(f"\n  Percentile level : {level}")
        print(f"  H grid           : {sd['H_vals']}")
        print(f"  C grid           : {sd['C_vals']}")
        print(f"  Significance threshold : z <= {Z_SIGNIFICANCE}")
        print(f"  Smoothing factors (LOO-CV on significant points only):")
        for H, cmin, s, mse in zip(sd['H_vals'], sd['C_mins'],
                                    sd['best_s'], sd['loo_mse']):
            print(f"    H={H:.2f}  C_min={cmin:.1f}  s={s:.5f}  LOO_MSE={mse:.5f}")

    print()
    # Inversion examples for the first available level
    first_level = sorted(spline_data.keys())[0]
    sd     = spline_data[first_level]
    spl0   = sd['splines'][0]
    c_min0 = float(sd['C_mins'][0])
    c_max  = float(sd['C_vals'][-1])
    print(f"  Inversion examples (level={first_level}, H={sd['H_vals'][0]:.2f}):")
    print(f"  {'z_var_obs':>10}  {'C_lower':>8}  status")
    for z in [-1.5, -1.96, -2.0, -3.0, -4.0, -5.0, -7.0, -10.0, -13.0]:
        c, st = _invert_single_spline(spl0, z, c_min0, c_max)
        print(f"  {z:>10.2f}  {c:>8.4f}  {st}")


# ──────────────────────────────────────────────────────────────────────────────
# 7.  CLI
# ──────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='conc_spline_final.py',
        description=(
            "Precompute concordance splines from tab-separated calibration files\n"
            "and estimate concordance bounds from observed z_var values.\n\n"
            "Two sub-commands:\n"
            "  precompute  — fit splines from TSV calibration files (slow, run once)\n"
            "  estimate    — query precomputed splines (fast)\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest='command')

    # ── precompute ─────────────────────────────────────────────────────────
    p_pre = sub.add_parser(
        'precompute',
        help="Fit splines from TSV calibration files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Load one or more TSV calibration files, run LOO cross-validation\n"
            "to select the smoothing factor for each H row, fit UnivariateSplines,\n"
            "and save the result to an HDF5 file.\n\n"
            "TSV format (tab-separated):\n"
            "  Line 1 (optional): # level: 50%\n"
            "  Line 2 (header)  : Noise level<TAB>C_0.1<TAB>...<TAB>C_1.0\n"
            "  Lines 3+         : 0.05<TAB>-2.03<TAB>...\n\n"
            "The level label is detected from the '# level:' comment,\n"
            "the filename (2.5, 5, 50, median), or --levels.\n\n"
            "Run this once; then use 'estimate' for each dataset."
        )
    )
    p_pre.add_argument(
        '--tsv', nargs='+', required=True, metavar='FILE',
        help="One TSV file per percentile level (space-separated or repeated).")
    p_pre.add_argument(
        '--levels', nargs='+', metavar='LABEL', default=None,
        help="Explicit level labels in the same order as --tsv, e.g. '2.5%%' '5%%' '50%%'.\n"
             "Only needed when the level cannot be detected automatically.")
    p_pre.add_argument(
        '--output', '-o', default='concordance_splines.h5', metavar='FILE',
        help="Output HDF5 file (default: concordance_splines.h5).")
    p_pre.add_argument(
        '--n-candidates', type=int, default=200, metavar='N',
        help="Number of smoothing factor candidates on log scale (default: 200).")
    p_pre.add_argument(
        '--quiet', '-q', action='store_true',
        help="Suppress progress output.")
    p_pre.add_argument(
        '--correction', nargs='+', metavar='LEVEL:FILE', default=None,
        help="2D correction matrix file(s), one per percentile level.\n"
             "Format: LEVEL:FILE  e.g.  5%%:C_by_splines.tsv\n"
             "FILE must be a 10x10 TSV (same layout as the calibration TSV)\n"
             "with cells = median raw C_est per (H, C_true) cell,\n"
             "as produced by validation experiments on synthetic datasets.\n"
             "The correction is applied automatically by estimate_concordance().")
    p_pre.add_argument(
        '--summary', action='store_true',
        help="Print a human-readable summary after fitting.")

    # ── estimate ───────────────────────────────────────────────────────────
    p_est = sub.add_parser(
        'estimate',
        help="Estimate concordance bound from z_var and noise.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Query precomputed splines to find a concordance bound C*.\n\n"
            "For the 2.5%% level (97.5%% confidence lower bound):\n"
            "  C* = minimum concordance consistent with the observation.\n\n"
            "For the 50%% level (point estimate, requires p50 calibration data):\n"
            "  C* = most likely concordance level."
        )
    )
    p_est.add_argument(
        '--splines', default='concordance_splines.h5', metavar='FILE',
        help="HDF5 splines file from precompute (default: concordance_splines.h5).")
    p_est.add_argument(
        '--z-var', type=float, required=True, metavar='Z',
        help="Observed z_var from optconn6 (typically negative).")
    p_est.add_argument(
        '--noise', '--noise-half', dest='noise', type=float, required=True,
        metavar='H',
        help="Estimated noise half-width from optconn6 (H_est column).")
    p_est.add_argument(
        '--level', default=None, metavar='LABEL',
        help="Level to report in verbose mode (e.g. '2.5%%', '50%%').\n"
             "Defaults to the first level in the splines file.\n"
             "Ignored with -q (quiet mode always reports all levels).")
    p_est.add_argument(
        '-q', '--quiet', action='store_true',
        help="Quiet mode: print a single TSV line with all available levels.\n"
             "Column order: z_var  H_est  C_<level1>  C_<level2>  …\n"
             "Levels are sorted alphabetically (2.5%% before 5%% before 50%%).")

    return parser


def cmd_precompute(args) -> None:
    hint_labels = args.levels if args.levels else [None] * len(args.tsv)

    # Parse --correction LEVEL:FILE pairs (split at the FIRST colon, so a Windows
    # path such as 5%:C:\\data\\corr.tsv keeps its drive letter)
    correction_files = {}
    if args.correction:
        for entry in args.correction:
            level_label, sep, path = entry.partition(':')
            level_label, path = level_label.strip(), path.strip()
            if not sep or not level_label or not path:
                sys.exit(f"Error: --correction entry '{entry}' must be LEVEL:FILE, "
                         f"e.g. 5%:corr_p05.tsv (in a Windows .bat/.cmd file write "
                         f"% as %%).")
            if level_label in correction_files:
                sys.exit(f"Error: level '{level_label}' is given more than once in "
                         f"--correction.")
            correction_files[level_label] = path

    try:
        spline_data = precompute(
            tsv_paths=args.tsv,
            hint_labels=hint_labels,
            output_path=args.output,
            n_s_candidates=args.n_candidates,
            correction_files=correction_files if correction_files else None,
            verbose=not args.quiet,
        )
    except (ValueError, FileNotFoundError) as exc:
        sys.exit(f"Error: {exc}")
    if args.summary:
        print_summary(spline_data)


def cmd_estimate(args) -> None:
    spline_data  = load_splines(args.splines)
    sorted_levels = sorted(spline_data.keys())

    if args.quiet:
        # Emit one TSV line: z_var  H_est  C_level1  C_level2  …
        parts = [f"{args.z_var:.8f}", f"{args.noise:.4f}"]
        for lv in sorted_levels:
            r = estimate_concordance(spline_data, args.z_var, args.noise, lv)
            parts.append(f"{r['C_lower']:.6f}")
        print('\t'.join(parts))
        return

    # Verbose mode
    level = args.level if args.level else sorted_levels[0]
    if level not in spline_data:
        print(f"Error: level '{level}' not in splines file. "
              f"Available: {sorted_levels}", file=sys.stderr)
        sys.exit(1)

    res = estimate_concordance(spline_data, args.z_var, args.noise, level)

    # Human-readable confidence label
    conf_labels = {'2.5%': '97.5%', '5%': '95.0%', '50%': 'point estimate (50%)'}
    conf = conf_labels.get(level, level)

    print(f"  z_var observed : {args.z_var}")
    print(f"  Noise H_est    : {args.noise}")
    print(f"  Available levels in splines file: {sorted_levels}")
    print(f"\n  [{conf}  |  percentile threshold {level}]")
    print(f"  Concordance C* : {res['C_lower']:.4f}")
    print(f"  Status         : {res['status']}")
    if res['status'] == 'not_significant':
        print(f"  Note: z_var={args.z_var:.4f} > {Z_SIGNIFICANCE}; "
              "data not significantly concordant.")
    elif res['status'] == 'above_range':
        print("  Note: z_var more extreme than C=1.0 predicts; C* capped at 1.0.")

    H_lo_idx = int(np.searchsorted(
        spline_data[level]['H_vals'], res['H_clipped'], side='right') - 1)
    H_lo = spline_data[level]['H_vals'][H_lo_idx]
    H_hi = spline_data[level]['H_vals'][
        min(H_lo_idx + 1, len(spline_data[level]['H_vals']) - 1)]
    print(f"  H interpolation: C*={res['C_at_H_lo']:.4f} at H={H_lo:.2f}, "
          f"C*={res['C_at_H_hi']:.4f} at H={H_hi:.2f}, "
          f"weight={res['interp_weight']:.3f}")


def main() -> None:
    parser = _build_parser()
    args   = parser.parse_args()

    if args.command == 'precompute':
        cmd_precompute(args)
    elif args.command == 'estimate':
        cmd_estimate(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
