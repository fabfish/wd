"""
E10 (reviewer xkCF follow-up, 2026-08-03): held-out test of the constant C.

The reviewer asked for "one held-out test in which C is estimated on one setting
and used to predict lambda* for another architecture, batch size, or learning
rate". This module does the fitting and reporting for the width-ladder version of
that test:

  * fit C independently on a ladder of MLP widths (a family of architectures with
    a clean parameter-count knob),
  * regress log C on log width and extrapolate,
  * predict lambda* at two larger widths *before* those grids are run,
  * compare against a per-setting oracle grid and against three competing rules.

`fit_C_grid` deliberately re-implements nothing: it uses the same log-parabola
trough and the same `C = lambda* * sum_t eta_t` definition as E5a/E5c, so the
numbers are comparable to the ones already in the response.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.nips26_lib import fit_loglog_slope, sum_lr  # noqa: E402
from mlp_wd.scripts.run_e5c_c_sensitivity import _parabola_trough  # noqa: E402

TABLE_DIR = REPO_ROOT / 'rebuttal' / 'nips_rebuttal' / '_data'
PLOT_DIR = REPO_ROOT / 'outputs' / 'plots' / 'nips26'

DATASET_N = {'mnist': 60000, 'cifar10': 50000}


# --------------------------------------------------------------------------
# fitting
# --------------------------------------------------------------------------

def fit_C_grid(df, *, lrs, wds, epochs, batch_size, n, num_layers=3,
               widths=None, metric='best_test_loss', min_points=3):
    """
    Per (hidden_dim, momentum, lr), fit lambda* and C = lambda* * sum_t eta_t.

    Only rows whose weight decay is on the probe ladder `wds` are used, so that
    single-point rule evaluations added later to the same CSV cannot pollute the
    fit. Same convention as E5c: optimum from the log-parabola trough of
    `metric`, `interior` flags optima that sit on the edge of the swept range.
    """
    d = df.copy()
    d = d[(d['epochs'].astype(int) == int(epochs))
          & (d['batch_size'].astype(int) == int(batch_size))
          & (d['num_layers'].astype(int) == int(num_layers))]
    if widths is not None:
        d = d[d['hidden_dim'].astype(int).isin([int(w) for w in widths])]
    d = d[d['lr'].astype(float).map(
        lambda v: any(np.isclose(v, t) for t in lrs))]
    d = d[d['wd'].astype(float).map(
        lambda v: any(np.isclose(v, t, rtol=1e-6, atol=1e-15) for t in wds))]
    if d.empty:
        raise RuntimeError('fit_C_grid: no rows matched the probe grid')

    records = []
    for (h, mom, lr), g in d.groupby(['hidden_dim', 'momentum', 'lr']):
        g = g.sort_values('wd').drop_duplicates(subset=['wd'], keep='last')
        if len(g) < min_points:
            continue
        vals = g[metric].astype(float).values
        wd_vals = g['wd'].astype(float).values
        wd_arg = float(wd_vals[int(np.argmin(vals))])
        wd_star = _parabola_trough(wd_vals, vals)
        S = float(sum_lr(float(lr), epochs, batch_size, 'cosine', n=n))
        srt = np.sort(wd_vals)
        records.append({
            'hidden_dim': int(h), 'momentum': float(mom), 'lr': float(lr),
            'wd_argmin': wd_arg, 'wd_interp': wd_star,
            'metric_min': float(np.min(vals)),
            'acc_max': float(g['best_test_acc'].astype(float).max()),
            'sum_lr': S, 'C': wd_star * S,
            'interior': bool(srt[0] < wd_arg < srt[-1]),
            'n_points': int(len(g)),
        })
    opt = pd.DataFrame(records)
    if opt.empty:
        raise RuntimeError('fit_C_grid: no group had enough weight-decay points')
    return opt.sort_values(['momentum', 'hidden_dim', 'lr']).reset_index(drop=True)


def geo(series):
    s = np.asarray(series, float)
    s = s[np.isfinite(s) & (s > 0)]
    return float(np.exp(np.mean(np.log(s)))) if len(s) else float('nan')


def C_by_width(opt, prefer_interior=True):
    """Geometric-mean C per (momentum, hidden_dim)."""
    rows = []
    for (mom, h), g in opt.groupby(['momentum', 'hidden_dim']):
        sub = g[g['interior']] if (prefer_interior and g['interior'].any()) else g
        rows.append({'momentum': float(mom), 'hidden_dim': int(h),
                     'C': geo(sub['C']), 'n_lr': int(len(sub)),
                     'n_interior': int(g['interior'].sum())})
    return pd.DataFrame(rows).sort_values(['momentum', 'hidden_dim'])


def fit_C_vs_width(cw):
    """log C on log hidden_dim, per momentum."""
    out = {}
    for mom, g in cw.groupby('momentum'):
        g = g.sort_values('hidden_dim')
        fit = fit_loglog_slope(g['hidden_dim'].values, g['C'].values)
        # intercept from fit_loglog_slope is in log space: log C = a + b log h
        out[float(mom)] = dict(
            slope=fit['slope'], intercept=fit['intercept'],
            lo=fit['lo'], hi=fit['hi'], n=fit['n'],
            widths=[int(v) for v in g['hidden_dim']],
            C=[float(v) for v in g['C']],
        )
    return out


def predict_C(fit, hidden_dim):
    return float(math.exp(fit['intercept']) * float(hidden_dim) ** fit['slope'])


# --------------------------------------------------------------------------
# competing rules
# --------------------------------------------------------------------------

def steps_per_epoch(batch_size, n):
    return math.ceil(float(n) / float(batch_size))


def rule_lambda(rule, *, lr, epochs, batch_size, n, C_pred, product_ref):
    """
    ours     lambda = C_pred(width) / sum_t eta_t, C extrapolated from the ladder
    default  the value people type, 5e-4
    wang     lambda = 1/(eta*T), T counted in optimizer steps
    kosson   the product eta*lambda held at its reference value
    """
    if rule == 'ours':
        return C_pred / float(sum_lr(lr, epochs, batch_size, 'cosine', n=n))
    if rule == 'default':
        return 5e-4
    if rule == 'wang':
        return 1.0 / (float(lr) * steps_per_epoch(batch_size, n) * float(epochs))
    if rule == 'kosson':
        return product_ref / float(lr)
    raise ValueError(rule)


RULES = ['default', 'wang', 'kosson', 'ours']


# --------------------------------------------------------------------------
# prediction bookkeeping
# --------------------------------------------------------------------------

def predictions_path(dataset):
    return TABLE_DIR / f'e10_predictions_{dataset}.json'


def write_predictions(payload, dataset):
    path = predictions_path(dataset)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return path


def read_predictions(dataset):
    path = predictions_path(dataset)
    if not path.exists():
        raise RuntimeError(
            f'{path} missing: run the predict phase before the heldout phase, '
            'otherwise the test is not blind')
    return json.loads(path.read_text())
