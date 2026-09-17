#!/usr/bin/env python
"""Paper-style heatmaps for the E12 crosstab grids (non-MLP settings).

One PNG per (setting, phase): rows = wd shape, columns = complete-grid
budget rungs (same 4/4 columns as e12_crosstab.py), cells = best test acc
(seed 42, coupled WD), annotated with the numeric value.

Colormap: RdYlBu_r (blue = low acc, red = high acc), white gridlines,
dark text on light cells / white text on dark cells. Compact, colorbar-free
(annotation carries the values; range is per-table).
Output: e12_multi_setting/figures/e12_heat_<model>_<phase>.png
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from e12_full_table import build_rows
from e12_crosstab import iter_grids, SHAPE_LABELS, DISPLAY_SHAPES

E12 = Path(__file__).resolve().parent
FIG = E12 / 'figures'
CMAP = 'RdYlBu_r'


def _exp_s(v, a, hi, exp_mR, den):
    """Exponential-in-value mapping anchored at the max: s = (e^{a(v-hi)} -
    e^{-aR}) / (1 - e^{-aR}) in [0, 1]. Color steps grow exponentially as v
    approaches the max, so closely-spaced top values get distinct shades."""
    return (np.exp(a * (v - hi)) - exp_mR) / den


def _text_color(rgb):
    """black on light cells, white on dark cells (perceived luminance)."""
    r, g, b = rgb
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    return 'white' if lum < 0.5 else 'black'


def make_heatmaps(include_mlp=False):
    rows = build_rows(include_ms=True)
    tab = pd.DataFrame(rows)
    if not include_mlp:
        tab = tab[~tab['setting'].str.startswith('mlp')]
    FIG.mkdir(parents=True, exist_ok=True)
    made = []
    for setting, phase, cols, lam_ref, acc in iter_grids(tab):
        shapes = list(DISPLAY_SHAPES)
        mat = np.full((len(shapes), len(cols)), np.nan)
        for i, s in enumerate(shapes):
            for j, c in enumerate(cols):
                mat[i, j] = acc[s].get(c, np.nan)

        # Multi-seed means per (shape, C) from ALL seeds present
        g = tab[(tab['setting'] == setting) & (tab['phase'] == phase)]
        ms_mean = {}
        ms_n = {}
        for i, s in enumerate(shapes):
            for j, c in enumerate(cols):
                sub = g[(g['wd_sched'] == s)
                        & (g['budget_c'].round(2) == c)]
                if len(sub):
                    ms_mean[(i, j)] = float(sub['best_acc'].mean())
                    ms_n[(i, j)] = len(sub)
        # Any cell with multi-seed data (>=2 seeds) shows the multi-seed
        # mean (bold); single-seed cells show the seed-42 value.
        display = mat.copy()
        bold_mask = np.zeros_like(mat, dtype=bool)
        for (i, j) in ms_mean:
            if ms_n[(i, j)] >= 2:
                display[i, j] = ms_mean[(i, j)]
                bold_mask[i, j] = True

        # Exponential-in-value scale anchored at the max: closely-spaced top
        # values (76-79.5) each get a clearly distinct shade, while the
        # collapse tail saturates to deep blue. a = 1 / (max - p75) so the
        # top quartile of values spans ~63% of the colormap. Colors follow
        # the DISPLAYED value (multi-seed mean where applied).
        flat = display[~np.isnan(display)]
        hi = float(np.nanmax(flat))
        vmin = float(np.nanmin(flat))
        span_top = max(hi - float(np.percentile(flat, 75)), 0.3)
        a = 1.0 / span_top
        R = hi - vmin
        if R < 1e-9:
            R = 1.0
        exp_mR = np.exp(-a * R)
        den = 1.0 - exp_mR
        s = _exp_s(display, a, hi, exp_mR, den)

        ncol = len(cols)
        fig, ax = plt.subplots(figsize=(max(3.2, ncol * 0.62),
                                        2.2 + len(shapes) * 0.42))
        cmap = plt.get_cmap(CMAP)
        im = ax.imshow(s, cmap=cmap, vmin=0.0, vmax=1.0,
                       aspect='auto', interpolation='nearest')
        for i in range(len(shapes)):
            for j in range(ncol):
                v = display[i, j]
                if np.isnan(v):
                    continue
                rgb = cmap(s[i, j])[:3]
                ax.text(j, i, f'{v:.1f}', ha='center', va='center',
                        fontsize=9, color=_text_color(rgb),
                        fontweight='bold' if bold_mask[i, j] else 'normal')
        ax.set_xticks(range(ncol))
        ax.set_xticklabels([f'{c:.2f}' for c in cols], fontsize=8)
        ax.set_yticks(range(len(shapes)))
        ax.set_yticklabels([SHAPE_LABELS[s] for s in shapes], fontsize=9)
        ax.tick_params(length=0)
        ax.set_xlabel('budget C', fontsize=9)
        # white gridlines
        for s in ['top', 'bottom', 'left', 'right']:
            ax.spines[s].set_visible(False)
        ax.set_xticks(np.arange(-0.5, ncol, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(shapes), 1), minor=True)
        ax.grid(which='minor', color='white', linewidth=1.2)
        ax.tick_params(which='minor', length=0)

        key = setting.replace('/', '_').replace('cifar', 'c') + '_' + phase.lower()
        out = FIG / f'e12_heat_{key}.png'
        fig.savefig(out, dpi=110, bbox_inches='tight',
                    facecolor='white')
        plt.close(fig)
        made.append((setting, phase, out.name))
        print('saved', out)
    return made


if __name__ == '__main__':
    make_heatmaps()
