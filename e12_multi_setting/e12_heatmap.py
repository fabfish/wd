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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from e12_full_table import build_rows
from e12_crosstab import iter_grids, SHAPE_LABELS

E12 = Path(__file__).resolve().parent
FIG = E12 / 'figures'
CMAP = 'RdYlBu_r'


def _text_color(rgb):
    """black on light cells, white on dark cells (perceived luminance)."""
    r, g, b = rgb
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    return 'white' if lum < 0.5 else 'black'


def make_heatmaps(include_mlp=False):
    rows = build_rows()
    tab = pd.DataFrame(rows)
    if not include_mlp:
        tab = tab[~tab['setting'].str.startswith('mlp')]
    FIG.mkdir(parents=True, exist_ok=True)
    made = []
    for setting, phase, cols, lam_ref, acc in iter_grids(tab):
        shapes = ['fixed', 'linear_up', 'linear', 'iso_product']
        mat = np.full((len(shapes), len(cols)), np.nan)
        for i, s in enumerate(shapes):
            for j, c in enumerate(cols):
                mat[i, j] = acc[s].get(c, np.nan)

        # per-table scale, padded slightly so extremes are not clipped
        lo, hi = np.nanmin(mat), np.nanmax(mat)
        pad = max(0.05, (hi - lo) * 0.08)
        vmin, vmax = lo - pad, hi + pad

        ncol = len(cols)
        fig, ax = plt.subplots(figsize=(max(3.2, ncol * 0.62),
                                        2.2 + len(shapes) * 0.42))
        cmap = plt.get_cmap(CMAP)
        im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax,
                       aspect='auto', interpolation='nearest')
        for i in range(len(shapes)):
            for j in range(ncol):
                v = mat[i, j]
                if np.isnan(v):
                    continue
                rgb = cmap((v - vmin) / (vmax - vmin))[:3]
                ax.text(j, i, f'{v:.1f}', ha='center', va='center',
                        fontsize=9, color=_text_color(rgb))
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
