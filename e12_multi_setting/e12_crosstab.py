#!/usr/bin/env python
"""E12 compact crosstab: per (setting, phase), rows = wd shape, cols = budget
rung (rounded to nearest ladder value), cells = best acc. Same sources as
e12_full_table.py.

Rendering: empty cell for missing rungs (no NaN), row maximum bolded.
Writes e12_multi_setting/tables/e12_crosstab.md and prints.
Also (re)embeds the tables into e12_multi_setting/README.md between the
markers "## 完整阶梯表" and "## 多种子结果".
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e12_full_table import build_rows  # reuse

ROOT = Path(__file__).resolve().parent.parent
E12 = ROOT / 'e12_multi_setting'
TABLES = E12 / 'tables'
README = E12 / 'README.md'
MARKER_START = '## 完整阶梯表'
MARKER_END = '## 多种子结果'

# Row labels for the rendered tables (budget math uses the raw keys).
SHAPE_LABELS = {'fixed': 'fixed',
                'linear_up': 'linear up',
                'linear': 'linear down',
                'iso_product': 'iso up'}


def render_table(piv, lam_piv):
    """Markdown table: header + one row per shape, bold per-row max, no NaN.

    Each cell shows acc and the lambda0 of that run, e.g. "58.15 (λ=0.01)".
    """
    cols = list(piv.columns)
    header = '| wd_sched | ' + ' | '.join(f'{c:.2f}' for c in cols) + ' |'
    sep = '|---|' + '---|' * len(cols)
    lines = [header, sep]
    for idx, row in piv.iterrows():
        row_max = row.max()
        cells = []
        for c in cols:
            v = row[c]
            if pd.isna(v):
                cells.append('')
            else:
                # Lambda annotation only on the fixed row: there it IS the
                # traditional const-WD value. On dynamic-shape rows lambda0 is
                # just a shape parameter and would only add noise.
                if idx == 'fixed':
                    lam = lam_piv.loc[idx, c]
                    s = f'{v:.2f} (λ={lam:.4g})'
                else:
                    s = f'{v:.2f}'
                if pd.notna(row_max) and v == row_max:
                    s = f'**{s}**'
                cells.append(s)
        label = SHAPE_LABELS.get(idx, idx)
        lines.append(f'| {label} | ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def build_md_body():
    rows = build_rows()
    tab = pd.DataFrame(rows)
    out_lines = [
        '# E12 crosstab: acc by shape x measured lambda (setting-local C)',
        '',
        'Row 1 = measured lambda0 of the fixed-const-WD ladder. '
        'Row 2 = realized budget C = integral(lambda*eta) / '
        'integral(lambda_ref*eta) (simple division).',
        'Cells = best test acc (seed 42, coupled WD). '
        'Dynamic-shape cells land on the column whose C matches (2 dp).',
        '',
    ]
    for (setting, phase), g in tab.groupby(['setting', 'phase'], sort=True):
        if g.empty:
            continue
        # Columns = every measured rung: one column per (lambda0, C) pair,
        # ordered by C. Row1 = lambda0, row2 = C = integral ratio (2 dp).
        g = g.assign(c2=g['budget_c'].round(2))
        col_keys = sorted(
            set(zip(g['lambda0'].round(6), g['c2'])),
            key=lambda t: (t[0], t[1]))
        header1 = '| λ | ' + ' | '.join(
            f'{lam:.3g}' for lam, c in col_keys) + ' |'
        header2 = '| C | ' + ' | '.join(
            f'{c:.2f}' for lam, c in col_keys) + ' |'
        sep = '|---|' + '---|' * len(col_keys)
        lines = [header1, header2, sep]
        for shape in ['fixed', 'linear_up', 'linear', 'iso_product']:
            h = g[g['wd_sched'] == shape]
            row_max = float(h['best_acc'].max()) if len(h) else None
            cells = []
            for lam, c in col_keys:
                sub = h[(h['lambda0'].round(6) == lam)
                        & (h['c2'] == c)]
                v = float(sub['best_acc'].max()) if len(sub) else None
                if v is None:
                    cells.append('')
                else:
                    bold = (row_max is not None and v == row_max)
                    cells.append(f'**{v:.2f}**' if bold else f'{v:.2f}')
            label = SHAPE_LABELS.get(shape, shape)
            lines.append('| ' + label + ' | ' + ' | '.join(cells) + ' |')
        out_lines.append(f'## {setting} / {phase}')
        out_lines.append('')
        out_lines.extend(lines)
        out_lines.append('')
    return '\n'.join(out_lines)

def main():
    text = build_md_body()
    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / 'e12_crosstab.md').write_text(text)

    # Embed into README (replace previous embedding, keep the rest intact).
    readme = README.read_text()
    start = readme.find(MARKER_START)
    end = readme.find(MARKER_END)
    if start == -1 or end == -1 or end <= start:
        print('WARN: README markers not found; skipping embed')
    else:
        body = '\n'.join(text.split('\n', 1)[1:])  # drop the H1
        new_readme = (readme[:start] + MARKER_START +
                      '（所有测过的份额，行=形状，列=预算 C 单位）\n\n' +
                      body.rstrip() + '\n\n' + readme[end:])
        README.write_text(new_readme)
        print('embedded into README (%d bytes)' % len(new_readme))
    print(text)


if __name__ == '__main__':
    main()
