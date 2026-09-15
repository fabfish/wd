#!/usr/bin/env python
"""E12 compact crosstab: per (setting, phase), rows = wd shape, columns =
measured budget rungs (realized C, 2 dp, union across shapes), cells = best
test acc. Same sources as e12_full_table.py.

Column semantics:
  - columns = every measured C rung (simple division: integral ratio vs the
    fixed-WD oracle lambda_ref), sorted ascending.
  - row 1 (lambda) = the const-WD lambda of that rung: the measured fixed-run
    lambda where one exists, else lambda_ref * C (the equivalent fixed lambda
    spending the same shrinkage budget).
  - row 2 (C) = the budget rung in setting-local C units.
  - shape rows = best test acc at the rung (empty = not measured). Dynamic
    shapes land on the column whose realized C matches (2 dp); their
    per-shape lambda0 lives in the full ladder table, not here.

Rendering: the lambda row is the markdown header (separator directly after
it) so it renders as the first row on GitHub, C is the second row, shape rows
follow; per-row maximum bolded, empty cells for missing rungs (no NaN).
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


def _anchor_of(g):
    """lambda_ref: fixed-run lambda at realized C == 1.00 (grid oracle);
    fallback to the best-acc fixed run if the 1.00 rung is absent."""
    fx = g[g['wd_sched'] == 'fixed']
    one = fx[fx['budget_c'].round(2) == 1.00]
    if one.empty:
        one = fx.loc[fx['best_acc'].idxmax()]
    return float(one['lambda0'].iloc[0])


def build_md_body():
    rows = build_rows()
    tab = pd.DataFrame(rows)
    out_lines = [
        '# E12 crosstab: acc by shape x budget rung (setting-local C)',
        '',
        'Columns = complete-grid budget rungs: C where all four shapes '
        '(fixed, linear up, linear down, iso up) were measured. '
        'Shape-specific probe rungs are excluded here; they live in the '
        'full ladder table.',
        'Row 1 (lambda) = const-WD value of that rung: measured fixed lambda '
        'where one exists, else lambda_ref x C (equivalent fixed lambda).',
        'Row 2 (C) = realized budget = integral(lambda*eta) / '
        'integral(lambda_ref*eta) (simple division).',
        'Cells = best test acc (seed 42, coupled WD); empty = not measured; '
        'per-row max bolded. Dynamic-shape cells land on the column whose '
        'realized C matches (2 dp); per-shape lambda0 is in the full ladder '
        'table.',
        '',
    ]
    for (setting, phase), g in tab.groupby(['setting', 'phase'], sort=True):
        if g.empty:
            continue
        lam_ref = _anchor_of(g)
        c2 = g['budget_c'].round(2)

        # Columns = complete-grid rungs only: C where ALL four shapes were
        # measured. Shape-specific probe rungs (single-shape extra density,
        # e.g. fill rungs) are excluded here; they live in the full ladder
        # table.
        cov = {}
        for _, r in g.iterrows():
            cov.setdefault(round(float(r['budget_c']), 2), set()).add(
                r['wd_sched'])
        all_shapes = set(['fixed', 'linear_up', 'linear', 'iso_product'])
        cols = sorted(c for c, s in cov.items()
                      if c >= 0.005 and all_shapes <= s)

        # measured fixed lambda per rounded C (for the lambda row)
        fixed_lam = {}
        for _, r in g[g['wd_sched'] == 'fixed'].iterrows():
            c = round(float(r['budget_c']), 2)
            fixed_lam.setdefault(c, float(r['lambda0']))

        lam_cells = [f"{fixed_lam.get(c, lam_ref * c):.3g}" for c in cols]
        header1 = '| λ | ' + ' | '.join(lam_cells) + ' |'
        sep = '|---|' + '---|' * len(cols)
        header2 = '| C | ' + ' | '.join(f'{c:.2f}' for c in cols) + ' |'
        lines = [header1, sep, header2]

        for shape in ['fixed', 'linear_up', 'linear', 'iso_product']:
            h = g[g['wd_sched'] == shape]
            acc = {}
            for c in cols:
                sub = h[h['budget_c'].round(2) == c]
                if len(sub):
                    acc[c] = float(sub['best_acc'].max())
            row_max = max(acc.values()) if acc else None
            cells = []
            for c in cols:
                if c not in acc:
                    cells.append('')
                else:
                    v = acc[c]
                    cells.append(f'**{v:.2f}**' if v == row_max else f'{v:.2f}')
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
