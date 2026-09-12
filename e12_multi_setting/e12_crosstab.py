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
SHAPE_LABELS = {'fixed': 'fixed (const λ)',
                'linear_up': 'linear up',
                'linear': 'linear down',
                'iso_product': 'iso up'}


def render_table(piv, lam_piv):
    """Markdown table: header + one row per shape, bold per-row max, no NaN.

    Each cell shows acc and the lambda0 of that run, e.g. "58.15 (λ=0.01)".
    """
    cols = list(piv.columns)
    header = '| wd_sched | ' + ' | '.join(f'{c:g}' for c in cols) + ' |'
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
    # Snap realized budgets onto the nominal ladder rungs so every table
    # shares the same integer-ish columns (no 0.94/0.96 rungs; 0.94C and
    # 1.04C both land on the 1C column).
    RUNG = [0.33, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0, 9.0, 15.0]
    tab['budget_r'] = [min(RUNG, key=lambda r: abs(b - r))
                       for b in tab['budget_c']]

    out_lines = [
        '# E12 crosstab: best acc by shape x budget rung (setting-local C)',
        '',
        'Rows = WD schedule, columns = realized budget in setting-local C units',
        '(lambda_ref = fixed oracle of that setting/phase, seed 42).',
        'Empty cells = budget not tested. Bold = best value in that row.',
        '',
    ]
    for (setting, phase), g in tab.groupby(['setting', 'phase'], sort=True):
        g = g[g['seed'] == 42] if 'seed' in g else g
        # One run per (shape, budget rung): the highest-acc one, so lambda0
        # shown is the lambda0 of the run whose acc is displayed.
        best = g.sort_values('best_acc', ascending=False).drop_duplicates(
            ['wd_sched', 'budget_r'])
        piv = best.pivot_table(index='wd_sched', columns='budget_r',
                               values='best_acc', aggfunc='max')
        lam_piv = best.pivot_table(index='wd_sched', columns='budget_r',
                                   values='lambda0', aggfunc='first')
        piv = piv.reindex(['fixed', 'linear_up', 'linear', 'iso_product'])
        lam_piv = lam_piv.reindex(['fixed', 'linear_up', 'linear',
                                   'iso_product'])
        cols = [c for c in [0.33, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0,
                            6.0, 9.0, 15.0] if c in piv.columns]
        piv = piv[cols]
        lam_piv = lam_piv[cols]
        out_lines.append(f'## {setting} / {phase}')
        out_lines.append('')
        out_lines.append(render_table(piv, lam_piv))
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
        print(f'embedded into README ({len(new_readme)} bytes)')
    print(text)


if __name__ == '__main__':
    main()
