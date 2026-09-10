#!/usr/bin/env python
"""E12 compact crosstab: per (setting, phase), rows = wd shape, cols = budget
rung (rounded to nearest ladder value), cells = best acc. Same sources as
e12_full_table.py. Writes e12_multi_setting/tables/e12_crosstab.md and prints."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e12_full_table import build_rows  # reuse


def main():
    rows = build_rows()
    tab = pd.DataFrame(rows)
    tab['budget_r'] = tab['budget_c'].round(1)

    out_lines = [
        '# E12 crosstab: best acc by shape x budget rung (setting-local C)',
        '',
        'Rows = WD schedule, columns = realized budget in setting-local C units',
        '(lambda_ref = fixed oracle of that setting/phase, seed 42).',
        'R50/C100 matched rows fill in as the queue progresses.',
        '',
    ]
    for (setting, phase), g in tab.groupby(['setting', 'phase'], sort=True):
        g = g[g['seed'] == 42] if 'seed' in g else g
        piv = g.pivot_table(index='wd_sched', columns='budget_r',
                            values='best_acc', aggfunc='max')
        piv = piv.reindex(['fixed', 'linear_up', 'linear', 'iso_product'])
        out_lines.append(f'## {setting} / {phase}')
        out_lines.append('')
        out_lines.append(piv.round(2).to_markdown())
        out_lines.append('')
    out = Path(__file__).resolve().parent.parent / 'e12_multi_setting' \
        / 'tables' / 'e12_crosstab.md'
    out.parent.mkdir(parents=True, exist_ok=True)
    text = '\n'.join(out_lines)
    out.write_text(text)
    print(text)


if __name__ == '__main__':
    main()
