#!/usr/bin/env python
"""
E4 blind-prediction points — standalone document.

The E4 arm of the rebuttal predicted lambda WITHOUT tuning (five strategies
calibrated once on the R18 reference: zero / default 5e-4 / kosson
(eta*lambda held) / wang (1/(eta*T_steps)) / ours (C/sum_lr)). Its rows,
notably the 9.62e-4 "ours" variant, are EXCLUDED from the main E12
grid-search tables and recorded here instead.

Note: the current predict_wd() no longer reproduces these numbers (the
reference C has since been refitted), so this table is a historical record
of what was actually run, not a description of current code.

Writes e12_multi_setting/e4_prediction_points.md
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
E11_CSV = ROOT / 'e11_raise_up' / 'results' / 'nips26_e11_runs.csv'
OUT = ROOT / 'e12_multi_setting' / 'e4_prediction_points.md'


def main():
    df = pd.read_csv(E11_CSV)
    g = df[df['exp'] == 'e4'].copy()
    g = g[['model', 'batch_size', 'lr', 'epochs', 'wd', 'best_test_acc',
           'diverged']].sort_values(['model', 'batch_size', 'lr', 'epochs',
                                     'wd'])

    lines = [
        '# E4 blind-prediction points (historical, excluded from E12 main tables)',
        '',
        'E4 predicted lambda with no tuning at the target setting: five',
        'strategies calibrated once on the R18 reference, applied blind to',
        'held-out settings. These rows are deliberately NOT part of the',
        'grid-search narrative of the main tables: 1C there means the best',
        'GRID-SEARCHED const WD, and these predicted points (e.g. 9.62e-4 on',
        'R50/VGG16) would otherwise pollute that definition.',
        '',
        'Protocol: cosine LR, momentum 0.9, seed 42.',
        '',
        '## Protocol-matched rows (B=128, lr=0.1, T=100)',
        '',
    ]
    main = g[(g['batch_size'] == 128) & (g['lr'] == 0.1) & (g['epochs'] == 100)]
    lines.append('| model | lambda | best acc | diverged |')
    lines.append('|---|---:|---:|---:|')
    for _, r in main.iterrows():
        lines.append(
            f"| {r['model']} | {r['wd']:.4g} | {r['best_test_acc']:.2f} | "
            f"{int(r['diverged'])} |")

    lines += [
        '',
        'The 9.62e-4 rows beat the grid oracle on R50 (grid 1.1e-3 = 77.72)',
        'and VGG16 (grid 1e-3 = 73.02) but are a prediction, not a search',
        'result, so they are kept out of the main comparison. The 2.56e-4',
        'rows are the wang strategy 1/(eta*T_steps); 5.98e-4 is ours/kosson',
        'under the cosine-calibrated C.',
        '',
        '## All E4 rows (including R18 protocol variants)',
        '',
        '| model | B | lr | T | lambda | best acc | diverged |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for _, r in g.iterrows():
        lines.append(
            f"| {r['model']} | {int(r['batch_size'])} | {r['lr']:g} | "
            f"{int(r['epochs'])} | {r['wd']:.4g} | "
            f"{r['best_test_acc']:.2f} | {int(r['diverged'])} |")

    lines.append('')
    OUT.write_text('\n'.join(lines) + '\n')
    print(f'wrote {OUT} ({len(g)} rows)')


if __name__ == '__main__':
    main()
