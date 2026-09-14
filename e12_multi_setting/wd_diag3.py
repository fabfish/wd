#!/usr/bin/env python
"""Fill7: intended C rungs per (setting, phase, shape) + progress vs CSV."""
import sys, json
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e12_full_table as ft

ROOT = Path(__file__).resolve().parent.parent
E12 = ROOT / 'e12_multi_setting'
CSV = E12 / 'results' / 'e12_runs.csv'

# anchor per (setting, phase): fixed seed42 C==1.00 lambda
rows = ft.build_rows()
tab = pd.DataFrame(rows)
anchors = {}
for (setting, phase), g in tab.groupby(['setting', 'phase'], sort=True):
    fx = g[g['wd_sched'] == 'fixed']
    one = fx[fx['budget_c'].round(2) == 1.00]
    if one.empty:
        one = fx.loc[fx['best_acc'].idxmax()]
    anchors[(setting, phase)] = float(one['lambda0'].iloc[0])

f7 = pd.DataFrame(json.load(open('/tmp/wd_queue/fill7_configs.json')))
f7['wd'] = f7['wd'].astype(float)

csv_df = pd.read_csv(CSV)
if 'wd_mode' not in csv_df.columns:
    csv_df['wd_mode'] = 'coupled'
csv_df = csv_df[(csv_df['wd_mode'].fillna('coupled') == 'coupled')
                & (csv_df['seed'] == 42) & (csv_df['exp'] == 'e12_fill')]
done = set()
for _, r in csv_df.iterrows():
    done.add((r['model'], r['dataset'], float(r['momentum']), r['wd_sched'],
              round(float(r['wd']), 6)))

shape_labels = {'fixed': 'fixed', 'linear_up': 'linear up',
                'linear': 'linear down', 'iso_product': 'iso up'}
total_rem = 0
for (setting, phase), g in tab.groupby(['setting', 'phase'], sort=True):
    model, ds = setting.split('/')
    mom = 0.9 if phase == 'SGDM' else 0.0
    lam_ref = anchors[(setting, phase)]
    f7sub = f7[(f7['model'] == model) & (f7['dataset'] == ds)
               & (f7['momentum'] == mom)]
    if f7sub.empty:
        continue
    print(f'== {setting} / {phase}  (lam_ref={lam_ref:.4g})')
    for shape, h in f7sub.groupby('wd_sched'):
        cs = sorted((h['wd'] / lam_ref).round(2))
        rem = [w for w in h['wd'] if (model, ds, mom, shape, round(w, 6)) not in done]
        total_rem += len(rem)
        print(f'   {shape_labels.get(shape, shape):12s} n={len(h):2d} '
              f'C={[f"{c:.2f}" for c in cs]}')
        if rem:
            print(f'        remaining {len(rem)}: {[f"{w:.4g}" for w in rem]}')
    # current dynamic C per shape in CSV (post-fill7 view for coverage)
    print()
print(f'TOTAL fill7 remaining: {total_rem}')
