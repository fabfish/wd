#!/usr/bin/env python
"""Diagnose column design for the new crosstab: fixed-ladder columns + dynamic
shapes mapped by realized C (2dp). Check per (setting, phase):
  - fixed C ladder (sorted by lambda)
  - dynamic C values per shape
  - which dynamic C values lack a fixed column (coverage report)
  - fill7 additions (fixed lambda still queued/not yet in CSV)
"""
import sys, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e12_full_table  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
E12 = ROOT / 'e12_multi_setting'

rows = e12_full_table.build_rows()
import pandas as pd
tab = pd.DataFrame(rows)

# fill7 queued configs (fixed wd only? inspect keys)
f7 = json.load(open('/tmp/wd_queue/fill7_configs.json'))
print('fill7 entries:', len(f7))
if f7:
    print('sample keys:', sorted(f7[0].keys()))
    f7df = pd.DataFrame(f7)
    print(f7df.head(3).to_string())
    print('---')

for (setting, phase), g in tab.groupby(['setting', 'phase'], sort=True):
    fixed = g[g['wd_sched'] == 'fixed'].sort_values('lambda0')
    fc = fixed['budget_c'].round(2).tolist()
    fl = fixed['lambda0'].round(6).tolist()
    print(f'== {setting} / {phase}  (anchor=lambda_ref)')
    print(f'   fixed C ladder: {[f"{c:.2f}" for c in fc]}')
    print(f'   fixed lambda:   {[f"{l:.3g}" for l in fl]}')
    dyn = g[g['wd_sched'] != 'fixed']
    for shape, h in dyn.groupby('wd_sched'):
        cs = sorted(h['budget_c'].round(2).unique())
        missing = [c for c in cs if c not in fc]
        print(f'   {shape:10s} C={[f"{c:.2f}" for c in cs]}'
              + (f'  MISSING={[f"{c:.2f}" for c in missing]}' if missing else ''))
    # fill7 queued fixed runs for this setting/phase (wd values not yet in CSV)
    if not f7df.empty and 'model' in f7df.columns:
        sub = f7df[(f7df['model'] == setting.split('/')[0])
                   & (f7df['dataset'] == setting.split('/')[1])]
        if not sub.empty:
            have = set(fl)
            new = sorted(set(sub['wd'].astype(float).round(6)) - have)
            print(f'   fill7 queued fixed lambda (not yet in CSV): '
                  f'{[f"{l:.3g}" for l in new]}')
