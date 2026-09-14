#!/usr/bin/env python
"""Fill7 vs coverage analysis, per (setting, phase)."""
import sys, json
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e12_full_table

ROOT = Path(__file__).resolve().parent.parent
E12 = ROOT / 'e12_multi_setting'
CSV = E12 / 'results' / 'e12_runs.csv'

rows = e12_full_table.build_rows()
tab = pd.DataFrame(rows)

f7 = pd.DataFrame(json.load(open('/tmp/wd_queue/fill7_configs.json')))
f7['wd'] = f7['wd'].astype(float)
f7['momentum'] = f7['momentum'].astype(float)
f7 = f7[f7['wd_sched'] == 'fixed']

# anchor per (setting, phase): fixed seed42 C==1.00 lambda
anchors = {}
for (setting, phase), g in tab.groupby(['setting', 'phase'], sort=True):
    fx = g[g['wd_sched'] == 'fixed']
    one = fx[fx['budget_c'].round(2) == 1.00]
    if one.empty:
        # fallback: max acc fixed
        one = fx.loc[fx['best_acc'].idxmax()]
    anchors[(setting, phase)] = float(one['lambda0'].iloc[0])

# CSV already-done fill7 runs: exp == e12_fill
csv_df = pd.read_csv(CSV)
if 'wd_mode' not in csv_df.columns:
    csv_df['wd_mode'] = 'coupled'
done_f7 = csv_df[(csv_df['exp'] == 'e12_fill') & (csv_df['wd_mode'].fillna('coupled') == 'coupled')
                 & (csv_df['seed'] == 42)]
done_keys = set()
for _, r in done_f7.iterrows():
    done_keys.add((r['model'], r['dataset'], float(r['momentum']),
                   round(float(r['wd']), 6)))

print(f'fill7 total configs: {len(f7)}; done in CSV: {len(done_keys)}')
print()

all_ok = True
for (setting, phase), g in tab.groupby(['setting', 'phase'], sort=True):
    model, ds = setting.split('/')
    mom = 0.9 if phase == 'SGDM' else 0.0
    lam_ref = anchors[(setting, phase)]
    fixed_c = sorted(set(g[g['wd_sched'] == 'fixed']['budget_c'].round(2)))
    dyn = g[g['wd_sched'] != 'fixed']
    dyn_c = sorted(set(dyn['budget_c'].round(2)))
    f7sub = f7[(f7['model'] == model) & (f7['dataset'] == ds)
               & (f7['momentum'] == mom)]
    f7c = sorted((f7sub['wd'] / lam_ref).round(2))
    remaining = sorted(f7sub[~f7sub.apply(
        lambda r: (r['model'], r['dataset'], r['momentum'],
                   round(r['wd'], 6)) in done_keys, axis=1)]['wd'].tolist())
    post_fixed = sorted(set(fixed_c) | set(f7c))
    missing = [c for c in dyn_c if c not in post_fixed]
    status = 'COVERED' if not missing else f'MISSING={[f"{c:.2f}" for c in missing]}'
    if missing:
        all_ok = False
    print(f'== {setting} / {phase}  lam_ref={lam_ref:.4g}')
    print(f'   fixed now:  {[f"{c:.2f}" for c in fixed_c]}')
    print(f'   fill7 adds: {[f"{c:.2f}" for c in f7c]} (n={len(f7c)})')
    print(f'   fill7 remaining wd: {[f"{w:.4g}" for w in remaining]} (n={len(remaining)})')
    print(f'   dyn C: {[f"{c:.2f}" for c in dyn_c]}')
    print(f'   -> {status}')
    print()
print('ALL COVERED' if all_ok else 'SOME DYNAMIC C NOT ON FIXED LADDER')
