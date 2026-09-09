#!/usr/bin/env python
"""
Build e12_multi_setting/anchors.json from the e12_fixed sweep rows.

For every (model, dataset, phase) present in the E12 CSV, the fixed-lambda
oracle is the e12_fixed row with the highest best_test_acc (diverged rows
excluded). Its lambda becomes that setting's local anchor: the local budget
unit C = lambda_ref * sum_t eta_t used by --sweep e12_matched.

Usage (from repo root):
  python e12_multi_setting/make_anchors.py \
      e12_multi_setting/results/e12_runs.csv \
      e12_multi_setting/anchors.json
"""
import argparse
import csv
import json
from pathlib import Path


def phase_of(momentum):
    return 'sgd' if float(momentum) == 0.0 else 'sgdm'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('csv_path')
    parser.add_argument('out_path', nargs='?',
                        default='e12_multi_setting/anchors.json')
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        raise SystemExit(f'CSV not found: {csv_path}. Run e12_fixed first.')

    best = {}  # (model, dataset, phase) -> (acc, lambda)
    with open(csv_path, newline='') as f:
        for row in csv.DictReader(f):
            if str(row.get('exp', '')) != 'e12_fixed':
                continue
            if str(row.get('diverged', '')).lower() in ('true', '1'):
                continue
            try:
                acc = float(row['best_test_acc'])
                lam = float(row['wd'])
            except (KeyError, ValueError):
                continue
            key = (row['model'], row['dataset'], phase_of(row['momentum']))
            if key not in best or acc > best[key][0]:
                best[key] = (acc, lam)

    if not best:
        raise SystemExit('No e12_fixed rows found in CSV.')

    anchors = {f'{m}|{d}|{p}': lam for (m, d, p), (acc, lam) in best.items()}
    out = Path(args.out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w') as f:
        json.dump(anchors, f, indent=2, sort_keys=True)
    print(f'Wrote {len(anchors)} anchors to {out}:')
    for key in sorted(anchors):
        print(f'  {key} -> {anchors[key]:.6g}')


if __name__ == '__main__':
    main()
