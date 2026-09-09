#!/usr/bin/env python
"""One-off wall_time probe for a new (model, dataset): runs one cfg and
appends the row to the E12 CSV (deduped by the later sweeps via RUN_KEY)."""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rebuttal.run_nips26_wd_sched import make_cfg, run_one, append_row  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--momentum', type=float, required=True)
    parser.add_argument('--lam', type=float, required=True)
    parser.add_argument('--csv', default='e12_multi_setting/results/e12_runs.csv')
    parser.add_argument('--exp', default='e12_probe')
    args = parser.parse_args()

    cfg = make_cfg('fixed', args.lam, args.momentum, lr=0.1, epochs=100,
                   batch_size=128, model=args.model, dataset=args.dataset,
                   lr_mode='cosine', exp=args.exp)
    t0 = time.time()
    row = run_one(cfg)
    dt = time.time() - t0
    append_row(args.csv, row)
    print(f'PROBE_DONE wall_time={dt:.0f}s best={row["best_test_acc"]:.2f}')


if __name__ == '__main__':
    main()
