#!/usr/bin/env python
"""
E12 multi-seed driver.

Runs the peak configs listed in e12_multi_setting/ms_configs.json across
--seeds, reusing the rebuttal runner's make_cfg / run_grid machinery (same
RUN_KEY dedupe, same CSV). seed=42 rows dedup against the grid runs already
in the CSV, so only the extra seeds cost GPU time.

ms_configs.json: list of {"model", "dataset", "momentum", "wd_sched", "wd"}.
  - wd_sched == "fixed"   -> lr_mode='cosine' (the fixed-WD oracle)
  - otherwise             -> lr_mode='cos_shape' (dynamic WD under cosine LR)

Usage (from repo root):
  python e12_multi_setting/run_e12_ms.py --seeds 42,123,2024 \
      --gpus 3 --workers_per_gpu 2 \
      --csv e12_multi_setting/results/e12_runs.csv
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rebuttal.run_nips26_wd_sched import make_cfg, run_grid  # noqa: E402
from wd_core.gpu_scheduler import parse_gpu_ids  # noqa: E402
from wd_core.logger import get_logger  # noqa: E402


DEFAULT_CONFIGS = Path(__file__).resolve().parent / 'ms_configs.json'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', type=str, default=str(DEFAULT_CONFIGS))
    parser.add_argument('--seeds', type=str, default='42,123,2024')
    parser.add_argument('--gpus', type=str, default='1,2')
    parser.add_argument('--workers_per_gpu', type=int, default=3)
    parser.add_argument('--csv', type=str,
                        default='e12_multi_setting/results/e12_runs.csv')
    args = parser.parse_args()

    configs_path = Path(args.configs)
    if not configs_path.exists():
        raise SystemExit(f'config file not found: {configs_path}')
    with open(configs_path) as f:
        raw = json.load(f)

    seeds = [int(s) for s in args.seeds.split(',') if s.strip()]
    cfgs = []
    for entry in raw:
        model = entry['model']
        dataset = entry['dataset']
        momentum = float(entry['momentum'])
        wd_sched = entry['wd_sched']
        lam0 = float(entry['wd'])
        lr_mode = 'cosine' if wd_sched == 'fixed' else 'cos_shape'
        exp = entry.get('exp', 'e12_ms')
        for seed in seeds:
            cfg = make_cfg(wd_sched, lam0, momentum, lr=0.1, epochs=100,
                           batch_size=128, model=model, seed=seed,
                           dataset=dataset, lr_mode=lr_mode, exp=exp)
            cfgs.append(cfg)

    logger = get_logger(f'e12_ms_{len(raw)}cfg')
    gpu_ids = parse_gpu_ids(args.gpus)
    logger.info(f'e12_ms: {len(raw)} configs x {len(seeds)} seeds = '
                f'{len(cfgs)} runs, gpus={gpu_ids}, '
                f'workers_per_gpu={args.workers_per_gpu}')
    run_grid(cfgs, gpu_ids, args.workers_per_gpu, args.csv, logger,
             label='e12_ms')


if __name__ == '__main__':
    main()
