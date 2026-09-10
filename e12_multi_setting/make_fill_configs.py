#!/usr/bin/env python
"""
Generate e12_multi_setting/fill_configs.json: the gap-filling E12 runs.

Gaps found after the main grid (all cosine LR, B=128, eta0=0.1, T=100):
 1. R18/C100 SGD: linear_down matched ladder is MISSING entirely
    (E9 'matched' only ran SGDM; E11 raise arms have no down shapes).
 2. VGG/C100 SGD: collapse boundary between 1C and 2C is unprobed
    (iso_product 75.89@1C -> 47.34@2C); add 1.2C/1.5C probes.
 3. VGG/C100 SGDM: ladder density at 1.2C (peak sits at 1.0-1.04C).
 4. R50/C100 SGDM: fixed-lambda curve is sharp between 6e-4 and 1e-3
    (9.62e-4 = 78.20 vs 6e-4 = 76.07, 1e-3 = 75.66); densify 8e-4/9e-4/1.1e-3.
 5. MLP/C10 SGD: sub-1C rungs missing (fixed wins at 1C; test whether
    less budget helps the dynamic shapes too).

lambda0 values are inverted from the budget via the runner's own
solve_lambda0_for_budget so the realized budgets match the ladder exactly.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rebuttal.run_nips26_wd_sched import solve_lambda0_for_budget, \
    contraction_sum  # noqa: E402
from wd_core.data import DATASET_TRAIN_N  # noqa: E402

LR0, EPOCHS, BS = 0.1, 100, 128
# Setting-local anchors (fixed oracle per phase, seed 42).
ANCHORS = {
    ('resnet18', 'cifar100', 'sgd'): 5e-3,
    ('vgg16', 'cifar100', 'sgd'): 1e-2,
    ('vgg16', 'cifar100', 'sgdm'): 9.62e-4,
    ('resnet50', 'cifar100', 'sgdm'): 9.62e-4,
    ('mlp', 'cifar10', 'sgd'): 1e-2,
}
MOMENTUM = {'sgd': 0.0, 'sgdm': 0.9}

# (model, dataset, phase, wd_sched, factors) — fixed uses lambda0 directly.
PLAN = [
    ('resnet18', 'cifar100', 'sgd', 'linear', [1 / 3, 1, 1.5, 2, 3, 4, 6]),
    ('vgg16', 'cifar100', 'sgd', 'iso_product', [1.2, 1.5]),
    ('vgg16', 'cifar100', 'sgd', 'linear_up', [1.2, 1.5]),
    ('vgg16', 'cifar100', 'sgdm', 'iso_product', [1.2]),
    ('vgg16', 'cifar100', 'sgdm', 'linear_up', [1.2]),
    ('mlp', 'cifar10', 'sgd', 'linear_up', [1 / 3, 0.5]),
    ('mlp', 'cifar10', 'sgd', 'linear', [1 / 3, 0.5]),
    ('mlp', 'cifar10', 'sgd', 'iso_product', [1 / 3, 0.5]),
]
# (model, dataset, phase, lambda0) fixed-densify entries.
FIXED_EXTRA = [
    ('resnet50', 'cifar100', 'sgdm', 8e-4),
    ('resnet50', 'cifar100', 'sgdm', 9e-4),
    ('resnet50', 'cifar100', 'sgdm', 1.1e-3),
]


def main():
    entries = []
    for model, dataset, phase, sched, factors in PLAN:
        lam_ref = ANCHORS[(model, dataset, phase)]
        n = DATASET_TRAIN_N[dataset]
        anchor = contraction_sum(LR0, lam_ref, EPOCHS, BS, 'fixed', n=n)
        for f in factors:
            budget = f * anchor
            lam0 = solve_lambda0_for_budget(budget, LR0, EPOCHS, BS, sched, n=n)
            entries.append({
                'model': model, 'dataset': dataset,
                'momentum': MOMENTUM[phase],
                'wd_sched': sched, 'wd': lam0, 'exp': 'e12_fill',
            })
    for model, dataset, phase, lam0 in FIXED_EXTRA:
        entries.append({
            'model': model, 'dataset': dataset,
            'momentum': MOMENTUM[phase],
            'wd_sched': 'fixed', 'wd': lam0, 'exp': 'e12_fill',
        })

    out = Path(__file__).resolve().parent / 'fill_configs.json'
    with open(out, 'w') as f:
        json.dump(entries, f, indent=2)
    print(f'wrote {len(entries)} fill configs to {out}')
    for e in entries:
        print(f"  {e['model']}/{e['dataset']} {e['momentum']:g} "
              f"{e['wd_sched']:>12s} wd={e['wd']:.4g}")


if __name__ == '__main__':
    main()
