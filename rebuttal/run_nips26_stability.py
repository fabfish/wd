"""
E7: does the stability mechanism itself show up in a deep network?

Three modes.

stability   Train pairs of networks on datasets differing in exactly one
            example, with identical initialization and batch ordering, and
            track ||theta_t - theta'_t||. This is the definition behind uniform
            stability rather than a proxy for it. Without weight decay the
            convex analysis says the divergence grows with t; with weight decay
            the contraction factor (1 - eta*lambda) per step says it saturates
            at a level set by 1/lambda.

equilibrium Log per-layer weight norms during ordinary training, to check the
            rotational-equilibrium prediction ||w|| ~ sqrt(eta/lambda) and to
            show that it is reached well before the end of training and so
            carries no information about the training horizon.

bn          Repeat the eta-lambda grid on an MLP with and without
            normalization. The equilibrium mechanism needs scale invariance;
            our stability argument does not. If the coupling survives without
            normalization, the two accounts are not the same claim.

    python rebuttal/run_nips26_stability.py --mode stability --gpu 1
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wd_core.data import get_cifar100_loaders, get_cifar100_neighbour_loaders  # noqa: E402
from wd_core.models import get_model  # noqa: E402
from wd_core.utils import (  # noqa: E402
    evaluate, make_grad_scaler, set_seed, train_model_ext, weight_norm_probe,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = str(ROOT / 'data')
OUT_DIR = ROOT / 'rebuttal' / 'nips_rebuttal' / '_data'


def param_distance(model_a, model_b):
    with torch.no_grad():
        total = sum(float(((pa - pb) ** 2).sum())
                    for pa, pb in zip(model_a.parameters(), model_b.parameters()))
    return float(np.sqrt(total))


def param_norm(model):
    with torch.no_grad():
        return float(np.sqrt(sum(float((p ** 2).sum()) for p in model.parameters())))


# --------------------------------------------------------------------------
# E7a
# --------------------------------------------------------------------------

def run_stability(args, device):
    """Train S and S' in lockstep and record how far the iterates drift apart."""
    rows = []
    for wd in args.wd:
        # Both models start from the same initialization and see the same batch
        # positions in the same order; the only difference is one example.
        loader_s, loader_sp, test_loader = get_cifar100_neighbour_loaders(
            batch_size=args.batch_size, num_workers=0, data_dir=DATA_DIR,
            subset_size=args.subset, replace_index=args.replace_index,
            seed=args.seed, augment=False,
        )

        set_seed(args.seed)
        model_a = get_model(args.model, num_classes=100).to(device)
        set_seed(args.seed)
        model_b = get_model(args.model, num_classes=100).to(device)
        assert param_distance(model_a, model_b) == 0.0, "pair did not start identical"

        opt_a = optim.SGD(model_a.parameters(), lr=args.lr,
                          momentum=args.momentum, weight_decay=wd)
        opt_b = optim.SGD(model_b.parameters(), lr=args.lr,
                          momentum=args.momentum, weight_decay=wd)
        criterion = nn.CrossEntropyLoss()

        print(f"\n=== lambda = {wd:g} | eta = {args.lr:g} | n = {args.subset} "
              f"| {args.epochs} epochs ===")
        step = 0
        for epoch in range(args.epochs):
            model_a.train()
            model_b.train()
            for (xa, ya), (xb, yb) in zip(loader_s, loader_sp):
                xa, ya = xa.to(device), ya.to(device)
                xb, yb = xb.to(device), yb.to(device)
                for model, opt, x, y in ((model_a, opt_a, xa, ya),
                                         (model_b, opt_b, xb, yb)):
                    opt.zero_grad(set_to_none=True)
                    criterion(model(x), y).backward()
                    opt.step()
                step += 1

            dist = param_distance(model_a, model_b)
            acc_a, loss_a = evaluate(model_a, test_loader, device)
            acc_b, loss_b = evaluate(model_b, test_loader, device)
            rows.append(dict(wd=wd, lr=args.lr, epoch=epoch + 1, step=step,
                             param_distance=dist,
                             relative_distance=dist / max(param_norm(model_a), 1e-12),
                             test_loss_gap=abs(loss_a - loss_b),
                             acc_a=acc_a, acc_b=acc_b,
                             subset=args.subset, momentum=args.momentum))
            print(f"  epoch {epoch+1:3d} | step {step:5d} | "
                  f"||theta - theta'|| = {dist:.4f} | "
                  f"relative {dist / max(param_norm(model_a), 1e-12):.2e} | "
                  f"|loss gap| = {abs(loss_a - loss_b):.4f}")

    out = OUT_DIR / 'e7a_stability.csv'
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {out}")


# --------------------------------------------------------------------------
# E7b
# --------------------------------------------------------------------------

def run_equilibrium(args, device):
    """Log weight norms during ordinary training for the Kosson comparison."""
    histories = []
    for wd in args.wd:
        for lr in args.eq_lr:
            set_seed(args.seed)
            train_loader, test_loader = get_cifar100_loaders(
                batch_size=args.batch_size, num_workers=4, data_dir=DATA_DIR)
            model = get_model(args.model, num_classes=100).to(device)
            optimizer = optim.SGD(model.parameters(), lr=lr,
                                  momentum=args.momentum, weight_decay=wd)
            scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

            print(f"\n=== equilibrium probe: eta={lr:g}, lambda={wd:g}, "
                  f"T={args.epochs} ===")
            result = train_model_ext(
                model, train_loader, test_loader, optimizer, scheduler, device,
                epochs=args.epochs, use_amp=True, log_interval=max(args.epochs // 4, 1),
                probe_fn=lambda epoch, m: weight_norm_probe(m),
                tag=f"[eq|lr={lr:g}|wd={wd:g}] ",
            )
            histories.append(dict(lr=lr, wd=wd, epochs=args.epochs,
                                  momentum=args.momentum,
                                  predicted_ratio=float(np.sqrt(lr / wd)) if wd > 0 else None,
                                  history=result.pop('history'), summary=result))

    out = OUT_DIR / 'e7b_equilibrium.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(histories, indent=1))
    print(f"\nwrote {out}")


# --------------------------------------------------------------------------
# E7c
# --------------------------------------------------------------------------

def run_bn(args, device):
    """The same eta-lambda grid with and without normalization, on an MLP."""
    from mlp_wd.mlp_core.runner import run_single_experiment  # noqa: E402

    lrs = [0.01, 0.03, 0.1, 0.3]
    wds = [1e-4, 5e-4, 2e-3, 1e-2, 5e-2]
    rows = []
    total = len(lrs) * len(wds) * 2
    i = 0
    for use_bn in (0, 1):
        for lr in lrs:
            for wd in wds:
                i += 1
                start = time.time()
                res = run_single_experiment(
                    method='SGDM+WD', batch_size=128, lr=lr, wd=wd,
                    momentum=0.9, epochs=args.epochs, seed=args.seed,
                    dataset='cifar10', use_bn=bool(use_bn), num_workers=4,
                )
                res['use_bn'] = use_bn
                rows.append(res)
                print(f"[{i}/{total}] bn={use_bn} lr={lr:g} wd={wd:g} -> "
                      f"{res['best_test_acc']:.2f}% ({time.time() - start:.0f}s)")

    out = OUT_DIR / 'e7c_bn_ablation.csv'
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {out}")


def main():
    p = argparse.ArgumentParser(description='E7 stability and equilibrium probes')
    p.add_argument('--mode', required=True, choices=['stability', 'equilibrium', 'bn'])
    p.add_argument('--model', default='resnet18')
    p.add_argument('--gpu', type=int, default=None)
    p.add_argument('--lr', type=float, default=0.05)
    p.add_argument('--wd', type=float, nargs='+', default=[0.0, 1e-3, 1e-2])
    p.add_argument('--eq_lr', type=float, nargs='+', default=[0.02, 0.1])
    p.add_argument('--momentum', type=float, default=0.0)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--subset', type=int, default=10000,
                   help='training set size for the paired runs')
    p.add_argument('--replace_index', type=int, default=0)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    if args.gpu is not None:
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    {'stability': run_stability,
     'equilibrium': run_equilibrium,
     'bn': run_bn}[args.mode](args, device)


if __name__ == '__main__':
    main()
