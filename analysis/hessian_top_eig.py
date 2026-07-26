"""
Top Hessian eigenvalue of the training loss, by power iteration.

E3 fits the divergence boundary to 1/eta_max = lambda + L/2 and reads off an
effective smoothness from the intercept. That is only meaningful if it can be
checked against a smoothness measured some other way, which is what this does.

The comparison is deliberately loose: L in the analysis is a global constant
while the measured quantity is the local curvature along the trajectory, so the
claim under test is that they agree in order of magnitude and move the same way
with the weight decay, not that they are equal.

    python -m analysis.hessian_top_eig --epochs 15 --gpu 1
"""
import argparse
import csv
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.nips26_lib import TABLE_DIR  # noqa: E402
from wd_core.data import get_cifar100_loaders  # noqa: E402
from wd_core.models import get_model  # noqa: E402
from wd_core.utils import set_seed, train_epoch  # noqa: E402


def top_eigenvalue(model, loader, device, n_batches=4, iters=25, tol=1e-4):
    """
    Largest eigenvalue of the loss Hessian by power iteration on
    Hessian-vector products, averaged over a few fixed minibatches.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    params = [p for p in model.parameters() if p.requires_grad]

    batches = []
    for i, (x, y) in enumerate(loader):
        if i >= n_batches:
            break
        batches.append((x.to(device), y.to(device)))

    v = [torch.randn_like(p) for p in params]
    norm = torch.sqrt(sum((vi * vi).sum() for vi in v))
    v = [vi / norm for vi in v]

    eig = None
    for _ in range(iters):
        hv = [torch.zeros_like(p) for p in params]
        for x, y in batches:
            model.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            grads = torch.autograd.grad(loss, params, create_graph=True)
            dot = sum((g * vi).sum() for g, vi in zip(grads, v))
            parts = torch.autograd.grad(dot, params, retain_graph=False)
            for acc, part in zip(hv, parts):
                acc += part.detach() / len(batches)

        new_eig = float(sum((h * vi).sum() for h, vi in zip(hv, v)))
        norm = torch.sqrt(sum((h * h).sum() for h in hv))
        if float(norm) == 0.0:
            break
        v = [h / norm for h in hv]
        if eig is not None and abs(new_eig - eig) / max(abs(new_eig), 1e-12) < tol:
            eig = new_eig
            break
        eig = new_eig

    model.zero_grad(set_to_none=True)
    return eig


def main():
    parser = argparse.ArgumentParser(description='Top Hessian eigenvalue during training')
    parser.add_argument('--model', default='resnet18')
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--wd', type=float, nargs='+', default=[0.0, 1e-3, 1e-2])
    parser.add_argument('--momentum', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--probe_every', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=6)
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    out_path = TABLE_DIR / 'e3_hessian.csv'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    for wd in args.wd:
        set_seed(args.seed)
        train_loader, _ = get_cifar100_loaders(
            batch_size=args.batch_size, num_workers=args.num_workers,
            data_dir=str(Path(__file__).resolve().parent.parent / 'data'),
        )
        # Curvature is measured without data augmentation so that repeated
        # probes see the same objective.
        probe_loader, _ = get_cifar100_loaders(
            batch_size=args.batch_size, num_workers=2, augment=False,
            data_dir=str(Path(__file__).resolve().parent.parent / 'data'),
        )
        model = get_model(args.model, num_classes=100).to(device)
        optimizer = optim.SGD(model.parameters(), lr=args.lr,
                              momentum=args.momentum, weight_decay=wd)

        eig = top_eigenvalue(model, probe_loader, device)
        print(f"wd={wd:g} epoch=0 top_eig={eig:.1f}  (2/L = {2 / eig:.4f})")
        rows.append(dict(wd=wd, epoch=0, top_eig=eig, two_over_L=2 / eig,
                         lr=args.lr, momentum=args.momentum, model=args.model))

        for epoch in range(args.epochs):
            train_epoch(model, train_loader, optimizer, None, device, use_amp=True)
            if (epoch + 1) % args.probe_every == 0 or epoch == args.epochs - 1:
                eig = top_eigenvalue(model, probe_loader, device)
                print(f"wd={wd:g} epoch={epoch+1} top_eig={eig:.1f}  "
                      f"(2/L = {2 / eig:.4f})")
                rows.append(dict(wd=wd, epoch=epoch + 1, top_eig=eig,
                                 two_over_L=2 / eig, lr=args.lr,
                                 momentum=args.momentum, model=args.model))

    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {out_path}")


if __name__ == '__main__':
    main()
