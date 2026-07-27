"""E5c: C stability + sensitivity on MNIST + small MLP (Fig.1 protocol).

Three phases, one CSV:

  A. Fig.1-lite — SGD/SGDM vs +WD LR ordering (eta* shift sanity check)
  B. C probe    — eta x lambda grids under SGD and SGDM; fit C = lambda* * S
  C. Wrong-C    — lambda = (f * C_sgdm) / S for f in {0.1, 1/3, 1, 3, 10}

Phase C is built only after B finishes, using the geo-mean C under SGDM and
exact sum_lr(..., n=60000). Resume-safe via mlp_wd.mlp_core.grid.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.nips26_lib import sum_lr  # noqa: E402
from mlp_wd.mlp_core.gpu_scheduler import parse_gpu_ids  # noqa: E402
from mlp_wd.mlp_core.grid import run_grid  # noqa: E402
from mlp_wd.scripts.run_exp1_lr_ordering import (  # noqa: E402
    DEFAULT_WD_A,
    DEFAULT_WD_B,
    GROUP_A_LRS,
    GROUP_B_LRS,
    build_grid as build_exp1_grid,
)

MNIST_N = 60000
C_PROBE_LRS = [0.05, 0.1, 0.2]
C_PROBE_WDS = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
SENS_FACTORS = [0.1, 1.0 / 3.0, 1.0, 3.0, 10.0]
SENS_LRS = [0.05, 0.1]


def _parabola_trough(wd, loss):
    """Min of a parabola through the argmin and its two neighbours in log wd."""
    wd, loss = np.asarray(wd, float), np.asarray(loss, float)
    i = int(np.argmin(loss))
    if i == 0 or i == len(wd) - 1:
        return float(wd[i])
    x = np.log(wd[i - 1:i + 2])
    y = loss[i - 1:i + 2]
    denom = (y[0] - 2 * y[1] + y[2])
    if abs(denom) < 1e-12:
        return float(wd[i])
    delta = 0.5 * (y[0] - y[2]) / denom
    delta = float(np.clip(delta, -1.0, 1.0))
    return float(np.exp(x[1] + delta * (x[1] - x[0])))


def build_phase_a(epochs, seed, batch_size):
    rows = build_exp1_grid(
        epochs, seed, batch_size, GROUP_A_LRS, GROUP_B_LRS,
        DEFAULT_WD_A, DEFAULT_WD_B,
    )
    for r in rows:
        r["run_tag"] = f"e5cA_{r['run_tag']}"
        r["phase"] = "A"
    return rows


def build_phase_b(epochs, seed, batch_size):
    rows = []
    for mom, method_base in ((0.0, "SGD"), (0.9, "SGDM")):
        for lr in C_PROBE_LRS:
            for wd in C_PROBE_WDS:
                method = f"{method_base}+WD"
                rows.append({
                    "method": method,
                    "batch_size": batch_size,
                    "lr": lr,
                    "wd": wd,
                    "momentum": mom,
                    "epochs": epochs,
                    "seed": seed,
                    "run_tag": f"e5cB_{method}_lr{lr}_wd{wd}",
                    "phase": "B",
                })
    return rows


def fit_C_from_phase_b(csv_path, epochs, batch_size):
    """Per (momentum, lr) fit lambda* from best_test_loss; return C table + geo means."""
    df = pd.read_csv(csv_path)
    # Restrict to the C-probe lambda ladder. Phase-C factor cells share
    # (method, lr) with phase B and must not pollute the fit.
    wd_probe = {float(w) for w in C_PROBE_WDS}
    mask = (
        (df["epochs"].astype(int) == epochs)
        & (df["batch_size"].astype(int) == batch_size)
        & (df["method"].isin(["SGD+WD", "SGDM+WD"]))
        & (df["lr"].isin(C_PROBE_LRS))
        & df["wd"].astype(float).map(lambda w: any(np.isclose(w, t, rtol=0, atol=1e-12) for t in wd_probe))
    )
    if "run_tag" in df.columns:
        tagged = df["run_tag"].astype(str).str.startswith("e5cB_")
        if tagged.any():
            mask = mask & tagged
    d = df.loc[mask].copy()
    if d.empty:
        raise RuntimeError(f"no phase-B rows in {csv_path}")

    records = []
    for (mom, lr), g in d.groupby(["momentum", "lr"]):
        g = g.sort_values("wd")
        if len(g) < 3:
            continue
        loss = g["best_test_loss"].astype(float).values
        wds = g["wd"].astype(float).values
        wd_arg = float(wds[int(np.argmin(loss))])
        wd_star = _parabola_trough(wds, loss)
        S = float(sum_lr(float(lr), epochs, batch_size, "cosine", n=MNIST_N))
        wds_sorted = np.sort(wds)
        interior = bool(wds_sorted[0] < wd_arg < wds_sorted[-1])
        records.append({
            "momentum": float(mom),
            "lr": float(lr),
            "wd_argmax": wd_arg,
            "wd_interp": wd_star,
            "loss_min": float(np.min(loss)),
            "S": S,
            "C": wd_star * S,
            "interior": interior,
            "n_points": int(len(g)),
        })
    opt = pd.DataFrame(records)
    if opt.empty:
        raise RuntimeError("phase-B fit produced no optima")

    def _geo(series):
        s = series[series > 0]
        return float(np.exp(np.mean(np.log(s)))) if len(s) else float("nan")

    c_sgd = _geo(opt.loc[np.isclose(opt["momentum"], 0.0) & opt["interior"], "C"])
    c_sgdm = _geo(opt.loc[np.isclose(opt["momentum"], 0.9) & opt["interior"], "C"])
    # Fall back to all points if no interior optima
    if not math.isfinite(c_sgd) or c_sgd <= 0:
        c_sgd = _geo(opt.loc[np.isclose(opt["momentum"], 0.0), "C"])
    if not math.isfinite(c_sgdm) or c_sgdm <= 0:
        c_sgdm = _geo(opt.loc[np.isclose(opt["momentum"], 0.9), "C"])
    return opt, c_sgd, c_sgdm


def build_phase_c(epochs, seed, batch_size, C_sgdm):
    rows = []
    for lr in SENS_LRS:
        S = float(sum_lr(lr, epochs, batch_size, "cosine", n=MNIST_N))
        for f in SENS_FACTORS:
            wd = float(f"{(f * C_sgdm) / S:.6g}")
            rows.append({
                "method": "SGDM+WD",
                "batch_size": batch_size,
                "lr": lr,
                "wd": wd,
                "momentum": 0.9,
                "epochs": epochs,
                "seed": seed,
                "run_tag": f"e5cC_SGDMWD_lr{lr}_f{f:g}_wd{wd}",
                "phase": "C",
                "factor": f,
            })
    return rows


def main():
    ap = argparse.ArgumentParser(description="E5c MNIST-MLP C sensitivity")
    ap.add_argument("--dataset", default="mnist", choices=["mnist", "cifar10"])
    ap.add_argument("--num_layers", type=int, default=3)
    ap.add_argument("--hidden_dim", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gpus", type=str, default="0,1,2,3")
    ap.add_argument("--workers_per_gpu", type=int, default=12)
    ap.add_argument("--loader_workers", type=int, default=0)
    ap.add_argument("--output", type=str,
                    default="mlp_wd/outputs/results/e5c_mnist.csv")
    ap.add_argument("--history_dir", type=str,
                    default="mlp_wd/outputs/history/e5c_mnist")
    ap.add_argument("--optima_csv", type=str,
                    default="rebuttal/nips_rebuttal/_data/e5c_optima.csv")
    ap.add_argument("--phases", type=str, default="A,B,C",
                    help="comma list of phases to run, e.g. A,B or C")
    ap.add_argument("--log_every", type=int, default=0)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    gpu_ids = parse_gpu_ids(args.gpus) if args.gpus and args.gpus != "all" else None
    phases = {p.strip().upper() for p in args.phases.split(",") if p.strip()}

    common = dict(
        output_file=args.output,
        history_dir=args.history_dir,
        dataset=args.dataset,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        gpu_ids=gpu_ids,
        workers_per_gpu=args.workers_per_gpu,
        log_every=args.log_every,
        loader_workers=args.loader_workers,
    )

    if "A" in phases:
        rows_a = build_phase_a(args.epochs, args.seed, args.batch_size)
        print(f"[e5c] phase A: {len(rows_a)} Fig.1-lite runs")
        if args.dry_run:
            print(rows_a[:2])
        else:
            run_grid(rows_a, **common)

    if "B" in phases:
        rows_b = build_phase_b(args.epochs, args.seed, args.batch_size)
        print(f"[e5c] phase B: {len(rows_b)} C-probe runs")
        if args.dry_run:
            print(rows_b[:2])
        else:
            run_grid(rows_b, **common)

    C_sgdm = None
    if "C" in phases:
        if args.dry_run:
            C_sgdm = 1.0
            print("[e5c] dry_run: using C_sgdm=1.0 for phase C plan")
        else:
            opt, c_sgd, C_sgdm = fit_C_from_phase_b(
                args.output, args.epochs, args.batch_size,
            )
            Path(args.optima_csv).parent.mkdir(parents=True, exist_ok=True)
            opt.to_csv(args.optima_csv, index=False)
            print(f"[e5c] fitted C_sgd={c_sgd:.4g}  C_sgdm={C_sgdm:.4g} "
                  f"(wrote {args.optima_csv})")
            if not math.isfinite(C_sgdm) or C_sgdm <= 0:
                raise RuntimeError(f"invalid C_sgdm={C_sgdm}")

        rows_c = build_phase_c(args.epochs, args.seed, args.batch_size, C_sgdm)
        print(f"[e5c] phase C: {len(rows_c)} wrong-C runs (C_sgdm={C_sgdm:g})")
        for r in rows_c:
            print(f"  lr={r['lr']} f={r['factor']:g} wd={r['wd']}")
        if not args.dry_run:
            run_grid(rows_c, **common)

    print("[e5c] done")


if __name__ == "__main__":
    main()
