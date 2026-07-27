"""Three-panel E5c figure + token CSV for xkCF Q5.

Left:   Fig.1-style LR ordering on MNIST-MLP (SGD/SGDM ± WD).
Middle: fitted C under SGD vs SGDM, with E5a CIFAR geo-C reference.
Right:  cost of mis-specifying C (Delta best_test_loss vs factor).
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.nips26_lib import sum_lr  # noqa: E402
from mlp_wd.scripts.run_e5c_c_sensitivity import (  # noqa: E402
    C_PROBE_LRS,
    C_PROBE_WDS,
    MNIST_N,
    SENS_FACTORS,
    SENS_LRS,
    _parabola_trough,
    fit_C_from_phase_b,
)

# Wave-0 E5a geo-means by architecture (CIFAR-100 / SGDM)
E5A_C = {"ResNet-18": 1.42, "ResNet-50": 1.42, "VGG-16": 1.72, "E5a geo": 1.48}


def _geo(vals):
    vals = np.asarray([v for v in vals if v is not None and np.isfinite(v) and v > 0], float)
    return float(np.exp(np.mean(np.log(vals)))) if len(vals) else float("nan")


def panel_fig1(ax, df):
    styles = [
        ("SGD", 0.0, "C0", "-"),
        ("SGD+WD", 0.0, "C1", "-"),
        ("SGDM", 0.9, "C2", "-"),
        ("SGDM+WD", 0.9, "C3", "-"),
    ]
    # Prefer phase-A tags when present; otherwise use method+momentum
    for method, mom, color, ls in styles:
        sub = df[
            (df["method"] == method)
            & np.isclose(df["momentum"].astype(float), mom)
        ].copy()
        if "run_tag" in df.columns:
            tagged = sub[sub["run_tag"].astype(str).str.startswith("e5cA_")]
            if not tagged.empty:
                sub = tagged
        if sub.empty:
            continue
        # For +WD curves in Fig.1, Exp1 uses a single lambda; collapse duplicates by lr
        g = sub.groupby("lr", as_index=False)["best_test_acc"].max().sort_values("lr")
        ax.plot(g["lr"], g["best_test_acc"], marker="o", color=color, ls=ls,
                label=method, lw=2)
        i = g["best_test_acc"].idxmax()
        ax.scatter([g.loc[i, "lr"]], [g.loc[i, "best_test_acc"]],
                   marker="*", s=160, color=color, edgecolors="k", zorder=5)
    ax.set_xscale("log")
    ax.set_xlabel(r"$\eta$")
    ax.set_ylabel("Best test acc (%)")
    ax.set_title("A. Fig.1 protocol (MNIST MLP)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)


def panel_C_stability(ax, opt, c_sgd, c_sgdm):
    labels, vals, colors = [], [], []
    for name, c, col in (
        ("MNIST-MLP\nSGD", c_sgd, "C0"),
        ("MNIST-MLP\nSGDM", c_sgdm, "C2"),
    ):
        labels.append(name)
        vals.append(c)
        colors.append(col)
    for name, c in E5A_C.items():
        labels.append(name)
        vals.append(c)
        colors.append("0.6")
    xs = np.arange(len(labels))
    ax.bar(xs, vals, color=colors, edgecolor="k", linewidth=0.6)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel(r"fitted $C=\lambda^\star S$")
    ax.set_title("B. Is $C$ stable?")
    ax.axhline(1.0, color="k", ls=":", lw=1)
    ax.set_yscale("log")
    ax.grid(True, axis="y", alpha=0.3)
    # scatter individual (mom, lr) points for MNIST
    for _, r in opt.iterrows():
        x = 0 if np.isclose(r["momentum"], 0.0) else 1
        ax.scatter([x], [r["C"]], color="k", s=18, zorder=5, alpha=0.7)


def panel_sensitivity(ax, df, c_sgdm, epochs, batch_size):
    """Match phase-C cells by rebuilding planned (lr, factor) -> wd from C_sgdm."""
    rows = []
    probe = set(float(w) for w in C_PROBE_WDS) | {1e-3}  # ladder + Fig.1 fixed lambda
    for lr in SENS_LRS:
        S = float(sum_lr(lr, epochs, batch_size, "cosine", n=MNIST_N))
        for f in SENS_FACTORS:
            wd_target = float(f"{(f * c_sgdm) / S:.6g}")
            cand = df[
                (df["method"] == "SGDM+WD")
                & np.isclose(df["momentum"].astype(float), 0.9)
                & np.isclose(df["lr"].astype(float), lr)
            ].copy()
            if cand.empty:
                continue
            # Prefer exact planned wd; else nearest non-probe wd
            exact = cand[np.isclose(cand["wd"].astype(float), wd_target, rtol=0, atol=0)]
            if exact.empty:
                exact = cand[np.isclose(cand["wd"].astype(float), wd_target, rtol=1e-4, atol=1e-12)]
            if exact.empty:
                non_probe = cand[~cand["wd"].astype(float).map(
                    lambda w: any(np.isclose(w, p, rtol=0, atol=1e-12) for p in probe))]
                if non_probe.empty:
                    continue
                idx = (non_probe["wd"].astype(float) - wd_target).abs().idxmin()
                if abs(float(non_probe.loc[idx, "wd"]) - wd_target) / max(wd_target, 1e-12) > 0.05:
                    continue
                exact = non_probe.loc[[idx]]
            rows.append({
                "lr": lr,
                "factor": f,
                "loss": float(exact["best_test_loss"].astype(float).min()),
                "acc": float(exact["best_test_acc"].astype(float).max()),
                "wd": float(exact["wd"].astype(float).iloc[0]),
                "wd_target": wd_target,
            })
    sens = pd.DataFrame(rows)
    if sens.empty:
        ax.text(0.5, 0.5, "no phase-C rows", ha="center", transform=ax.transAxes)
        ax.set_title("C. Cost of wrong $C$")
        return sens

    # Delta loss relative to f=1 per lr
    for lr, g in sens.groupby("lr"):
        g = g.sort_values("factor")
        base = float(g.loc[np.isclose(g["factor"], 1.0), "loss"].iloc[0]) \
            if (np.isclose(g["factor"], 1.0)).any() else float(g["loss"].min())
        ax.plot(g["factor"], g["loss"] - base, "o-", lw=2, label=rf"$\eta$={lr:g}")
    ax.set_xscale("log")
    ax.axvline(1.0, color="k", ls=":", lw=1)
    ax.axhline(0.0, color="k", ls=":", lw=1)
    ax.set_xlabel(r"factor $f$ in $\lambda = f\,C / S$")
    ax.set_ylabel(r"$\Delta$ best test loss (vs $f=1$)")
    ax.set_title("C. Cost of wrong $C$ (SGDM)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    return sens


def compute_tokens(opt, c_sgd, c_sgdm, sens):
    tokens = {
        "E5C-C-SGD": f"{c_sgd:.3g}",
        "E5C-C-SGDM": f"{c_sgdm:.3g}",
        "E5C-FIG": "outputs/plots/nips26/e5c_mnist_mlp_C.png",
    }
    vals = [c for c in (c_sgd, c_sgdm) if math.isfinite(c) and c > 0]
    tokens["E5C-C-RATIO"] = f"{max(vals)/min(vals):.2f}x" if len(vals) == 2 else "PENDING"

    def _cost(target):
        if sens is None or sens.empty:
            return None
        near = sens[
            np.isclose(sens["factor"], target, rtol=0.05)
            | np.isclose(sens["factor"], 1.0 / target, rtol=0.05)
        ]
        if near.empty:
            return None
        # loss relative to f=1 within each lr, then take max abs
        costs = []
        for lr, g in sens.groupby("lr"):
            if not (np.isclose(g["factor"], 1.0)).any():
                continue
            base = float(g.loc[np.isclose(g["factor"], 1.0), "loss"].iloc[0])
            sub = near[near["lr"] == lr]
            if sub.empty:
                continue
            costs.append(float((sub["loss"] - base).abs().max()))
        return max(costs) if costs else None

    for target, key in ((3.0, "E5C-3X"), (10.0, "E5C-10X")):
        c = _cost(target)
        tokens[key] = f"{c:.3g} test-loss" if c is not None else "PENDING"
    return tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="mlp_wd/outputs/results/e5c_mnist.csv")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--optima_csv",
                    default="rebuttal/nips_rebuttal/_data/e5c_optima.csv")
    ap.add_argument("--sens_csv",
                    default="rebuttal/nips_rebuttal/_data/e5c_sensitivity.csv")
    ap.add_argument("--tokens_md",
                    default="rebuttal/nips_rebuttal/_data/e5c_tokens.md")
    ap.add_argument("--output",
                    default="outputs/plots/nips26/e5c_mnist_mlp_C.png")
    args = ap.parse_args()

    df = pd.read_csv(args.results)
    opt, c_sgd, c_sgdm = fit_C_from_phase_b(args.results, args.epochs, args.batch_size)
    Path(args.optima_csv).parent.mkdir(parents=True, exist_ok=True)
    opt.to_csv(args.optima_csv, index=False)

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4))
    panel_fig1(axes[0], df)
    panel_C_stability(axes[1], opt, c_sgd, c_sgdm)
    sens = panel_sensitivity(axes[2], df, c_sgdm, args.epochs, args.batch_size)
    if sens is not None and not sens.empty:
        sens.to_csv(args.sens_csv, index=False)

    fig.suptitle(
        r"E5c: sensitivity of $\lambda^\star \approx C/S$ on MNIST + 3-layer MLP",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")

    tokens = compute_tokens(opt, c_sgd, c_sgdm, sens)
    lines = ["# E5c resolved tokens", "", f"C_sgd={c_sgd:.4g}, C_sgdm={c_sgdm:.4g}", ""]
    for k, v in tokens.items():
        lines.append(f"- `[[{k}]]` = {v}")
    Path(args.tokens_md).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
