"""Compare three Exp2 configurations side-by-side: plain MLP vs BN-hidden vs
BN-everywhere (norm_output). Shows the test_error twin panel for each, so we
can see how progressively more scale-invariance tightens the per-lambda
minima alignment along eta*lambda.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _draw_err_panel(ax, df, *, title):
    df = df.copy()
    df = df[df["wd"] > 0]
    df["eta_lambda"] = df["lr"] * df["wd"]
    acc = pd.to_numeric(df["best_test_acc"], errors="coerce")
    df["_y"] = 1.0 - acc / 100.0
    df = df.dropna(subset=["_y"])
    df = df[np.isfinite(df["_y"])]

    wds = sorted(df["wd"].unique())
    cmap = plt.get_cmap("turbo", max(len(wds), 2))
    per_x, per_y = [], []
    for i, wd in enumerate(wds):
        sub = df[df["wd"] == wd].sort_values("eta_lambda")
        if len(sub) < 2:
            continue
        color = cmap(i)
        ax.plot(sub["eta_lambda"], sub["_y"],
                marker="o", linewidth=1.7, markersize=4.5, alpha=0.92,
                color=color, label=fr"$\lambda$={wd:g}")
        argopt = sub["_y"].idxmin()
        per_x.append(float(sub.loc[argopt, "eta_lambda"]))
        per_y.append(float(sub.loc[argopt, "_y"]))
        ax.scatter([sub.loc[argopt, "eta_lambda"]], [sub.loc[argopt, "_y"]],
                   marker="*", s=110, facecolors="white",
                   edgecolors=color, linewidths=1.4, zorder=5)

    if per_x:
        i_global = int(np.argmin(per_y))
        ax.axvline(per_x[i_global], color="black", linestyle="--",
                   linewidth=1.0, alpha=0.5)
        ax.scatter([per_x[i_global]], [per_y[i_global]],
                   marker="*", s=220, color="red", zorder=6)
        spread = max(per_x) / max(min(per_x), 1e-20)
        ax.text(0.02, 0.98,
                f"per-$\\lambda$ min spread: {spread:.0f}x\n"
                f"global min @ $\\eta\\lambda$={per_x[i_global]:.1e}",
                transform=ax.transAxes, va="top", ha="left", fontsize=9,
                bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "0.7"})

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.set_xlabel(r"$\eta \times \lambda$", fontsize=10)
    ax.set_ylabel("Test Error (1 - acc)", fontsize=10)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.legend(fontsize=7, loc="lower right", ncol=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plain", required=True, help="plain MLP CSV")
    ap.add_argument("--bn", required=True, help="BN-hidden CSV")
    ap.add_argument("--bnfull", required=True, help="BN-everywhere CSV")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    df_plain = pd.read_csv(args.plain)
    df_bn = pd.read_csv(args.bn)
    df_bnf = pd.read_csv(args.bnfull)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.6))
    _draw_err_panel(axes[0], df_plain,
                    title=f"Plain MLP\n(no normalization, {len(df_plain)} rows)")
    _draw_err_panel(axes[1], df_bn,
                    title=f"BN on hidden layers\n(output unprotected, {len(df_bn)} rows)")
    _draw_err_panel(axes[2], df_bnf,
                    title=f"BN everywhere\n(scale-invariant, {len(df_bnf)} rows)")

    fig.suptitle("Effect of scale-invariance on $\\eta\\lambda$ alignment "
                 "(test error, log-log)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
