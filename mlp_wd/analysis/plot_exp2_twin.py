"""Twin-panel reproduction of the original ResNet Exp2 figure:
left = best_test_acc, right = test_error (1 - acc/100).
Both should hit their extremum at the same eta*lambda if the scaling holds.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _draw_panel(ax, df, *, mode, title, cap):
    df = df.copy()
    df = df[df["wd"] > 0]
    df["eta_lambda"] = df["lr"] * df["wd"]
    if mode == "acc":
        df["_y"] = pd.to_numeric(df["best_test_acc"], errors="coerce")
        ylabel = "Best Test Accuracy (%)"
        better_is = "max"
    else:
        acc = pd.to_numeric(df["best_test_acc"], errors="coerce")
        df["_y"] = 1.0 - acc / 100.0
        ylabel = "Test Error (1 - acc)"
        better_is = "min"
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
                marker="o", linewidth=1.7, markersize=5.0, alpha=0.92,
                color=color, label=fr"$\lambda$={wd:g}")
        argopt = sub["_y"].idxmax() if better_is == "max" else sub["_y"].idxmin()
        per_x.append(float(sub.loc[argopt, "eta_lambda"]))
        per_y.append(float(sub.loc[argopt, "_y"]))
        ax.scatter([sub.loc[argopt, "eta_lambda"]], [sub.loc[argopt, "_y"]],
                   marker="*", s=130, facecolors="white",
                   edgecolors=color, linewidths=1.6, zorder=5)

    if per_x:
        i_global = int(np.argmax(per_y) if better_is == "max" else np.argmin(per_y))
        ax.axvline(per_x[i_global], color="black", linestyle="--",
                   linewidth=1.0, alpha=0.5)
        ax.scatter([per_x[i_global]], [per_y[i_global]],
                   marker="*", s=260, color="red", zorder=6,
                   label=fr"global {better_is} @ $\eta\lambda$={per_x[i_global]:.1e}")
        spread = max(per_x) / max(min(per_x), 1e-20)
        ax.text(0.02, 0.97 if better_is == "min" else 0.03,
                f"per-$\\lambda$ {better_is} spread: {spread:.0f}x",
                transform=ax.transAxes,
                va="top" if better_is == "min" else "bottom",
                ha="left", fontsize=9,
                bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "0.7"})

    ax.set_xscale("log")
    if mode == "err":
        ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.set_xlabel(r"$\eta \times \lambda$", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="best", ncol=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--title_suffix", default="")
    args = ap.parse_args()

    df = pd.read_csv(args.results)
    print(f"[twin] loaded {len(df)} rows from {args.results}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.6))
    _draw_panel(ax1, df, mode="acc",
                title="Curves vs $\\eta\\lambda$ (best_test_acc)", cap=None)
    _draw_panel(ax2, df, mode="err",
                title="Same curves: test error (min aligns with max acc)", cap=None)
    suptitle = "Scaling check: maxima alignment in $\\eta\\lambda$"
    if args.title_suffix:
        suptitle += f" — {args.title_suffix}"
    fig.suptitle(suptitle, fontsize=13, fontweight="bold")
    fig.tight_layout()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
