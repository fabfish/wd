"""Side-by-side: plain-MLP gentle vs BN-MLP. Shows whether BN tightens minima alignment."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _draw_panel(ax, df, *, y_metric, title, cap):
    df = df.copy()
    df = df[df["wd"] > 0]
    df["eta_lambda"] = df["lr"] * df["wd"]
    y = pd.to_numeric(df[y_metric], errors="coerce")
    df = df.assign(_y=y)
    if cap is not None:
        df["_y"] = df["_y"].fillna(cap)
    df = df.dropna(subset=["_y"])
    df = df[np.isfinite(df["_y"])]

    wds = sorted(df["wd"].unique())
    cmap = plt.get_cmap("turbo", max(len(wds), 2))
    per_min_x, per_min_y = [], []
    for i, wd in enumerate(wds):
        sub = df[df["wd"] == wd].sort_values("eta_lambda")
        if len(sub) < 2:
            continue
        color = cmap(i)
        ax.plot(sub["eta_lambda"], sub["_y"],
                marker="o", linewidth=1.6, markersize=4.5, alpha=0.92,
                color=color, label=fr"$\lambda$={wd:g}")
        argmin = sub["_y"].idxmin()
        per_min_x.append(float(sub.loc[argmin, "eta_lambda"]))
        per_min_y.append(float(sub.loc[argmin, "_y"]))
        ax.scatter([sub.loc[argmin, "eta_lambda"]], [sub.loc[argmin, "_y"]],
                   marker="*", s=120, facecolors="white",
                   edgecolors=color, linewidths=1.5, zorder=5)
    if per_min_x:
        i_global = int(np.argmin(per_min_y))
        ax.axvline(per_min_x[i_global], color="black",
                   linestyle="--", linewidth=1.0, alpha=0.5)
        ax.scatter([per_min_x[i_global]], [per_min_y[i_global]],
                   marker="*", s=240, color="red", zorder=6)
        spread = max(per_min_x) / max(min(per_min_x), 1e-20)
        ax.text(0.02, 0.97,
                f"per-$\\lambda$ minima spread\nmax/min = {spread:.1f}x",
                transform=ax.transAxes, va="top", ha="left", fontsize=9,
                bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "0.7"})

    ax.set_xscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.set_xlabel(r"$\eta \times \lambda$", fontsize=11)
    ax.set_ylabel(y_metric.replace("_", " ").title(), fontsize=11)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="upper left", ncol=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plain", default="mlp_wd/outputs/results/exp2_gentle.csv")
    ap.add_argument("--bn", default="mlp_wd/outputs/results/exp2_bn.csv")
    ap.add_argument("--output", default="mlp_wd/outputs/plots/exp2_bn_vs_plain.png")
    ap.add_argument("--y_metric", default="best_test_loss")
    ap.add_argument("--cap_diverged_to", type=float, default=2.302585093)
    args = ap.parse_args()

    cap = args.cap_diverged_to if args.cap_diverged_to >= 0 else None
    df_plain = pd.read_csv(args.plain)
    df_bn = pd.read_csv(args.bn)
    print(f"[compare] plain={len(df_plain)} rows, bn={len(df_bn)} rows")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), sharey=False)
    _draw_panel(ax1, df_plain, y_metric=args.y_metric,
                title="Plain MLP (gentle: mom=0.5, clip=2.0)",
                cap=cap)
    _draw_panel(ax2, df_bn, y_metric=args.y_metric,
                title="BN-MLP (mom=0.9, clip=1.0)",
                cap=cap)
    fig.suptitle(
        rf"Exp2: $\eta\,\lambda$ scaling — does BN align the spoon minima?",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
