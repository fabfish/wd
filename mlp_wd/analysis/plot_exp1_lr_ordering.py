"""Exp1: paired LR ordering, two side-by-side panels.

Left  panel (Group A, mom=0):    SGD vs SGD+WD, x = eta, y = best_test_acc.
Right panel (Group B, mom=0.9):  SGDM vs SGDM+WD, same axes.
Each panel marks each method's peak with a star. Within a panel, the
WD-on curve is expected to peak at a lower eta than the WD-off curve.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PANEL_METHODS = {
    "A": (("SGD", "C0"), ("SGD+WD", "C1")),
    "B": (("SGDM", "C2"), ("SGDM+WD", "C3")),
}
PANEL_TITLES = {
    "A": "Group A (mom = 0): SGD vs SGD+WD",
    "B": "Group B (mom = 0.9): SGDM vs SGDM+WD",
}


def _draw_panel(ax, df_panel: pd.DataFrame, methods, title: str, y_metric: str = "best_test_acc"):
    for method, color in methods:
        sub = df_panel[df_panel["method"] == method].sort_values("lr")
        if sub.empty:
            continue
        y = pd.to_numeric(sub[y_metric], errors="coerce")
        ax.plot(sub["lr"], y, marker="o", linewidth=2.0, markersize=6, color=color, label=method)
        if y.notna().any():
            best_idx = y.idxmax()
            ax.scatter([sub.loc[best_idx, "lr"]], [y.loc[best_idx]],
                       marker="*", s=240, color=color, edgecolors="black",
                       linewidths=1.0, zorder=5)
            ax.annotate(
                fr"$\eta^*$={sub.loc[best_idx, 'lr']:g}",
                xy=(sub.loc[best_idx, "lr"], y.loc[best_idx]),
                xytext=(8, 8), textcoords="offset points",
                fontsize=9, color=color,
            )
    ax.set_xscale("log")
    ax.set_xlabel(r"$\eta$ (lr)", fontsize=11)
    ax.set_ylabel("Best test accuracy (%)", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=10)


def plot_lr_ordering(df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    df_a = df[df["momentum"].abs() < 1e-9]
    df_b = df[(df["momentum"] - 0.9).abs() < 1e-9]
    _draw_panel(axes[0], df_a, PANEL_METHODS["A"], PANEL_TITLES["A"])
    _draw_panel(axes[1], df_b, PANEL_METHODS["B"], PANEL_TITLES["B"])
    fig.suptitle("Exp1: WD-induced optimal-LR shift (paired ablations)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved: {output_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="mlp_wd/outputs/results/exp1.csv")
    ap.add_argument("--output", default="mlp_wd/outputs/plots/exp1_lr_ordering.png")
    args = ap.parse_args()
    df = pd.read_csv(args.results)
    plot_lr_ordering(df, Path(args.output))


if __name__ == "__main__":
    main()
