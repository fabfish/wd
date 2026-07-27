"""Exp2 secondary plot: 2D heatmap of test loss / accuracy on (eta, lambda)."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_heatmap(df: pd.DataFrame, output_path: Path, metric: str = "final_test_loss") -> None:
    df = df[df["wd"] > 0].copy()
    if df.empty:
        raise ValueError("No rows with wd > 0.")
    df = df.dropna(subset=[metric])

    pivot = df.pivot_table(index="wd", columns="lr", values=metric, aggfunc="mean")
    pivot = pivot.sort_index(axis=0).sort_index(axis=1)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    is_loss = "loss" in metric
    cmap = "viridis_r" if is_loss else "viridis"
    arr = pivot.to_numpy()
    im = ax.imshow(arr, aspect="auto", cmap=cmap)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{x:g}" for x in pivot.columns], rotation=30)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{y:g}" for y in pivot.index])
    ax.invert_yaxis()
    ax.set_xlabel(r"$\eta$ (lr)", fontsize=12)
    ax.set_ylabel(r"$\lambda$ (wd)", fontsize=12)
    ax.set_title(f"Exp2: {metric.replace('_', ' ').title()} on $(\\eta, \\lambda)$",
                 fontsize=12, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(metric.replace("_", " ").title())

    if is_loss:
        ij = np.unravel_index(np.nanargmin(arr), arr.shape)
    else:
        ij = np.unravel_index(np.nanargmax(arr), arr.shape)
    ax.scatter([ij[1]], [ij[0]], s=200, marker="*", color="red", edgecolors="white",
               linewidths=1.4, zorder=5)
    best_lr = pivot.columns[ij[1]]
    best_wd = pivot.index[ij[0]]
    print(f"best: lr={best_lr:g}, wd={best_wd:g}, {metric}={arr[ij]:.4f}")

    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            v = arr[i, j]
            if not np.isfinite(v):
                continue
            text = f"{v:.3f}" if is_loss else f"{v:.1f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=7,
                    color="white" if (not is_loss and v < np.nanmean(arr)) or (is_loss and v > np.nanmean(arr)) else "black")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved: {output_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="mlp_wd/outputs/results/exp2.csv")
    ap.add_argument("--output", default="mlp_wd/outputs/plots/exp2_heatmap.png")
    ap.add_argument("--metric", default="final_test_loss",
                    choices=["final_test_loss", "best_test_loss",
                             "best_test_acc", "final_test_acc"])
    args = ap.parse_args()
    df = pd.read_csv(args.results)
    plot_heatmap(df, Path(args.output), metric=args.metric)


if __name__ == "__main__":
    main()
