"""Exp3: visualize batch-size scaling with linear LR rule.

Two panels:
  Left  -- best test accuracy vs lambda, one curve per batch size; star marks optimal lambda(B).
  Right -- optimal lambda*(B) plotted against B in log-log; check for a power law.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_exp3(df: pd.DataFrame, output_path: Path, y_metric: str = "best_test_acc") -> None:
    df = df.copy()
    df = df[df["wd"] > 0].dropna(subset=[y_metric])
    if df.empty:
        raise ValueError("No data.")

    bs_list = sorted(df["batch_size"].unique())
    cmap = plt.get_cmap("plasma", max(len(bs_list), 2))

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5))
    optimal = []
    for i, bs in enumerate(bs_list):
        sub = df[df["batch_size"] == bs].sort_values("wd")
        if len(sub) < 2:
            continue
        color = cmap(i)
        y = pd.to_numeric(sub[y_metric], errors="coerce")
        axes[0].plot(sub["wd"], y, marker="o", color=color, label=f"BS={bs}")
        if y.notna().any():
            best_idx = y.idxmax() if "acc" in y_metric else y.idxmin()
            best_wd = float(sub.loc[best_idx, "wd"])
            best_y = float(y.loc[best_idx])
            axes[0].scatter([best_wd], [best_y], marker="*", s=180, color=color,
                            edgecolors="black", linewidths=0.8, zorder=5)
            optimal.append((bs, best_wd, best_y))

    axes[0].set_xscale("log")
    axes[0].set_xlabel(r"$\lambda$ (wd)", fontsize=11)
    axes[0].set_ylabel(y_metric.replace("_", " ").title(), fontsize=11)
    axes[0].set_title("Per batch size: metric vs $\\lambda$", fontsize=12, fontweight="bold")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend(fontsize=9)

    if optimal:
        Bs = np.array([o[0] for o in optimal], dtype=float)
        Ws = np.array([o[1] for o in optimal], dtype=float)
        axes[1].plot(Bs, Ws, marker="o", color="C3")
        axes[1].set_xscale("log")
        axes[1].set_yscale("log")
        log_b = np.log10(Bs)
        log_w = np.log10(Ws)
        if len(Bs) >= 2 and np.all(np.isfinite(log_w)):
            slope, intercept = np.polyfit(log_b, log_w, 1)
            xx = np.linspace(log_b.min(), log_b.max(), 50)
            yy = intercept + slope * xx
            axes[1].plot(10**xx, 10**yy, "--", color="black", alpha=0.6,
                         label=fr"fit: slope={slope:+.2f}")
            axes[1].legend(fontsize=10)
            print(f"optimal lambda(B): slope={slope:+.3f} (theory predicts ~ +1 for lambda* prop B)")
        axes[1].set_xlabel("Batch size B", fontsize=11)
        axes[1].set_ylabel(r"Optimal $\lambda^*(B)$", fontsize=11)
        axes[1].set_title(r"Scaling: $\lambda^*$ vs B", fontsize=12, fontweight="bold")
        axes[1].grid(True, which="both", alpha=0.3)

    fig.suptitle("Exp3: batch-size scaling with linear LR rule", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved: {output_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="mlp_wd/outputs/results/exp3.csv")
    ap.add_argument("--output", default="mlp_wd/outputs/plots/exp3_batchsize.png")
    ap.add_argument("--metric", default="best_test_acc",
                    choices=["best_test_acc", "final_test_acc",
                             "final_test_loss", "best_test_loss"])
    args = ap.parse_args()
    df = pd.read_csv(args.results)
    plot_exp3(df, Path(args.output), y_metric=args.metric)


if __name__ == "__main__":
    main()
