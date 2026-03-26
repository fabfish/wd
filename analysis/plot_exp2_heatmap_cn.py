#!/usr/bin/env python3
"""
Experiment 2 heatmap (Chinese annotations).

Based on analysis/plot_exp2_fixed.py -> exp2_heatmap_analysis.png

This script generates a NEW figure with updated Chinese labels:
- Left heatmap:
  * Left:  权重衰减（λ）
  * Bottom: 学习率（η）
  * Right (colorbar): 测试精度（%）
  * Top:  λ-η 对应性能的热图
- Right curve plot:
  * Left:  最优权重衰减
  * Bottom: 最优学习率
  * Top:  λ-η 近似关系

Output: outputs/plots/exp2_heatmap_analysis_cn.png
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

# Chinese font configuration (same as other CN figures)
matplotlib.rcParams["font.sans-serif"] = [
    "Noto Sans CJK SC",
    "Noto Sans CJK TC",
    "Droid Sans Fallback",
    "DejaVu Sans",
]
matplotlib.rcParams["axes.unicode_minus"] = False

import matplotlib.pyplot as plt


def plot_exp2_heatmap_cn(df: pd.DataFrame, output_dir: str) -> None:
    """
    Experiment 2: η-λ inverse relationship heatmap (Chinese labels).

    Logic is adapted from plot_exp2_heatmap_fixed in analysis/plot_exp2_fixed.py,
    but uses a 1×2 layout (left heatmap, right curve) and Chinese annotations.
    """
    # Filter SGDM data with batch_size=128
    exp2_df_all = df[(df["method"] == "SGDM") & (df["batch_size"] == 128)].copy()

    # Use only original grid for this figure (exclude very small wd/lr supplement)
    exp2_df = exp2_df_all[
        (exp2_df_all["wd"] >= 0.0001) & (exp2_df_all["lr"] >= 0.01)
    ].copy()

    if exp2_df.empty:
        print("No data for Experiment 2 (CN figure)")
        return

    pivot = exp2_df.pivot_table(
        values="best_test_acc",
        index="wd",
        columns="lr",
        aggfunc="mean",
    )

    # Sort index ascending; with origin='lower', this puts large wd at top visually
    pivot = pivot.sort_index(ascending=True)
    if pivot.empty:
        print("No pivot data for Experiment 2 (CN figure)")
        return

    # 1×2 layout: left heatmap, right λ-η relationship
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # -------------------------------------------------------------------------
    # Left: heatmap
    # -------------------------------------------------------------------------
    from matplotlib.colors import LinearSegmentedColormap
    from scipy.ndimage import zoom

    # Colormap similar to fixed script: emphasize 68–80区间
    cdict = {
        "red": [
            (0.0, 0.55, 0.55),
            (0.4375, 0.70, 0.70),
            (0.60, 0.85, 0.85),
            (0.85, 0.95, 0.95),
            (0.94, 1.0, 1.0),
            (0.95, 0.55, 0.55),
            (1.0, 0.0, 0.0),
        ],
        "green": [
            (0.0, 0.10, 0.10),
            (0.4375, 0.35, 0.35),
            (0.60, 0.55, 0.55),
            (0.85, 0.65, 0.65),
            (0.94, 0.90, 0.90),
            (0.95, 0.80, 0.80),
            (1.0, 0.55, 0.55),
        ],
        "blue": [
            (0.0, 0.10, 0.10),
            (0.4375, 0.15, 0.15),
            (0.60, 0.35, 0.35),
            (0.85, 0.30, 0.30),
            (0.94, 0.25, 0.25),
            (0.95, 0.35, 0.35),
            (1.0, 0.30, 0.30),
        ],
    }
    cmap_smooth = LinearSegmentedColormap("smooth_acc_cn", cdict, N=256)

    data = pivot.values
    extent = [0, len(pivot.columns), 0, len(pivot.index)]

    # Upsample for smooth transitions
    zoom_factor = 10
    data_zoomed = zoom(data, zoom_factor, order=3)

    im = ax1.imshow(
        data_zoomed,
        cmap=cmap_smooth,
        aspect="auto",
        vmin=0,
        vmax=80,
        extent=extent,
        origin="lower",
        interpolation="bilinear",
    )

    # Colorbar on the right of heatmap
    cbar = plt.colorbar(im, ax=ax1)
    cbar.set_label("测试精度（%）", fontsize=11, color="#1a1a1a")

    # Tick positions/labels
    ax1.set_xticks(np.arange(len(pivot.columns)) + 0.5)
    ax1.set_xticklabels([f"{x:.2f}" for x in pivot.columns])
    ax1.set_yticks(np.arange(len(pivot.index)) + 0.5)
    ax1.set_yticklabels([f"{y:.4f}" for y in pivot.index])

    # Value annotations
    for i, wd in enumerate(pivot.index):
        for j, lr in enumerate(pivot.columns):
            val = pivot.loc[wd, lr]
            if not np.isnan(val):
                ax1.text(
                    j + 0.5,
                    i + 0.5,
                    f"{val:.1f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="black",
                    fontweight="bold",
                )

    ax1.set_xlabel("学习率（$\\eta$）", fontsize=11, color="#1a1a1a")
    ax1.set_ylabel("权重衰减（$\\lambda$）", fontsize=11, color="#1a1a1a")
    ax1.set_title(" $\\lambda$-$\\eta$ 对应性能的热图", fontsize=13, fontweight="bold")

    # -------------------------------------------------------------------------
    # Right: optimal λ vs η 关系
    # -------------------------------------------------------------------------
    # Optimal λ for each η
    lr_values = sorted(exp2_df["lr"].unique())
    optimal_wds = []
    for lr in lr_values:
        subset = exp2_df[exp2_df["lr"] == lr]
        if not subset.empty:
            best_row = subset.loc[subset["best_test_acc"].idxmax()]
            optimal_wds.append(best_row["wd"])
        else:
            optimal_wds.append(np.nan)

    ax2.plot(
        lr_values,
        optimal_wds,
        "o-",
        markersize=8,
        linewidth=2,
        color="steelblue",
        label="观测到的最优 $\\lambda$",
    )

    # Fit λ = a / η^b
    valid_mask = ~np.isnan(optimal_wds)
    lr_fit = np.array(lr_values)[valid_mask]
    wd_fit = np.array(optimal_wds)[valid_mask]

    if len(lr_fit) >= 3:
        try:
            log_lr = np.log(lr_fit)
            log_wd = np.log(wd_fit)
            coeffs = np.polyfit(log_lr, log_wd, 1)
            b = -coeffs[0]
            a = np.exp(coeffs[1])

            lr_smooth = np.linspace(min(lr_values), max(lr_values), 200)
            wd_fitted = a / (lr_smooth ** b)
            ax2.plot(
                lr_smooth,
                wd_fitted,
                "r--",
                linewidth=2,
                label=r"拟合：$\lambda \propto \eta^{-%1.2f}$" % b,
            )
        except Exception as e:  # pragma: no cover - diagnostic only
            print(f"[CN] Fitting failed: {e}")

    ax2.set_xlabel("最优学习率", fontsize=11, color="#1a1a1a")
    ax2.set_ylabel("最优权重衰减", fontsize=11, color="#1a1a1a")
    ax2.set_title("$\\lambda$-$\\eta$ 近似关系", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(left=0)
    ax2.set_ylim(bottom=0)

    fig.tight_layout()
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(output_dir) / "exp2_heatmap_analysis_cn.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved CN figure: {out_path}")
    plt.close(fig)


def main() -> None:
    input_file = "outputs/results/results.csv"
    output_dir = "outputs/plots"

    try:
        df = pd.read_csv(input_file)
        print(f"Loaded {len(df)} results from {input_file}")
    except FileNotFoundError:
        print(f"Error: File not found: {input_file}")
        return

    plot_exp2_heatmap_cn(df, output_dir)


if __name__ == "__main__":
    main()

