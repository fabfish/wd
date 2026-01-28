
import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
from matplotlib.transforms import Bbox


def _load_results(file_path: str) -> pd.DataFrame:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    df = pd.read_csv(file_path)
    df["LR"] = df["LR"].astype(float)
    df["WD"] = df["WD"].astype(float)
    df["LR_x_WD"] = df["LR"] * df["WD"]
    return df


def create_gradient_pill(
    ax,
    bbox: Bbox,
    color_start: str,
    color_end: str,
    color_mid: str = None,
    horizontal: bool = True,
    alpha: float = 0.3,
    zorder: int = 0,
):
    """
    Draw a pill shape (两头圆中间矩形的椭圆矩形) filled with a smooth color gradient.
    """
    x, y = bbox.x0, bbox.y0
    w, h = bbox.width, bbox.height

    pill = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.0,rounding_size=0.5",
        ec="none",
        fc="none",
        transform=ax.transData,
        zorder=zorder,
    )
    ax.add_patch(pill)

    if color_mid is not None:
        colors = [color_start, color_mid, color_end]
    else:
        colors = [color_start, color_end]
    cmap = LinearSegmentedColormap.from_list("pill_grad", colors, N=128)

    extent = (x, x + w, y, y + h)
    if horizontal:
        grad = np.linspace(0, 1, 256).reshape(1, -1)
    else:
        grad = np.linspace(0, 1, 256).reshape(-1, 1)

    img = ax.imshow(
        grad,
        extent=extent,
        cmap=cmap,
        aspect="auto",
        alpha=alpha,
        zorder=zorder,
        origin="lower",
    )
    img.set_clip_path(pill)
    return pill


def _plot_context_strip(ax, df: pd.DataFrame):
    """左侧灰色概览条：统一灰色，但保留 small / large 形状。"""
    ax.set_xscale("log")

    x_min = df["LR_x_WD"].min()
    x_max = df["LR_x_WD"].max()
    y_min = df["Accuracy"].min()
    y_max = df["Accuracy"].max()
    ax.set_xlim(x_min * 0.7, x_max * 1.3)
    ax.set_ylim(y_min, y_max)

    markers = {"small": "o", "large": "^"}
    for bs_type, marker in markers.items():
        sub = df[df["BS_Type"] == bs_type]
        if sub.empty:
            continue
        ax.scatter(
            sub["LR_x_WD"],
            sub["Accuracy"],
            color="0.6",
            s=40,
            alpha=0.7,
            marker=marker,
            edgecolors="0.4",
            linewidths=0.5,
        )

    ax.tick_params(axis="both", which="both", length=0, labelsize=8)
    ax.set_xticklabels([])
    yticks = [y_min, y_max]
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{yticks[0]:.2f}", f"{yticks[1]:.2f}"], fontsize=8)

    for spine in ["top", "bottom"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "right"]:
        ax.spines[spine].set_linewidth(0.8)
        ax.spines[spine].set_color("0.7")


def _select_0shot_six_points(df0: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    tol = 1e-9
    targets_small = [
        (2e-4, 5e-3),
        (2e-4, 1e-2),
        (5e-4, 1e-2),
        (1e-3, 1e-2),
    ]
    targets_large = [
        (5e-3, 1e-2),
        (1e-3, 1e-2),
    ]

    def is_target(row, targets):
        for t_lr, t_wd in targets:
            if abs(row["LR"] - t_lr) < tol and abs(row["WD"] - t_wd) < tol:
                return True
        return False

    df0_small = df0[
        (df0["BS_Type"] == "small")
        & df0.apply(lambda r: is_target(r, targets_small), axis=1)
    ].copy()
    df0_large = df0[
        (df0["BS_Type"] == "large")
        & df0.apply(lambda r: is_target(r, targets_large), axis=1)
    ].copy()
    return df0_small, df0_large


def _plot_right_top_0shot(ax, df0: pd.DataFrame):
    df0_small, df0_large = _select_0shot_six_points(df0)

    if df0_small.empty or df0_large.empty:
        ax.text(
            0.5,
            0.5,
            "No target points found",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )
        return

    all_points = pd.concat([df0_small, df0_large])
    x_min = all_points["LR_x_WD"].min()
    x_max = all_points["LR_x_WD"].max()
    y_min = all_points["Accuracy"].min()
    y_max = all_points["Accuracy"].max()
    ax.set_xscale("log")
    ax.set_xlim(x_min * 0.5, x_max * 2.0)
    ax.set_ylim(y_min - 0.02, y_max + 0.02)

    norm_small = Normalize(vmin=df0_small["Accuracy"].min(), vmax=df0_small["Accuracy"].max())
    norm_large = Normalize(vmin=df0_large["Accuracy"].min(), vmax=df0_large["Accuracy"].max())

    sc_small = ax.scatter(
        df0_small["LR_x_WD"],
        df0_small["Accuracy"],
        c=df0_small["Accuracy"],
        cmap="Reds",
        norm=norm_small,
        s=220,
        marker="o",
        edgecolors="k",
        linewidths=0.7,
        zorder=10,
        label="Small batch",
    )
    sc_large = ax.scatter(
        df0_large["LR_x_WD"],
        df0_large["Accuracy"],
        c=df0_large["Accuracy"],
        cmap="Blues",
        norm=norm_large,
        s=260,
        marker="^",
        edgecolors="k",
        linewidths=0.7,
        zorder=10,
        label="Large batch",
    )

    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("0-shot: LR×WD vs Accuracy (six key points)", fontsize=14)
    ax.grid(True, linestyle=":", alpha=0.4)

    high_group = pd.concat(
        [
            df0_small[df0_small["Accuracy"] > all_points["Accuracy"].median()],
            df0_large[df0_large["Accuracy"] > all_points["Accuracy"].median()],
        ]
    )
    if not high_group.empty:
        hx_min, hx_max = high_group["LR_x_WD"].min(), high_group["LR_x_WD"].max()
        hy_min, hy_max = high_group["Accuracy"].min(), high_group["Accuracy"].max()
        pad_x = 0.8
        pad_y = 0.01
        bbox_high = Bbox.from_extents(
            hx_min / (1 + pad_x),
            hx_max * (1 + pad_x),
            hy_min - pad_y,
            hy_max + pad_y,
        )
        create_gradient_pill(
            ax,
            bbox_high,
            color_start="#ffebee",
            color_mid="#ffffff",
            color_end="#e3f2fd",
            horizontal=True,
            alpha=0.45,
            zorder=0,
        )

    low_group = pd.concat(
        [
            df0_small[df0_small["Accuracy"] <= all_points["Accuracy"].median()],
            df0_large[df0_large["Accuracy"] <= all_points["Accuracy"].median()],
        ]
    )
    if not low_group.empty:
        lx_min, lx_max = low_group["LR_x_WD"].min(), low_group["LR_x_WD"].max()
        ly_min, ly_max = low_group["Accuracy"].min(), low_group["Accuracy"].max()
        pad_x = 0.8
        pad_y = 0.01
        bbox_low = Bbox.from_extents(
            lx_min / (1 + pad_x),
            lx_max * (1 + pad_x),
            ly_min - pad_y,
            ly_max + pad_y,
        )
        create_gradient_pill(
            ax,
            bbox_low,
            color_start="#ffebee",
            color_mid="#ffffff",
            color_end="#e3f2fd",
            horizontal=True,
            alpha=0.45,
            zorder=-1,
        )

    small_sorted = df0_small.sort_values("Accuracy")
    small_low = small_sorted.head(2)
    small_high = small_sorted.tail(2)
    if len(small_low) == 2 and len(small_high) == 2:
        sx_min = min(small_low["LR_x_WD"].min(), small_high["LR_x_WD"].min())
        sx_max = max(small_low["LR_x_WD"].max(), small_high["LR_x_WD"].max())
        sy_min = small_low["Accuracy"].min()
        sy_max = small_high["Accuracy"].max()
        bbox_small_trend = Bbox.from_extents(
            sx_min / 1.2,
            sx_max * 1.2,
            sy_min - 0.01,
            sy_max + 0.01,
        )
        create_gradient_pill(
            ax,
            bbox_small_trend,
            color_start="#ffcdd2",
            color_end="#b71c1c",
            horizontal=False,
            alpha=0.22,
            zorder=-2,
        )

    if len(df0_large) == 2:
        large_sorted = df0_large.sort_values("Accuracy")
        l_xmin = large_sorted["LR_x_WD"].min()
        l_xmax = large_sorted["LR_x_WD"].max()
        l_ymin = large_sorted["Accuracy"].min()
        l_ymax = large_sorted["Accuracy"].max()
        bbox_large_trend = Bbox.from_extents(
            l_xmin / 1.2,
            l_xmax * 1.2,
            l_ymin - 0.01,
            l_ymax + 0.01,
        )
        create_gradient_pill(
            ax,
            bbox_large_trend,
            color_start="#bbdefb",
            color_end="#0d47a1",
            horizontal=False,
            alpha=0.22,
            zorder=-2,
        )

    ax.legend(loc="lower right", fontsize=10, frameon=False)


def _plot_right_bottom_8shot(ax, df8: pd.DataFrame):
    mask = (
        (df8["Accuracy"] >= 0.10)
        & (df8["Accuracy"] <= 0.14)
        & (df8["LR_x_WD"] >= 1e-6)
        & (df8["LR_x_WD"] <= 1e-4)
    )
    df_sel = df8[mask].copy()

    if df_sel.empty:
        mask_wide = (
            (df8["Accuracy"] >= 0.10)
            & (df8["Accuracy"] <= 0.14)
            & (df8["LR_x_WD"] >= 1e-6)
            & (df8["LR_x_WD"] <= 1e-3)
        )
        df_sel = df8[mask_wide].copy()

    df_small = df_sel[df_sel["BS_Type"] == "small"].copy()
    df_large = df_sel[df_sel["BS_Type"] == "large"].copy()

    if df_sel.empty:
        ax.text(
            0.5,
            0.5,
            "No points in selected 8-shot region",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )
        return

    ax.set_xscale("log")
    x_min = df_sel["LR_x_WD"].min()
    x_max = df_sel["LR_x_WD"].max()
    y_min = df_sel["Accuracy"].min()
    y_max = df_sel["Accuracy"].max()
    ax.set_xlim(x_min * 0.5, x_max * 2.0)
    ax.set_ylim(y_min - 0.01, y_max + 0.01)

    if not df_small.empty:
        norm_small = Normalize(df_small["Accuracy"].min(), df_small["Accuracy"].max())
        sc_small = ax.scatter(
            df_small["LR_x_WD"],
            df_small["Accuracy"],
            c=df_small["Accuracy"],
            cmap="Reds",
            norm=norm_small,
            s=180,
            marker="o",
            edgecolors="k",
            linewidths=0.7,
            label="Small batch",
            zorder=10,
        )
    else:
        sc_small = None

    if not df_large.empty:
        norm_large = Normalize(df_large["Accuracy"].min(), df_large["Accuracy"].max())
        sc_large = ax.scatter(
            df_large["LR_x_WD"],
            df_large["Accuracy"],
            c=df_large["Accuracy"],
            cmap="Blues",
            norm=norm_large,
            s=200,
            marker="^",
            edgecolors="k",
            linewidths=0.7,
            label="Large batch",
            zorder=10,
        )
    else:
        sc_large = None

    ax.set_xlabel("LR × WD (log scale)", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("8-shot: Same-accuracy band in LR×WD space", fontsize=14)
    ax.grid(True, linestyle=":", alpha=0.4)

    bbox = Bbox.from_extents(
        x_min / 1.5,
        x_max * 1.5,
        y_min - 0.01,
        y_max + 0.01,
    )
    create_gradient_pill(
        ax,
        bbox,
        color_start="#ffebee",
        color_mid="#ffffff",
        color_end="#e3f2fd",
        horizontal=True,
        alpha=0.4,
        zorder=-1,
    )

    if sc_small is not None:
        cbar = plt.colorbar(sc_small, ax=ax, pad=0.02, fraction=0.05)
        cbar.set_label("Accuracy (color intensity)", fontsize=10)
        cbar.ax.tick_params(labelsize=8)

    ax.legend(loc="lower right", fontsize=10, frameon=False)


def plot_complex_figure():
    file_0shot = "outputs/results/results_0shot.csv"
    file_8shot = "outputs/results/results_8shot.csv"

    try:
        df0 = _load_results(file_0shot)
        df8 = _load_results(file_8shot)
    except FileNotFoundError as e:
        print(e)
        return

    sns.set(style="white")

    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(
        4,
        2,
        width_ratios=[1, 3],
        height_ratios=[0.35, 0.35, 1.0, 1.0],
        wspace=0.15,
        hspace=0.18,
    )

    ax_left_top = fig.add_subplot(gs[0, 0])
    _plot_context_strip(ax_left_top, df0)

    ax_left_bottom = fig.add_subplot(gs[1, 0])
    _plot_context_strip(ax_left_bottom, df8)

    ax_main_top = fig.add_subplot(gs[0:2, 1])
    _plot_right_top_0shot(ax_main_top, df0)

    ax_main_bot = fig.add_subplot(gs[2:4, 1])
    _plot_right_bottom_8shot(ax_main_bot, df8)

    shape_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="0.6",
            markeredgecolor="0.3",
            markersize=7,
            linestyle="",
            label="Small batch (shape)",
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            color="none",
            markerfacecolor="0.6",
            markeredgecolor="0.3",
            markersize=8,
            linestyle="",
            label="Large batch (shape)",
        ),
    ]
    fig.legend(
        handles=shape_handles,
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
        frameon=False,
        fontsize=9,
        title="Batch type (shapes)",
        title_fontsize=9,
    )

    plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.96])

    outfile = "outputs/plots/complex_analysis.png"
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    plt.savefig(outfile, dpi=200)
    print(f"Saved complex figure to {outfile}")
    plt.close()


if __name__ == "__main__":
    plot_complex_figure()
