#!/usr/bin/env python3
"""
Experiment 2 scaling-law plot: one curve per weight decay (lambda).

X-axis: eta * lambda
Y-axis: test accuracy (or loss if loss-like column exists)

If eta * lambda = C explains the optimum when T is fixed, different lambda
curves should peak near a similar x-value.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _pick_y_column(df: pd.DataFrame) -> tuple[str, bool]:
    """
    Pick a y-axis metric from available columns.

    Returns:
        (column_name, is_loss)
    """
    # Prefer accuracy for this repository; fallback to a loss-like column if available.
    accuracy_candidates = ["best_test_acc", "final_test_acc"]
    loss_candidates = ["best_val_loss", "val_loss", "best_test_loss", "test_loss", "final_train_loss"]

    for col in accuracy_candidates:
        if col in df.columns:
            return col, False
    for col in loss_candidates:
        if col in df.columns:
            return col, True

    raise ValueError(
        "No supported y-axis metric found. Expected one of "
        f"{accuracy_candidates + loss_candidates}."
    )


def plot_exp2_scaling_curves(
    df: pd.DataFrame,
    output_dir: str = "outputs/plots",
    method: str = "SGDM",
    batch_size: int = 128,
) -> None:
    """
    Generate scaling-law plot for Experiment 2.
    """
    exp2_df = df[(df["method"] == method) & (df["batch_size"] == batch_size)].copy()
    if exp2_df.empty:
        print(f"No data found for method={method}, batch_size={batch_size}.")
        return

    y_col, is_loss = _pick_y_column(exp2_df)
    exp2_df["eta_lambda"] = exp2_df["lr"] * exp2_df["wd"]

    # Keep deterministic ordering for clean legend.
    unique_wd = sorted(exp2_df["wd"].unique())

    fig, ax = plt.subplots(figsize=(10, 6))

    peak_x_values = []

    for wd in unique_wd:
        wd_df = exp2_df[exp2_df["wd"] == wd].copy()
        # Average repeated runs at same eta*lambda point.
        curve_df = (
            wd_df.groupby("eta_lambda", as_index=False)[y_col]
            .mean()
            .sort_values("eta_lambda")
        )
        if curve_df.empty:
            continue

        ax.plot(
            curve_df["eta_lambda"],
            curve_df[y_col],
            marker="o",
            linewidth=1.8,
            markersize=5,
            label=fr"$\lambda$={wd:g}",
            alpha=0.9,
        )

        # Mark optimum point for each lambda-curve.
        peak_idx = curve_df[y_col].idxmin() if is_loss else curve_df[y_col].idxmax()
        peak_row = curve_df.loc[peak_idx]
        peak_x_values.append(float(peak_row["eta_lambda"]))
        ax.scatter(
            [peak_row["eta_lambda"]],
            [peak_row[y_col]],
            marker="*",
            s=120,
            color="black",
            zorder=5,
        )

    ax.set_xscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.set_xlabel(r"$\eta \times \lambda$", fontsize=12)
    if is_loss:
        ax.set_ylabel(y_col.replace("_", " ").title(), fontsize=12)
        ax.set_title(
            "Experiment 2: Loss Curves vs " + r"$\eta\times\lambda$" + "\n(one curve per weight decay)",
            fontsize=13,
            fontweight="bold",
        )
    else:
        ax.set_ylabel("Test Accuracy (%)", fontsize=12)
        ax.set_title(
            "Experiment 2: Test Accuracy vs " + r"$\eta\times\lambda$" + "\n(one curve per weight decay)",
            fontsize=13,
            fontweight="bold",
        )

    # Visual check for constant eta*lambda optimum across curves.
    if peak_x_values:
        median_peak = float(np.median(peak_x_values))
        ax.axvline(
            median_peak,
            linestyle="--",
            linewidth=2,
            color="crimson",
            alpha=0.9,
            label=fr"median optimum $\eta\lambda$={median_peak:.2e}",
        )
        print("Per-curve optimum eta*lambda values:")
        for v in sorted(peak_x_values):
            print(f"  {v:.3e}")
        spread = max(peak_x_values) / max(min(peak_x_values), 1e-20)
        print(f"Optimum spread (max/min): {spread:.2f}x")

    ax.legend(fontsize=9, ncol=2)
    fig.tight_layout()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save a dedicated file and replace the current exp2 figure path requested by user.
    out_main = output_path / "exp2_eta_lambda_scaling_curves.png"
    out_replace = output_path / "exp2_heatmap_analysis.png"
    fig.savefig(out_main, dpi=200, bbox_inches="tight")
    fig.savefig(out_replace, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out_main}")
    print(f"Updated: {out_replace}")


def main() -> None:
    input_file = Path("outputs/results/results.csv")
    if not input_file.exists():
        print(f"File not found: {input_file}")
        return

    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} rows from {input_file}")
    plot_exp2_scaling_curves(df)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Experiment 2 alternative visualization: scaling curves vs η×λ.

Each curve corresponds to a fixed weight decay λ (wd column).
X-axis is η×λ (lr * wd). Y-axis: best val accuracy (train/val split runs) or
best test accuracy (full-train runs), depending on protocol.

If the scaling law η×λ = C (for fixed T) holds, the maxima of different curves
should occur around the same η×λ.

Rows from the val supplement (train/val split, ~90% train) must not be mixed with
main Exp2 rows (full training set) on the same curve; see protocol=auto.
"""

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _safe_log_ticks(x_min: float, x_max: float) -> List[float]:
    if not (np.isfinite(x_min) and np.isfinite(x_max) and x_min > 0 and x_max > 0):
        return []
    p_min = int(np.floor(np.log10(x_min)))
    p_max = int(np.ceil(np.log10(x_max)))
    return [10.0**p for p in range(p_min, p_max + 1)]


def _train_val_split_mask(df: pd.DataFrame) -> np.ndarray:
    """Rows that belong to the val-supplement protocol (not comparable to full-train test acc)."""
    if "best_val_loss" not in df.columns:
        return np.zeros(len(df), dtype=bool)
    v = pd.to_numeric(df["best_val_loss"], errors="coerce")
    mask = np.isfinite(v.to_numpy())
    if "data_protocol" in df.columns:
        tagged = df["data_protocol"].astype(str).eq("train_val_split")
        mask = mask | tagged.fillna(False).to_numpy()
    return mask


def plot_exp2_scaling_curves(
    df: pd.DataFrame,
    output_dir: str = "outputs/plots",
    *,
    method: str = "SGDM",
    batch_size: int = 128,
    protocol: str = "auto",
    left_metric: str = "auto",
) -> Path:
    exp2 = df[(df["method"] == method) & (df["batch_size"] == batch_size)].copy()
    if exp2.empty:
        raise ValueError(f"No data for method={method}, batch_size={batch_size}")

    # Keep only the original Exp2 grid (exclude supplement runs with tiny wd/lr).
    exp2 = exp2[(exp2["wd"] >= 1e-4) & (exp2["lr"] >= 1e-2)].copy()
    if exp2.empty:
        raise ValueError("No data left after applying original-grid filter (wd>=1e-4, lr>=1e-2)")

    needed_cols = {"lr", "wd", "best_test_acc"}
    missing = needed_cols - set(exp2.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    val_row = _train_val_split_mask(exp2)
    if protocol == "auto":
        protocol = "val_split" if val_row.any() else "full_train"
    elif protocol not in ("full_train", "val_split"):
        raise ValueError("protocol must be 'auto', 'full_train', or 'val_split'")

    if protocol == "val_split":
        exp2 = exp2.loc[val_row].copy()
        if exp2.empty:
            raise ValueError(
                "protocol=val_split but no train/val-split rows found "
                "(finite best_val_loss or data_protocol=train_val_split). "
                "Use --protocol full_train or run the val supplement."
            )
    else:
        exp2 = exp2.loc[~val_row].copy()
        if exp2.empty:
            raise ValueError(
                "protocol=full_train but no full-train rows left after excluding "
                "train/val-split runs; use --protocol val_split or check results.csv."
            )

    if left_metric == "auto":
        left_metric = "best_val_acc" if protocol == "val_split" else "best_test_acc"
    if left_metric not in ("best_val_acc", "best_test_acc"):
        raise ValueError("left_metric must be 'auto', 'best_val_acc', or 'best_test_acc'")

    acc_col = left_metric
    if acc_col not in exp2.columns:
        raise ValueError(f"Missing column {acc_col!r} for left panel")
    if exp2[acc_col].notna().sum() == 0:
        raise ValueError(f"No finite values in {acc_col!r} for the selected protocol")

    exp2["eta_lambda"] = exp2["lr"] * exp2["wd"]
    exp2 = exp2[np.isfinite(exp2["eta_lambda"]) & (exp2["eta_lambda"] > 0)].copy()
    if exp2.empty:
        raise ValueError("No positive finite η×λ values to plot")

    # Right panel: validation loss when available; else test error (full-train only).
    _vl = (
        pd.to_numeric(exp2["best_val_loss"], errors="coerce")
        if "best_val_loss" in exp2.columns
        else pd.Series(np.nan, index=exp2.index)
    )
    if protocol == "val_split" or np.isfinite(_vl.to_numpy()).any():
        metric_col = "best_val_loss"
        metric_label = "Best Val Loss"
        metric_title = "Same curves: best validation loss (min alignment)"
    else:
        metric_col = "test_error"
        exp2[metric_col] = 1.0 - (exp2["best_test_acc"].astype(float) / 100.0)
        metric_label = "Test Error (1 - acc)"
        metric_title = "Same curves: test error (min aligns with max acc)"

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8), sharex=True)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    ax_acc = axes[0]
    ax_loss = axes[1]

    # Drop rows missing left-metric for plotting optima/lines
    exp2_plot = exp2[np.isfinite(pd.to_numeric(exp2[acc_col], errors="coerce"))].copy()

    wds = sorted(exp2_plot["wd"].unique())
    cmap = plt.get_cmap("turbo", max(len(wds), 2))

    per_curve_opt_x = []
    for i, wd in enumerate(wds):
        sub = exp2_plot[exp2_plot["wd"] == wd].copy()
        if len(sub) < 2:
            continue
        sub = sub.sort_values("eta_lambda")
        color = cmap(i)
        label = f"λ={wd:g}"

        y_acc = pd.to_numeric(sub[acc_col], errors="coerce")
        ax_acc.plot(
            sub["eta_lambda"],
            y_acc,
            marker="o",
            linewidth=1.8,
            markersize=4.5,
            alpha=0.9,
            color=color,
            label=label,
        )

        best_row = sub.loc[y_acc.idxmax()]
        per_curve_opt_x.append(float(best_row["eta_lambda"]))
        ax_acc.scatter(
            [best_row["eta_lambda"]],
            [float(best_row[acc_col])],
            s=55,
            marker="*",
            color=color,
            edgecolors="black",
            linewidths=0.6,
            zorder=5,
        )

        sub_metric = sub[np.isfinite(sub[metric_col])].copy()
        if len(sub_metric) >= 1:
            sub_metric = sub_metric.sort_values("eta_lambda")
            if len(sub_metric) >= 2:
                ax_loss.plot(
                    sub_metric["eta_lambda"],
                    sub_metric[metric_col],
                    marker="o",
                    linewidth=1.8,
                    markersize=4.5,
                    alpha=0.9,
                    color=color,
                    label=label,
                )
            else:
                ax_loss.scatter(
                    sub_metric["eta_lambda"],
                    sub_metric[metric_col],
                    s=35,
                    color=color,
                    alpha=0.9,
                    label=label,
                )

            min_row = sub_metric.loc[sub_metric[metric_col].idxmin()]
            ax_loss.scatter(
                [min_row["eta_lambda"]],
                [min_row[metric_col]],
                s=70,
                marker="*",
                color=color,
                edgecolors="black",
                linewidths=0.6,
                zorder=6,
            )

    y_acc_all = pd.to_numeric(exp2_plot[acc_col], errors="coerce")
    global_best = exp2_plot.loc[y_acc_all.idxmax()]
    x_star = float(global_best["eta_lambda"])
    y_star = float(global_best[acc_col])
    ax_acc.axvline(x_star, color="black", linestyle="--", linewidth=1.2, alpha=0.7)
    ax_acc.scatter([x_star], [y_star], s=140, marker="*", color="red", zorder=6)

    ax_acc.set_xscale("log")
    ax_acc.set_xlabel("η × λ", fontsize=12)
    if acc_col == "best_val_acc":
        ax_acc.set_ylabel("Best Val Accuracy (%)", fontsize=12)
    else:
        ax_acc.set_ylabel("Best Test Accuracy (%)", fontsize=12)
    ax_acc.set_title("Exp2: Curves vs η×λ (one curve per λ)", fontsize=13, fontweight="bold")
    ax_acc.grid(True, which="both", alpha=0.25)

    x_min = float(exp2_plot["eta_lambda"].min())
    x_max = float(exp2_plot["eta_lambda"].max())
    xticks = _safe_log_ticks(x_min, x_max)
    if xticks:
        ax_acc.set_xticks(xticks)

    ax_loss.set_xscale("log")
    ax_loss.set_xlabel("η × λ", fontsize=12)
    ax_loss.set_ylabel(metric_label, fontsize=12)
    ax_loss.set_title(metric_title, fontsize=13, fontweight="bold")
    ax_loss.grid(True, which="both", alpha=0.25)
    if xticks:
        ax_loss.set_xticks(xticks)

    loss_vals = exp2_plot[np.isfinite(exp2_plot[metric_col])][metric_col].values
    if loss_vals.size:
        y_min = float(np.nanmin(loss_vals))
        y_max = float(np.nanmax(loss_vals))
    else:
        y_min, y_max = 0.0, 1.0

    if metric_col == "best_val_loss":
        if y_max <= y_min:
            y_max = y_min + 1e-3
        pad = max(0.01, 0.2 * (y_max - y_min))
        ax_loss.set_ylim(bottom=y_min - pad, top=y_max + 5 * pad)
        ax_loss.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, pos: f"{v:.2f}"))
    else:
        y0 = 0.2
        k = 5.0
        if y_max <= y_min:
            y_max = y_min + 1e-3

        logpart = float(np.log10(1.0 + max(0.0, y_max - y0) * k)) if y_max > y0 else 0.0
        expand = (0.7 / 0.3) * logpart / y0 if logpart > 0 else 1.0
        expand = float(np.clip(expand, 1.0, 25.0))

        def _y_transform(y):
            y = np.asarray(y, dtype=float)
            y = np.maximum(y, 0.0)
            out = y * expand
            mask = y > y0
            if np.any(mask):
                out[mask] = y0 * expand + np.log10(1.0 + (y[mask] - y0) * k)
            return out

        def _y_inverse(yp):
            yp = np.asarray(yp, dtype=float)
            out = yp / expand
            cutoff = y0 * expand
            mask = yp > cutoff
            if np.any(mask):
                out[mask] = y0 + (10.0 ** (yp[mask] - cutoff) - 1.0) / k
            return out

        try:
            from matplotlib.scale import FuncScale

            ax_loss.set_yscale(FuncScale(ax_loss, (_y_transform, _y_inverse)))
        except Exception:
            ax_loss.set_yscale("symlog", linthresh=0.2, linscale=2.0)

        ax_loss.set_ylim(bottom=_y_transform([0.0])[0], top=_y_transform([y_max])[0] * 1.05)

        candidate_ticks = [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.8]
        candidate_ticks = [t for t in candidate_ticks if t <= y_max]
        if candidate_ticks:
            ax_loss.set_yticks(_y_transform(candidate_ticks))
            ax_loss.set_yticklabels([f"{t:g}" for t in candidate_ticks])

        loss_visible = exp2_plot[np.isfinite(exp2_plot[metric_col])].copy()
        if not loss_visible.empty:
            gmin = loss_visible.loc[loss_visible[metric_col].idxmin()]
            x_gmin = float(gmin["eta_lambda"])
            y_gmin = float(gmin[metric_col])
            ax_loss.axvline(x_gmin, color="black", linestyle="--", linewidth=1.2, alpha=0.7)
            ax_loss.scatter([x_gmin], [y_gmin], s=140, marker="*", color="red", zorder=7)
            ax_loss.text(
                x_gmin,
                min(0.98, y_gmin + 0.08),
                "min",
                ha="center",
                va="bottom",
                fontsize=9,
                color="black",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8, linewidth=0.0),
            )

    handles, labels_leg = ax_acc.get_legend_handles_labels()
    if handles:
        ax_acc.legend(
            handles,
            labels_leg,
            fontsize=8,
            ncol=2,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=True,
        )

    proto_note = f"protocol={protocol}, left={acc_col}"
    fig.suptitle(
        f"Scaling check: maxima alignment in η×λ ({proto_note}; method={method}, batch_size={batch_size})",
        y=1.02,
        fontsize=11,
    )
    fig.tight_layout()

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "exp2_eta_lambda_scaling_curves.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    if per_curve_opt_x:
        per_curve_opt_x_arr = np.array(sorted(per_curve_opt_x))
        print("\nExp2 scaling-curve summary (per-λ optimum η×λ):")
        print(f"  count_curves={len(per_curve_opt_x_arr)}")
        print(f"  min={per_curve_opt_x_arr.min():.3e}")
        print(f"  median={np.median(per_curve_opt_x_arr):.3e}")
        print(f"  max={per_curve_opt_x_arr.max():.3e}")
        print(
            f"  global_best η×λ={x_star:.3e} (η={global_best['lr']}, λ={global_best['wd']}, {acc_col}={y_star:.2f})"
        )

    print(f"Saved: {out_path}")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot Exp2 η×λ scaling curves (no mixed-protocol artifacts).")
    ap.add_argument("--results", default="outputs/results/results.csv", help="Path to results CSV")
    ap.add_argument("--output-dir", default="outputs/plots", help="Directory for PNG output")
    ap.add_argument("--method", default="SGDM")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument(
        "--protocol",
        choices=["auto", "full_train", "val_split"],
        default="auto",
        help="Train protocol: val_split = supplement rows only; full_train = exclude those rows; auto picks val_split if such rows exist.",
    )
    ap.add_argument(
        "--left-metric",
        choices=["auto", "best_val_acc", "best_test_acc"],
        default="auto",
        help="Left Y-axis: auto uses best_val_acc for val_split and best_test_acc for full_train.",
    )
    args = ap.parse_args()
    df = pd.read_csv(args.results)
    plot_exp2_scaling_curves(
        df,
        output_dir=args.output_dir,
        method=args.method,
        batch_size=args.batch_size,
        protocol=args.protocol,
        left_metric=args.left_metric,
    )


if __name__ == "__main__":
    main()
