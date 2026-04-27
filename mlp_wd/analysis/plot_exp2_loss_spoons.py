"""Headline plot: 'bunch of spoons' -- final test loss vs eta * lambda.

For each lambda we draw a curve over the eta grid. If the eta x lambda scaling
law (with cosine schedule + fixed T) is well-calibrated, every lambda curve
descends, hits a minimum at roughly the same eta * lambda, then rises again
when training becomes unstable on the right tail.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_spoons(
    df: pd.DataFrame,
    output_path: Path,
    *,
    y_metric: str = "final_test_loss",
    title_suffix: str = "MLP / CIFAR-10, SGDM, BS=128",
    y_log: bool = False,
    cap_diverged_to: float | None = None,
) -> None:
    is_accuracy = "acc" in y_metric.lower()
    df = df.copy()
    df = df[df["wd"] > 0]
    if df.empty:
        raise ValueError("No rows with wd > 0; cannot draw spoons.")
    df["eta_lambda"] = df["lr"] * df["wd"]

    if y_metric not in df.columns:
        raise ValueError(f"Column {y_metric!r} not found in CSV (have: {list(df.columns)})")
    y = pd.to_numeric(df[y_metric], errors="coerce")
    df = df.assign(_y=y)
    if cap_diverged_to is not None:
        # A diverged run produces NaN loss; in cross-entropy land the "effective"
        # ceiling is -log(1/K) = log(K) (uniform predictor). Filling with that
        # value exposes the rising tail of every spoon at the random-predictor
        # level instead of dropping the point.
        df["_y"] = df["_y"].fillna(cap_diverged_to)
    df = df.dropna(subset=["_y"])
    df = df[np.isfinite(df["_y"])]

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    wds = sorted(df["wd"].unique())
    cmap = plt.get_cmap("turbo", max(len(wds), 2))

    per_curve_min_x = []
    per_curve_min_y = []

    for i, wd in enumerate(wds):
        sub = df[df["wd"] == wd].sort_values("eta_lambda")
        if len(sub) < 2:
            continue
        color = cmap(i)
        ax.plot(
            sub["eta_lambda"], sub["_y"],
            marker="o", linewidth=1.8, markersize=5.0, alpha=0.92,
            color=color, label=fr"$\lambda$={wd:g}",
        )
        argopt = sub["_y"].idxmax() if is_accuracy else sub["_y"].idxmin()
        per_curve_min_x.append(float(sub.loc[argopt, "eta_lambda"]))
        per_curve_min_y.append(float(sub.loc[argopt, "_y"]))
        ax.scatter([sub.loc[argopt, "eta_lambda"]], [sub.loc[argopt, "_y"]],
                   marker="*", s=130, facecolors="white",
                   edgecolors=color, linewidths=1.6, zorder=5)

    if per_curve_min_x:
        i_global = int(np.argmax(per_curve_min_y) if is_accuracy
                       else np.argmin(per_curve_min_y))
        x_star = per_curve_min_x[i_global]
        y_star = per_curve_min_y[i_global]
        ax.axvline(x_star, color="black", linestyle="--", linewidth=1.2, alpha=0.6)
        opt_label = "max" if is_accuracy else "min"
        ax.scatter([x_star], [y_star], marker="*", s=260, color="red", zorder=6,
                   label=fr"global {opt_label} @ $\eta\lambda$={x_star:.2e}")
        spread = max(per_curve_min_x) / max(min(per_curve_min_x), 1e-20)
        print(f"per-lambda {opt_label} eta*lambda: {sorted(per_curve_min_x)}")
        print(f"global {opt_label}: eta*lambda={x_star:.3e}, {y_metric}={y_star:.4f}, spread(max/min)={spread:.2f}x")

    ax.set_xscale("log")
    if y_log:
        ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.set_xlabel(r"$\eta \times \lambda$", fontsize=12)
    pretty = y_metric.replace("_", " ").replace("test loss", "test loss")
    ax.set_ylabel(pretty.title(), fontsize=12)
    ax.set_title(
        rf"Exp2: {pretty.title()} vs $\eta\,\lambda$ — bunch of spoons" + "\n" + title_suffix,
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=8, loc="best", ncol=2)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved: {output_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="mlp_wd/outputs/results/exp2.csv")
    ap.add_argument("--output", default="mlp_wd/outputs/plots/exp2_loss_spoons.png")
    ap.add_argument("--y_metric", default="final_test_loss",
                    choices=["final_test_loss", "best_test_loss",
                             "final_train_loss",
                             "best_test_acc", "final_test_acc"])
    ap.add_argument("--y_log", action="store_true")
    ap.add_argument("--title_suffix", default="MLP / CIFAR-10, SGDM, BS=128")
    ap.add_argument("--cap_diverged_to", type=float, default=2.302585093,
                    help="Fill NaN losses with this value so divergent runs "
                         "show up at the random-predictor ceiling (log K). "
                         "Pass a negative number to disable.")
    args = ap.parse_args()

    cap = args.cap_diverged_to if args.cap_diverged_to >= 0 else None
    df = pd.read_csv(args.results)
    print(f"[plot] loaded {len(df)} rows from {args.results}")
    plot_spoons(df, Path(args.output),
                y_metric=args.y_metric, title_suffix=args.title_suffix,
                y_log=args.y_log, cap_diverged_to=cap)


if __name__ == "__main__":
    main()
