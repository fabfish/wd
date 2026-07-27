"""
Re-render the three legacy paper figures so their typography matches the
new figures in `wd/figures/`:

- performance.jpg : analysis/plot_exp1_v18.py
- batch.png       : analysis/plot_v3_supplementary_v2.py
- LLM (1).png     : top-row 0-shot view from scripts/plot_four_panel.py,
                    with the diagonal red/blue gradients lightened.

Only typography (font family, size) and the LLM gradient palette are
changed; data and overall layout remain as in the original scripts.
"""
import shutil
import sys
import warnings
from pathlib import Path

import matplotlib as mpl

warnings.filterwarnings('ignore')

ROOT = Path('.').resolve()
DEST = ROOT / 'wd' / 'figures'

# ── Unified rcParams (same recipe used by rebuttal/regen_main_figs.py) ──
SERIF_RC = {
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.weight': 'normal',
    'axes.titleweight': 'bold',
    'axes.labelweight': 'normal',
    'mathtext.fontset': 'stix',
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'pdf.fonttype': 42,
    'ps.fonttype':  42,
}


def with_serif_rc(fn):
    def wrapper():
        with mpl.rc_context(SERIF_RC):
            return fn()
    return wrapper


# ──────────────────────────────────────────────────────────────────────
# 1. performance.jpg
# ──────────────────────────────────────────────────────────────────────
@with_serif_rc
def regen_performance():
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / 'analysis'))
    import importlib
    if 'plot_exp1_v18' in sys.modules:
        importlib.reload(sys.modules['plot_exp1_v18'])
    import plot_exp1_v18  # noqa: E402

    plot_exp1_v18.main()
    src = ROOT / 'outputs/plots/exp1_lr_ordering_v18.png'
    if src.exists():
        shutil.copy(src, DEST / 'performance.jpg')
        print('Saved performance.jpg')


# ──────────────────────────────────────────────────────────────────────
# 2. batch.png  – aligned with the unified style:
#    * plasma colormap for the η / λ line colours (same family that
#      drives FOCUS_LAMBDA_COLORS in the smooth-panel figure);
#    * black axes spines on every subplot (line plots and heatmaps),
#      overriding the gray of seaborn-v0_8-whitegrid.
# ──────────────────────────────────────────────────────────────────────
@with_serif_rc
def regen_batch():
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / 'analysis'))
    import importlib
    if 'plot_v3_supplementary_v2' in sys.modules:
        importlib.reload(sys.modules['plot_v3_supplementary_v2'])
    import plot_v3_supplementary_v2 as pvs  # noqa: E402

    import matplotlib.pyplot as plt

    # ── Patch line-curve plotting: swap viridis → plasma ─────────────
    orig_plot_curves = pvs.plot_training_curves

    def plot_curves_plasma(ax_loss, ax_acc, histories, title_prefix,
                           legend_label_key='lr', legend_format=None):
        cmap = plt.cm.plasma
        # Take a slice of plasma that avoids the very-yellow tail (>0.85),
        # which is too close to white on a white background.
        items = sorted(histories.items())
        n = len(items)
        positions = [0.05 + 0.80 * (i / max(n - 1, 1)) for i in range(n)]
        colors = [cmap(p) for p in positions]
        for (key, data), color in zip(items, colors):
            history = data['history']
            epochs = [h['epoch'] for h in history]
            train_loss = [h['train_loss'] for h in history]
            test_acc = [h['test_acc'] for h in history]
            label = (legend_format(data['config'][legend_label_key])
                     if legend_format
                     else f"{legend_label_key}={data['config'][legend_label_key]}")
            ax_loss.plot(epochs, train_loss, color=color, label=label, linewidth=1.5)
            ax_acc.plot(epochs, test_acc,  color=color, label=label, linewidth=1.5)
        ax_loss.set_xlabel('Epoch')
        ax_loss.set_ylabel('Train Loss')
        ax_loss.set_title(f'{title_prefix} - Train Loss')
        ax_loss.legend(fontsize=10, loc='upper right')
        ax_loss.grid(True, linestyle='--', alpha=0.5)
        ax_loss.set_xlim(left=0); ax_loss.set_ylim(bottom=0)
        ax_acc.set_xlabel('Epoch')
        ax_acc.set_ylabel('Test Acc (%)')
        ax_acc.set_title(f'{title_prefix} - Test Accuracy')
        ax_acc.legend(fontsize=10, loc='lower right')
        ax_acc.grid(True, linestyle='--', alpha=0.5)
        ax_acc.set_xlim(left=0)

    pvs.plot_training_curves = plot_curves_plasma

    # ── Heatmap: replace pvs's narrow-band colormap with the unified
    # smooth red→orange→yellow→green palette + bicubic zoom used in
    # rebuttal/regen_main_figs.py.  The pre-existing pvs heatmap has
    # vmin≈60 (saturating the deep-red end), so we use the same range
    # but with the unified colormap shape so batch.png lines up with
    # the appendix/main heatmaps.
    sys.path.insert(0, str(ROOT / 'rebuttal'))
    if 'regen_main_figs' in sys.modules:
        importlib.reload(sys.modules['regen_main_figs'])
    import regen_main_figs as rmf  # noqa: E402

    import numpy as _np
    from scipy.ndimage import zoom as _zoom
    from scipy.ndimage import gaussian_filter as _gauss

    # Per-panel yellow→green thresholds requested by the user:
    #   BS=32   →  cells ≥ 75.9 should be green
    #   BS=128  →  cells ≥ 75.3 should be green
    BS_GREEN_FLOOR = {32: 75.9, 128: 75.3}

    def heatmap_smooth(ax, df, bs, title, cax=None, show_ylabel=True):
        bs_df = df[df['batch_size'] == bs].copy()
        if bs_df.empty:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    transform=ax.transAxes)
            ax.set_title(title)
            return None
        pivot = bs_df.pivot_table(values='best_test_acc', index='wd',
                                  columns='lr', aggfunc='max')
        pivot = pivot.sort_index(ascending=True)

        valid = pivot.values[~_np.isnan(pivot.values)]
        # Fixed [60, 80] range so both panels share an identical legend
        # and the green knee is reproducible across runs.
        vmin, vmax = 60.0, 80.0
        green_floor = BS_GREEN_FLOOR.get(bs, 75.0)
        cmap = rmf.make_smooth_cmap(vmin, vmax, green_floor)

        data = _np.nan_to_num(pivot.values.astype(float), nan=vmin)
        # Same recipe as rebuttal/regen_main_figs.py: nearest-neighbour
        # zoom keeps each cell's centre at its true value, Gaussian
        # rounds the boundaries for the soft topographic look.
        data_zoomed = _zoom(data, 10, order=0)
        data_zoomed = _gauss(data_zoomed, sigma=10 * 0.55)
        data_zoomed = _np.clip(data_zoomed, vmin, vmax)

        extent = [0, len(pivot.columns), 0, len(pivot.index)]
        im = ax.imshow(data_zoomed, cmap=cmap, aspect='auto',
                       vmin=vmin, vmax=vmax, extent=extent,
                       origin='lower', interpolation='bilinear')

        ax.grid(False)
        ax.set_xticks(_np.arange(len(pivot.columns)) + 0.5)
        ax.set_xticklabels([f'{x:.3g}' for x in pivot.columns],
                           rotation=45, ha='right', fontsize=10)
        ax.set_yticks(_np.arange(len(pivot.index)) + 0.5)
        ax.set_yticklabels([f'{y:.0e}' for y in pivot.index], fontsize=10)
        ax.set_xlabel('Learning Rate (η)')
        if show_ylabel:
            ax.set_ylabel('Weight Decay (λ)')
        ax.set_title(title)

        for i, wd in enumerate(pivot.index):
            for j, lr in enumerate(pivot.columns):
                val = pivot.loc[wd, lr]
                if not _np.isnan(val):
                    ax.text(j + 0.5, i + 0.5, f'{val:.1f}',
                            ha='center', va='center',
                            color='black', fontsize=9, fontweight='bold')

        max_val = valid.max()
        max_idx = _np.where(pivot.values == max_val)
        if len(max_idx[0]) > 0:
            r0, c0 = max_idx[0][0], max_idx[1][0]
            ax.add_patch(plt.Rectangle((c0, r0), 1, 1, fill=False,
                                       edgecolor='blue', linewidth=2.5))
        return im

    orig_heatmap = pvs.plot_heatmap_improved
    pvs.plot_heatmap_improved = heatmap_smooth

    # Black spines on every subplot. seaborn-v0_8-whitegrid (set at
    # pvs module load) hides the right/top spines and gray-tints the
    # remaining ones. Patch `plt.tight_layout` to walk the current
    # figure and force black, fully-visible spines just before save.
    def _force_black_frames(ax):
        for side in ('left', 'right', 'top', 'bottom'):
            sp = ax.spines[side]
            sp.set_visible(True)
            sp.set_color('black')
            sp.set_linewidth(1.2)

    orig_tight = plt.tight_layout

    def tight_with_black_frames(*args, **kwargs):
        fig = plt.gcf()
        for ax in fig.get_axes():
            _force_black_frames(ax)
        return orig_tight(*args, **kwargs)

    plt.tight_layout = tight_with_black_frames

    # Widen the horizontal gap between subplots in the 3×5 grid so the
    # left-side y-tick labels of the Test-Accuracy subplots don't crash
    # into the right spine of the Train-Loss subplots.
    from matplotlib.figure import Figure
    orig_add_gridspec = Figure.add_gridspec

    def add_gridspec_wider(self, *args, **kwargs):
        if kwargs.get('wspace', None) is not None and kwargs['wspace'] < 0.4:
            kwargs['wspace'] = 0.4
        return orig_add_gridspec(self, *args, **kwargs)

    Figure.add_gridspec = add_gridspec_wider
    try:
        pvs.main()
    finally:
        Figure.add_gridspec = orig_add_gridspec
        plt.tight_layout = orig_tight
        pvs.plot_heatmap_improved = orig_heatmap
        pvs.plot_training_curves = orig_plot_curves

    src = ROOT / 'outputs/plots/v3_supplementary_analysis_v2.png'
    if src.exists():
        shutil.copy(src, DEST / 'batch.png')
        print(f'Saved batch.png  (from {src.name})')
    else:
        print('WARN: batch.png source not found')


# ──────────────────────────────────────────────────────────────────────
# 3. LLM (1).png  — keep the original four-panel pipeline 100 % intact
#    and only swap the *colors* used by the diagonal Red/Blue gradients.
#    No layout, axes, capsules, or connecting lines are touched. After
#    rendering the full 2×2 figure we crop just the top row (the 0-shot
#    pair) which is what wd/figures/LLM (1).png is.
# ──────────────────────────────────────────────────────────────────────
@with_serif_rc
def regen_LLM():
    """Re-render only the colors of the diagonal strokes in the 0-shot
    panel, leaving everything else byte-for-byte identical to the
    upstream `scripts/plot_four_panel.py` rendering."""
    import importlib
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / 'scripts'))
    if 'plot_four_panel' in sys.modules:
        importlib.reload(sys.modules['plot_four_panel'])
    import plot_four_panel as pfp  # noqa: E402

    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba

    # Patch draw_diagonal_stroke to lighten only the two colors. Whether
    # a call is for the Red (small-batch) or Blue (large-batch) family
    # is recovered from the supplied colors themselves.
    orig_diag = pfp.draw_diagonal_stroke

    def lighter_diag(ax, low_points, high_points,
                     ellipse_width_log=0.25, ellipse_height_y=0.03,
                     color_start='#ffcccc', color_end='#cc0000',
                     alpha=0.35, zorder=0):
        r, g, b, _ = to_rgba(color_end)
        if r > b:                       # Red family
            cs, ce = plt.cm.Reds(0.10), plt.cm.Reds(0.55)
        else:                           # Blue family
            cs, ce = plt.cm.Blues(0.10), plt.cm.Blues(0.55)
        return orig_diag(
            ax, low_points, high_points,
            ellipse_width_log=ellipse_width_log,
            ellipse_height_y=ellipse_height_y,
            color_start=cs, color_end=ce,
            alpha=alpha, zorder=zorder,
        )

    # Patch ax.text so every "lr × wd" annotation (the only ones with
    # fontsize=6.5 and rotation=45 in plot_four_panel.py) is offset up
    # and right by ~1 character width using `annotate(..., textcoords=
    # 'offset points')`. Everything else passes through unchanged.
    from matplotlib.axes import Axes
    orig_text = Axes.text

    def shifted_text(self, x, y, s, *args, **kwargs):
        if (kwargs.get('fontsize') == 6.5
                and kwargs.get('rotation') == 45):
            return self.annotate(
                s, xy=(x, y), xytext=(6, 6),
                textcoords='offset points',
                fontsize=kwargs.get('fontsize', 6.5),
                rotation=kwargs.get('rotation', 45),
                ha=kwargs.get('ha', 'left'),
                va=kwargs.get('va', 'bottom'),
                color=kwargs.get('color', '#333333'),
                zorder=kwargs.get('zorder', 11),
            )
        return orig_text(self, x, y, s, *args, **kwargs)

    pfp.draw_diagonal_stroke = lighter_diag
    Axes.text = shifted_text
    try:
        pfp.main()
    finally:
        Axes.text = orig_text
        pfp.draw_diagonal_stroke = orig_diag

    src_png = ROOT / 'outputs/plots/four_panel_analysis.png'
    if not src_png.exists():
        print(f'WARN: {src_png} not produced')
        return

    # Crop the top row out of the 2-row layout. The original LLM.png
    # has aspect ≈ 2.31 (1024 × 444). We replicate that ratio by
    # taking the top portion of the rendered four-panel PNG.
    from PIL import Image
    img = Image.open(src_png)
    w, h = img.size
    target_aspect = 1024 / 444  # original LLM.png aspect
    crop_h = int(round(w / target_aspect))
    crop_h = min(crop_h, h)
    top_strip = img.crop((0, 0, w, crop_h))
    out = DEST / 'LLM (1).png'
    top_strip.save(out)
    print(f'Saved {out.name}  (cropped top {crop_h}/{h} from four_panel)')

if __name__ == '__main__':
    regen_performance()
    regen_batch()
    regen_LLM()
