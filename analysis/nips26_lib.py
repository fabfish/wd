"""
Shared data loading and fitting for the NeurIPS 17464 rebuttal analysis.

Two sources of runs are unified here:
  * legacy CSVs, all of which are 100-epoch cosine runs on CIFAR-100, and
  * rebuttal/results/nips26_runs.csv, which additionally varies T and schedule.

Everything downstream is expressed in terms of sum_lr = sum_t eta_t rather than
eta*T, because that is the quantity the coupling law is really about and it is
what reconciles the factor of two between Eq. 16 and the conclusion.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
LEGACY_REBUTTAL = ROOT / 'rebuttal' / 'results'
LEGACY_OUTPUTS = ROOT / 'outputs' / 'results'
NEW_RUNS = ROOT / 'rebuttal' / 'results' / 'nips26_runs.csv'
PLOT_DIR = ROOT / 'outputs' / 'plots' / 'nips26'
REBUTTAL_DIR = ROOT / 'rebuttal' / 'nips_rebuttal'
TABLE_DIR = REBUTTAL_DIR / '_data'

CIFAR100_TRAIN_N = 50000

# Legacy files, with the metadata that is implicit in the filename rather than
# stored in the CSV. Every one of these was run for 100 cosine epochs.
LEGACY_FILES = [
    ('rebuttal/results/results_resnet18_seed42_exp2_ext.csv', 'resnet18', 42),
    ('rebuttal/results/results_resnet18_seed42_exp2_ext2.csv', 'resnet18', 42),
    ('rebuttal/results/results_resnet18_seed42_exp2_fill.csv', 'resnet18', 42),
    ('rebuttal/results/results_resnet18_seed123_exp2_ext.csv', 'resnet18', 123),
    ('rebuttal/results/results_resnet18_exp2_supplement.csv', 'resnet18', 42),
    ('rebuttal/results/results_resnet18_seed123.csv', 'resnet18', 123),
    ('rebuttal/results/results_resnet18_seed123_run2.csv', 'resnet18', 123),
    ('rebuttal/results/results_resnet18_seed42_run2.csv', 'resnet18', 42),
    ('rebuttal/results/results_resnet50_seed42.csv', 'resnet50', 42),
    ('rebuttal/results/results_resnet50_seed123.csv', 'resnet50', 123),
    ('rebuttal/results/results_vgg16_seed42.csv', 'vgg16', 42),
    ('rebuttal/results/results_vgg16_seed123.csv', 'vgg16', 123),
    ('outputs/results/results.csv', 'resnet18', 42),
    ('outputs/results/results_v2.csv', 'resnet18', 42),
    ('outputs/results/momentum_search.csv', 'resnet18', 42),
    ('outputs/results/sgdm_no_wd_search.csv', 'resnet18', 42),
    ('outputs/results/sgdm_no_wd_search_low.csv', 'resnet18', 42),
    ('outputs/results/sgdm_extended.csv', 'resnet18', 42),
    ('outputs/results/three_methods_comparison.csv', 'resnet18', 42),
    ('outputs/results/wd_shift_search.csv', 'resnet18', 42),
    ('outputs/results/lr_extension.csv', 'resnet18', 42),
    ('outputs/results/sgdwd_supplement.csv', 'resnet18', 42),
]

# v3_supplementary.csv has no `method` column; it is all SGDM at momentum 0.9.
LEGACY_V3 = ('outputs/results/v3_supplementary.csv', 'resnet18', 42)


def steps_per_epoch(batch_size, n=CIFAR100_TRAIN_N):
    return np.ceil(np.asarray(n, dtype=float) / np.asarray(batch_size, dtype=float))


def sum_lr(lr, epochs, batch_size, scheduler='cosine', n=CIFAR100_TRAIN_N):
    """
    sum_t eta_t over optimizer steps.

    CosineAnnealingLR stepped once per epoch from eta towards zero contributes
    sum_{k<T} 0.5*(1 + cos(pi*k/T)) = (T+1)/2 epochs' worth of step size, i.e.
    about half what a constant schedule accumulates at matched eta and T.
    Carrying that factor explicitly is what removes the ambiguity between
    lambda = 1/(eta*T) and lambda = 2/(eta*T).
    """
    lr = np.asarray(lr, dtype=float)
    epochs = np.asarray(epochs, dtype=float)
    steps = steps_per_epoch(batch_size, n)
    if scheduler == 'const':
        return lr * steps * epochs
    return lr * steps * (epochs + 1.0) / 2.0


def _read_legacy(path, model, seed, method_default=None, momentum_default=None):
    full = ROOT / path
    if not full.exists():
        return None
    df = pd.read_csv(full)
    if 'method' not in df.columns:
        df['method'] = method_default if method_default else 'SGDM'
    if 'momentum' not in df.columns:
        df['momentum'] = momentum_default if momentum_default is not None else 0.9
    df['model'] = model
    df['seed'] = seed
    df['dataset'] = 'cifar100'
    df['epochs'] = 100
    df['scheduler'] = 'cosine'
    df['source'] = Path(path).name
    df['diverged'] = 0
    for col in ('final_test_loss', 'final_train_acc'):
        if col not in df.columns:
            df[col] = np.nan
    return df


def load_legacy(include_v3=True):
    """Every pre-existing run, normalized to one schema."""
    frames = []
    for path, model, seed in LEGACY_FILES:
        df = _read_legacy(path, model, seed)
        if df is not None:
            frames.append(df)
    if include_v3:
        path, model, seed = LEGACY_V3
        df = _read_legacy(path, model, seed, method_default='SGDM', momentum_default=0.9)
        if df is not None:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out['batch_size'] = out['batch_size'].astype(int)
    out['sum_lr'] = sum_lr(out['lr'].values, out['epochs'].values,
                           out['batch_size'].values, 'cosine')
    return out


def load_new(csv_path=NEW_RUNS):
    if not Path(csv_path).exists():
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    if df.empty:
        return df
    df['source'] = 'nips26'
    df['batch_size'] = df['batch_size'].astype(int)
    df['epochs'] = df['epochs'].astype(int)
    # Deduplicate: later rows win, since a re-run supersedes an earlier attempt.
    key = ['model', 'dataset', 'method', 'batch_size', 'lr', 'wd', 'momentum',
           'epochs', 'scheduler', 'seed']
    df = df.drop_duplicates(subset=key, keep='last').reset_index(drop=True)
    return df


def load_all(csv_path=NEW_RUNS):
    """Legacy plus new runs, with new runs taking precedence on collisions."""
    legacy, new = load_legacy(), load_new(csv_path)
    if new.empty:
        return legacy
    if legacy.empty:
        return new
    key = ['model', 'batch_size', 'lr', 'wd', 'momentum', 'epochs', 'scheduler', 'seed']
    merged = pd.concat([legacy, new], ignore_index=True)
    merged = merged.drop_duplicates(subset=key, keep='last').reset_index(drop=True)
    return merged


def load_legacy_grid(model='resnet18', seed=None, batch_size=128, momentum=0.9):
    """
    The dense eta-lambda heatmap at T=100 that the envelope figure is built on.

    Seeds 42 and 123 each cover 8 learning rates x 9 weight decays.
    """
    df = load_legacy()
    m = ((df['model'] == model) & (df['batch_size'] == batch_size)
         & (df['momentum'] == momentum) & (df['wd'] > 0)
         & (df['epochs'] == 100))
    if seed is not None:
        m &= df['seed'] == seed
    grid = df[m].copy()
    grid = grid.groupby(['seed', 'lr', 'wd'], as_index=False).agg(
        best_test_acc=('best_test_acc', 'max'),
        final_test_acc=('final_test_acc', 'mean'),
        final_train_loss=('final_train_loss', 'mean'),
    )
    grid['product'] = grid['lr'] * grid['wd']
    return grid


def optimal_wd(df, group_cols, acc_col='best_test_acc', min_points=3):
    """
    For each group, the weight decay that maximizes accuracy.

    Also returns a log-parabola interpolated optimum, which is less quantized
    than the grid argmax and is what the slope fits use.
    """
    rows = []
    for keys, g in df.groupby(group_cols):
        g = g[g['wd'] > 0].sort_values('wd')
        if len(g) < min_points:
            continue
        idx = g[acc_col].idxmax()
        best = g.loc[idx]
        rec = dict(zip(group_cols if isinstance(group_cols, list) else [group_cols],
                       keys if isinstance(keys, tuple) else (keys,)))
        rec['wd_argmax'] = float(best['wd'])
        rec['acc_max'] = float(best[acc_col])
        rec['n_points'] = int(len(g))
        rec['wd_interp'] = _parabola_peak(g['wd'].values, g[acc_col].values)
        # An optimum sitting on the edge of the swept range is a lower bound on
        # lambda*, not a measurement of it, so keep it flagged rather than
        # silently mixing it in with the resolved optima.
        wds = np.sort(g['wd'].values)
        rec['interior'] = bool(wds[0] < rec['wd_argmax'] < wds[-1])
        rows.append(rec)
    return pd.DataFrame(rows)


def _parabola_peak(wd, acc):
    """Peak of a parabola through the argmax and its two neighbours in log wd."""
    wd, acc = np.asarray(wd, float), np.asarray(acc, float)
    i = int(np.argmax(acc))
    if i == 0 or i == len(wd) - 1:
        return float(wd[i])
    x = np.log(wd[i - 1:i + 2])
    y = acc[i - 1:i + 2]
    denom = (y[0] - 2 * y[1] + y[2])
    if abs(denom) < 1e-12:
        return float(wd[i])
    delta = 0.5 * (y[0] - y[2]) / denom
    delta = float(np.clip(delta, -1.0, 1.0))
    return float(np.exp(x[1] + delta * (x[1] - x[0])))


def fit_loglog_slope(x, y, n_boot=2000, seed=0):
    """Least-squares slope of log y on log x, with a bootstrap interval."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = (x > 0) & (y > 0)
    x, y = np.log(x[ok]), np.log(y[ok])
    if len(x) < 2:
        return dict(slope=np.nan, intercept=np.nan, lo=np.nan, hi=np.nan, n=len(x))
    slope, intercept = np.polyfit(x, y, 1)
    rng = np.random.RandomState(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.randint(0, len(x), len(x))
        if len(np.unique(x[idx])) < 2:
            continue
        boots.append(np.polyfit(x[idx], y[idx], 1)[0])
    lo, hi = (np.percentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan))
    return dict(slope=float(slope), intercept=float(intercept),
                lo=float(lo), hi=float(hi), n=int(len(x)))


def fit_reference_C(csv_path=NEW_RUNS, model='resnet18', batch_size=128,
                    lr=0.1, epochs=100, momentum=0.9, seed=42):
    """
    Calibrate the constant in lambda* = C / sum_lr on one reference setting.

    Everything the transfer comparison predicts follows from this single number,
    so it is fitted once, from data that already exists, and never re-tuned.
    """
    df = load_all(csv_path)
    m = ((df['model'] == model) & (df['batch_size'] == batch_size)
         & (np.isclose(df['lr'], lr)) & (df['epochs'] == epochs)
         & (np.isclose(df['momentum'], momentum)) & (df['wd'] > 0)
         & (df['scheduler'] == 'cosine') & (df['seed'] == seed))
    ref = df[m]
    if ref.empty:
        raise RuntimeError(f"no reference runs found for lr={lr}, T={epochs}")
    peak = _parabola_peak(ref.sort_values('wd')['wd'].values,
                          ref.sort_values('wd')['best_test_acc'].values)
    return float(peak * sum_lr(lr, epochs, batch_size, 'cosine'))


# --------------------------------------------------------------------------
# competing rules
# --------------------------------------------------------------------------

def predict_wd(strategy, lr, epochs, batch_size, C, ref):
    """
    Weight decay predicted by each rule, given no tuning at the target setting.

    ours      lambda = C / sum_t eta_t, with C calibrated once
    kosson    the product eta*lambda is held at its reference value, so the rule
              carries no dependence on the training length
    wang      lambda = 1 / (eta * T) with T counted in optimizer steps
    default   the value people actually type, 5e-4
    zero      no weight decay at all
    """
    if strategy == 'zero':
        return 0.0
    if strategy == 'default':
        return 5e-4
    if strategy == 'ours':
        return C / sum_lr(lr, epochs, batch_size, 'cosine')
    if strategy == 'kosson':
        return ref['product'] / lr
    if strategy == 'wang':
        return 1.0 / (lr * steps_per_epoch(batch_size) * epochs)
    raise ValueError(strategy)


STRATEGIES = ['zero', 'default', 'kosson', 'wang', 'ours']

# Held-out settings. The learning rate follows the paper's own linear rule when
# the batch size changes, so no per-setting tuning enters through the back door.
TRANSFER_CONFIGS = [
    dict(name='T=25',        model='resnet18', batch_size=128, lr=0.1,   epochs=25),
    dict(name='T=200',       model='resnet18', batch_size=128, lr=0.1,   epochs=200),
    dict(name='B=32',        model='resnet18', batch_size=32,  lr=0.025, epochs=100),
    dict(name='B=512',       model='resnet18', batch_size=512, lr=0.4,   epochs=100),
    dict(name='VGG-16',      model='vgg16',    batch_size=128, lr=0.1,   epochs=100),
    dict(name='ResNet-50',   model='resnet50', batch_size=128, lr=0.1,   epochs=100),
]


def reference_point(csv_path=NEW_RUNS):
    """The reference setting itself: R18, B=128, eta=0.1, T=100."""
    C = fit_reference_C(csv_path)
    lr, epochs, bs = 0.1, 100, 128
    wd_star = C / sum_lr(lr, epochs, bs, 'cosine')
    return dict(C=C, lr=lr, epochs=epochs, batch_size=bs,
                wd_star=wd_star, product=lr * wd_star)


def build_transfer_plan(csv_path=NEW_RUNS, round_to=3):
    """Configurations E4 needs to train, after removing those already covered."""
    ref = reference_point(csv_path)
    df = load_all(csv_path)
    plan, seen = [], set()

    for cfg in TRANSFER_CONFIGS:
        for strategy in STRATEGIES:
            wd = predict_wd(strategy, cfg['lr'], cfg['epochs'], cfg['batch_size'],
                            ref['C'], ref)
            if wd <= 0:
                wd = 0.0
            else:
                wd = float(f"%.{round_to}g" % wd)
            key = (cfg['model'], cfg['batch_size'], cfg['lr'], wd, cfg['epochs'])
            if key in seen:
                continue
            seen.add(key)
            have = df[(df['model'] == cfg['model'])
                      & (df['batch_size'] == cfg['batch_size'])
                      & np.isclose(df['lr'], cfg['lr'])
                      & np.isclose(df['wd'], wd, rtol=1e-3, atol=1e-12)
                      & (df['epochs'] == cfg['epochs'])
                      & np.isclose(df['momentum'], 0.9)
                      & (df['scheduler'] == 'cosine')]
            if not have.empty:
                continue
            plan.append(dict(model=cfg['model'], batch_size=cfg['batch_size'],
                             lr=cfg['lr'], wd=wd, epochs=cfg['epochs'],
                             strategy=strategy, config=cfg['name']))
    return plan


def ensure_dirs():
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
