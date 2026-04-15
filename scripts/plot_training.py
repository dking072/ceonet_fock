"""
Plot training curves from Lightning metrics.log files.

By default iterates over all models found under lightning_logs/ and saves
one PNG per model next to its metrics.log.

Usage:
    python plot_training.py                          # all models, save PNGs
    python plot_training.py --logs lightning_logs    # explicit logs root
    python plot_training.py --log path/to/metrics.log  # single file
"""

import argparse
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # no display required
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

HA_TO_EV = 27.2114

# --- CLI ---------------------------------------------------------------------
parser = argparse.ArgumentParser()
group = parser.add_mutually_exclusive_group()
group.add_argument(
    '--logs', default=None,
    help='Root directory to search for metrics.log files (default: lightning_logs/ next to this script)',
)
group.add_argument(
    '--log', default=None,
    help='Path to a single metrics.log file',
)
args = parser.parse_args()

SCRIPT_DIR = Path(__file__).parent

if args.log:
    log_files = [Path(args.log)]
else:
    logs_root = Path(args.logs) if args.logs else SCRIPT_DIR / 'lightning_logs'
    if not logs_root.is_absolute():
        logs_root = SCRIPT_DIR / logs_root
    log_files = sorted(logs_root.rglob('metrics.log'))

if not log_files:
    print('No metrics.log files found.')
    raise SystemExit(1)

# --- Helpers -----------------------------------------------------------------
_line_re = re.compile(r'Epoch\s+(\d+)\s+([\w\-/]+):\s+([\d.eE+\-]+)')


def parse_log(path):
    """Return dict[metric_key -> dict[epoch -> float]]."""
    data = defaultdict(dict)
    for line in path.read_text().splitlines():
        m = _line_re.search(line)
        if m:
            epoch = int(m.group(1))
            key   = m.group(2)
            value = float(m.group(3))
            data[key][epoch] = value
    return data


def to_series(d):
    pairs = sorted(d.items())
    return [p[0] for p in pairs], [p[1] for p in pairs]


def plot_model(log_path, data):
    """Build and save a 4-panel figure; return the output path."""
    model_name = log_path.parent.name   # e.g. "fock_diagonal"

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(model_name, fontsize=14, fontweight='bold')
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    ax_loss = fig.add_subplot(gs[0, 0])
    ax_rmse = fig.add_subplot(gs[0, 1])
    ax_sub  = fig.add_subplot(gs[1, 0])
    ax_lr   = fig.add_subplot(gs[1, 1])

    # Infer metric prefix from logged keys (e.g. "fock_diag")
    rmse_keys = [k for k in data if k.startswith('train_') and k.endswith('_rmse')]
    prefix = rmse_keys[0][len('train_'):-len('_rmse')] if rmse_keys else ''

    # Loss
    for key, label, color in [
        ('val_loss',         'val loss',   'C0'),
        (f'train_{prefix}',  'train loss', 'C1'),
    ]:
        if key in data:
            ep, vals = to_series(data[key])
            ax_loss.semilogy(ep, vals, label=label, color=color)
    ax_loss.set_xlabel('Epoch')
    ax_loss.set_ylabel('MSE loss (Ha^2)')
    ax_loss.set_title('Loss')
    ax_loss.legend()
    ax_loss.grid(True, which='both', alpha=0.3)

    # Overall RMSE (eV)
    for key, label, color in [
        (f'train_{prefix}_rmse', 'train RMSE', 'C1'),
        (f'val_{prefix}_rmse',   'val RMSE',   'C0'),
    ]:
        if key in data:
            ep, vals = to_series(data[key])
            ax_rmse.semilogy(ep, [v * HA_TO_EV for v in vals], label=label, color=color)
    ax_rmse.set_xlabel('Epoch')
    ax_rmse.set_ylabel('RMSE (eV)')
    ax_rmse.set_title('Overall RMSE')
    ax_rmse.legend()
    ax_rmse.grid(True, which='both', alpha=0.3)

    # Per-subblock val RMSE (eV)
    subblock_keys = [
        (f'val_{prefix}_ss_rmse', 'ss (1s/2s)', 'C2'),
        (f'val_{prefix}_sp_rmse', 'sp',         'C3'),
        (f'val_{prefix}_pp_rmse', 'pp (pxp)',   'C4'),
    ]
    any_sub = False
    for key, label, color in subblock_keys:
        if key in data:
            ep, vals = to_series(data[key])
            ax_sub.semilogy(ep, [v * HA_TO_EV for v in vals], label=label, color=color)
            any_sub = True
    if any_sub:
        ax_sub.set_xlabel('Epoch')
        ax_sub.set_ylabel('RMSE (eV)')
        ax_sub.set_title('Val RMSE by subblock')
        ax_sub.legend()
        ax_sub.grid(True, which='both', alpha=0.3)
    else:
        ax_sub.set_visible(False)

    # Learning rate
    lr_keys = [k for k in data if k.startswith('lr')]
    for key in lr_keys:
        ep, vals = to_series(data[key])
        ax_lr.semilogy(ep, vals, label=key)
    ax_lr.set_xlabel('Epoch')
    ax_lr.set_ylabel('Learning rate')
    ax_lr.set_title('Learning rate schedule')
    ax_lr.legend()
    ax_lr.grid(True, which='both', alpha=0.3)

    out = log_path.parent / f'{model_name}_training.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out


# --- Main loop ---------------------------------------------------------------
for log_path in log_files:
    data = parse_log(log_path)
    if not data:
        print(f'  {log_path}: no epoch data, skipping')
        continue
    out = plot_model(log_path, data)
    epochs = max(max(v.keys()) for v in data.values()) + 1
    print(f'  {log_path.parent.name}: {epochs} epochs -> {out}')
