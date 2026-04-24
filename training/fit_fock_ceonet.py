"""
Train CEONet + FockDiagonalReadout + FockOffDiagonalReadout on STO-3G
Fock matrices (e.g. qm9_sto3g_1k.pt) via the Lightning trainer.

Usage
-----
    python fit_fock_ceonet.py
    python fit_fock_ceonet.py --pt path/to/data.pt --layers 4 --nc 64

The script fits both diagonal and off-diagonal 5x5 blocks:

  diagonal:
    FockDiagonalReadout with per-atom-type weights + bias (closes the
    element-agnostic readout gap identified in the CEONet/QHNet
    comparison experiments).

  off-diagonal:
    FockOffDiagonalReadout with a PolynomialCutoff-gated BesselRBF and
    per-TP-path radial weighting (RadialTensorProductLayer) so Fock
    couplings decay smoothly with distance.

Loss: MSE + MAE on each block (via MaskedFockLoss), summed.
Metrics (reported in eV by MaskedFockMetrics): per-block RMSE / MAE.
"""

import os
import glob
import argparse
import torch

import cace
from cace.representations import CEONet
from cace.modules import BesselRBF, PolynomialCutoff
from cace.models.atomistic import NeuralNetworkPotential
from cace.tasks import LightningTrainingTask

from ceonet_fock.modules import (
    FockDiagonalReadout, FockOffDiagonalReadout, CEONetWithAtomicNumbers,
)
from ceonet_fock.data import QM9FockData
from ceonet_fock.tools import MaskedFockLoss, MaskedFockMetrics

# ── CLI ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
_on_cluster = 'SLURM_JOB_CPUS_PER_NODE' in os.environ
_default_pt = (
    '/global/scratch/users/king1305/data/qm9_sto3g_1k.pt' if _on_cluster
    else '/home/king1305/AI-WORKSPACES/fock-predict/ceonet_fock/datasets/qm9_sto3g_1k.pt'
)
parser.add_argument('--pt',           default=_default_pt,
                    help='Path to preprocessed .pt dataset')
parser.add_argument('--cutoff',       type=float, default=8.5,
                    help='CEONet trunk cutoff in Bohr')
parser.add_argument('--pair-cutoff',  type=float, default=15.0,
                    help='FockOffDiagonalReadout cutoff in Bohr (must cover typical pair distances)')
parser.add_argument('--batch-size',   type=int,   default=5)
parser.add_argument('--max-epochs',   type=int,   default=1000)
parser.add_argument('--lr',           type=float, default=1e-3)
parser.add_argument('--nc',           type=int,   default=32, help='CEONet trunk channel width')
parser.add_argument('--layers',       type=int,   default=2,  help='CEONet MP layers')
parser.add_argument('--readout-channels', type=int, default=16,
                    help='n_channel inside the Fock readouts')
parser.add_argument('--diag-loss-weight',    type=float, default=1.0)
parser.add_argument('--offdiag-loss-weight', type=float, default=1.0)
parser.add_argument('--logs-name', default='fock_ceonet')
args = parser.parse_args()

cutoff      = args.cutoff
pair_cutoff = args.pair_cutoff
batch_size  = args.batch_size

on_cluster  = _on_cluster

# ── Data ─────────────────────────────────────────────────────────────────────
data = QM9FockData(args.pt, cutoff=cutoff, batch_size=batch_size)

# ── Representation ────────────────────────────────────────────────────────────
radial_basis = BesselRBF(cutoff=cutoff, n_rbf=6, trainable=True)
cutoff_fn    = PolynomialCutoff(cutoff=cutoff)

ceonet = CEONet(
    zs=[1, 6, 7, 8, 9],
    n_atom_basis=4,
    cutoff=cutoff,
    radial_basis=radial_basis,
    cutoff_fn=cutoff_fn,
    max_l_cace=3,
    max_l_ceonet=2,
    max_nu_cace=3,
    n_radial_basis=12,
    nc=args.nc,
    layers=args.layers,
    avg_neighbors=4,
    stacking=False,
)
# Wrap so atomic_numbers survives into the readouts (FockDiagonalReadout needs
# Z to apply per-atom readout weights / biases).
representation = CEONetWithAtomicNumbers(ceonet)

# ── Output modules ────────────────────────────────────────────────────────────
fock_diag_readout = FockDiagonalReadout(
    feature_key='node_feats_l',
    output_key='pred_hamiltonian_diagonal_blocks',
    n_channel=args.readout_channels,
    use_feed_forward=True,
    atom_bias=True,
    atom_weights=True,
    n_elements=10,
)

fock_offdiag_readout = FockOffDiagonalReadout(
    feature_key='node_feats_l',
    output_key='pred_hamiltonian_non_diagonal_blocks',
    n_channel=args.readout_channels,
    n_rbf=8,
    lomax=2,
    linmax=2,
    use_feed_forward=True,
    linear_messages=True,
    cutoff=pair_cutoff,
)

# ── Model ─────────────────────────────────────────────────────────────────────
model = NeuralNetworkPotential(
    representation=representation,
    output_modules=[fock_diag_readout, fock_offdiag_readout],
)

# ── Losses and metrics ────────────────────────────────────────────────────────
losses = [
    MaskedFockLoss(
        predict_name='pred_hamiltonian_diagonal_blocks',
        target_name='hamiltonian_diagonal_blocks',
        mask_name='hamiltonian_diagonal_block_masks',
        loss_weight=args.diag_loss_weight,
    ),
    MaskedFockLoss(
        predict_name='pred_hamiltonian_non_diagonal_blocks',
        target_name='hamiltonian_non_diagonal_blocks',
        mask_name='hamiltonian_non_diagonal_block_masks',
        loss_weight=args.offdiag_loss_weight,
    ),
]
# Give the two losses distinct names so Lightning can distinguish their logs.
losses[0].name = 'fock_diag'
losses[1].name = 'fock_offdiag'

metrics = [
    MaskedFockMetrics(
        predict_name='pred_hamiltonian_diagonal_blocks',
        target_name='hamiltonian_diagonal_blocks',
        mask_name='hamiltonian_diagonal_block_masks',
        name='fock_diag',
    ),
    MaskedFockMetrics(
        predict_name='pred_hamiltonian_non_diagonal_blocks',
        target_name='hamiltonian_non_diagonal_blocks',
        mask_name='hamiltonian_non_diagonal_block_masks',
        name='fock_offdiag',
    ),
]

# ── Initialise lazy layers ────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
with torch.no_grad():
    for batch in data.train_dataloader():
        batch = batch.to(device)
        out = model(batch)
        print("Model output keys:", list(out.keys()))
        print("  diag blocks   :", tuple(out['pred_hamiltonian_diagonal_blocks'].shape))
        print("  offdiag blocks:", tuple(out['pred_hamiltonian_non_diagonal_blocks'].shape))
        break

# ── Resume from checkpoint if available ──────────────────────────────────────
chkpt = None
dev_run = False
if os.path.isdir(f'lightning_logs/{args.logs_name}'):
    num = 0
    latest = None
    while os.path.isdir(f'lightning_logs/{args.logs_name}/version_{num}'):
        latest = f'lightning_logs/{args.logs_name}/version_{num}'
        num += 1
    if latest:
        chkpts = glob.glob(f'{latest}/checkpoints/*.ckpt')
        if chkpts:
            chkpt = chkpts[0]
if chkpt:
    print(f'Resuming from checkpoint: {chkpt}')

# ── Train ─────────────────────────────────────────────────────────────────────
progress_bar = not on_cluster
task = LightningTrainingTask(
    model,
    losses=losses,
    metrics=metrics,
    save_pkl=True,
    logs_directory='lightning_logs',
    name=args.logs_name,
    scheduler_args={'mode': 'min', 'factor': 0.8, 'patience': 10},
    optimizer_args={'lr': args.lr},
)
task.fit(data, dev_run=dev_run, max_epochs=args.max_epochs,
         chkpt=chkpt, progress_bar=progress_bar)
