"""
Training script: predict STO-3G diagonal Fock blocks with CEONet + FockDiagonalReadout.

Usage:
    python fit_fock_diagonal.py
    python fit_fock_diagonal.py --pt path/to/data.pt --cutoff 5.0 --batch-size 16
"""

import os
import glob
import argparse
import torch

import cace
from cace.representations import CEONet
from cace.modules import BesselRBF, PolynomialCutoff
from ceonet_fock.modules import FockDiagonalReadout
from cace.models.atomistic import NeuralNetworkPotential
from cace.tasks import LightningTrainingTask

from ceonet_fock.data import QM9FockData
from ceonet_fock.tools import MaskedFockLoss, MaskedFockMetrics

# ── CLI ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
_on_cluster = 'SLURM_JOB_CPUS_PER_NODE' in os.environ
_default_pt = (
    '/global/scratch/users/king1305/data/qm9_sto3g_1k.pt' if _on_cluster
    else '/home/king1305/AI-WORKSPACES/fock-predict/ceonet_fock/datasets/qm9_sto3g_1k.pt'
)
parser.add_argument('--pt', default=_default_pt,
                    help='Path to preprocessed .pt dataset')
parser.add_argument('--cutoff',     type=float, default=8.5,
                    help='Neighbour cutoff in Bohr')
parser.add_argument('--batch-size', type=int,   default=5)
parser.add_argument('--max-epochs', type=int,   default=1000)
parser.add_argument('--lr',         type=float, default=1e-3)
args = parser.parse_args()

cutoff     = args.cutoff
batch_size = args.batch_size

on_cluster = 'SLURM_JOB_CPUS_PER_NODE' in os.environ

# ── Data ─────────────────────────────────────────────────────────────────────
# cutoff is passed here so the dataset builds edge_index per molecule at load time
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
    nc=32,
    layers=2,
    avg_neighbors=4,
    stacking=False,
)

# ── Output module ─────────────────────────────────────────────────────────────
fock_readout = FockDiagonalReadout(
    feature_key='node_feats_l',
    output_key='hamiltonian_diagonal_blocks',
    n_channel=16,
    use_feed_forward=True,
)

# ── Model ─────────────────────────────────────────────────────────────────────
model = NeuralNetworkPotential(
    representation=ceonet,
    output_modules=[fock_readout],
)

# ── Loss and metrics ──────────────────────────────────────────────────────────
losses  = [MaskedFockLoss(loss_weight=1.0)]
metrics = [MaskedFockMetrics()]

# ── Initialise lazy layers ────────────────────────────────────────────────────
model.cuda()
for batch in data.train_dataloader():
    batch = batch.cuda()
    out = model(batch)
    print("Model output keys:", list(out.keys()))
    print("Predicted blocks shape:", out['hamiltonian_diagonal_blocks'].shape)
    break

# ── Resume from checkpoint if available ──────────────────────────────────────
logs_name = 'fock_diagonal'
chkpt     = None
dev_run   = False
if os.path.isdir(f'lightning_logs/{logs_name}'):
    num = 0
    latest = None
    while os.path.isdir(f'lightning_logs/{logs_name}/version_{num}'):
        latest = f'lightning_logs/{logs_name}/version_{num}'
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
    name=logs_name,
    scheduler_args={'mode': 'min', 'factor': 0.8, 'patience': 10},
    optimizer_args={'lr': args.lr},
)
task.fit(data, dev_run=dev_run, max_epochs=args.max_epochs,
         chkpt=chkpt, progress_bar=progress_bar)
