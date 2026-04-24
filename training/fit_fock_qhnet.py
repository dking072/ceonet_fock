"""
Train QHNet (from QHBench/QH9) on the same STO-3G dataset as fit_fock_ceonet.py,
using the same LightningTrainingTask, optimiser, and loss recipe so the two
models can be compared apples-to-apples at equal training budget.

Design
------
We keep QHNet's forward unchanged.  Around it we add:
  - a Data-object adapter that remaps field names (positions -> pos,
    atomic_numbers -> atoms) and re-embeds 5x5 STO-3G blocks into the
    14x14 def2-SVP slot layout QHNet's Expansion produces.
  - a light wrapper that renames QHNet's output dict keys to
    `pred_*` (so predictions do not overwrite target fields).
  - loss and metric modules that read QHNet's native 14x14 targets and
    masks (`diagonal_hamiltonian` / `non_diagonal_hamiltonian`).

Dependencies that QHNet's source imports (torch_cluster, torch_scatter)
are shimmed from torch_geometric, so a system without torch_cluster /
torch_scatter installed still runs.  For real cluster runs, install the
actual packages to get fast radius_graph / scatter kernels.

Usage
-----
    python fit_fock_qhnet.py
    python fit_fock_qhnet.py --pt path/to/data.pt --hidden-size 128 --num-layers 5

The script writes per-epoch metrics and the best model to
  lightning_logs/<--logs-name>/version_<n>/
"""

import os
import sys
import glob
import math
import types
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import lightning as L
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import scatter as _pyg_scatter

# ── shims for torch_cluster / torch_scatter ──────────────────────────────────
def _naive_radius_graph(x, r, batch=None, loop=False, max_num_neighbors=32,
                        flow='source_to_target', num_workers=1):
    """O(N^2) radius graph per batch segment — OK for small molecules."""
    device = x.device
    if batch is None:
        batch = torch.zeros(x.size(0), dtype=torch.long, device=device)
    dst_all, src_all = [], []
    for g in batch.unique(sorted=True).tolist():
        idx = (batch == g).nonzero(as_tuple=False).view(-1)
        pos = x[idx]
        d = torch.cdist(pos, pos)
        mask = d <= r
        if not loop:
            mask.fill_diagonal_(False)
        pair = mask.nonzero(as_tuple=False)
        dst_all.append(idx[pair[:, 0]])
        src_all.append(idx[pair[:, 1]])
    if not dst_all:
        return torch.zeros(2, 0, dtype=torch.long, device=device)
    return torch.stack([torch.cat(dst_all), torch.cat(src_all)], dim=0)


_tc = types.ModuleType('torch_cluster')
_tc.radius_graph = _naive_radius_graph
sys.modules.setdefault('torch_cluster', _tc)

_ts = types.ModuleType('torch_scatter')
_ts.scatter = _pyg_scatter
_ts.scatter_sum = lambda s, i, dim=0, dim_size=None: _pyg_scatter(
    s, i, dim=dim, dim_size=dim_size, reduce='sum')
sys.modules.setdefault('torch_scatter', _ts)

# QHBench must be importable. Override with env var for the cluster.
QHBENCH_ROOT = os.environ.get(
    'QHBENCH_ROOT', '/home/king1305/Apps/AIRS/OpenDFT/QHBench/QH9')
if QHBENCH_ROOT not in sys.path:
    sys.path.insert(0, QHBENCH_ROOT)

from models import QHNet  # noqa: E402
from cace.tasks import LightningTrainingTask  # noqa: E402


# ── STO-3G -> def2-SVP slot mapping used to pad 5x5 targets into 14x14 ──────
STO3G_SLOT_MAP = {
    1: torch.tensor([0],             dtype=torch.long),
    6: torch.tensor([0, 1, 3, 4, 5], dtype=torch.long),
    7: torch.tensor([0, 1, 3, 4, 5], dtype=torch.long),
    8: torch.tensor([0, 1, 3, 4, 5], dtype=torch.long),
    9: torch.tensor([0, 1, 3, 4, 5], dtype=torch.long),
}


def _canonical_full_edge_index(n):
    """src-major (dst, src) edge list matching our cut_matrix ordering."""
    dst_list, src_list = [], []
    for src in range(n):
        for dst in range(n):
            if dst != src:
                dst_list.append(dst)
                src_list.append(src)
    return torch.tensor([dst_list, src_list], dtype=torch.long)


class QM9FockQHNetDataset(Dataset):
    """
    Wraps the preprocessed STO-3G QM9 .pt dataset and emits each sample
    in the field names + block layout QHNet expects:

      pos                              (N, 3)       positions in Bohr
      atoms                            (N, 1)       atomic numbers
      num_nodes                        scalar
      edge_index_full                  (2, N*(N-1)) src-major [dst, src]
      diagonal_hamiltonian             (N, 14, 14)  STO-3G block in def2-SVP slots
      non_diagonal_hamiltonian         (N*(N-1), 14, 14)
      diagonal_hamiltonian_mask        same shapes, float
      non_diagonal_hamiltonian_mask    same shapes, float

    STO-3G AOs [1s, 2s, 2px, 2py, 2pz] are placed at def2-SVP slots
    [0, 1, 3, 4, 5] for heavy atoms; H's single 1s goes at slot [0].
    """

    def __init__(self, pt_path):
        super().__init__(str(Path(pt_path).parent))
        self.pt_path = pt_path
        self.raw = torch.load(pt_path, weights_only=False)

    def len(self):
        return len(self.raw)

    def get(self, idx):
        src = self.raw[idx]
        N = int(src.num_nodes)
        Zs = src.atomic_numbers.tolist()

        diag_h = torch.zeros(N, 14, 14, dtype=torch.float32)
        diag_m = torch.zeros(N, 14, 14, dtype=torch.float32)
        for i, Z in enumerate(Zs):
            slots = STO3G_SLOT_MAP[int(Z)]
            n = slots.numel()
            sub = src.hamiltonian_diagonal_blocks[i, :n, :n]
            diag_h[i, slots.unsqueeze(1), slots.unsqueeze(0)] = sub
            diag_m[i, slots.unsqueeze(1), slots.unsqueeze(0)] = 1.0

        edge_index_full = _canonical_full_edge_index(N)
        E = edge_index_full.shape[1]
        off_h = torch.zeros(E, 14, 14, dtype=torch.float32)
        off_m = torch.zeros(E, 14, 14, dtype=torch.float32)

        k = 0
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                slots_row = STO3G_SLOT_MAP[int(Zs[j])]
                slots_col = STO3G_SLOT_MAP[int(Zs[i])]
                n_r = slots_row.numel()
                n_c = slots_col.numel()
                sub = src.hamiltonian_non_diagonal_blocks[k, :n_r, :n_c]
                off_h[k, slots_row.unsqueeze(1), slots_col.unsqueeze(0)] = sub
                off_m[k, slots_row.unsqueeze(1), slots_col.unsqueeze(0)] = 1.0
                k += 1

        return Data(
            pos=src.positions.to(torch.float32),
            atoms=src.atomic_numbers.view(-1, 1).to(torch.long),
            num_nodes=N,
            edge_index_full=edge_index_full,
            diagonal_hamiltonian=diag_h,
            non_diagonal_hamiltonian=off_h,
            diagonal_hamiltonian_mask=diag_m,
            non_diagonal_hamiltonian_mask=off_m,
        )


class QHNetDataModule(L.LightningDataModule):
    def __init__(self, pt_path, batch_size=5, valid_p=0.05, test_p=0.05, seed=12345):
        super().__init__()
        self.pt_path    = pt_path
        self.batch_size = batch_size
        self.valid_p    = valid_p
        self.test_p     = test_p
        self.seed       = seed
        self.prepare_data()

    def prepare_data(self):
        ds = QM9FockQHNetDataset(self.pt_path)
        n_all = len(ds)
        idx = torch.randperm(n_all, generator=torch.Generator().manual_seed(self.seed)).tolist()
        n_val  = max(1, int(self.valid_p * n_all))
        n_test = max(1, int(self.test_p  * n_all))
        n_train = n_all - n_val - n_test
        self.train = [ds[i] for i in idx[:n_train]]
        self.val   = [ds[i] for i in idx[n_train:n_train + n_val]]
        self.test  = [ds[i] for i in idx[n_train + n_val:]]

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True,  drop_last=True)
    def val_dataloader(self):
        return DataLoader(self.val,   batch_size=self.batch_size, shuffle=False)
    def test_dataloader(self):
        return DataLoader(self.test,  batch_size=self.batch_size, shuffle=False)


# ── QHNet wrapper: rename outputs + swallow extra kwargs from Lightning ──────
class QHNetLightning(nn.Module):
    def __init__(self, qhnet: QHNet):
        super().__init__()
        self.qhnet = qhnet

    def forward(self, data, **kwargs):
        out = self.qhnet(data, keep_blocks=True)
        return {
            'pred_hamiltonian_diagonal_blocks':     out['hamiltonian_diagonal_blocks'],
            'pred_hamiltonian_non_diagonal_blocks': out['hamiltonian_non_diagonal_blocks'],
        }


# ── Losses / metrics (14x14 masked, MSE+MAE, RMSE/MAE in eV) ─────────────────
HA_TO_EV = 27.2114


class MaskedFock14Loss(nn.Module):
    """MSE+MAE on a named 14x14 (predict, target, mask) triple."""
    def __init__(self, predict_name, target_name, mask_name,
                 loss_weight=1.0, mae_weight=1.0, name='fock'):
        super().__init__()
        self.predict_name = predict_name
        self.target_name  = target_name
        self.mask_name    = mask_name
        self.loss_weight  = loss_weight
        self.mae_weight   = mae_weight
        self.name         = name

    def forward(self, pred, target, **kw):
        p = pred[self.predict_name]
        t = target[self.target_name]
        m = target[self.mask_name].bool()
        d = p[m] - t[m]
        return self.loss_weight * ((d ** 2).mean() + self.mae_weight * d.abs().mean())


class MaskedFock14Metrics(nn.Module):
    """RMSE + MAE in eV on a named 14x14 (predict, target, mask) triple."""
    def __init__(self, predict_name, target_name, mask_name, name='fock'):
        super().__init__()
        self.predict_name = predict_name
        self.target_name  = target_name
        self.mask_name    = mask_name
        self.name         = name

    def forward(self, pred, target, **kw):
        p = pred[self.predict_name].detach()
        t = target[self.target_name].detach()
        m = target[self.mask_name].bool()
        d = (p - t)[m] * HA_TO_EV
        return {'rmse': torch.sqrt((d ** 2).mean()), 'mae': d.abs().mean()}


# ── driver ───────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    on_cluster = 'SLURM_JOB_CPUS_PER_NODE' in os.environ
    default_pt = (
        '/global/scratch/users/king1305/data/qm9_sto3g_1k.pt' if on_cluster
        else '/home/king1305/AI-WORKSPACES/fock-predict/ceonet_fock/datasets/qm9_sto3g_1k.pt'
    )
    p.add_argument('--pt',            default=default_pt)
    p.add_argument('--batch-size',    type=int,   default=5)
    p.add_argument('--max-epochs',    type=int,   default=1000)
    p.add_argument('--lr',            type=float, default=5e-4)
    p.add_argument('--hidden-size',   type=int,   default=128)
    p.add_argument('--bottle-size',   type=int,   default=32)
    p.add_argument('--num-layers',    type=int,   default=5,
                   help='QHNet ConvNet layers (>=4 required: self/pair nets fire at layer_idx > 2)')
    p.add_argument('--sh-lmax',       type=int,   default=4)
    p.add_argument('--max-radius',    type=float, default=15.0,
                   help='QHNet trunk cutoff in Bohr')
    p.add_argument('--diag-loss-weight',    type=float, default=1.0)
    p.add_argument('--offdiag-loss-weight', type=float, default=1.0)
    p.add_argument('--logs-name', default='fock_qhnet')
    args = p.parse_args()

    # ── data ───────────────────────────────────────────────────────────────
    data = QHNetDataModule(args.pt, batch_size=args.batch_size)

    # ── model ──────────────────────────────────────────────────────────────
    qhnet = QHNet(
        in_node_features=1,
        sh_lmax=args.sh_lmax,
        hidden_size=args.hidden_size,
        bottle_hidden_size=args.bottle_size,
        num_gnn_layers=args.num_layers,
        max_radius=args.max_radius,
        num_nodes=10,
        radius_embed_dim=16,
    )
    model = QHNetLightning(qhnet)

    # Warm up on one batch (no lazy layers in QHNet, but this also puts the
    # model on GPU and validates the shapes end-to-end).
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    with torch.no_grad():
        batch = next(iter(data.train_dataloader())).to(device)
        out = model(batch)
        print("Model output keys:", list(out.keys()))
        print("  diag blocks   :", tuple(out['pred_hamiltonian_diagonal_blocks'].shape))
        print("  offdiag blocks:", tuple(out['pred_hamiltonian_non_diagonal_blocks'].shape))

    # ── losses & metrics ───────────────────────────────────────────────────
    losses = [
        MaskedFock14Loss(
            predict_name='pred_hamiltonian_diagonal_blocks',
            target_name='diagonal_hamiltonian',
            mask_name='diagonal_hamiltonian_mask',
            loss_weight=args.diag_loss_weight,
            name='fock_diag',
        ),
        MaskedFock14Loss(
            predict_name='pred_hamiltonian_non_diagonal_blocks',
            target_name='non_diagonal_hamiltonian',
            mask_name='non_diagonal_hamiltonian_mask',
            loss_weight=args.offdiag_loss_weight,
            name='fock_offdiag',
        ),
    ]
    metrics = [
        MaskedFock14Metrics(
            predict_name='pred_hamiltonian_diagonal_blocks',
            target_name='diagonal_hamiltonian',
            mask_name='diagonal_hamiltonian_mask',
            name='fock_diag',
        ),
        MaskedFock14Metrics(
            predict_name='pred_hamiltonian_non_diagonal_blocks',
            target_name='non_diagonal_hamiltonian',
            mask_name='non_diagonal_hamiltonian_mask',
            name='fock_offdiag',
        ),
    ]

    # ── resume from checkpoint if present ──────────────────────────────────
    chkpt = None
    if os.path.isdir(f'lightning_logs/{args.logs_name}'):
        num = 0
        latest = None
        while os.path.isdir(f'lightning_logs/{args.logs_name}/version_{num}'):
            latest = f'lightning_logs/{args.logs_name}/version_{num}'
            num += 1
        if latest:
            c = glob.glob(f'{latest}/checkpoints/*.ckpt')
            if c:
                chkpt = c[0]
    if chkpt:
        print(f'Resuming from checkpoint: {chkpt}')

    # ── fit ────────────────────────────────────────────────────────────────
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
    task.fit(data, max_epochs=args.max_epochs, chkpt=chkpt, progress_bar=progress_bar)


if __name__ == '__main__':
    main()
