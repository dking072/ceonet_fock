import os
import torch
import numpy as np
import lightning as L
from pathlib import Path
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader

from cace.data.neighborhood import get_neighborhood

from .dataset import mol_to_data


def from_molden(path):
    """Convert a single Molden file to a PyG Data object. Analogous to from_xyz."""
    from pyscf.tools import molden
    mol, mo_energy, mo_coeff, _mo_occ, _sym, _uhf = molden.load(str(path))
    mol.build()
    return mol_to_data(mol, mo_energy, mo_coeff, blocks_only=True)


# Angstroms
class QM9FockDataset(Dataset):
    def __init__(self, pt_path, cutoff=8.5, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(str(Path(pt_path).parent), transform, pre_transform, pre_filter)
        self.pt_path = pt_path
        self.cutoff  = cutoff
        self.prepare_data()

    def prepare_data(self):
        self.dataset = torch.load(self.pt_path, weights_only=False)

    def len(self):
        return len(self.dataset)

    def get(self, idx):
        data = self.dataset[idx]
        if self.cutoff is not None:
            data = self._add_edges(data)
        return data

    def _add_edges(self, data):
        """
        Build neighbour list via the ASE/matscipy backend (same as AtomicData.from_atoms).
        Positions are in Bohr; cutoff must be in the same units.
        For isolated molecules pbc=(False,False,False) and cell is ignored.
        """
        positions_np = data.positions.numpy()
        edge_index, shifts, unit_shifts = get_neighborhood(
            positions=positions_np,
            cutoff=self.cutoff,
        )
        dtype = data.positions.dtype
        data.edge_index  = torch.tensor(edge_index,   dtype=torch.long)
        data.shifts      = torch.tensor(shifts,        dtype=dtype)
        data.unit_shifts = torch.tensor(unit_shifts,   dtype=dtype)
        data.cell        = torch.zeros(3, 3,           dtype=dtype)
        return data


class QM9FockData(L.LightningDataModule):
    def __init__(self, pt_path, cutoff=8.5, batch_size=32, drop_last=True, shuffle=True,
                 valid_p=0.05, test_p=0.05):
        super().__init__()
        self.pt_path    = pt_path
        self.cutoff     = cutoff
        self.batch_size = batch_size
        self.valid_p    = valid_p
        self.test_p     = test_p
        self.drop_last  = drop_last
        self.shuffle    = shuffle
        self.num_cpus = 0
        self.prepare_data()

    def prepare_data(self):
        dataset = QM9FockDataset(self.pt_path, cutoff=self.cutoff)
        torch.manual_seed(12345)
        if self.shuffle:
            dataset = dataset.shuffle()
        cut1 = int(len(dataset) * (1 - self.valid_p - self.test_p))
        cut2 = int(len(dataset) * (1 - self.test_p))
        self.train = dataset[:cut1]
        self.val   = dataset[cut1:cut2]
        self.test  = dataset[cut2:]

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, drop_last=self.drop_last,
                          shuffle=True, num_workers=self.num_cpus)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, drop_last=False,
                          shuffle=False, num_workers=self.num_cpus)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, drop_last=False,
                          shuffle=False, num_workers=self.num_cpus)
