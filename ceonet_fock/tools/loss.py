from typing import Dict
import torch
import torch.nn as nn

__all__ = ['MaskedFockLoss']


class MaskedFockLoss(nn.Module):
    """
    MSE loss over valid AO entries, gated by a block mask.

    H atoms contribute only the (0,0) entry; C/N/O contribute all 25 entries.
    The mask (hamiltonian_diagonal_block_masks) encodes which entries are
    physically meaningful for each atom, so loss is not diluted by padding zeros.
    """

    def __init__(self,
                 predict_name: str = 'hamiltonian_diagonal_blocks',
                 target_name:  str = 'hamiltonian_diagonal_blocks',
                 mask_name:    str = 'hamiltonian_diagonal_block_masks',
                 loss_weight:  float = 1.0):
        super().__init__()
        self.predict_name = predict_name
        self.target_name  = target_name
        self.mask_name    = mask_name
        self.loss_weight  = loss_weight
        self.name         = 'fock_diag'

    def forward(self, pred: Dict[str, torch.Tensor],
                target: Dict[str, torch.Tensor],
                **kwargs) -> torch.Tensor:
        p    = pred[self.predict_name]           # (N, 5, 5)
        t    = target[self.target_name]          # (N, 5, 5)
        mask = target[self.mask_name].bool()     # (N, 5, 5)
        diff = p[mask] - t[mask]
        return self.loss_weight * (diff ** 2).mean()
