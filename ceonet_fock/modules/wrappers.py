"""
Small representation wrappers that augment the return dict of a CACE
representation with fields the base class drops.

``CEONet.forward`` returns only {positions, cell, batch, node_feats,
node_feats_l, displacement} — ``atomic_numbers`` is not in the output
dict, so downstream readouts cannot look up element type.
``CEONetWithAtomicNumbers`` puts it back.
"""

from typing import Dict
import torch
from torch import nn


__all__ = ['CEONetWithAtomicNumbers']


class CEONetWithAtomicNumbers(nn.Module):
    """Preserve ``atomic_numbers`` in the representation output dict.

    Also forwards any ``cutoff`` attribute the underlying module exposes,
    which ``NeuralNetworkPotential`` reads when assembling the model.
    """

    def __init__(self, ceonet: nn.Module):
        super().__init__()
        self.ceonet = ceonet

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        Z = data['atomic_numbers']
        out = self.ceonet(data)
        out['atomic_numbers'] = Z
        return out

    # Pass-through attributes that some callers (e.g. NeuralNetworkPotential)
    # may look up on the representation directly.
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.ceonet, name)
