from typing import Dict, Sequence
import torch
import torch.nn as nn

__all__ = ['MaskedFockMetrics']

HA_TO_EV = 27.2114


class MaskedFockMetrics(nn.Module):
    """
    RMSE and MAE over valid AO entries, reported in eV, with per-subblock breakdowns.

    Subblock labels (STO-3G AO order: 1s, 2s, px, py, pz):
      ss  — 1s/2s rows and columns (AO indices 0, 1)
      sp  — off-diagonal s/p coupling entries
      pp  — px/py/pz rows and columns (AO indices 2, 3, 4)

    The mask (hamiltonian_diagonal_block_masks) is AND-ed with each subblock
    selector so that H-atom padding zeros are never included.
    """

    def __init__(self,
                 predict_name: str = 'pred_hamiltonian_diagonal_blocks',
                 target_name:  str = 'hamiltonian_diagonal_blocks',
                 mask_name:    str = 'hamiltonian_diagonal_block_masks',
                 name: str = 'fock_diag',
                 metric_keys: Sequence[str] = ('rmse', 'mae')):
        super().__init__()
        self.predict_name = predict_name
        self.target_name  = target_name
        self.mask_name    = mask_name
        self.name         = name
        self.metric_keys  = list(metric_keys)

    @staticmethod
    def _rmse(diff: torch.Tensor) -> torch.Tensor:
        return torch.sqrt((diff ** 2).mean())

    @staticmethod
    def _mae(diff: torch.Tensor) -> torch.Tensor:
        return diff.abs().mean()

    def _compute(self, diff: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = {}
        if 'rmse' in self.metric_keys:
            out['rmse'] = self._rmse(diff)
        if 'mae' in self.metric_keys:
            out['mae']  = self._mae(diff)
        return out

    def forward(self, pred: Dict[str, torch.Tensor],
                target: Dict[str, torch.Tensor],
                **kwargs) -> Dict[str, torch.Tensor]:
        p    = pred[self.predict_name].detach()   # (N, 5, 5)
        t    = target[self.target_name].detach()  # (N, 5, 5)
        mask = target[self.mask_name].bool()      # (N, 5, 5)

        diff = (p - t) * HA_TO_EV
        metrics = self._compute(diff[mask])

        # Per-subblock breakdown
        ss_idx = [0, 1]
        pp_idx = [2, 3, 4]
        ss_mask = mask.clone(); ss_mask[:, pp_idx, :] = False; ss_mask[:, :, pp_idx] = False
        pp_mask = mask.clone(); pp_mask[:, ss_idx, :] = False; pp_mask[:, :, ss_idx] = False
        sp_mask = mask & ~ss_mask & ~pp_mask

        for tag, sub_mask in [('ss', ss_mask), ('sp', sp_mask), ('pp', pp_mask)]:
            if sub_mask.any():
                for k, v in self._compute(diff[sub_mask]).items():
                    metrics[f'{tag}_{k}'] = v

        return metrics
