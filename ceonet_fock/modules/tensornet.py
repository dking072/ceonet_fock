"""
Verbatim copy of cace/modules/tensornet.py, plus `RadialTensorProductLayer`
which weights each (x_way, y_way, z_way) path by a learnable projection of
the edge RBF.  This lets each angular TP path learn its own radial profile
— the key expressive advantage of QHNet's fc_node_pair(rbf) → TP-weights
pattern over a single post-TP multiplicative gate.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, List, Callable, Tuple, Union

from .tensornet_utils import expand_to, _aggregate_new

class TensorLinearMixing(nn.Module):
    def __init__(self,
                 n_out : int,
                 lomax : int,
                 ) -> None:
        super().__init__()
        self.linear_list = nn.ModuleList([
            nn.LazyLinear(n_out, bias=False) for l in range(lomax + 1)
        ])

    def forward(self,
                input_tensors : Dict[int, torch.Tensor],
                ) -> Dict[int, torch.Tensor]:
        output_tensors = {}
        for l, linear in enumerate(self.linear_list):
            input_tensor = torch.transpose(input_tensors[l], 1, -1)
            output_tensor = linear(input_tensor)
            output_tensors[l] = torch.transpose(output_tensor, 1, -1)
        return output_tensors

class TensorProductLayer(nn.Module):
    def __init__(self, nc,
                 max_x_way      : int=2,
                 max_y_way      : int=2,
                 max_z_way      : int=2,
                 zstack         : bool=False,
                 stacking       : bool=False,
                 ) -> None:
        #lin, lr, lout
        super().__init__()
        self.stacking = stacking
        self.zstack = zstack
        self.combinations = []
        for x_way in range(max_x_way + 1):
            for y_way in range(max_y_way + 1):
                for z_way in range(abs(y_way - x_way), min(max_z_way, x_way + y_way) + 1, 2):
                    self.combinations.append((x_way, y_way, z_way))

    def forward(self,
                x : Dict[int, torch.Tensor],
                y : Dict[int, torch.Tensor],
                ) -> Dict[int, torch.Tensor]:
        output_tensors = {}
        for x_way, y_way, z_way in self.combinations:
            if x_way not in x or y_way not in y:
                continue
            output_tensor = _aggregate_new(x[x_way], y[y_way], x_way, y_way, z_way)
            if z_way not in output_tensors:
                output_tensors[z_way] = output_tensor
            else:
                if self.stacking:
                    output_tensors[z_way] = torch.hstack([output_tensors[z_way],output_tensor])
                elif self.zstack and (z_way == 0):
                    output_tensors[z_way] = torch.hstack([output_tensors[z_way],output_tensor])
                else:
                    output_tensors[z_way] += output_tensor
        return output_tensors

class TensorActivationGate(nn.Module):
    def __init__(self,l_out_list:List[int]) -> None:
        super().__init__()
        self.lomax = len(l_out_list) - 1
        self.net0 = nn.Sequential(nn.LazyLinear(l_out_list[0],bias=True),nn.SiLU())
        self.norm_net_list = nn.ModuleList([
            nn.Sequential(nn.LazyLinear(nc,bias=True),nn.Sigmoid()) for nc in l_out_list[1:]
        ])

    def forward(self,input_tensors : Dict[int, torch.Tensor]) -> Dict[int, torch.Tensor]:
        #Make mlp feed
        mlp_feed = [input_tensors[0]]
        for l in range(1,self.lomax+1):
            input_tensor_ = input_tensors[l].reshape(input_tensors[l].shape[0], input_tensors[l].shape[1], -1)
            norm = torch.sum(input_tensor_ ** 2, dim=2)
            mlp_feed.append(norm)
        mlp_feed = torch.hstack(mlp_feed)

        output_tensors = {}
        output_tensors[0] = self.net0(mlp_feed)
        for l in range(1,self.lomax+1):
            mlp_out = self.norm_net_list[l-1](mlp_feed)
            output_tensors[l] = input_tensors[l] * expand_to(mlp_out,l+2)
        return output_tensors

class TensorFeedForward(nn.Module):
    def __init__(self,nc,lomax) -> None:
        super().__init__()
        self.lomax = lomax
        self.mix1 = TensorLinearMixing(nc,lomax)
        self.gate = TensorActivationGate([nc]*(lomax+1))
        self.mix2 = TensorLinearMixing(nc,lomax)

    def forward(self,input_tensors : Dict[int, torch.Tensor]) -> Dict[int, torch.Tensor]:
        output_tensors = self.mix1(input_tensors)
        output_tensors = self.gate(output_tensors)
        output_tensors = self.mix2(output_tensors)
        return output_tensors


class RadialTensorProductLayer(nn.Module):
    """Tensor product with per-path, per-channel radial weighting.

    For each (x_way, y_way, z_way) path, the aggregate is multiplied by a
    learnable scalar-per-channel that is a linear projection of the edge
    RBF.  This lets different angular paths learn different radial
    envelopes, matching QHNet's `fc_node_pair(rbf) -> TP-path weights`
    pattern.  It is a drop-in replacement for `TensorProductLayer`
    whenever an edge vector (direction × distance) is one of the TP
    inputs.

    Compared to the post-TP ``rbf_mixing_list`` gate used in CEONet's
    ``MessagePassingLayer``:
      - Here, each path produces a *radially modulated* tensor before
        being summed into the z_way output.  Different paths landing in
        the same z can therefore carry different radial profiles.
      - The post-gate scales the *combined* z output, so all paths
        landing in z share one radial profile per channel.
    """

    def __init__(self,
                 nc         : int,
                 n_rbf      : int,
                 max_x_way  : int = 2,
                 max_y_way  : int = 2,
                 max_z_way  : int = 2,
                 stacking   : bool = False,
                 ) -> None:
        super().__init__()
        self.nc = nc
        self.stacking = stacking
        self.combinations = []
        for x_way in range(max_x_way + 1):
            for y_way in range(max_y_way + 1):
                for z_way in range(abs(y_way - x_way),
                                   min(max_z_way, x_way + y_way) + 1, 2):
                    self.combinations.append((x_way, y_way, z_way))
        self.n_paths = len(self.combinations)
        # RBF -> (n_paths * nc) scalar weights; one per (path, channel).
        # bias=False so that rbf=0 (beyond cutoff) zeros the TP contribution.
        self.rbf_to_weight = nn.Linear(n_rbf, self.n_paths * nc, bias=False)

    def forward(self,
                x   : Dict[int, torch.Tensor],
                y   : Dict[int, torch.Tensor],
                rbf : torch.Tensor,
                ) -> Dict[int, torch.Tensor]:
        # w: (n_edges, n_paths, nc)
        w = self.rbf_to_weight(rbf).view(-1, self.n_paths, self.nc)
        output_tensors = {}
        for p, (x_way, y_way, z_way) in enumerate(self.combinations):
            if x_way not in x or y_way not in y:
                continue
            agg = _aggregate_new(x[x_way], y[y_way], x_way, y_way, z_way)
            # agg: (n_edges, nc, *[3]*z_way). Multiply by per-channel radial weight.
            w_p = expand_to(w[:, p, :], z_way + 2)  # (n_edges, nc, 1, ..., 1)
            agg = agg * w_p
            if z_way not in output_tensors:
                output_tensors[z_way] = agg
            else:
                if self.stacking:
                    output_tensors[z_way] = torch.hstack([output_tensors[z_way], agg])
                else:
                    output_tensors[z_way] = output_tensors[z_way] + agg
        return output_tensors
