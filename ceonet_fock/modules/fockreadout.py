from typing import Dict
import torch
from torch import nn

from cace.modules.tensornet import (
    TensorFeedForward, TensorProductLayer, TensorLinearMixing,
)
from cace.modules.tensornet_utils import expand_to, multi_outer_product

__all__ = ['FockDiagonalReadout', 'FockOffDiagonalReadout']


class FockDiagonalReadout(nn.Module):
    """
    Predicts atom-centred 5×5 STO-3G Fock diagonal blocks from CEONet node_feats_l.

    CEONet output shapes:
      l=0: (N, C)        — scalars
      l=1: (N, C, 3)     — vectors, axis ordering [x, y, z]
      l=2: (N, C, 3, 3)  — rank-2 Cartesian tensors, axes [x,y,z] × [x,y,z]

    Equivariant block decomposition (AO order: 1s, 2s, px, py, pz):
      ss (rows/cols 0,1) ← 3 scalars from l=0
      sp (rows 0,1 × cols 2-4) ← 2 vectors from l=1  (one per s orbital)
      pp (rows/cols 2-4) ← 3×3 Cartesian tensor from l=2

    CACE [x,y,z] matches PySCF [px,py,pz] directly — no axis reordering needed.
    """

    def __init__(
        self,
        feature_key: str = 'node_feats_l',
        output_key: str = 'hamiltonian_diagonal_blocks',
        n_channel: int = 8,
        use_feed_forward: bool = True,
    ):
        super().__init__()
        self.feature_key = feature_key
        self.output_key = output_key
        self.use_feed_forward = use_feed_forward

        self.model_outputs = [output_key]
        self.required_derivatives = []

        if use_feed_forward:
            self.tensor_feed_forward = TensorFeedForward(n_channel, lomax=2)

        # ss: 3 independent scalars (F_1s1s, F_1s2s, F_2s2s) — bias OK for scalars
        self.lin_ss = nn.LazyLinear(3, bias=True)

        # sp: 2 vectors (F_{1s,pα}, F_{2s,pα}) — no bias to preserve equivariance
        self.lin_sp = nn.LazyLinear(2, bias=False)

        # pp: 1 rank-2 tensor (3×3 symmetric) — no bias to preserve equivariance
        self.lin_pp = nn.LazyLinear(1, bias=False)

    def forward(self, data: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
        if self.feature_key not in data:
            raise ValueError(f"Feature key '{self.feature_key}' not found in data.")

        features = data[self.feature_key]

        for l in (0, 1, 2):
            if l not in features:
                raise ValueError(f"l={l} features required but not found under '{self.feature_key}'.")

        if self.use_feed_forward:
            features = self.tensor_feed_forward(features)

        f0 = features[0]  # (N, C)
        f1 = features[1]  # (N, C, 3)     CACE: [z, y, x]
        f2 = features[2]  # (N, C, 3, 3)  CACE: [z,y,x] × [z,y,x]

        N = f0.shape[0]
        out = torch.zeros(N, 5, 5, device=f0.device, dtype=f0.dtype)

        # --- ss sub-block (rows/cols 0, 1) ---
        # (N, C) → (N, 3): [F_1s1s, F_1s2s, F_2s2s]
        ss = self.lin_ss(f0)
        out[:, 0, 0] = ss[:, 0]
        out[:, 0, 1] = ss[:, 1]
        out[:, 1, 0] = ss[:, 1]  # symmetry
        out[:, 1, 1] = ss[:, 2]

        # --- sp sub-block (rows 0,1  ×  cols 2,3,4) ---
        # Mix channels: (N, C, 3) → transpose → (N, 3, C) → Linear(2) → (N, 3, 2)
        #               → transpose → (N, 2, 3)
        # CACE [x,y,z] == PySCF [px,py,pz] — no reordering needed.
        sp = self.lin_sp(f1.transpose(1, -1))   # (N, 3, 2)
        sp = sp.transpose(1, -1)                # (N, 2, 3)
        out[:, 0, 2:5] = sp[:, 0]   # F_{1s, pα}
        out[:, 2:5, 0] = sp[:, 0]   # symmetry
        out[:, 1, 2:5] = sp[:, 1]   # F_{2s, pα}
        out[:, 2:5, 1] = sp[:, 1]   # symmetry

        # --- pp sub-block (rows/cols 2,3,4 = px,py,pz) ---
        # Mix channels: (N, C, 3, 3) → permute → (N, 3, 3, C) → Linear(1) → (N, 3, 3, 1)
        #               → squeeze → (N, 3, 3)
        # CACE [x,y,z] == PySCF [px,py,pz] — no reordering needed.
        pp = self.lin_pp(f2.permute(0, 2, 3, 1))  # (N, 3, 3, 1)
        pp = pp.squeeze(-1)                         # (N, 3, 3)
        out[:, 2:5, 2:5] = pp

        data[self.output_key] = out
        return data

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"use_feed_forward={self.use_feed_forward})"
        )


class FockOffDiagonalReadout(nn.Module):
    """
    Predicts off-diagonal 5x5 STO-3G Fock blocks between all atom pairs
    within each molecule.

    Constructs all unique pairs (i < j) per graph as global edges, then applies
    the same RBF-weighted attention tensor product message as CEONet's
    MessagePassingLayer (calc_hi_r_hj + attention + RBF), but without
    scatter-aggregation.  The (j, i) block is recovered as block(i,j).T
    from Fock matrix symmetry (F = F.T).

    Block layout for pair (i < j), rows = atom-j AOs, cols = atom-i AOs:

      Rows/cols 0,1 = s-type (1s, 2s)   rows/cols 2-4 = p-type (px, py, pz)

      ss (rows 0,1 x cols 0,1):  4 independent scalars          <- l=0
      sp (rows 0,1 x cols 2-4):  2 vectors, F_sp' = F_sp @ R.T  <- l=1
      ps (rows 2-4 x cols 0,1):  2 vectors, F_ps' = R @ F_ps    <- l=1
      pp (rows 2-4 x cols 2-4):  rank-2 tensor, F_pp'=R F_pp R.T <- l=2

    The ss sub-block has 4 entries (not 3 like the diagonal) because the
    two s-type orbitals belong to different atoms so there is no on-site
    symmetry constraint within the sub-block.
    """

    def __init__(
        self,
        feature_key: str = 'node_feats_l',
        output_key: str = 'hamiltonian_non_diagonal_blocks',
        n_channel: int = 16,
        n_rbf: int = 8,
        lomax: int = 2,
        linmax: int = 2,
        use_feed_forward: bool = True,
        stacking: bool = False,
        linear_messages: bool = True,
        radial_basis: nn.Module = None,
    ):
        super().__init__()
        self.feature_key = feature_key
        self.output_key = output_key
        self.lomax = lomax
        self.linmax = linmax
        self.use_feed_forward = use_feed_forward
        self.linear_messages = linear_messages

        self.model_outputs = [output_key]
        self.required_derivatives = []

        if use_feed_forward:
            self.tensor_feed_forward = TensorFeedForward(n_channel, lomax=lomax)

        # Radial basis (no cutoff -- global edges span the full molecule)
        if radial_basis is not None:
            self.radial_basis = radial_basis
        else:
            from cace.modules import BesselRBF
            self.radial_basis = BesselRBF(cutoff=30.0, n_rbf=n_rbf, trainable=True)

        # Attention: scalar invariant from hi x hj
        self.att_hi_hj  = TensorProductLayer(n_channel, max_x_way=linmax, max_y_way=linmax,
                                              max_z_way=0, stacking=True)
        self.att_hi_mix = TensorLinearMixing(n_channel, linmax)
        self.att_hj_mix = TensorLinearMixing(n_channel, linmax)

        # Three-body: hi x r x hj
        self.tp_right          = TensorProductLayer(n_channel, max_x_way=lomax,  max_y_way=linmax,
                                                     max_z_way=lomax,  stacking=stacking)
        self.tp_left           = TensorProductLayer(n_channel, max_x_way=linmax, max_y_way=lomax,
                                                     max_z_way=lomax,  stacking=stacking)
        self.hi_r_hj_mix_hi    = TensorLinearMixing(n_channel, linmax)
        self.hi_r_hj_mix_r_hj  = TensorLinearMixing(n_channel, lomax)
        self.hi_r_hj_mix_hj    = TensorLinearMixing(n_channel, linmax)

        # Optional linear messages: hi x r  and  r x hj
        if linear_messages:
            self.tp_hi_r     = TensorProductLayer(n_channel, max_x_way=linmax, max_y_way=lomax,
                                                   max_z_way=lomax,  stacking=stacking)
            self.hi_r_mix_hi = TensorLinearMixing(n_channel, linmax)
            self.tp_r_hj     = TensorProductLayer(n_channel, max_x_way=lomax,  max_y_way=linmax,
                                                   max_z_way=lomax,  stacking=stacking)
            self.r_hj_mix_hj = TensorLinearMixing(n_channel, linmax)

        # Count channel multiplicity per l (mirrors MessagePassingLayer logic)
        combo_count = {l: 0 for l in range(lomax + 1)}
        tensor_layers = [self.tp_left]
        if linear_messages:
            tensor_layers += [self.tp_hi_r, self.tp_r_hj]
        for m in tensor_layers:
            for l in range(lomax + 1):
                hits = [t for t in m.combinations if t[-1] == l]
                combo_count[l] += len(hits) if stacking else (1 if hits else 0)
        self.combo_count = combo_count

        # RBF mixing: one Linear per l (no bias to preserve cutoff-to-zero behaviour)
        self.rbf_mixing_list = nn.ModuleList([
            nn.Linear(n_rbf, combo_count[l] * n_channel, bias=False)
            for l in range(lomax + 1)
        ])

        # Attention MLPs: one per l
        def build_mlp(nout):
            return nn.Sequential(
                nn.LazyLinear(nout), nn.SiLU(),
                nn.Linear(nout, nout), nn.SiLU(),
                nn.Linear(nout, nout),
            )
        self.attention_net_list = nn.ModuleList([
            build_mlp(combo_count[l] * n_channel) for l in range(lomax + 1)
        ])

        # Block readout linears
        # ss: 4 independent scalars (no on-site symmetry constraint for off-diagonal)
        self.lin_ss = nn.LazyLinear(4, bias=True)
        # sp (rows j-s x cols i-p): 2 vectors; F_sp' = F_sp @ R.T
        self.lin_sp = nn.LazyLinear(2, bias=False)
        # ps (rows j-p x cols i-s): 2 vectors; F_ps' = R @ F_ps
        self.lin_ps = nn.LazyLinear(2, bias=False)
        # pp: 1 rank-2 tensor
        self.lin_pp = nn.LazyLinear(1, bias=False)

    # -------------------------------------------------------------------------

    def _build_global_pairs(self, positions, batch):
        """
        For every graph in the batch, enumerate all atom pairs (i < j) within
        the same graph.

        Returns
        -------
        global_i, global_j : (n_pairs,) atom index tensors with global_i < global_j
        out_ij, out_ji     : (n_pairs,) indices into the flat offdiag output tensor
                             for block(i,j) and block(j,i) respectively
        n_offdiag_total    : total number of off-diagonal blocks across the batch
        """
        device = positions.device
        N_per_graph = torch.bincount(batch)
        ptr = torch.cat([
            torch.zeros(1, dtype=torch.long, device=device),
            N_per_graph.cumsum(0),
        ])
        # Cumulative offdiag block offset per graph: each graph with N atoms
        # contributes N*(N-1) off-diagonal blocks.
        mol_offsets = torch.cat([
            torch.zeros(1, dtype=torch.long, device=device),
            (N_per_graph * (N_per_graph - 1)).cumsum(0),
        ])
        n_offdiag_total = mol_offsets[-1].item()

        pair_i_list, pair_j_list = [], []
        out_ij_list, out_ji_list = [], []

        for m in range(ptr.shape[0] - 1):
            N_m   = N_per_graph[m].item()
            start = ptr[m].item()
            mol_off = mol_offsets[m].item()
            if N_m < 2:
                continue

            # Lower-triangle indices: row > col (so col is the smaller local atom)
            row, col = torch.tril_indices(N_m, N_m, offset=-1, device=device)
            local_i = col   # local index, i < j
            local_j = row   # local index, j > i

            pair_i_list.append(local_i + start)
            pair_j_list.append(local_j + start)

            # Flat offdiag ordering: for atom k, j != k runs as
            #   0,...,k-1, k+1,...,N-1  (position = j if j<k else j-1)
            # So flat index for (i,j) with j>i:  i*(N_m-1) + j - 1
            # And for (j,i) with i<j:            j*(N_m-1) + i
            out_ij_list.append(mol_off + local_i * (N_m - 1) + local_j - 1)
            out_ji_list.append(mol_off + local_j * (N_m - 1) + local_i)

        global_i = torch.cat(pair_i_list)
        global_j = torch.cat(pair_j_list)
        out_ij   = torch.cat(out_ij_list)
        out_ji   = torch.cat(out_ji_list)
        return global_i, global_j, out_ij, out_ji, n_offdiag_total

    def _calc_hi_r_hj(self, hi, u, hj):
        """Mirrors MessagePassingLayer.calc_hi_r_hj exactly."""
        # Scalar attention feed
        h1 = self.att_hi_mix(hi)
        h2 = self.att_hj_mix(hj)
        att_feed = [self.att_hi_hj(h1, h2)[0]]

        # Three-body: hi x (r x hj)
        h1 = self.hi_r_hj_mix_hi(hi)
        h2 = self.hi_r_hj_mix_hj(hj)
        msgs = self.tp_right(u, h2)
        msgs = self.hi_r_hj_mix_r_hj(msgs)
        msgs = self.tp_left(h1, msgs)

        # Optional linear messages
        if self.linear_messages:
            h1 = self.hi_r_mix_hi(hi)
            h1 = self.tp_hi_r(h1, u)
            for l in range(self.lomax + 1):
                msgs[l] = torch.hstack([msgs[l], h1[l]])

            h1 = self.r_hj_mix_hj(hj)
            h1 = self.tp_r_hj(u, h1)
            for l in range(self.lomax + 1):
                msgs[l] = torch.hstack([msgs[l], h1[l]])

        att_feed.append(msgs[0])
        att_feed = torch.hstack(att_feed)
        return msgs, att_feed

    def forward(self, data: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
        features  = data[self.feature_key]
        positions = data['positions']
        batch     = data['batch']
        dtype     = positions.dtype
        device    = positions.device

        if self.use_feed_forward:
            features = self.tensor_feed_forward(features)

        # Build all unique pairs (i < j) within each graph
        global_i, global_j, out_ij, out_ji, n_offdiag = self._build_global_pairs(
            positions, batch
        )
        n_pairs = global_i.shape[0]
        C = features[0].shape[1]    # channels after optional feed-forward

        # Per-pair node features
        hi = {l: features[l][global_i] for l in range(self.linmax + 1)}
        hj = {l: features[l][global_j] for l in range(self.linmax + 1)}

        # Global pair geometry: direction vector from i to j
        r_ij = positions[global_j] - positions[global_i]   # (n_pairs, 3)
        d_ij = r_ij.norm(dim=-1, keepdim=True)             # (n_pairs, 1)
        u_ij = r_ij / d_ij                                 # (n_pairs, 3)
        rbf  = self.radial_basis(d_ij)                     # (n_pairs, n_rbf)

        # Radial moment tensors u[l] = outer_product(u_ij, l), broadcast over C
        ones = torch.ones(n_pairs, C, device=device, dtype=dtype)
        u = {
            l: multi_outer_product(u_ij, l).unsqueeze(1) * expand_to(ones, l + 2)
            for l in range(self.lomax + 1)
        }

        # Edge messages via hi x r x hj (same as calc_edge_messages)
        msgs, att_feed = self._calc_hi_r_hj(hi, u, hj)

        for l in range(self.lomax + 1):
            attention = self.attention_net_list[l](att_feed)   # (n_pairs, combo*C)
            rbf_mixed = self.rbf_mixing_list[l](rbf)           # (n_pairs, combo*C)
            msgs[l] = msgs[l] * expand_to(attention, l + 2)
            msgs[l] = msgs[l] * expand_to(rbf_mixed, l + 2)

        f0 = msgs[0]   # (n_pairs, combo0*C)
        f1 = msgs[1]   # (n_pairs, combo1*C, 3)
        f2 = msgs[2]   # (n_pairs, combo2*C, 3, 3)

        # Build 5x5 block for each pair (rows = j AOs, cols = i AOs)
        block_ij = torch.zeros(n_pairs, 5, 5, device=device, dtype=dtype)

        # ss (rows/cols 0,1): 4 independent scalars (no on-site symmetry)
        ss = self.lin_ss(f0)                    # (n_pairs, 4)
        block_ij[:, 0, 0] = ss[:, 0]
        block_ij[:, 0, 1] = ss[:, 1]
        block_ij[:, 1, 0] = ss[:, 2]
        block_ij[:, 1, 1] = ss[:, 3]

        # sp (rows 0,1 x cols 2-4): F_sp' = F_sp @ R.T
        #   f1: (n_pairs, C, 3) -> transpose -> (n_pairs, 3, C) -> lin_sp -> (n_pairs, 3, 2)
        #   -> transpose -> (n_pairs, 2, 3)
        sp = self.lin_sp(f1.transpose(1, -1)).transpose(1, 2)   # (n_pairs, 2, 3)
        block_ij[:, 0, 2:5] = sp[:, 0]
        block_ij[:, 1, 2:5] = sp[:, 1]

        # ps (rows 2-4 x cols 0,1): F_ps' = R @ F_ps
        #   f1: (n_pairs, C, 3) -> transpose -> (n_pairs, 3, C) -> lin_ps -> (n_pairs, 3, 2)
        ps = self.lin_ps(f1.transpose(1, -1))                    # (n_pairs, 3, 2)
        block_ij[:, 2:5, 0] = ps[:, :, 0]
        block_ij[:, 2:5, 1] = ps[:, :, 1]

        # pp (rows/cols 2-4): F_pp' = R @ F_pp @ R.T
        pp = self.lin_pp(f2.permute(0, 2, 3, 1)).squeeze(-1)    # (n_pairs, 3, 3)
        block_ij[:, 2:5, 2:5] = pp

        # Assemble output tensor in the same ordering as cut_matrix / the dataset
        # block(j, i) = block(i, j).T  from Fock matrix symmetry
        output = torch.zeros(n_offdiag, 5, 5, device=device, dtype=dtype)
        output[out_ij] = block_ij
        output[out_ji] = block_ij.transpose(1, 2)

        data[self.output_key] = output
        return data

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"use_feed_forward={self.use_feed_forward}, "
            f"linear_messages={self.linear_messages})"
        )
