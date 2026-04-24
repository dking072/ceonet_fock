"""
STO-3G Fock/overlap matrix utilities for QHNet-style ML (adapted from QHNet ori_dataset.py).

AO basis convention (PySCF native ordering, matches Cartesian for l<=1):
  H  (Z=1): [1s]                   — 1 orbital
  C  (Z=6): [1s, 2s, 2px, 2py, 2pz] — 5 orbitals
  N  (Z=7): [1s, 2s, 2px, 2py, 2pz] — 5 orbitals
  O  (Z=8): [1s, 2s, 2px, 2py, 2pz] — 5 orbitals
  F  (Z=9): [1s, 2s, 2px, 2py, 2pz] — 5 orbitals

Fock reconstruction from MO data (as stored in a Molden file):
  The generalized eigenvalue problem is  F C = S C diag(eps),  C.T S C = I.
  Solving for F gives:  F = S C diag(eps) C.T S
"""

import numpy as np
import torch
from torch_geometric.data import Data


# ---------------------------------------------------------------------------
# STO-3G orbital layout
# ---------------------------------------------------------------------------

# Number of AOs per atom type in STO-3G
STO3G_NORB: dict[int, int] = {1: 1, 6: 5, 7: 5, 8: 5, 9: 5}

# Max orbitals in one atom block (5 for heavy atoms, 1 for H → pad to 5)
MAX_NORB = 5

# Which rows/cols of the MAX_NORB-padded block are populated for each element.
# Layout of the 5-wide block:  [1s | 2s | px | py | pz]
# H only occupies slot 0 (1s); heavy atoms occupy all 5.
ORBITAL_MASK: dict[int, torch.Tensor] = {
    1: torch.tensor([0]),               # H : 1s
    6: torch.tensor([0, 1, 2, 3, 4]),   # C : 1s 2s px py pz
    7: torch.tensor([0, 1, 2, 3, 4]),   # N : 1s 2s px py pz
    8: torch.tensor([0, 1, 2, 3, 4]),   # O : 1s 2s px py pz
    9: torch.tensor([0, 1, 2, 3, 4]),   # F : 1s 2s px py pz
}


# ---------------------------------------------------------------------------
# Fock reconstruction
# ---------------------------------------------------------------------------

def fock_from_mo(
    S: np.ndarray,
    mo_coeff: np.ndarray,
    mo_energy: np.ndarray,
) -> np.ndarray:
    """
    Reconstruct the Fock matrix from MO eigenvalues and coefficients.

    Derivation
    ----------
    Generalised eigenvalue problem:  F C = S C diag(eps),  C.T S C = I
    => C^{-1} = C.T S  (since C.T S C = I and C is square)
    => F = (C^{-1}).T diag(eps) C^{-1}
         = S C diag(eps) C.T S

    Parameters
    ----------
    S          : (nao, nao) overlap matrix
    mo_coeff   : (nao, nmo) MO coefficient matrix  C
    mo_energy  : (nmo,)     orbital energies        eps

    Returns
    -------
    F : (nao, nao) Fock matrix
    """
    return S @ mo_coeff @ np.diag(mo_energy) @ mo_coeff.T @ S


# ---------------------------------------------------------------------------
# Block decomposition  (adapted from Mixed_MD17_DFT.cut_matrix in QHNet)
# ---------------------------------------------------------------------------

def cut_matrix(
    matrix: torch.Tensor,
    atom_numbers: list[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Decompose a full (nao × nao) matrix into per-atom-pair blocks.

    Each block is zero-padded to (MAX_NORB × MAX_NORB) = (5 × 5), and a
    companion boolean mask marks the physically meaningful entries.

    Diagonal blocks (i == j) contain on-site (self-interaction) terms.
    Off-diagonal blocks (i != j) contain coupling terms between atom pairs.

    Parameters
    ----------
    matrix       : (nao, nao) tensor
    atom_numbers : list of atomic numbers Z for each atom (length N_atoms)

    Returns
    -------
    diag_blocks      : (N,      MAX_NORB, MAX_NORB)
    offdiag_blocks   : (N*(N-1), MAX_NORB, MAX_NORB)
    diag_masks       : same shape as diag_blocks,    dtype float64
    offdiag_masks    : same shape as offdiag_blocks, dtype float64
    """
    N = len(atom_numbers)
    dtype = matrix.dtype

    # Pre-compute AO slice start index for each atom
    offsets = [0]
    for Z in atom_numbers:
        offsets.append(offsets[-1] + STO3G_NORB[Z])

    diag_blocks, offdiag_blocks = [], []
    diag_masks,  offdiag_masks  = [], []

    for i in range(N):          # column atom (bra)
        for j in range(N):      # row atom    (ket)
            Zi, Zj = atom_numbers[i], atom_numbers[j]
            mi = ORBITAL_MASK[Zi]   # active col indices in padded block
            mj = ORBITAL_MASK[Zj]   # active row indices in padded block

            block = torch.zeros(MAX_NORB, MAX_NORB, dtype=dtype)
            mask  = torch.zeros(MAX_NORB, MAX_NORB, dtype=dtype)

            # Extract the physical sub-block from the full matrix
            sub = matrix[offsets[j]:offsets[j + 1], offsets[i]:offsets[i + 1]]

            # Scatter into the padded block
            tmp = block[mj]; tmp[:, mi] = sub; block[mj] = tmp
            tmp = mask[mj];  tmp[:, mi] = 1;   mask[mj]  = tmp

            if i == j:
                diag_blocks.append(block)
                diag_masks.append(mask)
            else:
                offdiag_blocks.append(block)
                offdiag_masks.append(mask)

    return (
        torch.stack(diag_blocks),
        torch.stack(offdiag_blocks),
        torch.stack(diag_masks),
        torch.stack(offdiag_masks),
    )


# ---------------------------------------------------------------------------
# Build PyG Data object
# ---------------------------------------------------------------------------

def mol_to_data(mol, mo_energy: np.ndarray, mo_coeff: np.ndarray,
                blocks_only: bool = False) -> Data:
    """
    Build a PyTorch Geometric Data object in the QHNet style from a PySCF
    Mole object and its MO information (as loaded from a Molden file).

    Parameters
    ----------
    blocks_only : if True, omit the full (nao × nao) hamiltonian and overlap
        matrices and return only the atom-pair block decomposition.  Use this
        when building a batched DataLoader over mixed molecules: the full
        matrices have different nao for each molecule and cannot be collated
        by PyG's default Batch.from_data_list.  The block tensors are padded
        to a fixed (5 × 5) size and can always be concatenated along dim 0.

    Fields (blocks_only=False)
    --------------------------
    positions                       : (N, 3)       atom positions in Bohr
    atomic_numbers                  : (N,)         atomic numbers (int64)
    hamiltonian                     : (nao, nao)   full Fock matrix
    overlap                         : (nao, nao)   full overlap matrix
    hamiltonian_diagonal_blocks     : (N, 5, 5)    on-site Fock blocks
    hamiltonian_non_diagonal_blocks : (N*(N-1), 5, 5)  coupling Fock blocks
    *_block_masks                   : same shapes, float64 validity masks
    overlap_diagonal_blocks         : (N, 5, 5)
    overlap_non_diagonal_blocks     : (N*(N-1), 5, 5)
    """
    S = mol.intor('int1e_ovlp')
    F = fock_from_mo(S, mo_coeff, mo_energy)

    atom_numbers = [int(mol.atom_charge(i)) for i in range(mol.natm)]
    positions       = torch.tensor(mol.atom_coords(), dtype=torch.float32)   # Bohr
    atomic_numbers  = torch.tensor(atom_numbers, dtype=torch.int64)          # (N,)

    F_t = torch.tensor(F, dtype=torch.float32)
    S_t = torch.tensor(S, dtype=torch.float32)

    F_diag, F_off, F_diag_mask, F_off_mask = cut_matrix(F_t, atom_numbers)
    S_diag, S_off, S_diag_mask, S_off_mask = cut_matrix(S_t, atom_numbers)

    kwargs = dict(
        num_nodes=len(atom_numbers),
        positions=positions,
        atomic_numbers=atomic_numbers,
        hamiltonian_diagonal_blocks=F_diag,
        hamiltonian_non_diagonal_blocks=F_off,
        hamiltonian_diagonal_block_masks=F_diag_mask,
        hamiltonian_non_diagonal_block_masks=F_off_mask,
        overlap_diagonal_blocks=S_diag,
        overlap_non_diagonal_blocks=S_off,
        overlap_diagonal_block_masks=S_diag_mask,
        overlap_non_diagonal_block_masks=S_off_mask,
    )
    if not blocks_only:
        kwargs['hamiltonian'] = F_t
        kwargs['overlap']     = S_t

    return Data(**kwargs)
