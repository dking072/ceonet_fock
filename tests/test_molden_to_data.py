"""
Tests for molden_to_data and the block decomposition in dataset.py.

Run with:
    pytest ceonet-fock/tests/test_molden_to_data.py -v
or from this directory:
    python -m pytest test_molden_to_data.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from pyscf.tools import molden

# Make the data package importable regardless of working directory
sys.path.insert(0, str(Path(__file__).parent.parent / "data"))

from dataset import (
    MAX_NORB,
    ORBITAL_MASK,
    STO3G_NORB,
    cut_matrix,
    fock_from_mo,
    mol_to_data,
)

MOLDEN = Path(__file__).parent.parent.parent / "dsgdb9nsd_033841_sto-3g.molden"
ATOL = 1e-10


@pytest.fixture(scope="module")
def molden_data():
    """Load the molden file once for the whole module."""
    mol, mo_energy, mo_coeff, mo_occ, _sym, _uhf = molden.load(str(MOLDEN))
    mol.build()
    return mol, mo_energy, mo_coeff, mo_occ


@pytest.fixture(scope="module")
def data(molden_data):
    """Build the PyG Data object once for the whole module."""
    mol, mo_energy, mo_coeff, _mo_occ = molden_data
    return mol_to_data(mol, mo_energy, mo_coeff)


# ---------------------------------------------------------------------------
# Fock reconstruction from MO data
# ---------------------------------------------------------------------------

class TestFockReconstruction:
    def test_generalized_eigenvalue_residual(self, molden_data):
        """F C = S C diag(eps) should hold to numerical precision."""
        mol, mo_energy, mo_coeff, _ = molden_data
        S = mol.intor("int1e_ovlp")
        F = fock_from_mo(S, mo_coeff, mo_energy)
        residual = np.max(np.abs(F @ mo_coeff - S @ mo_coeff @ np.diag(mo_energy)))
        assert residual < ATOL, f"max|FC - SCε| = {residual:.2e}"

    def test_mo_orthonormality(self, molden_data):
        """C.T S C = I (MOs are orthonormal in the metric S)."""
        mol, mo_energy, mo_coeff, _ = molden_data
        S = mol.intor("int1e_ovlp")
        gram = mo_coeff.T @ S @ mo_coeff
        off_diag = np.max(np.abs(gram - np.eye(gram.shape[0])))
        assert off_diag < ATOL, f"max|C.T S C - I| = {off_diag:.2e}"

    def test_fock_symmetry(self, data):
        """The reconstructed Fock matrix must be symmetric."""
        F = data.hamiltonian.numpy()
        assert np.allclose(F, F.T, atol=ATOL), "Fock matrix is not symmetric"

    def test_overlap_symmetry(self, data):
        """The overlap matrix must be symmetric."""
        S = data.overlap.numpy()
        assert np.allclose(S, S.T, atol=ATOL), "Overlap matrix is not symmetric"


# ---------------------------------------------------------------------------
# Data object shape and dtype
# ---------------------------------------------------------------------------

class TestDataObjectFields:
    def test_pos_shape(self, data, molden_data):
        mol = molden_data[0]
        assert data.pos.shape == (mol.natm, 3)

    def test_atoms_shape(self, data, molden_data):
        mol = molden_data[0]
        assert data.atoms.shape == (mol.natm, 1)

    def test_hamiltonian_shape(self, data, molden_data):
        mol = molden_data[0]
        nao = mol.nao
        assert data.hamiltonian.shape == (nao, nao)
        assert data.overlap.shape == (nao, nao)

    def test_diagonal_blocks_shape(self, data, molden_data):
        mol = molden_data[0]
        N = mol.natm
        assert data.hamiltonian_diagonal_blocks.shape == (N, MAX_NORB, MAX_NORB)
        assert data.overlap_diagonal_blocks.shape == (N, MAX_NORB, MAX_NORB)

    def test_offdiagonal_blocks_shape(self, data, molden_data):
        mol = molden_data[0]
        N = mol.natm
        assert data.hamiltonian_non_diagonal_blocks.shape == (N * (N - 1), MAX_NORB, MAX_NORB)
        assert data.overlap_non_diagonal_blocks.shape == (N * (N - 1), MAX_NORB, MAX_NORB)

    def test_all_float64(self, data):
        for key in ("hamiltonian", "overlap",
                    "hamiltonian_diagonal_blocks", "hamiltonian_non_diagonal_blocks",
                    "overlap_diagonal_blocks", "overlap_non_diagonal_blocks"):
            assert data[key].dtype == torch.float64, f"{key} is not float64"

    def test_masks_binary(self, data):
        """Mask tensors should contain only 0 and 1."""
        for key in ("hamiltonian_diagonal_block_masks",
                    "hamiltonian_non_diagonal_block_masks",
                    "overlap_diagonal_block_masks",
                    "overlap_non_diagonal_block_masks"):
            vals = data[key].unique()
            assert set(vals.tolist()) <= {0.0, 1.0}, f"{key} has non-binary values"


# ---------------------------------------------------------------------------
# Block decomposition correctness — reconstruct full matrix from blocks
# ---------------------------------------------------------------------------

def reconstruct_from_blocks(
    atom_numbers: list[int],
    diag_blocks: torch.Tensor,
    offdiag_blocks: torch.Tensor,
) -> torch.Tensor:
    """
    Reassemble the full (nao × nao) matrix from padded atom-pair blocks.

    Inverse of cut_matrix: strip the zero-padding and place each sub-block
    back at the correct AO row/column offsets.
    """
    N = len(atom_numbers)
    nao = sum(STO3G_NORB[Z] for Z in atom_numbers)
    full = torch.zeros(nao, nao, dtype=diag_blocks.dtype)

    offsets = [0]
    for Z in atom_numbers:
        offsets.append(offsets[-1] + STO3G_NORB[Z])

    off_idx = 0
    for i in range(N):
        for j in range(N):
            mi = ORBITAL_MASK[atom_numbers[i]]
            mj = ORBITAL_MASK[atom_numbers[j]]
            if i == j:
                block = diag_blocks[i]
            else:
                block = offdiag_blocks[off_idx]
                off_idx += 1
            # Extract only the physically meaningful rows/cols
            sub = block[mj][:, mi]
            full[offsets[j]:offsets[j + 1], offsets[i]:offsets[i + 1]] = sub

    return full


class TestBlockRoundTrip:
    def test_fock_roundtrip(self, data):
        """Reconstruct full Fock matrix from blocks; must match the original."""
        atom_numbers = data.atoms.squeeze(1).tolist()
        F_rec = reconstruct_from_blocks(
            atom_numbers,
            data.hamiltonian_diagonal_blocks,
            data.hamiltonian_non_diagonal_blocks,
        )
        diff = (F_rec - data.hamiltonian).abs().max().item()
        assert diff < ATOL, f"Fock round-trip max error = {diff:.2e}"

    def test_overlap_roundtrip(self, data):
        """Reconstruct full overlap matrix from blocks; must match the original."""
        atom_numbers = data.atoms.squeeze(1).tolist()
        S_rec = reconstruct_from_blocks(
            atom_numbers,
            data.overlap_diagonal_blocks,
            data.overlap_non_diagonal_blocks,
        )
        diff = (S_rec - data.overlap).abs().max().item()
        assert diff < ATOL, f"Overlap round-trip max error = {diff:.2e}"

    def test_padded_entries_are_zero(self, data):
        """Entries outside the orbital mask must be zero (zero-padding check)."""
        for name, blocks, masks in [
            ("Fock diag",    data.hamiltonian_diagonal_blocks,     data.hamiltonian_diagonal_block_masks),
            ("Fock offdiag", data.hamiltonian_non_diagonal_blocks, data.hamiltonian_non_diagonal_block_masks),
            ("S diag",       data.overlap_diagonal_blocks,         data.overlap_diagonal_block_masks),
            ("S offdiag",    data.overlap_non_diagonal_blocks,     data.overlap_non_diagonal_block_masks),
        ]:
            leak = (blocks * (1.0 - masks)).abs().max().item()
            assert leak == 0.0, f"{name}: non-zero value in padded region ({leak})"

    def test_h_atom_block_size(self, data, molden_data):
        """H atom diagonal block should have exactly 1 active entry (1s only)."""
        mol = molden_data[0]
        atom_numbers = [int(mol.atom_charge(i)) for i in range(mol.natm)]
        h_indices = [i for i, Z in enumerate(atom_numbers) if Z == 1]
        assert h_indices, "No hydrogen atoms found"
        for idx in h_indices:
            mask = data.hamiltonian_diagonal_block_masks[idx]
            assert mask.sum().item() == 1.0, \
                f"H atom {idx} diagonal block mask sum = {mask.sum().item()}, expected 1"

    def test_heavy_atom_block_size(self, data, molden_data):
        """C/N/O atom diagonal blocks should have 5x5 = 25 active entries."""
        mol = molden_data[0]
        atom_numbers = [int(mol.atom_charge(i)) for i in range(mol.natm)]
        heavy_indices = [i for i, Z in enumerate(atom_numbers) if Z != 1]
        for idx in heavy_indices:
            mask = data.hamiltonian_diagonal_block_masks[idx]
            assert mask.sum().item() == 25.0, \
                f"Heavy atom {idx} diagonal block mask sum = {mask.sum().item()}, expected 25"
