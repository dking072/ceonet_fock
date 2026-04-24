"""
Equivariance test for the diagonal Fock block model.

Under a global rotation R applied to all atomic positions, the predicted
5×5 STO-3G diagonal Fock blocks must satisfy:

    F'_i = U(R) @ F_i @ U(R).T   for every atom i

where U(R) = block_diag(I_2, R_3x3) is the AO transformation matrix:
  rows/cols 0,1  (1s, 2s) are scalar — unchanged by R
  rows/cols 2,3,4 (px, py, pz) transform as a 3-vector under R

This decomposes into three independently verifiable conditions:

    ss block  (rows/cols 0,1):        F'_ss = F_ss              invariant
    sp block  (rows 0,1 × cols 2-4):  F'_sp = F_sp @ R.T        vector covariance
    pp block  (rows/cols 2-4):        F'_pp = R @ F_pp @ R.T    rank-2 tensor covariance

Run with:
    pytest tests/test_equivariance.py -v
"""

import os
import sys
from pathlib import Path

import pytest
import torch

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT   = Path(__file__).parent.parent            # ceonet_fock/
CACE   = ROOT.parent.parent / 'dispersion/cace'  # ../dispersion/cace/
sys.path.insert(0, str(CACE))
sys.path.insert(0, str(ROOT))

from cace.representations import CEONet
from cace.modules import BesselRBF, PolynomialCutoff
from ceonet_fock.modules import FockDiagonalReadout, CEONetWithAtomicNumbers
from cace.models.atomistic import NeuralNetworkPotential
from ceonet_fock.data import QM9FockData

# ── constants ─────────────────────────────────────────────────────────────────
_FILENAME = 'qm9_sto3g_100.pt'
if 'SLURM_JOB_CPUS_PER_NODE' in os.environ:
    PT_PATH = Path('/global/scratch/users/king1305/data') / _FILENAME
else:
    PT_PATH = Path('/home/king1305/AI-WORKSPACES/fock-predict/ceonet_fock/datasets') / _FILENAME

CUTOFF  = 8.5   # Bohr  (matches QM9FockDataset default)
ATOL    = 1e-4


# ── helpers ───────────────────────────────────────────────────────────────────

def build_model(cutoff: float) -> NeuralNetworkPotential:
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
    fock_readout = FockDiagonalReadout(
        feature_key='node_feats_l',
        output_key='pred_hamiltonian_diagonal_blocks',
        n_channel=16,
        use_feed_forward=True,
    )
    return NeuralNetworkPotential(
        representation=CEONetWithAtomicNumbers(ceonet),
        output_modules=[fock_readout],
    )


def random_rotation(dtype) -> torch.Tensor:
    """Random SO(3) rotation matrix via QR decomposition."""
    Q, _ = torch.linalg.qr(torch.randn(3, 3))
    if torch.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q.to(dtype)


def ao_transform(R: torch.Tensor) -> torch.Tensor:
    """
    5×5 AO transformation matrix U(R) = block_diag(I_2, R).
    Scalar orbitals (1s, 2s) occupy rows/cols 0,1.
    p orbitals (px, py, pz) occupy rows/cols 2,3,4.
    """
    U = torch.eye(5, dtype=R.dtype, device=R.device)
    U[2:5, 2:5] = R
    return U


def rotate_batch(batch, R: torch.Tensor):
    """
    Clone batch and apply rotation R to all real-space vector fields.
    edge_index, atomic_numbers, masks and block targets are left unchanged.
    """
    batch = batch.clone()
    R = R.to(batch.positions.dtype)
    batch.positions = batch.positions @ R.T
    batch.shifts    = batch.shifts    @ R.T
    return batch


def predict(model, batch) -> torch.Tensor:
    with torch.no_grad():
        return model(batch.clone())['pred_hamiltonian_diagonal_blocks']  # (N, 5, 5)


# ── fixture ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope='module')
def model_and_batch():
    data  = QM9FockData(str(PT_PATH), cutoff=CUTOFF, batch_size=4)
    batch = next(iter(data.val_dataloader()))

    model = build_model(CUTOFF)
    model.eval()

    # Trigger lazy-layer initialisation
    with torch.no_grad():
        model(batch.clone())

    return model, batch


# ── subblock tests ────────────────────────────────────────────────────────────

def test_ss_block_is_invariant(model_and_batch):
    """1s/2s block (rows/cols 0,1) must be unchanged by any rotation."""
    model, batch = model_and_batch
    R = random_rotation(batch.positions.dtype)

    F  = predict(model, batch)
    F2 = predict(model, rotate_batch(batch, R))

    diff = (F[:, :2, :2] - F2[:, :2, :2]).abs().max().item()
    assert diff < ATOL, f"ss not invariant — max |ΔF_ss| = {diff:.3e}"


def test_sp_block_transforms_as_vector(model_and_batch):
    """s×p block must satisfy F'_sp = F_sp @ R.T (the p-orbital column index rotates)."""
    model, batch = model_and_batch
    R = random_rotation(batch.positions.dtype)

    F  = predict(model, batch)
    F2 = predict(model, rotate_batch(batch, R))

    sp          = F[:, :2, 2:5]       # (N, 2, 3)
    sp_expected = sp @ R.T            # rotate p-orbital axis
    diff = (F2[:, :2, 2:5] - sp_expected).abs().max().item()
    assert diff < ATOL, f"sp not equivariant — max |ΔF_sp| = {diff:.3e}"


def test_ps_block_transforms_as_vector(model_and_batch):
    """p×s block (transpose of sp) must satisfy F'_ps = R @ F_ps."""
    model, batch = model_and_batch
    R = random_rotation(batch.positions.dtype)

    F  = predict(model, batch)
    F2 = predict(model, rotate_batch(batch, R))

    ps          = F[:, 2:5, :2]       # (N, 3, 2)
    ps_expected = R @ ps              # rotate p-orbital row index
    diff = (F2[:, 2:5, :2] - ps_expected).abs().max().item()
    assert diff < ATOL, f"ps not equivariant — max |ΔF_ps| = {diff:.3e}"


def test_pp_block_transforms_as_rank2_tensor(model_and_batch):
    """p×p block must satisfy F'_pp = R @ F_pp @ R.T."""
    model, batch = model_and_batch
    R = random_rotation(batch.positions.dtype)

    F  = predict(model, batch)
    F2 = predict(model, rotate_batch(batch, R))

    pp          = F[:, 2:5, 2:5]     # (N, 3, 3)
    pp_expected = R @ pp @ R.T       # broadcasts over N
    diff = (F2[:, 2:5, 2:5] - pp_expected).abs().max().item()
    assert diff < ATOL, f"pp not equivariant — max |ΔF_pp| = {diff:.3e}"


# ── full-block test ───────────────────────────────────────────────────────────

def test_full_block_transforms_as_UFU_T(model_and_batch):
    """Full 5×5 block must satisfy F'_i = U(R) F_i U(R).T for all atoms i."""
    model, batch = model_and_batch
    R = random_rotation(batch.positions.dtype)
    U = ao_transform(R)               # (5, 5)

    F  = predict(model, batch)        # (N, 5, 5)
    F2 = predict(model, rotate_batch(batch, R))

    F_expected = U @ F @ U.T         # (5,5) @ (N,5,5) @ (5,5) broadcasts over N
    diff = (F2 - F_expected).abs().max().item()
    assert diff < ATOL, f"Full-block equivariance failed — max |ΔF| = {diff:.3e}"


# ── element-specific tests ────────────────────────────────────────────────────

def test_hydrogen_entry_is_invariant(model_and_batch):
    """H atoms: the only physical entry F[0,0] must be rotationally invariant."""
    model, batch = model_and_batch
    R = random_rotation(batch.positions.dtype)
    h_mask = (batch.atomic_numbers == 1)

    if not h_mask.any():
        pytest.skip("No H atoms in test batch")

    F  = predict(model, batch)
    F2 = predict(model, rotate_batch(batch, R))

    diff = (F[h_mask, 0, 0] - F2[h_mask, 0, 0]).abs().max().item()
    assert diff < ATOL, f"H-atom F(1s|1s) not invariant — max |ΔF| = {diff:.3e}"


def test_heavy_atom_full_block_equivariance(model_and_batch):
    """Heavy atoms (C/N/O): full 5×5 block must satisfy F' = U F U.T."""
    model, batch = model_and_batch
    R = random_rotation(batch.positions.dtype)
    U = ao_transform(R)
    heavy_mask = (batch.atomic_numbers != 1)

    if not heavy_mask.any():
        pytest.skip("No heavy atoms in test batch")

    F  = predict(model, batch)
    F2 = predict(model, rotate_batch(batch, R))

    F_h  = F[heavy_mask]             # (M, 5, 5)
    F2_h = F2[heavy_mask]
    F_expected = U @ F_h @ U.T

    diff = (F2_h - F_expected).abs().max().item()
    assert diff < ATOL, f"Heavy-atom equivariance failed — max |ΔF| = {diff:.3e}"


# ── multiple random rotations ─────────────────────────────────────────────────

@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_equivariance_multiple_rotations(model_and_batch, seed):
    """Full-block equivariance must hold for several independent rotations."""
    model, batch = model_and_batch
    torch.manual_seed(seed)
    R = random_rotation(batch.positions.dtype)
    U = ao_transform(R)

    F  = predict(model, batch)
    F2 = predict(model, rotate_batch(batch, R))

    F_expected = U @ F @ U.T
    diff = (F2 - F_expected).abs().max().item()
    assert diff < ATOL, f"Equivariance failed (seed={seed}) — max |ΔF| = {diff:.3e}"
