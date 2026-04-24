"""
Equivariance test for the off-diagonal Fock block model.

For a pair (i, j) with atoms i (src, columns) and j (dst, rows), the 5x5
STO-3G off-diagonal block F_{ij} represents <AO_j | F | AO_i>.  Under a
global rotation R applied to all atomic positions, both endpoint AOs
transform by the single-atom AO matrix U(R) = block_diag(I_2, R), so the
block must satisfy:

    F'_{ij} = U(R) @ F_{ij} @ U(R).T     for every pair (i, j)

This decomposes into four independently verifiable conditions:

    ss block  (rows/cols 0,1):        F'_ss = F_ss              invariant
    sp block  (rows 0,1 × cols 2-4):  F'_sp = F_sp @ R.T        vector covariance
    ps block  (rows 2-4 × cols 0,1):  F'_ps = R @ F_ps          vector covariance
    pp block  (rows/cols 2-4):        F'_pp = R @ F_pp @ R.T    rank-2 tensor covariance

Also tested: translation invariance (only r_ij enters the readout) and
the inner-pair ordering (the readout's own pair enumeration is
rotation-independent so pair indices must line up between original and
rotated batches).

Run with:
    pytest tests/test_equivariance_offdiag.py -v
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
from ceonet_fock.modules import FockOffDiagonalReadout
from cace.models.atomistic import NeuralNetworkPotential
from ceonet_fock.data import QM9FockData

# ── constants ─────────────────────────────────────────────────────────────────
_FILENAME = 'qm9_sto3g_100.pt'
if 'SLURM_JOB_CPUS_PER_NODE' in os.environ:
    PT_PATH = Path('/global/scratch/users/king1305/data') / _FILENAME
else:
    PT_PATH = Path('/home/king1305/AI-WORKSPACES/fock-predict/ceonet_fock/datasets') / _FILENAME

CUTOFF     = 8.5    # Bohr, CEONet trunk cutoff
PAIR_CUTOFF = 15.0  # Bohr, FockOffDiagonalReadout cutoff
ATOL       = 1e-4
OUT_KEY    = 'pred_hamiltonian_non_diagonal_blocks'


# ── helpers ───────────────────────────────────────────────────────────────────

def build_model(cutoff: float, pair_cutoff: float) -> NeuralNetworkPotential:
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
    fock_readout = FockOffDiagonalReadout(
        feature_key='node_feats_l',
        output_key=OUT_KEY,
        n_channel=16,
        n_rbf=8,
        lomax=2,
        linmax=2,
        use_feed_forward=True,
        linear_messages=True,
        cutoff=pair_cutoff,
    )
    return NeuralNetworkPotential(
        representation=ceonet,
        output_modules=[fock_readout],
    )


def random_rotation(dtype) -> torch.Tensor:
    """Random SO(3) rotation matrix via QR decomposition."""
    Q, _ = torch.linalg.qr(torch.randn(3, 3))
    if torch.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q.to(dtype)


def ao_transform(R: torch.Tensor) -> torch.Tensor:
    """5×5 AO transformation matrix U(R) = block_diag(I_2, R)."""
    U = torch.eye(5, dtype=R.dtype, device=R.device)
    U[2:5, 2:5] = R
    return U


def rotate_batch(batch, R: torch.Tensor):
    """Clone batch and apply rotation R to all real-space vector fields."""
    batch = batch.clone()
    R = R.to(batch.positions.dtype)
    batch.positions = batch.positions @ R.T
    if hasattr(batch, 'shifts') and batch.shifts is not None:
        batch.shifts = batch.shifts @ R.T
    return batch


def translate_batch(batch, t: torch.Tensor):
    """Clone batch and translate all atoms by t (applied per-graph so
    the per-molecule geometry is unchanged)."""
    batch = batch.clone()
    # Translate every atom by the same vector.  Intra-molecular distances are
    # unaffected so any translation-invariant model must be unchanged.
    batch.positions = batch.positions + t.to(batch.positions.dtype)
    return batch


def predict(model, batch) -> torch.Tensor:
    with torch.no_grad():
        return model(batch.clone())[OUT_KEY]   # (n_pairs_total, 5, 5)


# ── fixture ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope='module')
def model_and_batch():
    data  = QM9FockData(str(PT_PATH), cutoff=CUTOFF, batch_size=4)
    batch = next(iter(data.val_dataloader()))

    model = build_model(CUTOFF, PAIR_CUTOFF)
    model.eval()

    # Trigger lazy-layer initialisation
    with torch.no_grad():
        model(batch.clone())

    return model, batch


# ── subblock tests ────────────────────────────────────────────────────────────

def test_ss_block_is_invariant(model_and_batch):
    """Pair ss block (rows/cols 0,1) must be unchanged by rotation."""
    model, batch = model_and_batch
    R = random_rotation(batch.positions.dtype)

    F  = predict(model, batch)
    F2 = predict(model, rotate_batch(batch, R))

    diff = (F[:, :2, :2] - F2[:, :2, :2]).abs().max().item()
    assert diff < ATOL, f"offdiag ss not invariant — max |ΔF_ss| = {diff:.3e}"


def test_sp_block_transforms_as_vector(model_and_batch):
    """Pair sp block (rows 0,1 × cols 2-4) must satisfy F'_sp = F_sp @ R.T."""
    model, batch = model_and_batch
    R = random_rotation(batch.positions.dtype)

    F  = predict(model, batch)
    F2 = predict(model, rotate_batch(batch, R))

    sp          = F[:, :2, 2:5]       # (n_pairs, 2, 3)
    sp_expected = sp @ R.T
    diff = (F2[:, :2, 2:5] - sp_expected).abs().max().item()
    assert diff < ATOL, f"offdiag sp not equivariant — max |ΔF_sp| = {diff:.3e}"


def test_ps_block_transforms_as_vector(model_and_batch):
    """Pair ps block (rows 2-4 × cols 0,1) must satisfy F'_ps = R @ F_ps."""
    model, batch = model_and_batch
    R = random_rotation(batch.positions.dtype)

    F  = predict(model, batch)
    F2 = predict(model, rotate_batch(batch, R))

    ps          = F[:, 2:5, :2]       # (n_pairs, 3, 2)
    ps_expected = R @ ps
    diff = (F2[:, 2:5, :2] - ps_expected).abs().max().item()
    assert diff < ATOL, f"offdiag ps not equivariant — max |ΔF_ps| = {diff:.3e}"


def test_pp_block_transforms_as_rank2_tensor(model_and_batch):
    """Pair pp block (rows/cols 2-4) must satisfy F'_pp = R @ F_pp @ R.T."""
    model, batch = model_and_batch
    R = random_rotation(batch.positions.dtype)

    F  = predict(model, batch)
    F2 = predict(model, rotate_batch(batch, R))

    pp          = F[:, 2:5, 2:5]      # (n_pairs, 3, 3)
    pp_expected = R @ pp @ R.T
    diff = (F2[:, 2:5, 2:5] - pp_expected).abs().max().item()
    assert diff < ATOL, f"offdiag pp not equivariant — max |ΔF_pp| = {diff:.3e}"


# ── full-block test ───────────────────────────────────────────────────────────

def test_full_block_transforms_as_UFU_T(model_and_batch):
    """Full 5×5 pair block must satisfy F'_{ij} = U(R) F_{ij} U(R).T."""
    model, batch = model_and_batch
    R = random_rotation(batch.positions.dtype)
    U = ao_transform(R)

    F  = predict(model, batch)        # (n_pairs, 5, 5)
    F2 = predict(model, rotate_batch(batch, R))

    F_expected = U @ F @ U.T
    diff = (F2 - F_expected).abs().max().item()
    assert diff < ATOL, f"offdiag full-block equivariance failed — max |ΔF| = {diff:.3e}"


# ── pair-symmetry sanity check ────────────────────────────────────────────────

def test_pair_transpose_symmetry(model_and_batch):
    """The readout enforces block(j, i) = block(i, j).T by construction,
    and that symmetry must be preserved under rotation.  With N atoms per
    molecule the offdiag output has N*(N-1) blocks; for every edge k with
    (dst_k, src_k), there is a partner k' with (src_k, dst_k), and the
    blocks should be transposes of one another.
    """
    model, batch = model_and_batch
    F = predict(model, batch)

    # Reconstruct (dst, src) pair ordering used by _build_global_pairs.
    # For each graph, pairs are enumerated as:
    #   for src in 0..N-1:
    #     for dst in 0..N-1 if dst != src: block[src*(N-1) + dst - (dst>src)]
    # This matches cut_matrix in ceonet_fock.data.dataset.
    ptr = batch.ptr
    max_abs = 0.0
    for g in range(ptr.shape[0] - 1):
        N_m  = int(ptr[g + 1] - ptr[g])
        # Offset into flat block tensor for graph g
        per_graph = N_m * (N_m - 1)
        # Cumulative offset
        off = sum(
            int(ptr[h + 1] - ptr[h]) * (int(ptr[h + 1] - ptr[h]) - 1)
            for h in range(g)
        )
        # Check symmetry for every (i, j) with j > i
        for i in range(N_m):
            for j in range(i + 1, N_m):
                # Flat index of (src=i, dst=j):  i*(N-1) + (j - 1 since j>i)
                k_ij = off + i * (N_m - 1) + (j - 1)
                # Flat index of (src=j, dst=i):  j*(N-1) + i  (i<j means no shift)
                k_ji = off + j * (N_m - 1) + i
                diff = (F[k_ij] - F[k_ji].T).abs().max().item()
                if diff > max_abs:
                    max_abs = diff
    assert max_abs < ATOL, f"offdiag transpose symmetry broken — max = {max_abs:.3e}"


# ── translation invariance ────────────────────────────────────────────────────

def test_translation_invariance(model_and_batch):
    """Off-diagonal blocks depend only on relative geometry; translating all
    atoms by the same vector must not change the predictions."""
    model, batch = model_and_batch
    t = torch.randn(3, dtype=batch.positions.dtype)

    F  = predict(model, batch)
    F2 = predict(model, translate_batch(batch, t))

    diff = (F - F2).abs().max().item()
    assert diff < ATOL, f"offdiag not translation-invariant — max |ΔF| = {diff:.3e}"


# ── element-specific tests ────────────────────────────────────────────────────

def test_hh_pair_entry_is_invariant(model_and_batch):
    """For H–H pairs the only physical entry F[0,0] must be rotation-invariant."""
    model, batch = model_and_batch
    R = random_rotation(batch.positions.dtype)

    # Find H–H pair indices.  Pair k has (src_k, dst_k) following the
    # src-major enumeration inside _build_global_pairs.
    Z = batch.atomic_numbers
    ptr = batch.ptr
    hh_idx = []
    for g in range(ptr.shape[0] - 1):
        N_m  = int(ptr[g + 1] - ptr[g])
        start = int(ptr[g])
        off = sum(
            int(ptr[h + 1] - ptr[h]) * (int(ptr[h + 1] - ptr[h]) - 1)
            for h in range(g)
        )
        for i in range(N_m):
            for j in range(N_m):
                if i == j:
                    continue
                if Z[start + i] == 1 and Z[start + j] == 1:
                    # Flat idx for (src=i, dst=j)
                    k = off + i * (N_m - 1) + (j if j < i else j - 1)
                    hh_idx.append(k)
    if not hh_idx:
        pytest.skip("No H–H pairs in test batch")

    hh = torch.tensor(hh_idx, dtype=torch.long)
    F  = predict(model, batch)
    F2 = predict(model, rotate_batch(batch, R))

    diff = (F[hh, 0, 0] - F2[hh, 0, 0]).abs().max().item()
    assert diff < ATOL, f"H–H F(1s|1s) not invariant — max |ΔF| = {diff:.3e}"


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
    assert diff < ATOL, f"offdiag equivariance failed (seed={seed}) — max |ΔF| = {diff:.3e}"
