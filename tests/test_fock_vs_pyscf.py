"""
Cross-check: compare the Fock matrix reconstructed from Molden MO data
against the Fock matrix produced by a fresh PySCF RHF calculation.

The two routes to the Fock matrix:
  (A) Reconstruction from stored MOs  — F = S C diag(eps) C.T S
  (B) PySCF mf.kernel() + mf.get_fock() — builds F from the converged density

Both start from the same geometry and STO-3G basis, so they should agree
up to the SCF convergence threshold (~1e-6 Ha for Fock matrix elements;
orbital energies agree to ~1e-9 Ha).

Run with:
    pytest ceonet-fock/tests/test_fock_vs_pyscf.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from pyscf import scf
from pyscf.tools import molden

sys.path.insert(0, str(Path(__file__).parent.parent / "data"))
from dataset import fock_from_mo

MOLDEN = Path(__file__).parent.parent.parent / "dsgdb9nsd_033841_sto-3g.molden"

# Tolerances
# Fock matrix elements: limited by SCF convergence (default conv_tol=1e-9 in energy,
# which translates to ~1e-6 in the Fock matrix).
FOCK_ATOL = 1e-5
# Orbital energies are eigenvalues of F; they agree more tightly.
EPS_ATOL  = 1e-7


@pytest.fixture(scope="module")
def molden_results():
    """Load MO data from the Molden file."""
    mol, mo_energy, mo_coeff, mo_occ, _sym, _uhf = molden.load(str(MOLDEN))
    mol.build()
    return mol, mo_energy, mo_coeff, mo_occ


@pytest.fixture(scope="module")
def rhf_results(molden_results):
    """Run a fresh RHF calculation on the same geometry + basis."""
    mol = molden_results[0]
    mf = scf.RHF(mol)
    mf.verbose = 0          # suppress PySCF output
    mf.kernel()
    assert mf.converged, "RHF did not converge"
    return mf


class TestFockVsPySCF:
    def test_rhf_converged(self, rhf_results):
        """Sanity: the fresh RHF must reach convergence."""
        assert rhf_results.converged

    def test_fock_matrix_agrees(self, molden_results, rhf_results):
        """
        The Fock matrix reconstructed from Molden MO data must agree with
        the Fock matrix built by PySCF from the converged RHF density.

        F_rec = S C diag(eps) C.T S   (Molden route)
        F_rhf = mf.get_fock()          (PySCF route)

        Tolerance: {FOCK_ATOL} Ha (set by SCF convergence, not numerical noise).
        """
        mol, mo_energy, mo_coeff, _ = molden_results
        S     = mol.intor("int1e_ovlp")
        F_rec = fock_from_mo(S, mo_coeff, mo_energy)
        F_rhf = rhf_results.get_fock()

        max_diff = np.max(np.abs(F_rhf - F_rec))
        assert max_diff < FOCK_ATOL, (
            f"max|F_rhf - F_rec| = {max_diff:.2e}  (tolerance {FOCK_ATOL:.0e})"
        )

    def test_fock_matrix_symmetry(self, molden_results, rhf_results):
        """Both Fock matrices must be symmetric."""
        mol, mo_energy, mo_coeff, _ = molden_results
        S     = mol.intor("int1e_ovlp")
        F_rec = fock_from_mo(S, mo_coeff, mo_energy)
        F_rhf = rhf_results.get_fock()

        assert np.allclose(F_rec, F_rec.T, atol=1e-12), "Reconstructed F is not symmetric"
        assert np.allclose(F_rhf, F_rhf.T, atol=1e-12), "PySCF F is not symmetric"

    def test_orbital_energies_agree(self, molden_results, rhf_results):
        """
        Orbital energies from both routes must agree.
        These are eigenvalues of the same converged Fock, so they agree
        more tightly than the Fock matrix elements themselves.
        """
        _, mo_energy_molden, _, _ = molden_results
        mo_energy_rhf = rhf_results.mo_energy

        # Sort both sets (ordering may differ)
        eps_mol = np.sort(mo_energy_molden)
        eps_rhf = np.sort(mo_energy_rhf)

        max_diff = np.max(np.abs(eps_rhf - eps_mol))
        assert max_diff < EPS_ATOL, (
            f"max|eps_rhf - eps_molden| = {max_diff:.2e}  (tolerance {EPS_ATOL:.0e})"
        )

    def test_fock_diagonalises_correctly_rhf(self, rhf_results):
        """
        PySCF Fock: F C = S C diag(eps) must hold for the RHF solution.
        """
        mf  = rhf_results
        mol = mf.mol
        S   = mol.intor("int1e_ovlp")
        F   = mf.get_fock()
        C   = mf.mo_coeff
        eps = mf.mo_energy

        residual = np.max(np.abs(F @ C - S @ C @ np.diag(eps)))
        # Bound is SCF convergence, not numerical noise (~1e-6 for default conv_tol)
        assert residual < FOCK_ATOL, f"RHF: max|FC - SCε| = {residual:.2e}"

    def test_fock_diagonalises_correctly_molden(self, molden_results):
        """
        Reconstructed Fock: F C = S C diag(eps) must hold for the Molden MOs.
        """
        mol, mo_energy, mo_coeff, _ = molden_results
        S     = mol.intor("int1e_ovlp")
        F_rec = fock_from_mo(S, mo_coeff, mo_energy)

        residual = np.max(np.abs(F_rec @ mo_coeff - S @ mo_coeff @ np.diag(mo_energy)))
        assert residual < 1e-10, f"Molden: max|FC - SCε| = {residual:.2e}"

    def test_total_energy_agrees(self, molden_results, rhf_results):
        """
        The RHF total energy should be consistent with the Molden calculation.
        Both used the same geometry and basis, so energies should match to ~1e-6 Ha.
        """
        mol, mo_energy, mo_coeff, mo_occ = molden_results
        mf = rhf_results

        # Recompute RHF energy from Molden density: E = tr(P (h + F/2))
        dm     = mf.make_rdm1(mo_coeff, mo_occ)
        h_core = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
        S      = mol.intor("int1e_ovlp")
        F_rec  = fock_from_mo(S, mo_coeff, mo_energy)
        e_elec = 0.5 * np.einsum("ij,ji->", dm, h_core + F_rec)
        e_nuc  = mf.energy_nuc()
        e_molden = e_elec + e_nuc

        assert abs(e_molden - mf.e_tot) < 1e-5, (
            f"E_molden = {e_molden:.8f}, E_rhf = {mf.e_tot:.8f}, "
            f"diff = {abs(e_molden - mf.e_tot):.2e}"
        )
