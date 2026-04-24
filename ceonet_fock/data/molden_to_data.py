"""
Convert a PySCF-generated Molden file to a QHNet-style PyTorch Geometric Data object.

Usage
-----
    python molden_to_data.py molecule.molden
    python molden_to_data.py molecule.molden -o out.pt

The Fock matrix is reconstructed from the MO eigenvalues and coefficients
stored in the Molden file:   F = S C diag(eps) C.T S

For STO-3G (max angular momentum l=1), the spherical and Cartesian AO bases
are identical — no basis transformation is needed.  For bases with d or higher
shells, uncomment the sph2cart block in load_molden().
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from pyscf.tools import molden

from dataset import mol_to_data, STO3G_NORB, MAX_NORB


def load_molden(fname: str):
    """
    Load a PySCF Molden file.

    Returns mol, mo_energy, mo_coeff, mo_occ.

    Cartesian conversion note
    -------------------------
    PySCF writes Molden files in the spherical AO basis by default.
    For STO-3G (only s and p shells) spherical == Cartesian, so the
    MO coefficients are valid as-is.

    For a basis with d/f shells you would need:
        T = mol.cart2sph_coeff()          # (nao_sph, nao_cart)
        mo_coeff_cart = np.linalg.pinv(T) @ mo_coeff   # or T.T if T is orthogonal
    and rebuild the mol with mol.cart = True before calling mol_to_data.
    """
    mol, mo_energy, mo_coeff, mo_occ, _mo_sym, _is_uhf = molden.load(fname)
    mol.build()     # ensure integrals are ready
    return mol, mo_energy, mo_coeff, mo_occ


def main():
    parser = argparse.ArgumentParser(
        description="Convert a PySCF Molden file to a QHNet PyG Data object (.pt)")
    parser.add_argument("molden_file", help="Path to the .molden file")
    parser.add_argument("-o", "--output", default=None,
                        help="Output .pt path (default: <stem>.pt)")
    args = parser.parse_args()

    molden_path = Path(args.molden_file)
    out_path = Path(args.output) if args.output else molden_path.with_suffix(".pt")

    print(f"Loading  {molden_path}")
    mol, mo_energy, mo_coeff, mo_occ = load_molden(str(molden_path))

    # Sanity check: only elements with a known STO-3G layout
    atom_Zs = [int(mol.atom_charge(i)) for i in range(mol.natm)]
    unknown = set(atom_Zs) - set(STO3G_NORB)
    if unknown:
        raise ValueError(
            f"Atoms with Z={unknown} are not in STO3G_NORB. "
            "Add them to dataset.py before proceeding.")

    n_occ = int(mo_occ.sum() // 2)    # number of occupied MOs (closed-shell)
    print(f"  {mol.natm} atoms  |  {mol.nao} AOs  |  {n_occ} occ / {mol.nao} MOs")
    print(f"  Block size: {MAX_NORB}×{MAX_NORB} (padded)")

    data = mol_to_data(mol, mo_energy, mo_coeff)

    print(f"\nData object fields:")
    print(f"  pos                             {tuple(data.pos.shape)}")
    print(f"  atoms                           {tuple(data.atoms.shape)}")
    print(f"  hamiltonian                     {tuple(data.hamiltonian.shape)}")
    print(f"  overlap                         {tuple(data.overlap.shape)}")
    print(f"  hamiltonian_diagonal_blocks     {tuple(data.hamiltonian_diagonal_blocks.shape)}")
    print(f"  hamiltonian_non_diagonal_blocks {tuple(data.hamiltonian_non_diagonal_blocks.shape)}")

    torch.save(data, out_path)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
