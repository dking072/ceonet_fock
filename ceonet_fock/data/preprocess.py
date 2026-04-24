"""
Preprocess a directory of PySCF Molden files into a single .pt dataset file.

Each Molden file is converted to a PyG Data object (see dataset.py) and the
full list is saved with torch.save().  Processing runs in parallel across
CPU workers.

Usage
-----
    python preprocess.py <molden_dir> <output.pt> [--n 100] [--workers 4]

Example
-------
    python preprocess.py ../../qm9_moldens ../../ceonet-fock/datasets/qm9_sto3g_100.pt --n 100
"""

import argparse
import sys
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import torch
from pyscf.tools import molden
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from dataset import mol_to_data


def _process_one(molden_path: str):
    """Worker function: load one Molden file and return a Data object (or None on error)."""
    try:
        mol, mo_energy, mo_coeff, _mo_occ, _sym, _uhf = molden.load(molden_path)
        mol.build()
        return mol_to_data(mol, mo_energy, mo_coeff, blocks_only=True)
    except Exception as exc:
        print(f"\n[WARN] skipping {molden_path}: {exc}", flush=True)
        return None


def main():
    parser = argparse.ArgumentParser(description="Molden directory → .pt dataset")
    parser.add_argument("molden_dir",  help="Directory containing *.molden files")
    parser.add_argument("output_pt",   help="Output .pt file path")
    parser.add_argument("--n",       type=int, default=None,
                        help="Max number of files to process (default: all)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel worker processes (default: 4)")
    args = parser.parse_args()

    molden_dir = Path(args.molden_dir)
    paths = sorted(molden_dir.glob("*.molden"))
    if args.n:
        paths = paths[: args.n]

    print(f"Processing {len(paths)} Molden files with {args.workers} workers …")

    with Pool(processes=args.workers) as pool:
        results = list(tqdm(
            pool.imap(_process_one, [str(p) for p in paths]),
            total=len(paths),
        ))

    data_list = [d for d in results if d is not None]
    n_skip = len(results) - len(data_list)
    if n_skip:
        print(f"[WARN] {n_skip} files were skipped due to errors.")

    output = Path(args.output_pt)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data_list, output)
    print(f"Saved {len(data_list)} molecules → {output}")


if __name__ == "__main__":
    main()
