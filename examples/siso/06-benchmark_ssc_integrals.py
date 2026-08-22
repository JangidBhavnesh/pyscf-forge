#!/usr/bin/env python
# Copyright 2014-2026 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Benchmark density-fitted spin--spin coupling integral generation.

The default benchmark keeps 16 dense active orbitals while increasing the AO
basis size through 100, 200, 300, 400, and 500 functions.  A grid of hydrogen
atoms in def2-SVP is used.  Each atom contributes five spherical AOs, making
the requested default sizes exact while allowing PySCF to select the standard
matching def2-SVP-RI correlation-fitting auxiliary basis automatically.  This
is the PySCF analogue of the ``/C`` auxiliary family used in the RI-SSC paper.

The active orbitals are random orthonormal vectors used only for timing.  They
are deliberately dense over the full AO basis to represent a delocalized
active space and avoid benchmarking a specially sparse transformation.

Examples:

    python examples/siso/06-benchmark_ssc_integrals.py
    python examples/siso/06-benchmark_ssc_integrals.py --sizes 100 200
    python examples/siso/06-benchmark_ssc_integrals.py --threads 8 --csv ssc.csv

Only the standard three- and two-center auxiliary-index quantities are used.
The three-center factors are transformed directly to the 16-orbital active
space, so the benchmark never constructs a four-center AO SSC tensor.
"""

import argparse
import csv
import gc
import math
import time

import numpy as np

from pyscf import df, gto, lib
from pyscf.siso.ss_int_helper import compute_ssc_integrals_ri


DEFAULT_SIZES = (100, 200, 300, 400, 500)
ACTIVE_ORBITALS = 16
PRIMARY_BASIS = 'def2-svp'
AOS_PER_HYDROGEN = 5


def make_benchmark_molecule(nao, spacing=6.0):
    """Build a def2-SVP hydrogen grid with exactly ``nao`` spherical AOs."""
    if nao % AOS_PER_HYDROGEN:
        raise ValueError(
            f'basis size {nao} must be divisible by {AOS_PER_HYDROGEN}')
    natom = nao // AOS_PER_HYDROGEN
    side = math.ceil(natom ** (1.0 / 3.0))
    atoms = []
    for atom_id in range(natom):
        ix = atom_id % side
        iy = (atom_id // side) % side
        iz = atom_id // side**2
        atoms.append(('H', (spacing * ix, spacing * iy, spacing * iz)))

    # Hydrogen contributes one electron.  This spin choice is needed only to
    # make Mole's electron-count validation consistent; no SCF is performed.
    mol = gto.M(
        atom=atoms,
        basis=PRIMARY_BASIS,
        unit='Bohr',
        spin=natom % 2,
        verbose=0,
    )
    if mol.nao_nr() != nao:
        raise RuntimeError(
            f"requested {nao} AOs, but molecule contains {mol.nao_nr()}")
    return mol


def make_active_orbitals(nao, nactive, seed):
    """Return dense Euclidean-orthonormal timing orbitals."""
    if nactive > nao:
        raise ValueError(
            f"active-space size {nactive} exceeds AO size {nao}")
    rng = np.random.default_rng(seed)
    coefficients = rng.standard_normal((nao, nactive))
    coefficients, _ = np.linalg.qr(coefficients, mode='reduced')
    return coefficients


def benchmark_one(nao, nactive, seed):
    """Time one standard-DF SSC integral generation."""
    mol = make_benchmark_molecule(nao)
    mo_coeff = make_active_orbitals(nao, nactive, seed)
    # SSC fits complete AO-pair densities, as in correlated wave-function
    # methods.  Select PySCF's standard correlation-fitting (/C-style) basis,
    # rather than a JK-specific auxiliary basis.
    auxbasis = df.addons.make_auxbasis(mol, mp2fit=True)
    auxmol = df.addons.make_auxmol(mol, auxbasis)

    gc.collect()

    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    hss = compute_ssc_integrals_ri(
        mol, mo_coeff=mo_coeff, auxbasis=auxbasis)
    cpu_seconds = time.process_time() - cpu_start
    wall_seconds = time.perf_counter() - wall_start

    trace = hss[0, 0] + hss[1, 1] + hss[2, 2]
    result = {
        'requested_nao': nao,
        'actual_nao': mol.nao_nr(),
        'nbas': mol.nbas,
        'naux': auxmol.nao_nr(),
        'aux_nbas': auxmol.nbas,
        'nactive': nactive,
        'active_pair_factors': auxmol.nao_nr() * nactive**2,
        'wall_seconds': wall_seconds,
        'cpu_seconds': cpu_seconds,
        'max_trace_error': float(np.max(np.abs(trace))),
        'integral_norm': float(np.linalg.norm(hss)),
    }

    del hss, mo_coeff, auxmol, mol
    gc.collect()
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--sizes', type=int, nargs='+', default=DEFAULT_SIZES,
        help='requested AO basis sizes (default: 100 200 300 400 500)')
    parser.add_argument(
        '--nactive', type=int, default=ACTIVE_ORBITALS,
        help='number of active orbitals (default: 16)')
    parser.add_argument(
        '--threads', type=int, default=1,
        help='number of PySCF/OpenMP threads (default: 1)')
    parser.add_argument(
        '--seed', type=int, default=12,
        help='random seed for the dense active orbitals (default: 12)')
    parser.add_argument(
        '--csv', type=str,
        help='optional CSV output path; results are always printed')
    return parser.parse_args()


def main():
    args = parse_args()
    if any(size <= 0 for size in args.sizes):
        raise ValueError('all basis sizes must be positive')
    if args.nactive <= 0:
        raise ValueError('nactive must be positive')
    if args.threads <= 0:
        raise ValueError('threads must be positive')

    lib.num_threads(args.threads)
    fields = (
        'requested_nao', 'actual_nao', 'nbas', 'naux', 'aux_nbas',
        'nactive', 'active_pair_factors', 'wall_seconds', 'cpu_seconds',
        'max_trace_error', 'integral_norm',
    )
    print(
        f"DF-SSC integral benchmark: basis={PRIMARY_BASIS}, "
        "auxbasis=standard-RI, "
        f"nactive={args.nactive}, "
        f"threads={args.threads}", flush=True)
    print(
        f"{'NAO':>6} {'Naux':>7} {'AO shells':>10} {'aux shells':>11} "
        f"{'wall/s':>12} {'cpu/s':>12} {'trace error':>14}",
        flush=True)

    results = []
    for index, nao in enumerate(args.sizes):
        if args.nactive > nao:
            raise ValueError(
                f"nactive={args.nactive} exceeds requested basis size {nao}")
        print(f"Starting basis size {nao} ...", flush=True)
        result = benchmark_one(nao, args.nactive, args.seed + index)
        results.append(result)
        print(
            f"{result['actual_nao']:6d} {result['naux']:7d} "
            f"{result['nbas']:10d} {result['aux_nbas']:11d} "
            f"{result['wall_seconds']:12.3f} "
            f"{result['cpu_seconds']:12.3f} "
            f"{result['max_trace_error']:14.3e}",
            flush=True)

    if args.csv:
        with open(args.csv, 'w', newline='', encoding='utf-8') as output:
            writer = csv.DictWriter(output, fieldnames=fields)
            writer.writeheader()
            writer.writerows(results)
        print(f"Wrote {args.csv}", flush=True)


if __name__ == '__main__':
    main()
