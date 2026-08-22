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

"""Compare full and RI spin--spin coupling integrals in the AO basis.

This is an integral-level validation of the RI factorization in

    D. Ganyushin, N. Gilka, P. R. Taylor, C. M. Marian, and F. Neese,
    J. Chem. Phys. 132, 144111 (2010), DOI: 10.1063/1.3367718.

The small water/STO-3G system keeps the full four-center reference inexpensive.
Several increasingly large standard Coulomb-fitting bases illustrate the
dependence of the SSC error on the auxiliary space.
"""

import time

import numpy as np

from pyscf import df, gto
from pyscf.siso.ss_int_helper import (compute_ssc_integrals,
                                      compute_ssc_integrals_ri)


AUXILIARY_BASES = (
    'weigend',
    'def2-universal-jkfit',
    'cc-pvdz-jkfit',
)


def error_statistics(reference, approximation):
    error = approximation - reference
    return {
        'rms': np.sqrt(np.mean(np.abs(error) ** 2)),
        'maximum': np.max(np.abs(error)),
        'relative_frobenius': np.linalg.norm(error) / np.linalg.norm(reference),
    }


def main():
    mol = gto.M(
        atom='''
            O  0.000000  0.000000  0.000000
            H  0.000000 -0.757000  0.587000
            H  0.000000  0.757000  0.587000
        ''',
        basis='sto-3g',
        unit='Angstrom',
        verbose=0,
    )

    start = time.perf_counter()
    full = compute_ssc_integrals(mol)
    full_time = time.perf_counter() - start

    print(f'Water/STO-3G: nao={mol.nao_nr()}, nbas={mol.nbas}')
    print(f'Full AO SSC tensor: {full_time:.6f} s, '
          f'norm={np.linalg.norm(full):.10e}')
    print()
    print(f"{'auxiliary basis':>24} {'naux':>6} {'time/s':>10} "
          f"{'RMS error':>14} {'max error':>14} {'relative Fro.':>14}")

    for auxbasis in AUXILIARY_BASES:
        auxmol = df.addons.make_auxmol(mol, auxbasis)
        start = time.perf_counter()
        hss_ri = compute_ssc_integrals_ri(mol, auxbasis=auxbasis)
        elapsed = time.perf_counter() - start
        error = error_statistics(full, hss_ri)
        print(
            f'{auxbasis:>24} {auxmol.nao_nr():6d} {elapsed:10.6f} '
            f"{error['rms']:14.6e} {error['maximum']:14.6e} "
            f"{error['relative_frobenius']:14.6e}")


if __name__ == '__main__':
    main()
