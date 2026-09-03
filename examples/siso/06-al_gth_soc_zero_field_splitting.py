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

'''Zero-field SOC fine-structure splitting of the Al 2P ground term.

Neutral aluminum has a 3s2 3p1, S=1/2 ground state.  Consequently, its
conventional spin-Hamiltonian ZFS parameters D and E are zero by Kramers
symmetry.  The meaningful zero-field observable for the free atom is instead
the SOC fine-structure gap between the 2P_1/2 and 2P_3/2 manifolds.  This
example computes that gap with GTH-SOC-PADE and GTH-DZVP.
'''

import numpy as np

from pyscf import gth_soc, gto, mcscf, scf, siso
from pyscf.data.nist import HARTREE2WAVENUMBER


mol = gto.M(
    atom='Al 0 0 0',
    basis='gth-dzvp',
    pseudo='gth-soc-pade',
    spin=1,
    symmetry=False,
    verbose=4,
)

# The Al GTH pseudopotential leaves three explicit valence electrons.  The
# active space contains the 3s orbital and all three components of 3p.
mf = scf.ROHF(mol).run()
modelspace = [(3, 2)]
mc = mcscf.CASSCF(mf, 4, 3)
mc = siso.state_average_solver(mc, modelspace)
mc.kernel()

# For a GTH pseudopotential, SISO automatically selects the GTH-SOC operator.
my_siso = siso.SISO(mc, modelspace, ham='BP')
my_siso.kernel()

soc_hamiltonian = my_siso.compute_soc_hamiltonian()
hermiticity_error = np.max(
    np.abs(soc_hamiltonian - soc_hamiltonian.conj().T))

energies = np.sort(my_siso.si_energies)
e_j12 = energies[:2].mean()
e_j32 = energies[2:].mean()
fine_structure = (e_j32 - e_j12) * HARTREE2WAVENUMBER
# https://physics.nist.gov/PhysRefData/Handbook/Tables/aluminumtable5.htm
nist_fine_structure = 112.061  # NIST Atomic Spectra Database, Al I

print(f'Al 2P_1/2 -> 2P_3/2 splitting: {fine_structure:.6f} cm^-1')
print(f'Difference from NIST ({nist_fine_structure:.3f} cm^-1): '
      f'{fine_structure - nist_fine_structure:+.6f} cm^-1')
print('Conventional S=1/2 ZFS parameters: D = E = 0')
print('SOC Hamiltonian Hermiticity error:', hermiticity_error)

# Useful internal checks for this atomic benchmark: Kramers degeneracy of the
# J=1/2 pair and fourfold rotational degeneracy of the J=3/2 manifold.
print('2P_1/2 spread:',
      np.ptp(energies[:2]) * HARTREE2WAVENUMBER, 'cm^-1')
print('2P_3/2 spread:',
      np.ptp(energies[2:]) * HARTREE2WAVENUMBER, 'cm^-1')
