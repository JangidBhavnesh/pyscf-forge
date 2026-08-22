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

"""Compute the O2 ground-triplet zero-field splitting from SOC and SSC.

CAS(8,6) spans the six O 2p orbitals.  The model space contains the four
lowest singlet and four lowest triplet roots, giving 16 spin-projection states
after expansion over M_S.  The excited roots represent the second-order SOC
splitting of X 3-Sigma-g-minus.

The SSC Hamiltonian uses RI integrals with PySCF's standard
correlation-fitting auxiliary basis.  The three SOC-mixed levels belonging to
the ground triplet are identified by their overlap with its spin-free M_S
components.
"""

import numpy as np

from pyscf import gto, mcpdft, scf, siso, mcscf
from pyscf.data.nist import HARTREE2WAVENUMBER
from pyscf.mcscf import avas
from pyscf.siso.ss_int_helper import (ground_triplet_levels,
                                      triplet_zfs_parameters)



def print_zfs(label, hamiltonian, reference_indices):
    energies, weights = ground_triplet_levels(
        hamiltonian, reference_indices)
    d_value, e_value, levels = triplet_zfs_parameters(energies)
    conversion = HARTREE2WAVENUMBER
    print(f'{label:>12}  D = {d_value * conversion: .9f} cm^-1  '
          f'E = {e_value * conversion: .9f} cm^-1')
    print(f'{"":>12}  centered levels = '
          f'{np.array2string(levels * conversion, precision=9)} cm^-1')
    print(f'{"":>12}  ground-triplet weights = '
          f'{np.array2string(weights, precision=9)}')


mol = gto.M(
    atom='O 0 0 -0.6035; O 0 0 0.6035',
    unit='Angstrom',
    basis='ano@4s3p2d1f',  # ANO-RCC-VTZP for oxygen
    spin=2,
    symmetry=False,
    verbose=4,
)

mf = scf.ROHF(mol).density_fit().sfx2c1e().run()
mo_coeff = avas.kernel(mf, ['O 2p'], minao=mol.basis)[2]

# Four singlet and four triplet roots in a CAS(8,6) O 2p active space.
modelspace = [(4, 1), (4, 3)]
mc = mcscf.CASSCF(mf, 6, 8)
mc = siso.state_average_solver(mc, modelspace, )
mc.conv_tol = 1e-9
mc.run(mo_coeff)

my_siso = siso.SISO(mc, modelspace, ham='DKH')
my_siso.build_imds()
h_soc_interaction = my_siso.compute_soc_hamiltonian()
h_soc = my_siso.compute_hamiltonian()
h_spin_free = h_soc - h_soc_interaction
h_ssc = my_siso.compute_ssc_hamiltonian(use_df=True)

h_soc = 0.5 * (h_soc + h_soc.conj().T)
h_spin_free = 0.5 * (h_spin_free + h_spin_free.conj().T)
h_ssc = 0.5 * (h_ssc + h_ssc.conj().T)
h_ssc_only = h_spin_free + h_ssc
h_total = h_soc + h_ssc

# The four singlets precede the three M_S components of the lowest triplet.
reference_indices = np.arange(4, 7)
triplet_block = np.ix_(reference_indices, reference_indices)

np.set_printoptions(precision=9, suppress=True)
print('\nDirect ground-triplet interaction blocks in cm^-1')
print('SOC:')
print(np.real_if_close(
    h_soc_interaction[triplet_block] * HARTREE2WAVENUMBER))
print('SSC:')
print(np.real_if_close(h_ssc[triplet_block] * HARTREE2WAVENUMBER))

print('\nTriplet zero-field-splitting parameters')
print_zfs('SOC', h_soc, reference_indices)
print_zfs('SSC', h_ssc_only, reference_indices)
print_zfs('SOC + SSC', h_total, reference_indices)
