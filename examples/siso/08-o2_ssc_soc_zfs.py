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

from pyscf import gto, mcpdft, scf, siso, mcscf
from pyscf.mcscf import avas


mol = gto.M(
    atom='O 0 0 -0.6035; O 0 0 0.6035',
    unit='Angstrom',
    basis='ano@4s3p2d1f',  # ANO-RCC-VTZP for oxygen
    spin=2,
    symmetry=False,
    verbose=4,
)

mf = scf.ROHF(mol).density_fit().sfx2c1e()
# mf.chkfile = 'mf.chk'
mf.kernel()

mo_coeff = avas.kernel(mf, ['O 2p'], minao=mol.basis)[2]

# Four singlet and four triplet roots in a CAS(8,6) O 2p active space.
modelspace = [(4, 1), (4, 3)]
mc = mcpdft.CASSCF(mf, 'tPBE', 6, 8)
mc = siso.state_average_solver(mc, modelspace,ms='lin')
mc.kernel(mo_coeff)

my_siso = siso.SISO(mc, modelspace, ham='DKH', ssc=True)
my_siso.kernel()

# With no arguments, analyze the first root of every supported multiplicity
# in the model space.  Here that is the lowest triplet root.
siso.compute_D_and_E(my_siso, mltp=[3,], nroots=[1,])
