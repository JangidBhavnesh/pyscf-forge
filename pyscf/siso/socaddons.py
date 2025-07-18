#!/usr/bin/env python
# Copyright 2014-2025 The PySCF Developers. All Rights Reserved.
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

# Author: Bhavnesh Jangid <jangidbhavnesh@uchicago.edu>

'''
Helper Functions for SOC
'''

import numpy as np
from pyscf import lib, mcscf
from pyscf.csf_fci import csf_solver
from pyscf.siso import amfi as amfIntegrals

logger = lib.logger

def socintegrals(mol, somf=True, amf=True, mmf=False, soc1e=True, soc2e=True, ham='DK', dm=None):
    '''
    Wrapper for the SOC integral generation
    1e and 2e integrals are generated using the amfi module.
    In case of mmfi, the parent wavefunction density matrix is used.
    args:
        mol:
            molecule object
        somf: bool
            spin-orbit mean field integrals.
        amf: bool
            atomic mean field integrals.
            In this case, amf dm is generated.
        mmf: bool
            molecular mean field integrals.
        soc1e: bool
            include 1e SOC integrals.
        soc2e: bool
            include 2e SOC integrals.
        ham: str
            SOC Hamiltonian (BP or DK)
        dm: np.array (nao, nao), optional
            density matrix of parent wavefunction.
    returns:
        hso:
            SOC integrals of dimension (3, nao, nao)
    '''

    # Sanity checks
    assert ham in ('BP', 'DK'),\
        "Only Breit-Pauli or Douglas-Kroll Hamiltonian are available."

    assert somf, "Explicit 2e SOC integrals are implemented yet."

    if mol.has_ecp and len(mol._pseudo) > 0:
        raise NotImplementedError("SOC with GTH-PP is not implemented yet.")

    if mol.has_ecp and mol.has_ecp_soc:
        hso =  -0.5j*mol.intor('ECPso', comp=3)
        return hso

    assert soc1e or soc2e, "Atleast one of the SOC integrals should be included."

    if amf:
        dm0 = amfIntegrals.compute_amfi_dm(mol)
    elif mmf:
        assert dm is not None, \
            "For mmf, the density matrix of the parent wavefunction must be provided."
        dm0 = dm

    log = logger.Logger(mol.stdout, mol.verbose)
    cpu0 = logger.process_clock(), logger.perf_counter()
    hso1e, hso2e = amfIntegrals.compute_soc_integrals(mol, dm0, ham=ham)
    log.timer("SOC integrals generation took: ", *cpu0)

    if soc1e and soc2e:
        hso = hso1e+hso2e
    elif soc1e:
        hso = hso1e
    elif soc2e:
        hso = hso2e
    return hso

def sacasscf_solver(mc, states, weights=None, ms=None):
    '''
    Wrapper for the generating the SACASSCF solver.
    args:
        mc: pyscf.mcscf.CASSCF object
            CASSCF object to be used for SACASSCF.
        states: list of tuples
            Each tuple contains (nroots, spinmult, wfnsym).
            nroots: int, number of roots for the state.
            spinmult: int, spin multiplicity of the state.
            wfnsym: int or None, symmetry of the wavefunction.
        weights: np.array or None
            Weights for each state. If None, equal weights are assigned.
        ms: str or None
            Method for mixing states. If 'lin', linear mixing is used.
            Otherwise, state average mixing is used.
    returns:
        mc: state-averaged/mix CAS object
    '''
    mol = mc._scf.mol

    def _construct_solver(mol, smult, wfnsym, nroots):
        solver = csf_solver(mol, smult=smult)
        solver.wfnsym = wfnsym
        solver.nroots = nroots
        solver.spin = smult - 1
        return solver

    if not isinstance(states, (list, tuple)):
        raise TypeError("states must be a list of tuples (nroots, spinmult, wfnsym)")

    states= sorted(states, key=lambda x: x[1])

    states = [state if len(state) > 2
              else state + (None,) * (3 - len(state))
              for state in states]

    solvers = [_construct_solver(mol, smult, wfnsym, nroots)
                          for (nroots, smult, wfnsym) in states]

    statetot = sum(state[0] for state in states)
    weights = np.ones(statetot) / statetot if weights is None else weights

    if ms == 'lin':
        return mc.multi_state_mix(solvers, weights, "lin")
    else:
        return mcscf.state_average_mix_(mc, solvers, weights)

if __name__ == "__main__":
    from pyscf import scf, gto
    xyz ='''O  0.00000000   0.08111156   0.00000000
            H  0.78620605   0.66349738   0.00000000
            H -0.78620605   0.66349738   0.00000000'''
    mol = gto.M(atom=xyz, basis='cc-pvtz-dk', verbose=5)
    mf = scf.RHF(mol).run()
    dm = mf.make_rdm1()

    # AMFI Integrals
    hso = socintegrals(mol, ham='DK')

    # MMFI Integrals
    # hso_mmfi = socintegrals(mol, amf=False, mmf=True, ham='DK', dm=dm)

