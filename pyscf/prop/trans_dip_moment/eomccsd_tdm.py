#!/usr/bin/env python
# Copyright 2014-2024 The PySCF Developers. All Rights Reserved.
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
#
# Author: Bhavnesh Jangid <jangidbhavnesh@uchicago.edu>

import numpy as np
from pyscf import scf, cc, gto

# The TDM for the EE-EOM-CCSD Method
# J. Chem. Theory Comput. 2019, 15, 4036−4043; has given the equations in the SO-basis
# I have converted it to MO-basis and implemented it here.

# Note that: I have using left vectors as the inverse of the right eigen vectors.
# I have to compare the tdm with some other code to make sure this implementation is correct.

def trans_rdm(r0,r1,r2,l0,l1,l2,t1,t2,deltaij):
    '''
    Transition density matrix for the EE-EOM-CCSD method.
    args:
        r0:
        r1: (nocc, nvir)
        r2: (nocc, nocc, nvir, nvir)
        l0:
        l1: (nvir, nocc)
        l2:
        t1: (nocc, nvir)
        t2: (nocc, nocc, nvir, nvir)
        deltaij:

    returns:
        TDM (nao*nao)
    '''

    # Occ-Occ block
    Dij = np.einsum('ei,je->ij', l1, t1)
    Dij += ( 2.*np.einsum('efmi,mjef->ij', l2, t2) -
            np.einsum('efmi,mjfe->ij', l2, t2))
    Dij *= -r0
    Dij -= np.einsum('je,ei->ij', r1, l1)
    tau = ( np.einsum('me,jf->mjef', r1, t1) + 0.5 * r2)
    Dij -= (2.*np.einsum('efmi,mjef->ij', l2, tau)
            - np.einsum('efmi, mjfe->ij', l2, tau))
    Dij += deltaij
    del tau

    # Vir-Vir block
    Dab = np.einsum('bm,ma->ab', l1, t1)
    Dab += ( 2.*np.einsum('ebmn,mnea->ab', l2, t2) -
            np.einsum('ebmn,mnae->ab', l2, t2))
    Dab *= r0
    Dab += np.einsum('ma,bm->ab', r1, l1)
    tau = (np.einsum('me,na->mnea', r1, t1) + 0.5 * r2)
    Dab += (2.*np.einsum('ebmn,mnea->ab', l2, tau) -
            np.einsum('ebmn, mnae->ab', l2, tau))
    del tau

    # Occ-Vir block
    Dia = r0 * l1.T
    Dia += np.einsum('me,eami->ia', r1, l2)

    # Vir-Occ block
    Dai = deltaij * t1.T
    Dai += r0 * (2.*np.einsum('em, miea->ai', l1, t2) -
                 np.einsum('em, miae->ai', l1, t2))
    Dai -= r0 * (np.einsum('em, ie, ma -> ai', l1, t1, t1))
    Dai -= r0 * (2.*np.einsum('efmn, mnea, if->ai', l2, t2, t1) -
                 np.einsum('efmn, mnae, if->ai', l2, t2, t1))
    Dai -= r0 * (2.*np.einsum('efmn, mief, na->ai', l2, t2, t1) -
                 np.einsum('efmn, mife, na->ai', l2, t2, t1))
    Dai += l0 * r1.T
    Dai += (2.*np.einsum('em, miea->ai', l1, r2) -
            np.einsum('em, miae->ai', l1, r2))
    Dai -= np.einsum('em , ma, ie->ai', l1, r1, t1)
    Dai -= np.einsum('em , ie, ma->ai', l1, r1, t1)
    tau = ( 2. * np.einsum('efmn, me->nf', l2, r1) -
           np.einsum('efnm, me->nf', l2, r1))
    Dai += (2.*np.einsum('nf, nifa->ai', tau, t2) -
            np.einsum('nf, niaf->ai', tau, t2))
    Dai -= np.einsum('nf, if, na->ai', tau, t1, t1)
    del tau

    Dai -= (2.*np.einsum('efmn, na, mief->ai', l2, r1, t2) -
            np.einsum('efmn, na, mife->ai', l2, r1, t2))
    Dai -= (2.*np.einsum('efmn, mief, na->ai', l2, r2, t1) -
            np.einsum('efmn, mife, na->ai', l2, r2, t1))
    Dai -= (2.*np.einsum('efmn, if, mnea->ai', l2, r1, t2) -
            np.einsum('efmn, if, miae->ai', l2, r1, t2))
    Dai -= (2.*np.einsum('efmn, mnea, if->ai', l2, r2, t1) -
            np.einsum('efmn, mnae, if->ai', l2, r2, t1))

    # Now assemble the tdm
    nocc, nvir = t1.shape
    nmo = nocc + nvir
    D = np.zeros([nmo, nmo], dtype=t1.dtype)
    D[:nocc, :nocc] += Dij
    D[:nocc, nocc:] += Dia
    D[nocc:, :nocc] += Dai
    D[nocc:, nocc:] += Dab

    del Dij, Dia, Dai, Dab

    return D


def compute_r0(t1, r1, r2, deltaE, eris):
    '''
    Compute the r0 vector for the EOM-CCSD method.
    args:
        t1: (nocc, nvir)
        r1: (nocc, nvir)
        r2: (nocc, nocc, nvir, nvir)
        deltaE: Energy difference between the excited state and the ground state
        eris: Eris object
    return:
        r0: float
    '''
    nocc = t1.shape[0]

    # Compute the fock contribution
    fov = eris.fock[:nocc, nocc:]
    ovov = np.asarray(eris.ovov)
    Fov = 2.*np.einsum('kcld,ld->kc', ovov, t1)
    Fov -= np.einsum('kdlc,ld->kc', ovov, t1)
    Fov += fov

    del ovov

    # Compute the matrix-vector product
    oovv = np.asarray(eris.oovv)
    mat = 2.*np.einsum('kc, kc->', Fov, r1)
    mat += 2.*np.einsum('klcd, klcd->', oovv, r2)
    mat -= np.einsum('lkcd, klcd->', oovv.transpose(1,0,2,3), r2)

    # Now compute r0
    r0 = mat / deltaE
    return r0

if __name__ == '__main__':

    # Define the geometry
    mol = gto.Mole(atom = '''
    H        0.61473       -0.02651        0.47485
    O        0.13157        0.02998       -0.34219
    H       -0.79537       -0.00348       -0.13266
    ''',
    basis = 'cc-pvdz')
    mol.verbose=4
    mol.build()

    # Mean-field calculation
    mf = scf.RHF(mol)
    mf.kernel()

    # Solve the CCSD
    mycc = cc.CCSD(mf)
    mycc.kernel()

    # Solve the lambda equation
    mycc.solve_lambda()

    # Solve the EOM-CCSD (First singlet)
    extene, vee = mycc.eomee_ccsd_singlet(nroots=1)

    # Construct the TDMs:
    # Note that, I am using left vectors as the inverse of right vector
    r1, r2 = cc.eom_rccsd.vector_to_amplitudes_ee(np.array(vee), mycc.nmo, mycc.nocc)
    eris = mycc.ao2mo()

    r0 = compute_r0(mycc.t1, r1, r2, extene, eris)
    l1 = r1.T
    l2 = r2.transpose(3,2,1,0)
    tdm_left = trans_rdm(1, np.zeros_like(r1), np.zeros_like(r2),0,l1,l2,mycc.t1,mycc.t2,deltaij=0)
    tdm_right = trans_rdm(r0,r1,r2,1,mycc.l1.T,mycc.l2.transpose(3,2,1,0),mycc.t1,mycc.t2,deltaij=0)


    # Dipole integrals: be careful with the gauge origin of the dipole integrals
    charges = mol.atom_charges()
    coords = mol.atom_coords()  # in a.u.
    nuc_charge_center = np.einsum('z,zx->x', charges, coords) / charges.sum()
    mol.set_common_orig_(nuc_charge_center)
    dip_ints = mol.intor('cint1e_r_sph', comp=3)


    # Now calculate the tdm
    lefttdm = np.einsum('xpq, pq->x', dip_ints, tdm_left)
    righttdm = np.einsum('xpq, pq->x', dip_ints, tdm_right)
    tdm = lefttdm*righttdm
    print(tdm)
    transition_dipole_moment = np.linalg.norm(tdm)

    # Osci. Str.
    osc = 2./3 * extene * transition_dipole_moment
    print(osc)
