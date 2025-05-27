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
Quasi-Degenerate Perturbation Theory Based Spin-Orbit Coupling Treatment
'''

import numpy as np
from sympy import symbols
from sympy.physics.quantum.cg import CG
from itertools import product
from functools import reduce
from pyscf import scf, lib, fci
from pyscf.siso import socaddons
import ctypes

libsiso = lib.load_library('libsiso')
libsiso.SOCcompute_ss.restype = None
libsiso.SOCcompute_ssp.restype = None
libsiso.SOCcompute_ssm.restype = None

logger = lib.logger

au2ev = 27.21138602
au2cminv = 219474.6313705

# Forked from: https://github.com/IrisA144/liblan_preview

def calculate_zmat(siso, somf=True, amf=True, mmf=False, soc1e=True, soc2e=True, ham='DK'):
    """
    Computing the SOC integrals.

    Args:
        siso:
            instance of class SISO
        somf: 
            spin-orbit mean field integrals.
        amf:
            atomic mean field integrals.
        mmf:
            molecular mean field integrals.
        soc1e:
            include 1e SOC integrals.
        soc2e:
            include 2e SOC integrals.
        ham:
            SOC Hamiltonian (BP or DK)
    Returns:
        zsoc: 
            SOC integrals of dimension (3, ncas, ncas)    
    """
    mc = siso.mc
    mol = mc._scf.mol
    ncas = mc.ncas
    ncore = mc.ncore

    dm = None
    if mmf:
        dm = mc.make_rdm1()
    
    # Generate the SOC integrals
    hso = socaddons.socintegrals(mol, somf=somf, amf=amf, mmf=mmf, 
                                 soc1e=soc1e, soc2e=soc2e, ham=ham, dm=dm)
    
    # Basis transformation
    mo_cas = mc.mo_coeff[:, ncore:ncore+ncas]
    h1 = np.asarray([reduce(np.dot, (mo_cas.T, hso_, mo_cas)) for hso_ in hso])

    # Cartesian to Spherical tensor transformation
    zsoc = np.zeros((3, ncas, ncas), dtype=hso[0].dtype)
    zsoc[0] = 1./np.sqrt(2)  * (h1[0] - 1.j*h1[1])
    zsoc[1] = h1[2]
    zsoc[2] = -1./np.sqrt(2) * (h1[0] + 1.j*h1[1])

    del hso, h1

    return zsoc

def assemble_amat(siso):
    '''
    Computing the intermediate a tensor

    Args:
        siso:
            instance of class SISO

    Returns:
        amat:
            a list of nSt elements each of dimernsions 2 x ncia||b x 4  
    '''
    
    def _gen_linkstr_index(ms, ncasorb, alphae, betae):
        ms_map = {-1:fci.cistring.gen_des_str_index,
                 0:fci.cistring.gen_linkstr_index,
                 1:fci.cistring.gen_cre_str_index}
        return [ms_map[ms[0]](ncasorb, alphae), 
                ms_map[ms[1]](ncasorb, betae)]
    
    def _get_alpha_beta(nelec, S):
        alphae = (nelec + S) // 2
        betae = nelec - alphae
        return alphae, betae
    
    Stuples = siso.stuples
    mc = siso.mc
    ncasorb = list(range(mc.ncas))
    nelec = sum(mc.nelecas)

    amat = []
    for (s1, s2) in Stuples:
        if s1 == s2:  # SS part
            nelec_a, nelec_b = _get_alpha_beta(nelec, s1)
            amat.append(_gen_linkstr_index((0,0)), ncasorb, nelec_a, nelec_b)
        elif s1 + 2 == s2:  # SS+1 part
            nelec_a, nelec_b = _get_alpha_beta(nelec, s1+2)
            amat.append(_gen_linkstr_index((-1,1)), ncasorb, nelec_a, nelec_b)
        elif s1 == s2 + 2: # SS-1 part
            nelec_a, nelec_b = _get_alpha_beta(nelec, s1-2)
            amat.append(_gen_linkstr_index((1,-1)), ncasorb, nelec_a, nelec_b)
        else: # No connection
            amat.append(0)

    imds = siso.imds
    imds.a = amat
    return amat

def assemble_civecs(siso):
    '''
    
    '''
    mc = siso.mc
    ci_mc = mc.ci
    twoslst = siso.twoslst
    statelst = siso.statelis
    assert len(statelst) == len(ci_mc)

    cimat = []
    for i in range(len(twoslst)):
        try:
            cimat.append(np.asarray(ci_mc[int(np.sum(statelst[:twoslst[i]])):int(np.sum(statelst[:twoslst[i] + 1]))]))
        except IndexError:
            cimat.append(np.asarray(ci_mc[int(np.sum(statelst[:twoslst[i]])):int(np.sum(statelst))]))
    imds = siso.imds
    imds.c = cimat
    return cimat

def assemble_energy(siso):
    '''
    '''
    mc = siso.mc
    e_states = mc.e_states
    twoslst = siso.twoslst
    tottwos = len(twoslst)
    statelst = siso.statelis
    assert len(statelst) == len(e_states)

    e = []
    for i in range(tottwos):
        try:
            e.append(np.asarray(e_states[int(np.sum(statelst[:twoslst[i]])):int(np.sum(statelst[:twoslst[i] + 1]))]))
        except IndexError:
            e.append(np.asarray(e_states[int(np.sum(statelst[:twoslst[i]])):int(np.sum(statelst))]))
    imds = siso.imds
    imds.e = e
    return e

def compute_cg_coefficients(S, Ms=0):
    """
    Compute the Clebsch-Gordan coefficients for the spin-orbit coupling.

    Args:
        siso:
            instance of class SISO
    Returns:
        cg_coeffs:
            a list of Clebsch-Gordan coefficients for each state
    """
    assert Ms in (0, 1, -1), "Ms should be either 0, 1 or -1."
    Ms_map = {0:S+1, 1:S+3, -1:S-1}
    
    ss = symbols('S')
    sbra = S + 1
    sket = Ms_map[Ms]

    g = np.zeros((3, sbra, sket), dtype='complex')

    for g0 in range(3):
        for g1 in range(sbra):
            phase = (-1) ** (g0 + S / 2 - (g1 - S / 2))
            for g2 in range(sket):
                g[g0, g1, g2] += phase * CG.Wigner3j(ss/2, -(g1 - ss/2), 
                                                     1, 1 - g0, 
                                                     ss/2 + Ms, g2 - ss/2 - Ms
                                                    ).subs(ss,S).doit()
    return g

def compute_dmat(siso):
    '''
    '''
    twoslst = siso.twoslst
    stuples = siso.stuples
    imds = siso.imds

    d_col = []

    def flatten_to_ptr(array, dtype):
        return np.ascontiguousarray(array.flatten()).ctypes.data_as(ctypes.POINTER(dtype))

    for i, (s1,s2) in enumerate(stuples):

        if s1 == s2:  # SS part
            S = s1
            iS = np.where(twoslst == S)[0][0]
            zmat = imds.z[iS] # SOC integrals
            civec = imds.c[iS] # CI vectors
            civec = civec.astype(np.complex128) # Change the dtype to complex128
            alphamat, betamat = imds.a[iS] # coupling coefficients
            b = np.zeros((3, *civec.shape), dtype=np.complex128)

            # Instead of passing all of these shapes, I am passing a single shape array
            shapearray = np.array([*b.shape, 
                                   alphamat.shape[1], 
                                   betamat.shape[1], 
                                   zmat.shape[1], 
                                   civec.shape[1], 
                                   civec.shape[2]],
                                   dtype=np.int32, order='C')


            libsiso.SOCcompute_ss.argtypes = [
                ctypes.POINTER(ctypes.c_double),  # zmat
                ctypes.POINTER(ctypes.c_double),  # civecs
                ctypes.POINTER(ctypes.c_double),  # bdata
                ctypes.POINTER(ctypes.c_int),     # alphadet
                ctypes.POINTER(ctypes.c_int),     # betadet
                ctypes.POINTER(ctypes.c_int)      # shapearray
            ]

            libsiso.SOCcompute_ss(
                flatten_to_ptr(zmat, ctypes.c_double),
                flatten_to_ptr(civec, ctypes.c_double),
                flatten_to_ptr(b, ctypes.c_double),
                flatten_to_ptr(alphamat, ctypes.c_int),
                flatten_to_ptr(betamat, ctypes.c_int),
                flatten_to_ptr(shapearray, ctypes.c_int)
                )
            
            b = b.reshape(3, *civec.shape)
            g = compute_cg_coefficients(S, Ms=0)

            w = np.einsum('kij, mnij->mnk', civec, b)
            d = np.einsum('mij, mkl->kilj', g, w)
            d_col.append(d)
        
        elif s1 + 2 == s2:  # SS+1 part
            S = s1
            iS = np.where(twoslst == S)[0][0]
            iSp = np.where(twoslst == S + 2)[0][0]
            zmat = imds.z[iS] # SOC integrals

            civec = imds.c[iS] # CI vectors
            civec = civec.astype(np.complex128) # Change the dtype to complex128
            civecbra = imds.c[iSp]
            civecbra = civecbra.astype(np.complex128)

            alphamat, betamat = imds.a[iS] # coupling coefficients
            b = np.zeros((3, civec.shape[0], civecbra.shape[1], civecbra.shape[2]), dtype='complex')

            # Instead of passing all of these shapes, I am passing a single shape array
            shapearray = np.array([*b.shape, 
                                   alphamat.shape[1], 
                                   betamat.shape[1], 
                                   zmat.shape[1], 
                                   civec.shape[1], 
                                   civec.shape[2]],
                                   dtype=np.int32, order='C')
            
            libsiso.SOCcompute_ssp.argtypes = [
                ctypes.POINTER(ctypes.c_double),  # zmat
                ctypes.POINTER(ctypes.c_double),  # civecs
                ctypes.POINTER(ctypes.c_double),  # bdata
                ctypes.POINTER(ctypes.c_int),     # alphadet
                ctypes.POINTER(ctypes.c_int),     # betadet
                ctypes.POINTER(ctypes.c_int)      # shapearray
            ]

            libsiso.SOCcompute_ssp(
                flatten_to_ptr(zmat, ctypes.c_double),
                flatten_to_ptr(civec, ctypes.c_double),
                flatten_to_ptr(b, ctypes.c_double),
                flatten_to_ptr(alphamat, ctypes.c_int),
                flatten_to_ptr(betamat, ctypes.c_int),
                flatten_to_ptr(shapearray, ctypes.c_int)
                )
            
            b = b.reshape(3, civec.shape[0], civecbra.shape[1], civecbra.shape[2])

            g = compute_cg_coefficients(S, Ms=1)
            w = np.einsum('kij, mnij->mnk', civecbra, b)
            d = np.einsum('mij, mkl->kilj', g, w)
            d_col.append(d)
        
        elif s1 == s2 + 2: # SS-1 part
            S = s1
            iS = np.where(twoslst == S)[0][0]
            iSm = np.where(twoslst == S - 2)[0][0]
            zmat = imds.z[iS] # SOC integrals

            civec = imds.c[iS] # CI vectors
            civec = civec.astype(np.complex128) # Change the dtype to complex128
            civecbra = imds.c[iSm]
            civecbra = civecbra.astype(np.complex128)

            alphamat, betamat = imds.a[iS] # coupling coefficients
            b = np.zeros((3, civec.shape[0], civecbra.shape[1], civecbra.shape[2]), dtype='complex')

            # Instead of passing all of these shapes, I am passing a single shape array
            shapearray = np.array([*b.shape, 
                                   alphamat.shape[1], 
                                   betamat.shape[1], 
                                   zmat.shape[1], 
                                   civec.shape[1], 
                                   civec.shape[2]],
                                   dtype=np.int32, order='C')
            
            libsiso.SOCcompute_ssm.argtypes = [
                ctypes.POINTER(ctypes.c_double),  # zmat
                ctypes.POINTER(ctypes.c_double),  # civecs
                ctypes.POINTER(ctypes.c_double),  # bdata
                ctypes.POINTER(ctypes.c_int),     # alphadet
                ctypes.POINTER(ctypes.c_int),     # betadet
                ctypes.POINTER(ctypes.c_int)      # shapearray
            ]
            libsiso.SOCcompute_ssm(
                flatten_to_ptr(zmat, ctypes.c_double),
                flatten_to_ptr(civec, ctypes.c_double),
                flatten_to_ptr(b, ctypes.c_double),
                flatten_to_ptr(alphamat, ctypes.c_int),
                flatten_to_ptr(betamat, ctypes.c_int),
                flatten_to_ptr(shapearray, ctypes.c_int)
                )
            b = b.reshape(3, civec.shape[0], civecbra.shape[1], civecbra.shape[2])
            g = compute_cg_coefficients(S, Ms=-1)
            w = np.einsum('kij, mnij->mnk', civecbra, b)
            d = np.einsum('mij, mkl->kilj', g, w)
            d_col.append(d)

        else: # No connection
            d_col.append(0)
        
    return d_col

def compute_hamiltonian(siso):
    """
    Compute the Hamiltonian matrix for the spin-orbit coupling.

    Args:
        siso:
            instance of class SISO
    Returns:
        hamiltonian:
            a list of Hamiltonian matrices for each state
    """
    statelst = siso.statelis
    twoslst = siso.twoslst
    totalspins = len(twoslst)
    stuples = siso.stuples
    imds = siso.imds
    d = imds.d
    e = imds.e
    
    h_col = [[0] * totalspins for i in range(totalspins)]

    for i in range(totalspins):
        for j in range(totalspins):
            s1, s2 = twoslst[i], twoslst[j]
            indt = stuples.index((s1, s2))
            counter = [(i+1)*(x) for i, x in enumerate(statelst)]

            if s1 == s2:
                S = s1
                nstates = counter[S]
                if S!=0:
                    coeff = np.sqrt((S / 2 + 1) * (S + 1) / (S / 2)) / 2
                else:
                    coeff = 0.0
                
                h = coeff * d[indt].reshpae(nstates, nstates, order='C')

                for ns in range(nstates):
                    n1 = divmod(ns, S+1)[0]
                    h[ns, ns] += e[i][n1]

            elif s1 + 2 == s2:
                S = s1
                nstatesa = counter[S]
                nstatesb = counter[S + 2]
                coeff = np.sqrt((S + 3) / 2)
                h = coeff * d[indt].reshape((nstatesa, nstatesb), order='C')
               
            elif s1 == s2 + 2:  # SS-1 part
                S = s1
                nstatesa = counter[S]
                nstatesb = counter[S - 2]
                coeff = -np.sqrt((S + 1) / 2)
                h = coeff * d[indt].reshape((nstatesa, nstatesb), order='C')
            
            else:  # No connection
                nstatesa = counter[s1]
                nstatesb = counter[s2]
                h = np.zeros((nstatesa, nstatesb), dtype=np.complex128)
            h_col[i][j] = h

    return np.block(h_col)

def build_imds(siso):
    imds = siso.imds
    somf = siso.somf
    amf = siso.amf
    mmf = siso.mmf
    soc1e = siso.soc1e
    soc2e = siso.soc2e
    ham = siso.ham
    
    imds.z = calculate_zmat(siso, somf=somf, amf=amf, mmf=mmf,
                            soc1e=soc1e, soc2e=soc2e, ham=ham)
    imds.a = assemble_amat(siso)
    imds.c = assemble_civecs(siso)
    imds.e = assemble_energy(siso)
    imds.d = compute_dmat(siso)
    return siso

def kernel(siso):
    """
    Driver function for SI-SO
    """
    logger.debug(siso, 'Starting SI-SO kernel')
    siso.build_imds()
    hso = siso.compute_hamiltonian()
    # Sanity checks
    assert np.allclose(hso, hso.conj().T),\
        f"Hamiltonian is not Hermitian, max deviation: {np.max(np.abs(hso - hso.conj().T))}"
    siso.si_energies, siso.si_vecs = np.linalg.eigh(hso)
    siso._finalize()
    return siso.si_energies, siso.si_vecs

class _IMDS:
    """
    SI-SO intermediates
    Attributes:
    ----------
        self.z   :   (3,ncas,ncas)
        self.a   :   (nSt,2,ncia||b,4)
        self.b          (3, nstates nciam ncib) for given nS
        self.c      :   (nS,nstates,ncia,ncib)
        self.d      :   (nS,nstates,nstates)
        self.e      :   (nS,nstates)
    """
    def __init__(self):
        self.zmat = None
        self.a = None
        self.c = None
        self.e = None
        self.d = None

class SISO(lib.StreamObject):
    """
    Quasi-Degenerate Perturbation Theory Based Spin-Orbit Coupling Treatment.: 
    State interaction Spin-Orbit Coupling (SISO) class.
    """
    _keys=['statelst', 'twoslst', 'stuples','si_energies', 'si_vecs']

    def __init__(self, mc, modelspace, somf=True, amf=True, mmf=False, soc1e=True, soc2e=True, ham='DK'):
        self.mc = mc
        self.somf = somf
        self.amf = amf
        self.mmf = mmf
        self.soc1e = soc1e
        self.soc2e = soc2e
        self.ham = ham
        self.imds = _IMDS()
        self.initialize(modelspace)
        self.sanity_checks()
        self.dump_flags()

    def initialize(self, modelspace):
        statelis= sorted(modelspace, key=lambda x: x[1])
        SMlst = [state[1] - 1 for state in statelis]
        statelis_ = np.zeros(max(SMlst) + 1, dtype=int)
        for state in statelis:
            statelis_[state[1] - 1] = state[0]
        self.statelis = statelis_.tolist()
        self.twoslst = np.nonzero(self.statelis)[0]
        self.stuples = [x for x in product(self.twoslst, self.twoslst)]
        return self

    def sanity_checks(self):
        """
        Perform sanity checks on the input parameters.
        """
        assert self.ham in ('BP', 'DK'), "Only Breit-Pauli or Douglas-Kroll Hamiltonian are available."
        assert self.somf, "Explicit 2e SOC integrals are not implemented yet."
        if self.mc._scf.mol.has_ecp():
            raise NotImplementedError("ECP is not supported yet.")
        assert self.soc1e or self.soc2e, "At least one of the SOC integrals should be included."
        return self
    
    def dump_flags(self):
        log = logger.Logger(self.mc.stdout, self.mc.verbose)
        log.note(" ")
        log.note("Spin-Orbit Coupling using Quasi-Degenerate Perturbation Theory")
        log.note("Spin-Orbit Mean Field Integrals: %s", self.somf)
        log.note("Atomic Mean Field Integrals: %s", self.amf)
        log.note("Molecular Mean Field Integrals: %s", self.mmf)
        log.note("Include 1e SOC integrals: %s", self.soc1e)
        log.note("Include 2e SOC integrals: %s", self.soc2e)
        log.note("SOC Hamiltonian: %s", self.ham)
        log.note("Speed of ligt: %.2f m/s", lib.param.LIGHT_SPEED)
        log.note(" ")
        return self
    
    def build_imds(self):
        return build_imds(self)

    def _calc_h(self):
        return compute_hamiltonian(self)

    def kernel(self):
        return kernel(self)
    
    def _finalize(self):
        mc = self.mc

        log = logger.Logger(mc.stdout, mc.verbose)
        nroots=len(self.mag_energy)
        log.note(" ")
        log.note("*** Spin Orbit Coupling Energetics ***")
        log.note(" ")
        for i in range(nroots):
            log.note(" SO-CASSI State %d Total Energy = %.12g ",
                    i,
                    self.mag_energy[i])

        log.note(" ")
        log.note("*** Relative Spin Orbit Coupling Energetics ***")
        log.note(" ")
        log.note("SO State       Relative Energy(au)   Relative Energy(eV)   Relative Energy(cm$^{-1}$)")
        for i in range(nroots):
            log.note(" {:<10} {:>20.9f} {:>20.5f} {:>20.5f}".format(
                i,
                self.mag_energy[i] - self.mag_energy[0],
                au2ev * (self.mag_energy[i] - self.mag_energy[0]),
                au2cminv * (self.mag_energy[i] - self.mag_energy[0])))
       

