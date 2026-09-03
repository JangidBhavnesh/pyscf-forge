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

'''Analytic molecular and periodic GTH pseudopotential SOC integrals.'''

import numpy as np
from pyscf import gto
from pyscf.gto.mole import intor_cross
from pyscf.pbc.gto.pseudo import pp_int as pbc_pp_int


def _soc_projectors(pp):
    projectors = getattr(pp, 'soc_projectors', None)
    if projectors is not None:
        return projectors

    # Compatibility with the original prototype, which appended the number
    # of SOC blocks and the blocks themselves to the scalar parameter list.
    first = pp[4] + 5
    if len(pp) > first:
        count = int(pp[first])
        return pp[first+1:first+1+count]
    return ()


def fake_cell_vnl_so(cell):
    '''Generate the auxiliary cell for the spin-orbit part of V_nl.'''
    fake_env = [cell.atom_coords().ravel()]
    fake_atm = cell._atm.copy()
    fake_atm[:, gto.PTR_COORD] = np.arange(0, cell.natm * 3, 3)
    ptr = cell.natm * 3
    fake_bas = []
    kl_blocks = []

    for atom_id in range(cell.natm):
        if cell.atom_charge(atom_id) == 0:
            continue
        symbol = cell.atom_symbol(atom_id)
        if symbol not in cell._pseudo:
            continue

        for l, (radius, nproj, coupling) in enumerate(
                _soc_projectors(cell._pseudo[symbol]), start=1):
            if nproj == 0:
                continue
            alpha = .5 / radius**2
            norm = gto.gto_norm(l, alpha)
            fake_env.append([alpha, norm])
            fake_bas.append([atom_id, l, 1, 1, 0, ptr, ptr+1, 0])
            factors = np.array([
                pbc_pp_int._PLI_FAC[l, i] / radius**(i*2)
                for i in range(nproj)
            ])
            coupling = np.einsum(
                'i,ij,j->ij', factors, np.asarray(coupling), factors)
            kl_blocks.append(coupling)
            ptr += 2

    fakecell = cell.copy(deep=False)
    fakecell._atm = np.asarray(fake_atm, dtype=np.int32)
    fakecell._bas = np.asarray(fake_bas, dtype=np.int32).reshape(
        -1, gto.BAS_SLOTS)
    fakecell._env = np.asarray(np.hstack(fake_env), dtype=np.double)
    return fakecell, kl_blocks


def _contract_soc(half_integrals, fakecell, coupling_blocks, nao, nkpts):
    result = np.zeros((nkpts, 3, nao, nao), dtype=np.complex128)
    for kpoint_id in range(nkpts):
        offsets = [0] * 3
        for shell_id, coupling in enumerate(coupling_blocks):
            l = fakecell.bas_angular(shell_id)
            angular_momentum = fakecell.intor(
                'int1e_cg_irxp',
                shls_slice=(shell_id, shell_id+1, shell_id, shell_id+1))
            nd = 2 * l + 1
            nproj = coupling.shape[0]
            projected_ao = np.empty((nproj, nd, nao), dtype=np.complex128)
            for projector_id in range(nproj):
                start = offsets[projector_id]
                projected_ao[projector_id] = half_integrals[projector_id][
                    kpoint_id, start:start+nd]
                offsets[projector_id] = start + nd
            result[kpoint_id] += np.einsum(
                'inp,ij,jmq,smn->spq', projected_ao.conj(), coupling,
                projected_ao, angular_momentum)
    return result


def get_gth_pp_so_pbc(cell, kpts=None):
    '''Return periodic GTH pseudopotential SOC integrals.'''
    if kpts is None:
        kpts_array = np.zeros((1, 3))
    else:
        kpts_array = np.reshape(kpts, (-1, 3))

    fakecell, coupling_blocks = fake_cell_vnl_so(cell)
    if not coupling_blocks:
        result = np.zeros(
            (len(kpts_array), 3, cell.nao_nr(), cell.nao_nr()),
            dtype=np.complex128)
    else:
        half_integrals = pbc_pp_int._int_vnl(
            cell, fakecell, coupling_blocks, kpts_array)
        result = _contract_soc(
            half_integrals, fakecell, coupling_blocks, cell.nao_nr(),
            len(kpts_array))

    if kpts is None or np.shape(kpts) == (3,):
        return result[0]
    return result


def _molecular_half_integrals(mol, fakemol, coupling_blocks):
    dimensions = np.asarray([len(block) for block in coupling_blocks])
    original_bas = fakemol._bas
    half_integrals = []
    intors = ('int1e_ovlp', 'int1e_r2_origi', 'int1e_r4_origi')
    for projector_id, intor in enumerate(intors):
        fakemol._bas = original_bas[dimensions > projector_id]
        if fakemol.nbas > 0:
            half_integrals.append(intor_cross(intor, fakemol, mol)[None])
        else:
            half_integrals.append(None)
    fakemol._bas = original_bas
    return half_integrals


def get_gth_pp_so_mol(mol):
    '''Return open-boundary GTH pseudopotential SOC integrals.'''
    fakecell, coupling_blocks = fake_cell_vnl_so(mol)
    if not coupling_blocks:
        return np.zeros((3, mol.nao, mol.nao), dtype=np.complex128)
    half_integrals = _molecular_half_integrals(
        mol, fakecell, coupling_blocks)
    return _contract_soc(
        half_integrals, fakecell, coupling_blocks, mol.nao, 1)[0]


def get_gth_pp_so(mol_or_cell, kpts=None):
    '''Return molecular or periodic GTH pseudopotential SOC integrals.'''
    from pyscf.pbc.gto.cell import Cell
    if isinstance(mol_or_cell, Cell):
        return get_gth_pp_so_pbc(mol_or_cell, kpts)
    if kpts is not None:
        raise ValueError('kpts are only supported for periodic cells')
    return get_gth_pp_so_mol(mol_or_cell)

