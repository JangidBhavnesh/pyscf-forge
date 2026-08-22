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

"""Tests for Breit--Pauli spin--spin coupling integrals."""

import unittest

import numpy as np

from pyscf import gto
from pyscf.siso import ss_int_helper


def _full_ao_reference(mol, traceless=True):
    """Reference construction used by pyscf.prop.zfs.uhf."""
    nao = mol.nao_nr()
    hss = mol.intor('int2e_ip1ip2', comp=9)
    hss = hss.reshape(3, 3, nao, nao, nao, nao)
    hss = hss + hss.transpose(0, 1, 3, 2, 4, 5)
    hss = hss + hss.transpose(0, 1, 2, 3, 5, 4)
    hss = 0.5 * (hss + hss.swapaxes(0, 1))
    if traceless:
        trace = hss[0, 0] + hss[1, 1] + hss[2, 2]
        for axis in range(3):
            hss[axis, axis] -= trace / 3.0
    return hss


class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mol = gto.M(
            atom='O 0 0 0; H 0 0 1.1',
            basis='sto-3g', spin=1, verbose=0)

    def test_shellwise_ao_integrals_against_property_reference(self):
        for traceless in (False, True):
            with self.subTest(traceless=traceless):
                reference = _full_ao_reference(self.mol, traceless)
                hss = ss_int_helper.compute_ssc_integrals(
                    self.mol, traceless=traceless)
                np.testing.assert_allclose(
                    hss, reference, atol=2e-13, rtol=0.0)

    def test_shellwise_mo_transformation(self):
        rng = np.random.default_rng(12)
        nao = self.mol.nao_nr()
        mo = rng.standard_normal((nao, 3))
        ao_reference = _full_ao_reference(self.mol)
        reference = np.einsum(
            'xyijkl,ip,jq,kr,ls->xypqrs',
            ao_reference, mo, mo, mo, mo, optimize=True)

        hss = ss_int_helper.compute_ssc_integrals(self.mol, mo)
        np.testing.assert_allclose(hss, reference, atol=2e-13, rtol=0.0)

    def test_complex_mo_transformation(self):
        rng = np.random.default_rng(13)
        nao = self.mol.nao_nr()
        mo = (rng.standard_normal((nao, 2))
              + 1j * rng.standard_normal((nao, 2)))
        ao_reference = _full_ao_reference(self.mol)
        reference = np.einsum(
            'xyijkl,ip,jq,kr,ls->xypqrs', ao_reference,
            mo.conj(), mo, mo.conj(), mo, optimize=True)

        hss = ss_int_helper.compute_ssc_integrals(self.mol, mo)
        np.testing.assert_allclose(hss, reference, atol=1e-11, rtol=0.0)

    def test_cartesian_symmetry_and_trace(self):
        rng = np.random.default_rng(14)
        mo = rng.standard_normal((self.mol.nao_nr(), 3))
        hss = ss_int_helper.compute_ssc_integrals(self.mol, mo)

        np.testing.assert_allclose(
            hss, hss.swapaxes(0, 1), atol=1e-14, rtol=0.0)
        np.testing.assert_allclose(
            hss[0, 0] + hss[1, 1] + hss[2, 2],
            0.0, atol=2e-14, rtol=0.0)
        np.testing.assert_allclose(
            hss, hss.transpose(1, 0, 4, 5, 2, 3),
            atol=2e-13, rtol=0.0)

    def test_spherical_components(self):
        cart = np.array([
            [1.0, 2.0, 3.0],
            [2.0, -4.0, 5.0],
            [3.0, 5.0, 3.0],
        ])
        spherical = ss_int_helper.cartesian_to_spherical(cart)
        reference = np.array([
            2.5 - 2.0j,
            3.0 - 5.0j,
            9.0 / np.sqrt(6.0),
            -3.0 - 5.0j,
            2.5 + 2.0j,
        ])
        np.testing.assert_allclose(spherical, reference, atol=1e-14)
        for iq, q in enumerate(ss_int_helper.SPHERICAL_COMPONENTS):
            jq = ss_int_helper.SPHERICAL_COMPONENTS.index(-q)
            np.testing.assert_allclose(
                spherical[iq].conj(), (-1) ** q * spherical[jq],
                atol=1e-14)

    def test_triplet_zfs_parameters(self):
        d_reference = 2.4e-5
        e_reference = 0.2e-5
        levels = np.asarray([
            d_reference / 3.0 - e_reference,
            -2.0 * d_reference / 3.0,
            d_reference / 3.0 + e_reference,
        ]) + 10.0
        d_value, e_value, centered = (
            ss_int_helper.triplet_zfs_parameters(levels))

        self.assertAlmostEqual(d_value, d_reference, places=14)
        self.assertAlmostEqual(e_value, e_reference, places=14)
        self.assertAlmostEqual(centered.mean(), 0.0, places=14)

    def test_ground_triplet_level_selection(self):
        hamiltonian = np.diag([5.0, 1.0, 7.0, 1.2, 0.8])
        reference_indices = np.asarray([1, 3, 4])
        levels, weights = ss_int_helper.ground_triplet_levels(
            hamiltonian, reference_indices)

        np.testing.assert_allclose(levels, [0.8, 1.0, 1.2])
        np.testing.assert_allclose(weights, 1.0)

    def test_ri_integrals_against_full_integrals(self):
        full = ss_int_helper.compute_ssc_integrals(self.mol)
        hss_ri = ss_int_helper.compute_ssc_integrals_ri(
            self.mol, auxbasis='cc-pvdz-jkfit')
        error = hss_ri - full

        self.assertEqual(hss_ri.shape, full.shape)
        self.assertLess(np.linalg.norm(error) / np.linalg.norm(full), 0.02)
        np.testing.assert_allclose(
            hss_ri, hss_ri.swapaxes(0, 1), atol=1e-14, rtol=0.0)
        np.testing.assert_allclose(
            hss_ri[0, 0] + hss_ri[1, 1] + hss_ri[2, 2],
            0.0, atol=2e-14, rtol=0.0)
        np.testing.assert_allclose(
            hss_ri, hss_ri.transpose(1, 0, 4, 5, 2, 3),
            atol=2e-13, rtol=0.0)

    def test_ri_active_space_transformation(self):
        rng = np.random.default_rng(15)
        mo = rng.standard_normal((self.mol.nao_nr(), 3))
        ao_ri = ss_int_helper.compute_ssc_integrals_ri(
            self.mol, auxbasis='cc-pvdz-jkfit')
        reference = np.einsum(
            'abijkl,ip,jq,kr,ls->abpqrs', ao_ri,
            mo, mo, mo, mo, optimize=True)
        active_ri = ss_int_helper.compute_ssc_integrals_ri(
            self.mol, mo_coeff=mo, auxbasis='cc-pvdz-jkfit')
        np.testing.assert_allclose(
            active_ri, reference, atol=2e-11, rtol=0.0)

    def test_input_validation(self):
        with self.assertRaisesRegex(ValueError, 'mo_coeff'):
            ss_int_helper.compute_ssc_integrals(
                self.mol, np.empty((self.mol.nao_nr() + 1, 2)))
        with self.assertRaisesRegex(ValueError, 'leading shape'):
            ss_int_helper.cartesian_to_spherical(np.empty((2, 3)))
        with self.assertRaisesRegex(ValueError, 'lindep'):
            ss_int_helper.compute_ssc_integrals_ri(
                self.mol, auxbasis='weigend', lindep=-1.0)
        with self.assertRaisesRegex(ValueError, 'three triplet'):
            ss_int_helper.triplet_zfs_parameters([0.0, 1.0])
        with self.assertRaisesRegex(ValueError, 'three integers'):
            ss_int_helper.ground_triplet_levels(
                np.eye(3), [0, 1])


if __name__ == '__main__':
    unittest.main()
