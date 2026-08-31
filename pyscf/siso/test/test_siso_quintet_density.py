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

"""Tests for the rank-2, q=0 spin transition density."""

import unittest
from types import SimpleNamespace

import numpy as np

from pyscf import fci, gto
from pyscf.siso.ss_coupling import (assemble_ssc_hamiltonian_block,
                                    compute_ssc_hamiltonian,
                                    make_quintet_density_q0,
                                    rank2_cg_coefficients)
from pyscf.siso.sscint import cartesian_to_spherical, compute_ssc_integrals


def _spin_excitation(ci, spin, p, q, norb, nelec):
    """Apply a_p^dagger a_q or b_p^dagger b_q to a real CI vector."""
    if spin == 'a':
        intermediate = fci.addons.des_a(ci, norb, nelec, q)
        intermediate_nelec = (nelec[0] - 1, nelec[1])
        return fci.addons.cre_a(
            intermediate, norb, intermediate_nelec, p)
    intermediate = fci.addons.des_b(ci, norb, nelec, q)
    intermediate_nelec = (nelec[0], nelec[1] - 1)
    return fci.addons.cre_b(intermediate, norb, intermediate_nelec, p)


def _density_excitation(ci, p, q, norb, nelec):
    return (_spin_excitation(ci, 'a', p, q, norb, nelec)
            + _spin_excitation(ci, 'b', p, q, norb, nelec))


def _spin_z_excitation(ci, p, q, norb, nelec):
    return (_spin_excitation(ci, 'a', p, q, norb, nelec)
            - _spin_excitation(ci, 'b', p, q, norb, nelec))


def _direct_q0(cibra, ciket, norb, nelec):
    """Directly apply the defining Q0 operator for a small test space."""
    q0 = np.empty((norb,) * 4)
    zero = np.zeros_like(ciket)
    for p in range(norb):
        for q in range(norb):
            for r in range(norb):
                for s in range(norb):
                    result = (_density_excitation(
                        ciket, p, q, norb, nelec) if s == r else zero.copy())
                    result -= _spin_z_excitation(
                        _spin_z_excitation(ciket, r, q, norb, nelec),
                        p, s, norb, nelec)
                    result += 0.5 * _spin_z_excitation(
                        _spin_z_excitation(ciket, r, s, norb, nelec),
                        p, q, norb, nelec)
                    result -= 0.5 * _density_excitation(
                        _density_excitation(ciket, r, s, norb, nelec),
                        p, q, norb, nelec)
                    q0[p, q, r, s] = (
                        np.vdot(cibra, result).real
                        / (4.0 * np.sqrt(6.0)))
    return q0


class KnownValues(unittest.TestCase):
    def setUp(self):
        self.norb = 3
        self.nelec = (2, 1)
        shape = (
            fci.cistring.num_strings(self.norb, self.nelec[0]),
            fci.cistring.num_strings(self.norb, self.nelec[1]),
        )
        rng = np.random.default_rng(21)
        self.cibra = rng.standard_normal(shape)
        self.ciket = rng.standard_normal(shape)

    def test_against_direct_operator(self):
        reference = _direct_q0(
            self.cibra, self.ciket, self.norb, self.nelec)
        q0 = make_quintet_density_q0(
            self.cibra, self.ciket, self.norb, self.nelec)
        np.testing.assert_allclose(q0, reference, atol=2e-15, rtol=0.0)

    def test_transition_hermiticity(self):
        q_bk = make_quintet_density_q0(
            self.cibra, self.ciket, self.norb, self.nelec)
        q_kb = make_quintet_density_q0(
            self.ciket, self.cibra, self.norb, self.nelec)
        np.testing.assert_allclose(
            q_bk, q_kb.transpose(1, 0, 3, 2).conj(),
            atol=2e-15, rtol=0.0)

    def test_complex_ci_vectors(self):
        q0 = make_quintet_density_q0(
            self.cibra, self.ciket, self.norb, self.nelec)
        q0_complex = make_quintet_density_q0(
            1j * self.cibra, (2.0 - 0.5j) * self.ciket,
            self.norb, self.nelec)
        np.testing.assert_allclose(
            q0_complex, (-0.5 - 2.0j) * q0,
            atol=3e-15, rtol=0.0)

    def test_singlet_density_vanishes(self):
        # A doubly occupied spatial orbital is a pure two-electron singlet.
        ci = np.zeros((self.norb, self.norb))
        ci[0, 0] = 1.0
        q0 = make_quintet_density_q0(
            ci, ci, self.norb, (1, 1))
        np.testing.assert_allclose(q0, 0.0, atol=2e-15, rtol=0.0)

    def test_input_validation(self):
        with self.assertRaisesRegex(ValueError, 'coefficients'):
            make_quintet_density_q0(
                self.cibra.ravel()[:-1], self.ciket,
                self.norb, self.nelec)
        with self.assertRaisesRegex(ValueError, 'electron counts'):
            make_quintet_density_q0(
                self.cibra, self.ciket, self.norb, (4, 0))

    def test_rank2_cg_axial_pattern(self):
        coefficients = rank2_cg_coefficients(2)
        np.testing.assert_allclose(
            coefficients[2], np.diag([1.0, -2.0, 1.0]),
            atol=1e-14, rtol=0.0)
        self.assertEqual(np.count_nonzero(coefficients[0]), 1)
        self.assertEqual(np.count_nonzero(coefficients[1]), 2)
        self.assertEqual(np.count_nonzero(coefficients[3]), 2)
        self.assertEqual(np.count_nonzero(coefficients[4]), 1)
        np.testing.assert_allclose(rank2_cg_coefficients(1), 0.0)

    def test_mixed_spin_singlet_quintet_density(self):
        norb = 4
        strings = fci.cistring.make_strings(range(norb), 2)
        closed_shell_address = int(np.flatnonzero(strings == 0b0011)[0])
        singlet = np.zeros((6, 6))
        singlet[closed_shell_address, closed_shell_address] = 1.0
        quintet = np.ones((1, 1))

        q_singlet_quintet = make_quintet_density_q0(
            singlet, quintet, norb, 4,
            two_s_bra=0, two_s_ket=4)
        q_quintet_singlet = make_quintet_density_q0(
            quintet, singlet, norb, 4,
            two_s_bra=4, two_s_ket=0)

        self.assertGreater(np.linalg.norm(q_singlet_quintet), 0.1)
        np.testing.assert_allclose(
            q_singlet_quintet,
            q_quintet_singlet.transpose(1, 0, 3, 2).conj(),
            atol=2e-15, rtol=0.0)

        # For S=0 <- S=2, each q component selects one of the five ket
        # projections and the common-M q=0 element has unit coefficient.
        coefficients = rank2_cg_coefficients(0, 4)
        np.testing.assert_allclose(
            coefficients[:, 0, :], np.eye(5), atol=1e-14, rtol=0.0)

    def test_mixed_half_integer_spin_density(self):
        norb = 3
        alpha_strings = fci.cistring.make_strings(range(norb), 2)
        beta_strings = fci.cistring.make_strings(range(norb), 1)
        alpha_address = int(np.flatnonzero(alpha_strings == 0b011)[0])
        beta_address = int(np.flatnonzero(beta_strings == 0b001)[0])

        # A closed shell in orbital 0 plus one alpha electron in orbital 1
        # is a highest-weight doublet.  Three alpha electrons form the
        # highest-weight quartet.
        doublet = np.zeros((3, 3))
        doublet[alpha_address, beta_address] = 1.0
        quartet = np.ones((1, 1))
        q_doublet_quartet = make_quintet_density_q0(
            doublet, quartet, norb, 3,
            two_s_bra=1, two_s_ket=3)
        q_quartet_doublet = make_quintet_density_q0(
            quartet, doublet, norb, 3,
            two_s_bra=3, two_s_ket=1)

        self.assertGreater(np.linalg.norm(q_doublet_quartet), 0.1)
        np.testing.assert_allclose(
            q_doublet_quartet,
            q_quartet_doublet.transpose(1, 0, 3, 2).conj(),
            atol=2e-15, rtol=0.0)
        coefficients = rank2_cg_coefficients(1, 3)
        self.assertAlmostEqual(coefficients[2, -1, 2], 1.0, places=14)

    def test_mixed_spin_selection_rules(self):
        # S=0 <-> S=1 satisfies Delta S <= 2 but fails the lower rank-2
        # triangle condition, so its quintet density is identically zero.
        singlet = np.zeros((3, 3))
        singlet[0, 0] = 1.0
        triplet = np.zeros((3, 1))
        triplet[0, 0] = 1.0
        q0 = make_quintet_density_q0(
            singlet, triplet, 3, 2,
            two_s_bra=0, two_s_ket=2)
        np.testing.assert_allclose(q0, 0.0, atol=0.0, rtol=0.0)

        # A sextet-to-singlet gap is Delta S = 5/2 and is rejected before
        # attempting to form a common-M transition density.
        with self.assertRaisesRegex(ValueError, r'\|S_bra - S_ket\| <= 2'):
            make_quintet_density_q0(
                np.ones((1, 1)), np.ones((1, 1)), 5, 5,
                two_s_bra=5, two_s_ket=0)

    def test_assemble_axial_root_block(self):
        reduced = np.zeros((2, 2, 5))
        reduced[:, :, 2] = np.asarray([[1.0, 0.25], [0.25, -0.5]])
        block = assemble_ssc_hamiltonian_block(reduced, two_s=2)
        reference = np.kron(
            reduced[:, :, 2], np.diag([1.0, -2.0, 1.0]))
        np.testing.assert_allclose(block, reference, atol=1e-14, rtol=0.0)

    def test_siso_interface_against_two_electron_triplet(self):
        mol = gto.M(
            atom='H 0 0 -0.7; H 0 0 0.7', basis='sto-3g',
            spin=2, verbose=0)
        cartesian = compute_ssc_integrals(mol)
        spherical = cartesian_to_spherical(cartesian)
        ci = np.ones((1, 1))
        fake_siso = SimpleNamespace(
            mc=SimpleNamespace(
                _scf=SimpleNamespace(mol=mol),
                ncas=2,
                ncore=0,
                nelecas=(2, 0),
                mo_coeff=np.eye(2),
            ),
            twoslst=np.asarray([2]),
            imds=SimpleNamespace(c=[ci.reshape(1, 1, 1)]),
        )
        hssc = compute_ssc_hamiltonian(
            fake_siso, ssc_integrals=spherical, prefactor=1.0)

        _, (dm2aa, _, _, _) = fci.direct_spin1.trans_rdm12s(
            ci, ci, 2, (2, 0))
        ms_one_energy = np.einsum(
            'pqrs,pqrs', cartesian[2, 2], dm2aa) / 8.0
        reference = np.diag([
            ms_one_energy, -2.0 * ms_one_energy, ms_one_energy])
        np.testing.assert_allclose(hssc, reference, atol=2e-13, rtol=0.0)
        np.testing.assert_allclose(hssc, hssc.conj().T, atol=1e-14)
        self.assertEqual(fake_siso.imds.q0[0].shape, (1, 1, 2, 2, 2, 2))

    def test_nonaxial_cg_phases_against_explicit_spins(self):
        mol = gto.M(
            atom='H 0 0 0; H 1.1 0.2 0; H 0.3 1.2 0.4',
            basis='sto-3g', charge=1, spin=2, verbose=0)
        cartesian = compute_ssc_integrals(mol)
        spherical = cartesian_to_spherical(cartesian)
        ci = np.zeros((3, 1))
        ci[0, 0] = 1.0
        fake_siso = SimpleNamespace(
            mc=SimpleNamespace(
                _scf=SimpleNamespace(mol=mol),
                ncas=3,
                ncore=0,
                nelecas=(2, 0),
                mo_coeff=np.eye(3),
            ),
            twoslst=np.asarray([2]),
            imds=SimpleNamespace(c=[ci.reshape(1, 3, 1)]),
        )
        hssc = compute_ssc_hamiltonian(
            fake_siso, ssc_integrals=spherical, prefactor=1.0)

        _, (dm2aa, _, _, _) = fci.direct_spin1.trans_rdm12s(
            ci, ci, 3, (2, 0))
        spatial_tensor = 0.5 * np.einsum(
            'abpqrs,pqrs->ab', cartesian, dm2aa)
        sx = np.asarray([[0, 1], [1, 0]], dtype=complex) / 2.0
        sy = np.asarray([[0, -1j], [1j, 0]], dtype=complex) / 2.0
        sz = np.diag([1.0, -1.0]) / 2.0
        spin_half = (sx, sy, sz)
        product_hamiltonian = sum(
            spatial_tensor[a, b] * np.kron(spin_half[a], spin_half[b])
            for a in range(3) for b in range(3))

        # Columns contain |1,-1>, |1,0>, |1,+1> in the product-spin basis
        # |aa>, |ab>, |ba>, |bb>, matching SISO's increasing-M_S ordering.
        triplet_projector = np.asarray([
            [0.0, 0.0, 1.0],
            [0.0, 1.0 / np.sqrt(2.0), 0.0],
            [0.0, 1.0 / np.sqrt(2.0), 0.0],
            [1.0, 0.0, 0.0],
        ], dtype=complex)
        reference = (triplet_projector.conj().T
                     @ product_hamiltonian @ triplet_projector)
        np.testing.assert_allclose(hssc, reference, atol=2e-13, rtol=0.0)

    def test_siso_interface_couples_singlet_and_quintet(self):
        mol = gto.M(
            atom=('H 0 0 0; H 1 0 0; H 0.2 1.1 0; '
                  'H 0.1 0.3 1.2'),
            basis='sto-3g', spin=4, verbose=0)
        spherical = cartesian_to_spherical(compute_ssc_integrals(mol))

        strings = fci.cistring.make_strings(range(4), 2)
        closed_shell_address = int(np.flatnonzero(strings == 0b0011)[0])
        singlet = np.zeros((6, 6))
        singlet[closed_shell_address, closed_shell_address] = 1.0
        quintet = np.ones((1, 1))
        fake_siso = SimpleNamespace(
            mc=SimpleNamespace(
                _scf=SimpleNamespace(mol=mol),
                ncas=4,
                ncore=0,
                nelecas=(2, 2),
                mo_coeff=np.eye(4),
            ),
            twoslst=np.asarray([0, 4]),
            imds=SimpleNamespace(c=[singlet[None], quintet[None]]),
        )
        hssc = compute_ssc_hamiltonian(
            fake_siso, ssc_integrals=spherical, prefactor=1.0)

        self.assertEqual(hssc.shape, (6, 6))
        self.assertGreater(np.linalg.norm(hssc[0, 1:]), 1e-6)
        np.testing.assert_allclose(
            hssc, hssc.conj().T, atol=2e-13, rtol=0.0)
        self.assertIn((0, 4), fake_siso.imds.q0_pairs)
        self.assertIn((4, 0), fake_siso.imds.q0_pairs)


if __name__ == '__main__':
    unittest.main()
