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

"""Prototype tests for Gamma-point periodic SSC auxiliary integrals."""

import unittest

import numpy as np

from pyscf import gto, lib
from pyscf.pbc import df as pbcdf
from pyscf.pbc import gto as pbcgto
from pyscf.pbc.df import aft
from pyscf.siso import pbcsscint
from pyscf.siso.sscint import cartesian_to_spherical
from pyscf.siso.sscint import (
    compute_ssc_integrals_ri as compute_molecular_ssc_integrals_ri,
)


PRIMARY_BASIS = {"H": [[0, [1.1, 1.0]]]}
AUXILIARY_BASIS = {
    "H": [
        [0, [0.7, 1.0]],
        [0, [1.8, 1.0]],
    ]
}


def _make_cell(lattice, mesh):
    cell = pbcgto.Cell()
    cell.a = np.asarray(lattice)
    cell.atom = "H 0.2 0.3 0.4; H 1.3 1.1 0.8"
    cell.unit = "Bohr"
    cell.basis = PRIMARY_BASIS
    cell.spin = 0
    cell.precision = 1e-7
    cell.mesh = mesh
    cell.verbose = 0
    return cell.build()


def _reconstruct_ao(metric, factors):
    factors = lib.unpack_tril(factors)
    return np.einsum(
        "Pij,abPQ,Qkl->abijkl",
        factors,
        metric,
        factors,
        optimize=True,
    )


class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # A triclinic cell and displaced centers exercise off-diagonal
        # Cartesian components that a one-center cubic example cannot test.
        cls.cell = _make_cell(
            [
                [4.8, 0.2, 0.1],
                [0.1, 5.3, 0.3],
                [0.2, 0.1, 5.9],
            ],
            [15, 17, 19],
        )
        cls.gdf = pbcsscint._make_gamma_gdf(
            cls.cell, AUXILIARY_BASIS, mesh=[15, 17, 19]
        )

    def test_reciprocal_metric_shape_and_symmetry(self):
        metric = pbcsscint._build_aux_ssc_metric(
            self.gdf, mesh=[15, 17, 19]
        )
        naux = self.gdf.auxcell.nao_nr()

        self.assertEqual(metric.shape, (3, 3, naux, naux))
        self.assertEqual(metric.dtype, np.float64)
        self.assertTrue(np.all(np.isfinite(metric)))
        np.testing.assert_allclose(
            metric, metric.swapaxes(0, 1), atol=1e-14, rtol=0.0
        )
        np.testing.assert_allclose(
            metric,
            metric.transpose(0, 1, 3, 2),
            atol=1e-14,
            rtol=0.0,
        )
        self.assertGreater(np.linalg.norm(metric[0, 1]), 1e-3)
        self.assertGreater(np.linalg.norm(metric[0, 2]), 1e-3)
        self.assertGreater(np.linalg.norm(metric[1, 2]), 1e-3)

    def test_reciprocal_mesh_convergence(self):
        meshes = ([7, 9, 11], [9, 11, 13], [11, 13, 15], [15, 17, 19])
        spherical = [
            cartesian_to_spherical(
                pbcsscint._build_aux_ssc_metric(self.gdf, mesh=mesh)
            )
            for mesh in meshes
        ]
        changes = [
            np.linalg.norm(fine - coarse)
            for coarse, fine in zip(spherical[:-1], spherical[1:])
        ]

        self.assertLess(changes[1], changes[0])
        self.assertLess(changes[2], changes[1])
        self.assertLess(changes[-1], 1e-6)

    def test_automatic_derivative_mesh(self):
        density_fitter = pbcsscint._make_gamma_gdf(
            self.cell, AUXILIARY_BASIS
        )
        estimated_cutoff = aft.estimate_ke_cutoff(
            density_fitter.auxcell, precision=self.cell.precision
        )
        auxiliary_mesh = self.cell.cutoff_to_mesh(estimated_cutoff)
        self.assertTrue(
            np.all(density_fitter.mesh >= np.asarray(self.cell.mesh))
        )
        self.assertTrue(
            np.all(density_fitter.mesh >= np.asarray(auxiliary_mesh))
        )

        automatic = pbcsscint._build_aux_ssc_metric(density_fitter)
        tighter_cutoff = aft.estimate_ke_cutoff(
            density_fitter.auxcell, precision=self.cell.precision * 0.01
        )
        tighter_mesh = np.maximum(
            density_fitter.mesh,
            self.cell.cutoff_to_mesh(tighter_cutoff),
        )
        tighter_mesh = self.cell.symmetrize_mesh(tighter_mesh)
        reference = pbcsscint._build_aux_ssc_metric(
            density_fitter, mesh=tighter_mesh
        )

        difference = np.linalg.norm(
            cartesian_to_spherical(automatic)
            - cartesian_to_spherical(reference)
        )
        self.assertLess(difference, 10 * self.cell.precision)

        # An explicit mesh remains an exact user override.
        np.testing.assert_array_equal(self.gdf.mesh, [15, 17, 19])

    def test_analytical_layout_sign_and_cutoff_convergence(self):
        reciprocal = pbcsscint._build_aux_ssc_metric(
            self.gdf, mesh=[15, 17, 19]
        )
        reciprocal_rank2 = cartesian_to_spherical(reciprocal)
        auxcell = self.gdf.auxcell
        original_rcut = auxcell.rcut
        try:
            analytical_coarse = pbcsscint._build_aux_ssc_metric_pbc_intor(
                auxcell
            )
            auxcell.rcut = 18.0
            analytical_fine = pbcsscint._build_aux_ssc_metric_pbc_intor(
                auxcell
            )
        finally:
            auxcell.rcut = original_rcut

        naux = auxcell.nao_nr()
        self.assertEqual(analytical_fine.shape, (3, 3, naux, naux))
        np.testing.assert_allclose(
            analytical_fine,
            analytical_fine.transpose(1, 0, 3, 2),
            atol=1e-13,
            rtol=0.0,
        )

        coarse_rank2 = cartesian_to_spherical(analytical_coarse)
        fine_rank2 = cartesian_to_spherical(analytical_fine)
        coarse_error = np.linalg.norm(coarse_rank2 - reciprocal_rank2)
        fine_error = np.linalg.norm(fine_rank2 - reciprocal_rank2)
        self.assertLess(fine_error, coarse_error)
        self.assertLess(fine_error / np.linalg.norm(reciprocal_rank2), 0.01)

        # A missing minus sign in the Fourier derivative convention makes this
        # overlap negative while leaving all shape tests untouched.
        overlap = np.vdot(fine_rank2, reciprocal_rank2).real
        self.assertGreater(overlap, 0.0)

    def test_large_vacuum_ri_ao_tensor_approaches_molecule(self):
        lattice = [
            [11.0, 0.2, 0.1],
            [0.1, 11.55, 0.3],
            [0.2, 0.1, 12.1],
        ]
        cell = _make_cell(lattice, [49, 51, 53])
        periodic_metric, periodic_factors = (
            pbcsscint.compute_ssc_integrals_ri(
                cell, AUXILIARY_BASIS, mesh=[49, 51, 53]
            )
        )
        molecular_metric, molecular_factors = (
            compute_molecular_ssc_integrals_ri(
                cell.to_mol(), AUXILIARY_BASIS
            )
        )

        periodic_rank2 = cartesian_to_spherical(
            _reconstruct_ao(periodic_metric, periodic_factors)
        )
        molecular_rank2 = cartesian_to_spherical(
            _reconstruct_ao(molecular_metric, molecular_factors)
        )
        relative_error = np.linalg.norm(
            periodic_rank2 - molecular_rank2
        ) / np.linalg.norm(molecular_rank2)
        self.assertLess(relative_error, 0.01)

    def test_end_to_end_ri_mesh_convergence(self):
        meshes = ([9, 11, 13], [11, 13, 15], [15, 17, 19])
        spherical = []
        for mesh in meshes:
            metric, factors = pbcsscint.compute_ssc_integrals_ri(
                self.cell, AUXILIARY_BASIS, mesh=mesh
            )
            spherical.append(
                cartesian_to_spherical(_reconstruct_ao(metric, factors))
            )

        coarse_change = np.linalg.norm(spherical[1] - spherical[0])
        fine_change = np.linalg.norm(spherical[2] - spherical[1])
        self.assertLess(fine_change, coarse_change)
        self.assertLess(fine_change, 10 * self.cell.precision)

    def test_ri_factor_contract_and_ao_symmetry(self):
        two_center, three_center = pbcsscint.compute_ssc_integrals_ri(
            self.cell, AUXILIARY_BASIS, mesh=[15, 17, 19]
        )
        nao = self.cell.nao_nr()
        nao_pair = nao * (nao + 1) // 2
        nfit = three_center.shape[0]

        self.assertEqual(two_center.shape, (3, 3, nfit, nfit))
        self.assertEqual(three_center.shape, (nfit, nao_pair))
        self.assertEqual(two_center.dtype, np.float64)
        self.assertEqual(three_center.dtype, np.float64)
        np.testing.assert_allclose(
            two_center,
            two_center.swapaxes(0, 1),
            atol=1e-14,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            two_center,
            two_center.transpose(0, 1, 3, 2),
            atol=1e-14,
            rtol=0.0,
        )

        ao_tensor = _reconstruct_ao(two_center, three_center)
        np.testing.assert_allclose(
            ao_tensor, ao_tensor.swapaxes(2, 3), atol=2e-13, rtol=0.0
        )
        np.testing.assert_allclose(
            ao_tensor, ao_tensor.swapaxes(4, 5), atol=2e-13, rtol=0.0
        )
        np.testing.assert_allclose(
            ao_tensor,
            ao_tensor.transpose(1, 0, 4, 5, 2, 3),
            atol=2e-13,
            rtol=0.0,
        )

    def test_captured_metric_coordinates_match_gamma_cderi(self):
        density_fitter = pbcsscint._make_gamma_gdf(
            self.cell, AUXILIARY_BASIS, mesh=[15, 17, 19]
        )
        transform = pbcsscint._get_metric_transform(density_fitter)
        three_center = pbcsscint._collect_gamma_three_center(density_fitter)

        self.assertEqual(transform.shape[0], three_center.shape[0])
        eri_from_factors = three_center.T @ three_center
        eri_from_gdf = density_fitter.get_eri()
        np.testing.assert_allclose(
            eri_from_factors, eri_from_gdf, atol=2e-13, rtol=0.0
        )

    def test_compensated_charge_gdf_coordinates(self):
        density_fitter = pbcsscint._make_gamma_gdf(
            self.cell, AUXILIARY_BASIS, mesh=[15, 17, 19]
        )
        density_fitter._prefer_ccdf = True
        transform = pbcsscint._get_metric_transform(density_fitter)
        factors = pbcsscint._collect_gamma_three_center(density_fitter)

        self.assertEqual(transform.shape[0], factors.shape[0])
        np.testing.assert_allclose(
            factors.T @ factors,
            density_fitter.get_eri(),
            atol=2e-13,
            rtol=0.0,
        )

    def test_negative_metric_block_is_rejected(self):
        nao_pair = self.cell.nao_nr() * (self.cell.nao_nr() + 1) // 2

        class NegativeMetricDF:
            cell = self.cell
            max_memory = self.cell.max_memory

            @staticmethod
            def sr_loop(**kwargs):
                del kwargs
                real = np.zeros((1, nao_pair))
                yield real, np.zeros_like(real), -1

        with self.assertRaisesRegex(NotImplementedError, "negative"):
            pbcsscint._collect_gamma_three_center(NegativeMetricDF())

    def test_ri_linearly_dependent_auxiliary_basis(self):
        duplicate_auxbasis = {
            "H": [
                [0, [0.7, 1.0]],
                [0, [0.7, 1.0]],
            ]
        }
        two_center, three_center = pbcsscint.compute_ssc_integrals_ri(
            self.cell,
            duplicate_auxbasis,
            mesh=[15, 17, 19],
            linear_dep_threshold=1e-10,
        )

        # There are four raw auxiliary functions (two duplicates on each H).
        self.assertLess(three_center.shape[0], 4)
        self.assertEqual(
            two_center.shape[2:],
            (three_center.shape[0], three_center.shape[0]),
        )
        self.assertTrue(np.all(np.isfinite(two_center)))
        self.assertTrue(np.all(np.isfinite(three_center)))

        # The exact duplicate must not change the reconstructed AO tensor.
        reference_two_center, reference_three_center = (
            pbcsscint.compute_ssc_integrals_ri(
                self.cell,
                {"H": [[0, [0.7, 1.0]]]},
                mesh=[15, 17, 19],
                linear_dep_threshold=1e-10,
            )
        )

        np.testing.assert_allclose(
            _reconstruct_ao(two_center, three_center),
            _reconstruct_ao(reference_two_center, reference_three_center),
            atol=5e-13,
            rtol=0.0,
        )

    def test_standard_and_default_auxiliary_bases(self):
        cell = pbcgto.Cell(
            a=np.eye(3) * 5.0,
            atom="H 0 0 0; H 0 0 1.4",
            unit="Bohr",
            basis="sto-3g",
            spin=0,
            precision=1e-5,
            mesh=[15, 15, 15],
            verbose=0,
        ).build()
        nao_pair = cell.nao_nr() * (cell.nao_nr() + 1) // 2
        for auxbasis in (None, "weigend"):
            with self.subTest(auxbasis=auxbasis):
                metric, factors = pbcsscint.compute_ssc_integrals_ri(
                    cell, auxbasis=auxbasis, mesh=[15, 15, 15]
                )
                self.assertEqual(factors.shape[1], nao_pair)
                self.assertEqual(
                    metric.shape, (3, 3, factors.shape[0], factors.shape[0])
                )
                self.assertTrue(np.all(np.isfinite(metric)))
                self.assertTrue(np.all(np.isfinite(factors)))

    def test_input_validation(self):
        molecule = gto.M(atom="H 0 0 0", basis="sto-3g", spin=1, verbose=0)
        with self.assertRaisesRegex(TypeError, "pyscf.pbc.gto.Cell"):
            pbcsscint._make_gamma_gdf(molecule, AUXILIARY_BASIS)

        with self.assertRaisesRegex(ValueError, "must be built"):
            pbcsscint._make_gamma_gdf(pbcgto.Cell(), AUXILIARY_BASIS)

        with self.assertRaisesRegex(ValueError, "three positive integers"):
            pbcsscint._make_gamma_gdf(
                self.cell, AUXILIARY_BASIS, mesh=[15, 17]
            )
        with self.assertRaisesRegex(TypeError, "three positive integers"):
            pbcsscint._make_gamma_gdf(
                self.cell, AUXILIARY_BASIS, mesh=[15.0, 17.0, 19.0]
            )
        with self.assertRaisesRegex(ValueError, "three positive integers"):
            pbcsscint._make_gamma_gdf(
                self.cell, AUXILIARY_BASIS, mesh=[15, 0, 19]
            )

        for dimension in (0, 1, 2):
            low_dimensional_cell = self.cell.copy()
            low_dimensional_cell.dimension = dimension
            with self.subTest(dimension=dimension):
                with self.assertRaisesRegex(
                    NotImplementedError, "only 3D cells"
                ):
                    pbcsscint.compute_ssc_integrals_ri(
                        low_dimensional_cell,
                        AUXILIARY_BASIS,
                        mesh=[15, 17, 19],
                    )

        with self.assertRaisesRegex(TypeError, "positive real scalar"):
            pbcsscint.compute_ssc_integrals_ri(
                self.cell,
                AUXILIARY_BASIS,
                mesh=[15, 17, 19],
                linear_dep_threshold="invalid",
            )
        with self.assertRaisesRegex(ValueError, "positive finite scalar"):
            pbcsscint.compute_ssc_integrals_ri(
                self.cell,
                AUXILIARY_BASIS,
                mesh=[15, 17, 19],
                linear_dep_threshold=0.0,
            )

        non_gamma = pbcdf.GDF(self.cell, kpts=np.asarray([[0.1, 0.0, 0.0]]))
        non_gamma.auxcell = self.gdf.auxcell
        with self.assertRaisesRegex(NotImplementedError, "Gamma point"):
            pbcsscint._build_aux_ssc_metric(non_gamma)

    def test_gamma_real_check_rejects_complex_residual(self):
        almost_real = np.asarray([1.0 + 1e-12j])
        np.testing.assert_allclose(
            pbcsscint._real_if_gamma(almost_real, "test"), [1.0]
        )
        with self.assertRaisesRegex(ValueError, "not real at Gamma"):
            pbcsscint._real_if_gamma(
                np.asarray([1.0 + 1e-5j]), "test"
            )


if __name__ == "__main__":
    unittest.main()
