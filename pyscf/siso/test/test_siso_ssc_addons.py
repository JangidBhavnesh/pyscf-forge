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

"""Tests for effective-spin fitting of SISO Hamiltonians."""

import io
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from pyscf.siso import sscaddons
from pyscf.siso.anisoaddons import spin_operators


def _zfs_hamiltonian(spin, d_value, e_value):
    sx, sy, sz = spin_operators(spin)
    dimension = int(2 * spin + 1)
    return (
        d_value * (
            sz @ sz
            - spin * (spin + 1) / 3.0 * np.eye(dimension))
        + e_value * (sx @ sx - sy @ sy)
    )


def _fake_siso(include_ssc=True):
    multiplicities = (3, 4, 5, 7)
    spins = (1.0, 1.5, 2.0, 3.0)
    spin_free_energies = (-2.0, -1.5, -1.0, -0.5)
    soc_parameters = (
        (1.2e-4, 0.2e-4),
        (-0.8e-4, 0.1e-4),
        (0.6e-4, 0.05e-4),
        (-0.4e-4, 0.06e-4),
    )
    ssc_parameters = (
        (0.3e-4, 0.04e-4),
        (0.2e-4, 0.03e-4),
        (-0.1e-4, 0.02e-4),
        (0.15e-4, 0.01e-4),
    )

    dimension = sum(multiplicities)
    soc = np.zeros((dimension, dimension), dtype=np.complex128)
    ssc = np.zeros_like(soc)
    offset = 0
    for mult, spin, soc_de, ssc_de in zip(
            multiplicities, spins, soc_parameters, ssc_parameters):
        root_slice = slice(offset, offset + mult)
        soc[root_slice, root_slice] = _zfs_hamiltonian(spin, *soc_de)
        ssc[root_slice, root_slice] = _zfs_hamiltonian(spin, *ssc_de)
        offset += mult

    output = io.StringIO()
    mysiso = SimpleNamespace(
        statelis=[0, 0, 1, 1, 1, 0, 1],
        twoslst=np.asarray([2, 3, 4, 6]),
        ssc=include_ssc,
        stdout=output,
        verbose=4,
        mc=SimpleNamespace(stdout=output, verbose=4),
        imds=SimpleNamespace(
            e=[np.asarray([energy]) for energy in spin_free_energies],
            d=object(), c=object(), hssc=ssc),
        compute_soc_hamiltonian=mock.Mock(return_value=soc),
        compute_ssc_hamiltonian=mock.Mock(return_value=ssc),
    )
    return mysiso, soc_parameters, ssc_parameters


class KnownValues(unittest.TestCase):

    def test_triplet_quartet_quintet_and_septet_fits(self):
        mysiso, soc_parameters, ssc_parameters = _fake_siso()
        results = sscaddons.compute_D_and_E(
            mysiso, mltp=[3, 4, 5, 7], nroots=[1, 1, 1, 1])

        self.assertIs(mysiso.d_and_e, results)
        self.assertEqual(len(results), 4)
        for result, soc_de, ssc_de in zip(
                results, soc_parameters, ssc_parameters):
            np.testing.assert_allclose(
                [result['soc']['D'], result['soc']['E']],
                soc_de, atol=2e-15, rtol=0.0)
            np.testing.assert_allclose(
                [result['ssc']['D'], result['ssc']['E']],
                ssc_de, atol=2e-15, rtol=0.0)
            np.testing.assert_allclose(
                [result['total']['D'], result['total']['E']],
                np.add(soc_de, ssc_de), atol=2e-15, rtol=0.0)
            self.assertLess(result['total']['relative_residual'], 1e-10)
            self.assertTrue(result['total']['projection_reliable'])
            self.assertEqual(
                result['total']['D_tensor'].shape, (3, 3))

        output = mysiso.stdout.getvalue()
        for mult in (3, 4, 5, 7):
            heading = f'Multiplicity (2S+1): {mult}'
            self.assertIn(f'{heading}\n{"-" * len(heading)}', output)
        self.assertIn('Root no.', output)
        self.assertIn('\n        0 |', output)
        self.assertIn('D(total)', output)
        self.assertIn('D(SOC)', output)
        self.assertIn('D(SSC)', output)
        mysiso.compute_soc_hamiltonian.assert_called_once_with()
        self.assertIsNotNone(mysiso.imds.hsoc)
        self.assertIsNotNone(mysiso.imds.htotal)

    def test_ssc_column_is_omitted_when_disabled(self):
        mysiso, soc_parameters, _ = _fake_siso(include_ssc=False)
        results = sscaddons.compute_D_and_E(
            mysiso, mltp=[4], nroots=[1])

        self.assertNotIn('ssc', results[0])
        np.testing.assert_allclose(
            [results[0]['total']['D'], results[0]['total']['E']],
            soc_parameters[1], atol=2e-15, rtol=0.0)
        self.assertNotIn('D(SSC)', mysiso.stdout.getvalue())

    def test_unreliable_projection_is_not_reported_as_zfs(self):
        mysiso, _, _ = _fake_siso()
        results = sscaddons.compute_D_and_E(
            mysiso, mltp=[3], nroots=[1])
        for component in ('total', 'soc'):
            results[0][component]['projection_reliable'] = False
        mysiso.stdout.seek(0)
        mysiso.stdout.truncate()

        sscaddons.finalize(mysiso, results)
        output = mysiso.stdout.getvalue()
        self.assertIn('N/A', output)
        self.assertIn('D(SSC)', output)

    def test_request_validation(self):
        mysiso, _, _ = _fake_siso()
        multiplicities, root_counts = sscaddons._validate_requests(
            SimpleNamespace(statelis=[0, 0, 3, 2, 1, 0, 2]), (), None)
        self.assertEqual(multiplicities, [3, 4, 5, 7])
        self.assertEqual(root_counts, [1, 1, 1, 1])

        with self.assertRaisesRegex(ValueError, '3, 4, 5, and 7'):
            sscaddons.compute_D_and_E(mysiso, mltp=[2], nroots=[1])
        with self.assertRaisesRegex(ValueError, 'one count'):
            sscaddons.compute_D_and_E(
                mysiso, mltp=[3, 4], nroots=[1])
        with self.assertRaisesRegex(ValueError, 'contains 1'):
            sscaddons.compute_D_and_E(mysiso, mltp=[3], nroots=[2])

        mysiso.modelspace = [(1, 3, None)]
        with self.assertRaisesRegex(ValueError, 'not part of'):
            sscaddons.compute_D_and_E(mysiso, mltp=[4], nroots=[1])


if __name__ == '__main__':
    unittest.main()
