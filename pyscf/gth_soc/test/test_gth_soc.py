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

import unittest

import numpy as np
from pyscf import gth_soc
from pyscf.pbc import gto
from pyscf.pbc.gto.pseudo import pp_int


def make_cell(pseudo):
    return gto.Cell(
        atom='C 0 0 0',
        a=np.eye(3) * 6,
        unit='B',
        basis='gth-dzvp',
        pseudo=pseudo,
        verbose=0,
    ).build()


class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cell = make_cell('gth-soc-pade')

    def test_parser_keeps_soc_projectors_separate(self):
        parameters = self.cell._pseudo['C']
        self.assertEqual(len(parameters), 5 + parameters[4])
        self.assertEqual(len(parameters.soc_projectors), parameters[4] - 1)

    def test_scalar_potential_compatibility(self):
        reference = make_cell('gth-pade')
        np.testing.assert_allclose(
            pp_int.get_pp_nl(self.cell), pp_int.get_pp_nl(reference),
            atol=1e-14, rtol=0.0)

    def test_periodic_and_molecular_soc(self):
        periodic = gth_soc.get_gth_pp_so(self.cell)
        self.assertEqual(periodic.shape, (3, self.cell.nao, self.cell.nao))
        np.testing.assert_allclose(
            periodic, -periodic.conj().transpose(0, 2, 1), atol=1e-14)

        kpts = np.array([[0., 0., 0.], [.1, .2, .3]])
        self.assertEqual(
            gth_soc.get_gth_pp_so(self.cell, kpts).shape,
            (2, 3, self.cell.nao, self.cell.nao))

        mol = self.cell.to_mol().build(False, False)
        molecular = gth_soc.get_gth_pp_so(mol)
        self.assertEqual(molecular.shape, (3, mol.nao, mol.nao))
        np.testing.assert_allclose(
            molecular, -molecular.conj().transpose(0, 2, 1), atol=1e-14)


if __name__ == '__main__':
    unittest.main()
