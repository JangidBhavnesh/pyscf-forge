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

'''GTH pseudopotential spin-orbit coupling support.'''

from os.path import basename

from pyscf.gto import basis
from pyscf.gto.basis import parse_cp2k_pp
from pyscf.gto import pp_int as molecular_pp_int
from pyscf.pbc.gto.pseudo import pp_int as periodic_pp_int

from pyscf.gth_soc.parser import DATA_FILES, GTHSOCParameters, load
from pyscf.gth_soc.pp_int import (fake_cell_vnl_so, get_gth_pp_so,
                                 get_gth_pp_so_mol, get_gth_pp_so_pbc)


def enable():
    '''Register GTH-SOC aliases and compatibility entry points in PySCF.'''
    for name, path in DATA_FILES.items():
        basis.PP_ALIAS[name] = str(path)

    if not hasattr(parse_cp2k_pp, '_gth_soc_original_load'):
        parse_cp2k_pp._gth_soc_original_load = parse_cp2k_pp.load

        def load_with_soc(pseudofile, symbol, suffix=None):
            if basename(str(pseudofile)) in {
                    'gth-soc-pade.dat', 'gth-soc-pbe.dat'}:
                return load(pseudofile, symbol, suffix)
            return parse_cp2k_pp._gth_soc_original_load(
                pseudofile, symbol, suffix)

        parse_cp2k_pp.load = load_with_soc

    molecular_pp_int.get_gth_pp_so = get_gth_pp_so_mol
    periodic_pp_int.fake_cell_vnl_so = fake_cell_vnl_so
    periodic_pp_int.get_gth_pp_so = get_gth_pp_so_pbc


enable()


__all__ = [
    'GTHSOCParameters', 'enable', 'fake_cell_vnl_so', 'get_gth_pp_so',
    'get_gth_pp_so_mol', 'get_gth_pp_so_pbc', 'load',
]
