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

'''Parser for CP2K GTH pseudopotentials with spin-orbit projectors.'''

from pathlib import Path

import numpy as np
from pyscf.gto.basis import parse_cp2k_pp
from pyscf.lib.exceptions import BasisNotFoundError


_DATA_DIR = Path(__file__).resolve().parents[1] / 'pbc' / 'gto' / 'pseudo'
DATA_FILES = {
    'gthsocpade': _DATA_DIR / 'gth-soc-pade.dat',
    'gthsocpbe': _DATA_DIR / 'gth-soc-pbe.dat',
}


class GTHSOCParameters(list):
    '''Scalar GTH parameters with SOC projectors stored separately.

    Keeping the SOC projectors out of the list makes this object fully
    compatible with PySCF routines that iterate over ``pp[5:]``.
    '''

    def __init__(self, scalar_parameters, soc_projectors):
        super().__init__(scalar_parameters)
        self.soc_projectors = soc_projectors


def _format_name(name):
    return name.lower().replace('-', '').replace('_', '').replace(' ', '')


def resolve_data_file(name_or_path):
    path = Path(name_or_path).expanduser()
    if path.is_file():
        return path.resolve()
    try:
        return DATA_FILES[_format_name(str(name_or_path))]
    except KeyError as err:
        raise BasisNotFoundError(
            f'Unknown GTH-SOC pseudopotential {name_or_path!r}') from err


def _symmetric_matrix(values, dimension):
    matrix = np.zeros((dimension, dimension))
    matrix[np.triu_indices(dimension)] = values
    return (matrix + matrix.T - np.diag(matrix.diagonal())).tolist()


def _parse(lines):
    line_iter = iter(lines)
    try:
        next(line_iter)  # Header containing element and potential names
        nelecs = [int(nelec) for nelec in next(line_iter).split()]
        local = next(line_iter).split()
        rloc = float(local[0])
        nexp = int(local[1])
        cexp = [float(coefficient) for coefficient in local[2:]]
        projector_line = next(line_iter)
        nproj_types = int(projector_line.split()[0])
    except (IndexError, StopIteration, TypeError, ValueError) as err:
        raise BasisNotFoundError('Not pseudopotential data') from err

    if 'SOC' not in projector_line.upper():
        return parse_cp2k_pp._parse(list(lines))

    scalar_projectors = []
    soc_projectors = []
    for l in range(nproj_types):
        projector = next(line_iter).split()
        radius = float(projector[0])
        nproj = int(projector[1])

        h_values = [float(value) for value in projector[2:]]
        for _ in range(1, nproj):
            h_values.extend(float(value) for value in next(line_iter).split())
        scalar_projectors.append(
            [radius, nproj, _symmetric_matrix(h_values, nproj)])

        if l > 0:
            k_values = []
            for _ in range(nproj):
                k_values.extend(float(value) for value in next(line_iter).split())
            soc_projectors.append(
                [radius, nproj, _symmetric_matrix(k_values, nproj)])

    scalar_parameters = [nelecs, rloc, nexp, cexp, nproj_types]
    scalar_parameters.extend(scalar_projectors)
    return GTHSOCParameters(scalar_parameters, soc_projectors)


def load(name_or_path, symbol, suffix=None):
    '''Load one element from a GTH-SOC pseudopotential database.'''
    path = resolve_data_file(name_or_path)
    return _parse(parse_cp2k_pp.search_seg(str(path), symbol, suffix))

