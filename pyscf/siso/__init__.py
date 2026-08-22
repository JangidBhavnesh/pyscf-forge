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

# Author: Bhavnesh Jangid <jangidbhavnesh@uchicago.edu>

from pyscf.siso.socaddons import (compute_nevpt2_energies, sacasscf_solver,
                                 socintegrals, state_average_solver)
from pyscf.siso.ss_coupling import (SSC_PHYSICAL_PREFACTOR,
                                    assemble_ssc_hamiltonian_block,
                                    compute_ssc_hamiltonian,
                                    contract_ssc_integrals_q0,
                                    make_quintet_density_q0,
                                    rank2_cg_coefficients)
from pyscf.siso.ss_int_helper import (cartesian_to_spherical,
                                      compute_ssc_integrals,
                                      compute_ssc_integrals_ri,
                                      ground_triplet_levels,
                                      triplet_zfs_parameters)
from pyscf.siso.siso import SISO
