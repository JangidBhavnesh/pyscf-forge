import numpy as np
from pyscf import gto, scf
from pyscf.mcscf import avas
from pyscf.siso import sscint

# Author: Bhavnesh Jangid

# Spin-spin coupling (SSC) integrals for O2 in four flavours:
#   1. exact, in the AO basis
#   2. exact, in the active MO basis
#   3. RI approximation, in the AO basis (returned as factorized 2c/3c blocks)
#   4. RI approximation, in the active MO basis
# Finally the spherical rank-2 form used by the SISO code is built.
#
# Note, no physical prefactor (alpha^2, g_e^2, spin normalization) is included. 
# The isotropic trace (contact term) is kept
# in the Cartesian tensors and removed by the Cartesian -> spherical
# transformation.

mol = gto.Mole(
    atom = 'O 0 0 -0.6035; O 0 0 0.6035',
    unit = 'Angstrom',
    basis = 'cc-pvtz-dk',
    spin = 2,
    symmetry = False,
    verbose = 4,
    output = 'O2_ssc_integrals.out'
)
mol.build()

mf = scf.ROHF(mol).density_fit().sfx2c1e()
mf.kernel()

mo_coeff = avas.kernel(mf, ['O 2p'], minao=mol.basis)[2]

ncas, nelecas = 6, 8
ncore = (mol.nelectron - nelecas) // 2
mo_act = mo_coeff[:, ncore:ncore + ncas]

nao = mol.nao_nr()
print('nao = %d, nact = %d' % (nao, ncas))

# 1. Exact integrals in the AO basis.
# The tensor has shape (3, 3, nao, nao, nao, nao). If it does not fit in
# mol.max_memory it is stored in a temporary HDF5 file and an h5py.Dataset is
# returned instead of an ndarray, so the caller does not need to care.
hss_ao = sscint.compute_ssc_integrals(mol)
print('exact AO:', hss_ao.shape)

# 2. Exact integrals directly in the active MO basis.
# The AO shell blocks are transformed on the fly, so the full AO tensor is never
# built. This is the cheaper and the recommended exact route.
hss_mo = sscint.compute_ssc_integrals_mo(mol, mo_act)
print('exact MO:', hss_mo.shape)

# Consistency check: transform the AO tensor to the MO basis by hand.
hss_ao_to_mo = np.einsum('abijkl,ip,jq,kr,ls->abpqrs', np.asarray(hss_ao),
                         mo_act, mo_act, mo_act, mo_act, optimize=True)
print('max |AO->MO - direct MO| = %.2e' % abs(hss_ao_to_mo - hss_mo).max())
del hss_ao, hss_ao_to_mo

# 3. RI approximation in the AO basis.
# The four-center tensor is never formed. Instead two factors are returned:
#   int2c_xyz: (3, 3, naux, naux) metric-transformed two-center SSC integrals
#   int3c    : (naux, nao*(nao+1)//2) metric-orthogonalized 3c Coulomb
#              integrals with the AO pair packed as i >= j
# such that  d^{ab}_{ijkl} ~ sum_PQ int3c[P,ij] int2c_xyz[a,b,P,Q] int3c[Q,kl].
#
# With auxbasis=None the default SSC auxiliary basis is used: an even-tempered
# basis (beta = 2.5, see sscint.make_ssc_auxbasis) whose angular momentum
# is extended by one above the highest l of the orbital basis. Any other
# auxiliary basis accepted by pyscf.df.addons.make_auxmol can be passed.
int2c_xyz, int3c = sscint.compute_ssc_integrals_ri(mol, auxbasis=None)
print('RI AO factors:', int2c_xyz.shape, int3c.shape)

# 4. RI approximation in the active MO basis.
# Only (nfit, nact, nact) three-center factors and the final tensor are built.
hss_mo_ri = sscint.compute_ssc_integrals_ri_mo(mol, mo_act, auxbasis=None)
print('RI MO:', hss_mo_ri.shape)
print('max |exact MO - RI MO| = %.2e (max element %.2e)'
      % (abs(hss_mo - hss_mo_ri).max(), abs(hss_mo).max()))

# Spherical rank-2 components, q = -2, -1, 0, +1, +2 (Condon-Shortley phase).
# Only the symmetric traceless part of the Cartesian tensor contributes.
hss_sph = sscint.cartesian_to_spherical(hss_mo)
print('spherical (from exact MO):', hss_sph.shape)

# get_ssc_integrals is the one-call version: MO integrals + spherical
# transformation. use_df=False uses the exact integrals unless they do not fit
# in mol.max_memory, in which case it warns and switches to RI. use_df=True
# selects RI directly.
hss_sph_exact = sscint.get_ssc_integrals(mol, mo_act, use_df=False)
hss_sph_ri = sscint.get_ssc_integrals(mol, mo_act, use_df=True)
print('max |exact - RI| (spherical) = %.2e' % abs(hss_sph_exact - hss_sph_ri).max())
