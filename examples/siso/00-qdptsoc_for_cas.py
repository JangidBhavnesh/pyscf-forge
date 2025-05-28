import numpy as np
from pyscf import gto, scf, mcscf
from pyscf.mcscf import avas
from pyscf import siso

mol = gto.Mole(atom="Al 0 0 0", 
               spin=1,
               max_memory=10000, 
               basis="ano@5s4p2d1f", #=ANO_RCC_VTZP
               verbose=4) 
mol.build()

mf = scf.ROHF(mol).sfx2c1e()
mf.kernel()

mo_coeff = avas.kernel(mf, ['Al 3s', 'Al 3p'], minao=mol.basis)[2]

# SA-CAS
mc = mcscf.CASSCF(mf, 4, 3)
mc = siso.sacasscf_solver(mc, [(3, 2),])
mc.max_cycle_macro = 100
mc.conv_tol = 1e-8
mc.kernel(mo_coeff)

# QDPT-SOC For SA-CAS
mysiso = siso.SISO(mc,  [(3, 2),], ham='BP', amf=True)
mysiso.kernel()

