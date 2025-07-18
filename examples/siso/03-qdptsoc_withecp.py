import numpy as np
from pyscf import gto, scf, mcscf
from pyscf.mcscf import avas
from pyscf import siso

'''
When computing the SOC integrals with ECP, generally the scalar relativistic effects are
included in the ECP, so one does not need to include them explicitly using SFX2C-1e.
'''
mol = gto.Mole(atom="Al 0 0 0", 
               spin=1,
               max_memory=10000, 
               basis='crenbl',
               ecp='crenbl',
               verbose=4) 
mol.build()

mf = scf.ROHF(mol)
mf.kernel()

# SA-CAS
mc = mcscf.CASSCF(mf, 4, 3)
mc = siso.sacasscf_solver(mc, [(3, 2),])
mc.max_cycle_macro = 100
mc.conv_tol = 1e-8
mc.kernel()

# QDPT-SOC For SA-CAS
mysiso = siso.SISO(mc,  [(3, 2),], amf=True)
mysiso.kernel()

