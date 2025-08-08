import numpy as np
from pyscf import gto, scf
from pyscf.mcscf import avas
from pyscf import siso, mcpdft
from functools import reduce
from pyscf.siso.anisoaddons import generate_aniso_data, write_aniso_file

mol = gto.Mole(atom="Al 0 0 0",
            spin=1,
            max_memory=100000,
            basis="ano@5s4p2d1f",
            verbose=4)
mol.build()

mf = scf.ROHF(mol).sfx2c1e()
mf.chkfile = 'Al.chk'
mf.kernel()

mo_coeff = avas.kernel(mf, ['Al 3s', 'Al 3p'], minao=mol.basis)[2]

modelspace = [(3, 2), (4,4)]
mc = mcpdft.CASSCF(mf, 'tPBE0', 4, 3)
mc = siso.sacasscf_solver(mc, modelspace, ms='lin')
mc.max_cycle_macro = 1
mc.conv_tol = 1e-8
mc.kernel(mo_coeff)

mysiso = siso.SISO(mc,  modelspace, ham='DK', amf=True)
sienergy, sivec, hso = mysiso.kernel()

# Save the data to an aniso file
mydata = generate_aniso_data(mol, mc, modelspace, mysiso, hso, origin='CHARGE_CENTER', ham='DK')
write_aniso_file('Al.aniso', data = mydata, backend='Orca')

