from pyscf import gto, scf, mcpdft
from pyscf.mcscf import avas
from pyscf import siso

mol = gto.Mole(atom="N 0 0 0", 
               spin=3,
               max_memory=10000, 
               basis="ano@5s4p2d1f", #=ANO_RCC_VTZP
               verbose=4) 
mol.build()

mf = scf.ROHF(mol).sfx2c1e()
mf.kernel()

mo_coeff = avas.kernel(mf, ['N 2s', 'N 2p', 'N 3s', 'N 3p', 'N 3d'], minao=mol.basis)[2]

# L-PDFT
mc = mcpdft.CASSCF(mf, 'tPBE0', 13, 5)
mc = siso.sacasscf_solver(mc, [(8, 2),(1,4)], ms='lin')
mc.max_cycle_macro = 100
mc.conv_tol = 1e-8
mc.kernel(mo_coeff)

# QDPT-SOC for L-PDFT
mysiso = siso.SISO(mc,  [(8, 2),(1,4)], ham='DK', amf=True)
mysiso.kernel()

