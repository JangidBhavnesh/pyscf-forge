from pyscf import gto, scf, solvent, mcscf, mcpdft
import numpy as np
from pyscf.mcscf import avas
from mrh.my_pyscf.gto import ANO_RCC_VTZP
from mrh.my_pyscf.fci import csf_solver

mol = gto.Mole()
mol.atom='''
H 0 0 0
H 0 0 0.9
'''

mol.basis='CC-PVDZ'
mol.verbose=4
mol.max_memory = 100000
mol.build()

from pyscf import dft, dft2
dft.libxc = dft2.libxc
dft.numint.libxc = dft2.libxc
dft.numint.LibXCMixin.libxc = dft2.libxc

# Reparametrized-M06L: rep-M06L
# MC23 = { '0.2952*HF + (1-0.2952)*rep-M06L, 0.2952*HF + (1-0.2952)*rep-M06L'}}

# M06L_C has 27 parameters

MC23_C =  np.array([0.06,0.0031,0.00515088, 0.00304966, 2.427648e+00, 3.707473e+00, -7.943377e+00, -2.521466e+00,
     2.658691e+00, 2.932276e+00, -8.832841e-01, -1.895247e+00, -2.899644e+00, -5.068570e-01,
       -2.712838e+00, 9.416102e-02, -3.485860e-03, -5.811240e-04, 6.668814e-04, 0.0, 2.669169e-01,
      -7.563289e-02, 7.036292e-02, 3.493904e-04, 6.360837e-04, 0.0, 1e-10])

    # M06L_X has 18 parameters
MC23_X =  np.array([3.352197e+00, 6.332929e-01, -9.469553e-01, 2.030835e-01,
                         2.503819e+00, 8.085354e-01, -3.619144e+00, -5.572321e-01,
                         -4.506606e+00, 9.614774e-01, 6.977048e+00, -1.309337e+00, -2.426371e+00,
                           -7.896540e-03, 1.364510e-02, -1.714252e-06, -4.698672e-05, 0.0])

# Source: https://github.com/ElectronicStructureLibrary/libxc
XC_ID_MGGA_C_M06_L = 233
XC_ID_MGGA_X_M06_L = 203
libxc_register_code = 'repM06LT'.lower ()
dft2.libxc.register_custom_functional_(libxc_register_code, 'M06_L',
                                           ext_params={XC_ID_MGGA_C_M06_L: MC23_C,
                                                       XC_ID_MGGA_X_M06_L: MC23_X}) #,hyb=(0.2952,0,0))
#mf = scf.RKS(mol)
#ac = 'repM06LT'.lower()
#mf.xc = 'pbe, repm06lt' #'0.25*HF+0.75*repm06lt, 0.25*HF+0.75*repm06lt'
#mf.kernel()

#exit()
#tM06L0 = 't' + mcpdft.hyb('M06L',0.25, hyb_type='average')
#tMC23 = 't' + mcpdft.hyb('MC23',0.25, hyb_type='average')

mc = mcpdft.CASSCF(mf,'tPBE', 2, 2, grids_level=9)
mc.max_cycle_macro = 100
mc.conv_tol=1e-10
mc.fcisolver = csf_solver(mol, smult=1)
mc.kernel()

