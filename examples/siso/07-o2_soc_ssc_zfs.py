from pyscf import gto, scf, mcscf, siso
from pyscf.mcscf import avas
from pyscf.siso.anisoaddons import generate_aniso_data, write_aniso_file

# Author: Bhavnesh Jangid

# Example to perform state interaction for O2 with SOC, SSC and SOC+SSC,
# followed by generating three data files for the SINGLE_ANISO module.
# The same state-averaged CASSCF orbitals, CI vectors and model space are
# used in all three calculations so that only the coupling terms change.
# SINGLE_ANISO can use these files to compute magnetic properties, including
# g-tensors and zero-field splitting parameters.

# 1. Define the molecule
mol = gto.Mole()
mol.atom = '''
O 0 0 -0.6035
O 0 0  0.6035'''
mol.unit = 'Angstrom'
mol.basis = 'ano@3s2p1d'  # ANO-RCC-VDZP
mol.spin = 2
mol.charge = 0
mol.verbose = 4
mol.output = 'O2_soc_ssc_aniso.out'
mol.build()

# 2. SCF calculation: include scalar relativistic effects through spin-free X2C.
mf = scf.ROHF(mol).sfx2c1e()
mf.kernel()

# 3. Active space selection via AVAS:
# CAS(12e, 8o) contains the O 2s and 2p valence orbitals.
mo_coeff = avas.kernel(mf, ['O 2s', 'O 2p'], minao=mol.basis)[2]

# 4. State-averaged CASSCF:
# Three singlets and one triplet give six states after expansion over M_S.
# This small model space is for demonstration; converge the basis and number
# of states for production magnetic-property calculations.
modelspace = [(3, 1), (1, 3)]
mc = mcscf.CASSCF(mf, 8, 12)
mc = siso.sacasscf_solver(mc, modelspace)
mc.max_cycle_macro = 100
mc.conv_tol = 1e-8
mc.kernel(mo_coeff)

# 5. State interaction: compute each Hamiltonian separately.
# SISO reads the model space from the state-average solvers attached to mc.
# ham='DKH' and amf=True select the DKH SOC operator with AMFI.
# SSC is the dipolar rank-2 electron spin-spin interaction; its physical
# prefactor is included automatically. ham selects SOC, not the SSC operator.
# For this small system, use the exact active-space SSC integrals. To use
# density fitting instead, set ssc_use_df=True and optionally ssc_auxbasis.

# SOC only: SSC is disabled by default.
mysiso_soc = siso.SISO(mc, ham='DKH', amf=True, ssc=False)
mysiso_soc.kernel()

# SSC only: disable both one-electron and two-electron SOC contributions.
mysiso_ssc = siso.SISO(mc, ham='DKH', amf=True, ssc=True,
                      soc1e=False, soc2e=False, ssc_use_df=False)
mysiso_ssc.kernel()

# SOC+SSC: retain both SOC contributions and enable SSC.
mysiso_soc_ssc = siso.SISO(mc, ham='DKH', amf=True, ssc=True,
                          ssc_use_df=False)
mysiso_soc_ssc.kernel()

# 6. Generate the SINGLE_ANISO data files:
# Pass the corresponding SISO object for each calculation. The exporter uses
# its energies and eigenvectors to reconstruct the full Hamiltonian and to
# transform the magnetic, spin and electric-dipole operators consistently.
# The SSC-only file has zero AMFI matrices; SSC is contained in the exported
# Hamiltonian. Choose backend='OpenMolcas' to write the OpenMolcas format.
mydata_soc = generate_aniso_data(
    mol, mc, mysiso=mysiso_soc, origin='CHARGE_CENTER', ham='DKH')
write_aniso_file('O2_soc.aniso', data=mydata_soc, backend='Orca')

mydata_ssc = generate_aniso_data(
    mol, mc, mysiso=mysiso_ssc, origin='CHARGE_CENTER', ham='DKH')
write_aniso_file('O2_ssc.aniso', data=mydata_ssc, backend='Orca')

mydata_soc_ssc = generate_aniso_data(
    mol, mc, mysiso=mysiso_soc_ssc, origin='CHARGE_CENTER', ham='DKH')
write_aniso_file('O2_soc_ssc.aniso', data=mydata_soc_ssc, backend='Orca')

print('SINGLE_ANISO data files: O2_soc.aniso, O2_ssc.aniso, O2_soc_ssc.aniso')

# 7. Run SINGLE_ANISO separately with its input file and executable, as in
# 04-anisodatagen.py. ZFS and other magnetic properties are evaluated there.
