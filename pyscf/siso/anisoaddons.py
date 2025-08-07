# Generate the required data used by ANISO to compute the static magnetic properties.

import numpy as np
from itertools import product
from functools import reduce
from scipy.linalg import block_diag
from pyscf import lib
import sympy as sp

# Constants
ge = 2.00231930436182   # Taken from pyscf

def _basis_transformation(mat, sivec):
    '''
    Convert the given matrix to spin orbit basis.
    '''
    return np.array([reduce(np.dot, (sivec.conj().T, m, sivec)) for m in mat])

def get_ms_values(mult):
    '''
    Get the magnetic quantum numbers for a given multiplicity.
    args:
        mult: int
            Multiplicity of the state
        returns: list
            (-S, -S + 1, ..., S - 1, S)
    '''
    spin = (mult - 1)/2
    return list(2*np.arange(-spin, spin + 1))

def unpack_sos_basis(mat):
    orbang = np.stack([block_diag(*blk) for blk in [[x[i] for x in mat] for i in range(3)]])
    return orbang

unpack_sfs_basis = unpack_sos_basis

def generate_sos_basis(mat, mult):
    '''
    It convert the NR matrix to the spin orbit basis.
    args:
        mat: np.ndarray
            Matrix containing the NR integrals
        mult: int
            Multiplicity of the state
    returns:
        mat: np.ndarray of shspae (k, n*deg, n*deg)
            deg = multiplicity
            k = 3 (x, y, z components)
    '''
    return np.array([np.kron(m, np.eye(mult)) for m in mat])

def spin_operators(S_val):
    """
    Return (Sx, Sy, Sz) as a numpy array of shape (3, n, n)
    for a given spin S (can be int or half-int).
    """
    S = sp.Rational(S_val)
    hbar = sp.S(1)
    nstates = int(2 * S + 1)
    msvals = sorted([S - i for i in range(nstates)])
    dim = len(msvals)
    ms_index = {ms: i for i, ms in enumerate(msvals)}

    Sx = sp.Matrix.zeros(dim)
    Sy = sp.Matrix.zeros(dim)
    Sz = sp.Matrix.zeros(dim)

    for ms in msvals:
        i = ms_index[ms]
        Sz[i, i] = hbar * ms
        for delta, sign in [(+1, 1), (-1, 1)]:
            ms_new = ms + delta
            if ms_new in ms_index:
                j = ms_index[ms_new]
                coeff = hbar * sp.sqrt(S * (S + 1) - ms * ms_new) / 2
                Sx[j, i] += coeff
                Sy[j, i] += -sp.I * coeff if delta == +1 else sp.I * coeff

    Sx_np = np.array(Sx.evalf(), dtype=np.complex128)
    Sy_np = np.array(Sy.evalf(), dtype=np.complex128)
    Sz_np = np.array(Sz.evalf(), dtype=np.complex128)
    return Sx_np, Sy_np, Sz_np

def get_guage_origin(mol,origin):
    '''
    Adding this function here, to avoid depandancies on pyscf-forge
    '''
    charges = mol.atom_charges()
    coords  = mol.atom_coords()
    mass    = mol.atom_mass_list()
    if isinstance(origin,str):
        if origin.upper() == 'COORD_CENTER':
            center = (0,0,0)
        elif origin.upper() == 'MASS_CENTER':
            center = mass.dot(coords) / mass.sum()
        elif origin.upper() == 'CHARGE_CENTER':
            center = charges.dot(coords)/charges.sum()
        else:
            raise RuntimeError ("Gauge origin is not recognized")
    elif isinstance(origin,str):
        center = origin
    else:
        raise RuntimeError ("Gauge origin must be a string or tuple")
    return center

def _get_lxyz_integrals(mol, origin):
    '''
    Note these integrals are antisymm.
    '''
    center = get_guage_origin(mol,origin)
    with mol.with_common_orig(center):
        ints = mol.intor('int1e_cg_irxp', comp=3, hermi=2)
    return ints

def _get_soc_integrals(mol, origin='CHARGE_CENTER', ham='DK'):
    from pyscf.siso import socaddons
    hso = socaddons.socintegrals(mol, somf=True, amf=True, mmf=False,
                                 soc1e=True, soc2e=True, ham=ham, dm=None)
    hso /= 1j
    return hso.real

def _get_dipole_integrals(mol, origin='CHARGE_CENTER'):
    center = get_guage_origin(mol,origin)
    with mol.with_common_orig(center):
        ao_dip = mol.intor_symmetric('int1e_r', comp=3)
    return ao_dip

def get_1e_prop(mc, modelspace, origin='CHARGE_CENTER', ham='DK'):
    '''
    Get the one-electron properties for the given model space.
    Basically it computes <\Psi_i|\hat{O}|\Psi_j> for the one electron 
    operator \hat{O}.
    args:
        mc: mcscf object
            SA-CAS or L-PDFT object
        modelspace: list
            List of tuples (nroots, mult, symm)
        origin: str
            Origin for the gaude dependent integrals, default is 'CHARGE_CENTER'
    returns:
        orbangmoment: list
            List of orbital angular momentum matrices for each multiplicity group
        amfiinterac: list
            List of spin-orbit interaction matrices for each multiplicity group
        edipinterac: list
            List of electric dipole interaction matrices for each multiplicity group
    '''
    ncas = mc.ncas
    nelecas = mc.nelecas

    mo_cas = mc.mo_coeff[:, mc.ncore:mc.ncas+mc.ncore]
    ints_mo = _basis_transformation(_get_lxyz_integrals(mc._scf.mol, origin), mo_cas)
    ints_so = _basis_transformation(_get_soc_integrals(mc._scf.mol, origin, ham=ham), mo_cas)
    ints_dip = _basis_transformation(_get_dipole_integrals(mc._scf.mol, origin), mo_cas)

    modelspace = sorted(modelspace, key=lambda x: x[1])
    modelspace = [x[:2] for x in modelspace]  # Keep only nroots and imult
    
    orbangmoment = []
    amfiinterac = []
    edipinterac = []

    nroot0 = 0
    for i, (nroots, imult) in enumerate(modelspace):
        solver = mc.fcisolver.fcisolvers[i]

        orbLmat = np.zeros((3, nroots, nroots), dtype=ints_mo.dtype)
        amfimat = np.zeros((3, nroots, nroots), dtype=ints_mo.dtype)
        edipmat = np.zeros((3, nroots, nroots), dtype=ints_mo.dtype)

        ci_slice = mc.ci[nroot0:nroot0+nroots]

        ijpairs = list(product(range(nroots), repeat=2))
        for i, j in ijpairs:
            tdm1 = solver.trans_rdm1(ci_slice[i], ci_slice[j],ncas, nelecas)
            orbLmat[:, i, j] = np.tensordot(ints_mo, tdm1, axes=([1, 2], [0, 1])).real
            amfimat[:, i, j] = np.tensordot(ints_so, tdm1, axes=([1, 2], [0, 1])).real
            edipmat[:, i, j] = -np.tensordot(ints_dip, tdm1, axes=([1, 2], [0, 1])).real

        orbangmoment.append(orbLmat)
        amfiinterac.append(amfimat)
        edipinterac.append(edipmat)
        
        nroot0 += nroots

    assert sum([x[0] for x in modelspace]) == len(mc.ci), "Something went wrong."

    return orbangmoment, amfiinterac, edipinterac

def generate_aniso_data(mol, mc, modelspace, mysiso, hso, origin='CHARGE_CENTER', ham='DK'):
    '''
    args:
        mol: instance of mol.gto
            Molecule object containing the molecular geometry and basis set
        mc: mcscf object
            SA-CAS or L-PDFT
        modelspace: list
            List of tuples (nroots, mult, symm)
        mysiso: siso.SISO object
            SISO object containing sivec, sienergy, and ham
        hso: np.ndarray
            Spin-orbit Hamiltonian matrix
        origin: str
            Origin for the integrals, default is 'CHARGE_CENTER'
        ham: str
            SOC Hamiltonia: 'BP' or 'DK'
    returns:
        data: dict
            Dictionary containing the required data for ANISO calculations
    '''

    data = {}

    # Basic headings
    heading = 'PySCF Interface to SINGLEANISO'
    data['source'] = heading
    data['format'] =  '2021'

    # Geometrical data
    data['natoms'] =  int(mol.natm)
    atomlabels = [mol.atom_symbol(i) for i in range(mol.natm)]
    data['atomlbl'] =  atomlabels
    coords = mol.atom_coords()
    atom_list = [[i, label, coord[0], coord[1], coord[2]]
        for i, (label, coord) in enumerate(zip(atomlabels, coords), 1)]
    atom_list.insert(0, [mol.natm])
    data['coords (in angstrom)'] = atom_list

    # Model space data
    modelspacearr = np.array([x[:2] for x in modelspace], dtype=int)
    nroots, imult = modelspacearr.T
    szproj = np.concatenate([np.tile(get_ms_values(m), n) for n, m in modelspacearr], axis=0)
    multiplicity = np.array(np.repeat(imult, nroots), dtype=int)
    data['nss'] = int(np.sum(nroots * imult))
    data['nstate'] = int(np.sum(nroots))
    data['nmult'] = int(len(modelspacearr))
    data['imult'] = [int(x) for x in imult]
    data['nroot'] = [int(r) for r in nroots]
    data['szproj'] = [int(x) for x in szproj]
    data['multiplicity'] = [int(x) for x in multiplicity]

    # Energy data
    data['eso'] = mysiso.si_energies
    data['esfs'] = np.array(mc.e_states)
    
    # Generate the required operators
    sfs_lmat, sfs_amfi, sfs_edip = get_1e_prop(mc, modelspace, origin)
    
    sos_spin = []
    sos_magmom = []
    sos_edipmat = []
    for i, (nroots, mult) in enumerate(modelspace):
        spinstates = [(mult-1)/2 for _ in range(nroots)]
        spininter = [np.stack(spin_operators(spin), axis=0) for spin in spinstates]
        lmatsos = generate_sos_basis(sfs_lmat[i], mult)
        edipsos = generate_sos_basis(sfs_edip[i], mult)
        
        sos_edipmat.append(edipsos)
        
        sos_spin.append(np.stack([block_diag(*[a[i] for a in spininter]) for i in range(3)], axis=0))
        
        sos_magneticmoment_ =  -ge *np.stack([block_diag(*[a[i] for a in spininter]) for i in range(3)], axis=0)
        sos_magneticmoment_ -= 1j * lmatsos
        sos_magmom.append(sos_magneticmoment_)

    
    # Spin orbit free data
    sfs_lmat = unpack_sfs_basis(sfs_lmat)
    sfs_amfi = unpack_sfs_basis(sfs_amfi)
    sfs_edip = unpack_sfs_basis(sfs_edip)

    data['angmom_x'] = sfs_lmat[0]
    data['angmom_y'] = sfs_lmat[1]
    data['angmom_z'] = sfs_lmat[2]
    data['amfi_x'] = sfs_amfi[0]
    data['amfi_y'] = sfs_amfi[1]
    data['amfi_z'] = sfs_amfi[2]
    data['edmom_x'] = sfs_edip[0]
    data['edmom_y'] = sfs_edip[1]
    data['edmom_z'] = sfs_edip[2]

    # Spin orbit coupled data
    sivec = mysiso.si_vecs
    sos_edipmat = unpack_sos_basis(sos_edipmat)
    sos_spin = unpack_sos_basis(sos_spin)
    sos_magneticmoment = unpack_sos_basis(sos_magmom)

    sos_spin = _basis_transformation(sos_spin, sivec)
    sos_magneticmoment = _basis_transformation(sos_magneticmoment, sivec)
    sos_edipmat = _basis_transformation(sos_edipmat, sivec)

    data['magn_xr'] = sos_magneticmoment[0].real
    data['magn_xi'] = sos_magneticmoment[0].imag
    data['magn_yr'] = sos_magneticmoment[1].real
    data['magn_yi'] = sos_magneticmoment[1].imag
    data['magn_zr'] = sos_magneticmoment[2].real
    data['magn_zi'] = sos_magneticmoment[2].imag
    data['spin_xr'] = sos_spin[0].real
    data['spin_xi'] = sos_spin[0].imag
    data['spin_yr'] = sos_spin[1].real
    data['spin_yi'] = sos_spin[1].imag
    data['spin_zr'] = sos_spin[2].real
    data['spin_zi'] = sos_spin[2].imag
    data['edipm_xr'] = sos_edipmat[0].real
    data['edipm_xi'] = sos_edipmat[0].imag
    data['edipm_yr'] = sos_edipmat[1].real
    data['edipm_yi'] = sos_edipmat[1].imag
    data['edipm_zr'] = sos_edipmat[2].real
    data['edipm_zi'] = sos_edipmat[2].imag

    # Hamiltonian data
    data['eigenr'] = mysiso.si_vecs.real
    data['eigeni'] = mysiso.si_vecs.imag
    data['hsor'] = hso.real
    data['hsoi'] = hso.imag
    return data

class ANISOFileWriter:
    '''
    This class provides methods to write various sections of the ANISO file
    including source, format, number of atoms, atom labels, coordinates,
    and various properties related to the calculation.
    '''
    def __init__(self, filename, data):
        '''
        args:
            filename (str): The name of the file to write.
            data (dict): A dictionary containing the data to write.
                          The keys should match the expected ANISO file format.
        '''
        self.filename = filename
        self.data = data

    def write_general(self, ky, val):
        '''
        args:
            ky (str): The key for the data to write.
            val (any): The value associated with the key.
        returns:
            str: Formatted string in ASCII format.
        '''

        if isinstance(val, int):
            return f"${ky}\n{val}\n\n"

        elif isinstance(val, str):
            return f"${ky}\n{val}\n\n"

        elif isinstance(val, list):
            if all(isinstance(i, (int, float)) for i in val):
                return f"${ky}\n{len(val)}\n{' '.join(map(str, val))}\n\n"

            elif all(isinstance(i, list) for i in val):
                val_str = '\n'.join(' '.join(map(str, sublist)) for sublist in val)
                return f"${ky}\n{val_str}\n\n"

            elif all(isinstance(i, str) for i in val):
                return f"${ky}\n{len(val)}\n{' '.join(val)}\n\n"

            else:
                raise ValueError(f"Unsupported list format for key: {ky}")
        elif isinstance(val, np.ndarray):
            shape = list(val.shape)
            lines = []
            arr = val
            if arr.ndim == 1:
                for i in range(0, arr.shape[0], 5):
                    line = ' '.join(f"{v:22.14E}" for v in arr[i:i+5])
                    lines.append(line)
            elif arr.ndim == 2:
                for row in arr:
                    row_strs = []
                    for i in range(0, len(row), 5):
                        row_strs.append(' '.join(f"{v:22.14E}" for v in row[i:i+5]))
                    lines.append('\n'.join(row_strs))
            else:
                raise ValueError(f"Unsupported ndarray dimension for key: {ky}")
            return f"${ky}\n{' '.join(map(str, shape))}\n" + '\n'.join(lines) + '\n\n'
        else:
            raise ValueError(f"Unsupported data type for key: {ky}")

    def save_to_file(self):
        with open(self.filename, 'w', encoding='ascii') as f:
            for ky, val in self.data.items():
                f.write(self.write_general(ky, val))

def write_aniso_file(filename, data):
    writer = ANISOFileWriter(filename, data)
    writer.save_to_file()
