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

"""Spin--spin integrals and zero-field-splitting analysis helpers."""

import numpy as np
import scipy.linalg

from pyscf import ao2mo, df, lib


SPHERICAL_COMPONENTS = (-2, -1, 0, 1, 2)


def triplet_zfs_parameters(levels):
    r"""Extract conventional ``D`` and ``E`` from three spin-one levels.

    The levels are matched to the eigenvalues of

    .. math::

        H_\mathrm{ZFS} = D[S_z^2-S(S+1)/3]
        + E(S_x^2-S_y^2), \qquad S=1,

    using the conventional principal-axis choice ``|D| >= 3 E``.

    Args:
        levels : array_like
            Three energies in any order and in any consistent energy unit.

    Returns:
        d_value : float
            Axial ZFS parameter in the same unit as ``levels``.
        e_value : float
            Non-negative rhombic ZFS parameter in the same unit.
        centered_levels : ndarray
            Sorted energies after subtraction of their barycenter.
    """
    levels = np.asarray(levels)
    if levels.shape != (3,):
        raise ValueError("three triplet energy levels are required")
    if not np.issubdtype(levels.dtype, np.number):
        raise TypeError("triplet energy levels must be numeric")
    if not np.all(np.isfinite(levels)):
        raise ValueError("triplet energy levels must be finite")
    if np.iscomplexobj(levels):
        if not np.allclose(levels.imag, 0.0, atol=1e-12, rtol=0.0):
            raise ValueError("triplet energy levels must be real")
        levels = levels.real

    centered_levels = np.sort(levels.astype(float, copy=True))
    centered_levels -= centered_levels.mean()
    axial = np.argmax(np.abs(centered_levels))
    transverse = np.delete(centered_levels, axial)
    d_value = -1.5 * centered_levels[axial]
    e_value = 0.5 * abs(transverse[1] - transverse[0])
    return d_value, e_value, centered_levels


def ground_triplet_levels(hamiltonian, reference_indices):
    """Select three SI eigenstates by overlap with a spin-free triplet root.

    Args:
        hamiltonian : ndarray
            State-interaction Hamiltonian.  Small numerical non-Hermiticity
            is removed before diagonalization.
        reference_indices : array_like
            Three basis indices corresponding to the ``M_S`` components of
            the target spin-free triplet root.

    Returns:
        energies : ndarray
            Selected energies in ascending order.
        weights : ndarray
            Projection weights of the selected eigenstates onto the three
            reference functions.
    """
    hamiltonian = np.asarray(hamiltonian)
    if (hamiltonian.ndim != 2
            or hamiltonian.shape[0] != hamiltonian.shape[1]):
        raise ValueError("hamiltonian must be a square matrix")
    if not np.issubdtype(hamiltonian.dtype, np.number):
        raise TypeError("hamiltonian must contain numeric values")
    if not np.all(np.isfinite(hamiltonian)):
        raise ValueError("hamiltonian must contain only finite values")

    reference_indices = np.asarray(reference_indices)
    if (reference_indices.shape != (3,)
            or not np.issubdtype(reference_indices.dtype, np.integer)):
        raise ValueError("reference_indices must contain three integers")
    if (len(np.unique(reference_indices)) != 3
            or np.any(reference_indices < 0)
            or np.any(reference_indices >= hamiltonian.shape[0])):
        raise ValueError("reference_indices must be unique matrix indices")

    hermitian = 0.5 * (hamiltonian + hamiltonian.conj().T)
    energies, vectors = np.linalg.eigh(hermitian)
    weights = np.sum(np.abs(vectors[reference_indices]) ** 2, axis=0)
    selected = np.argsort(weights)[-3:]
    selected = selected[np.argsort(energies[selected])]
    return energies[selected], weights[selected]


def _symmetrized_mo_pair(mo_i, mo_j):
    r"""MO factors for the derivative of an AO product.

    ``int2e_ip1ip2`` differentiates only the first AO in each AO pair.  The
    derivative of the complete AO product is recovered by contracting it with

    .. math::

        C^*_{ip} C_{jq} + C^*_{jp} C_{iq}.

    This is equivalent to explicitly adding the integral with the two AOs in
    the pair interchanged, but does not require another integral evaluation.
    """
    return (np.einsum('ip,jq->ijpq', mo_i.conj(), mo_j)
            + np.einsum('iq,jp->ijpq', mo_i, mo_j.conj()))


def _project_cartesian_rank2(hss, traceless):
    """Enforce Cartesian symmetry and optionally remove the scalar part."""
    hss = 0.5 * (hss + hss.swapaxes(0, 1))
    if traceless:
        trace = hss[0, 0] + hss[1, 1] + hss[2, 2]
        hss[0, 0] -= trace / 3.0
        hss[1, 1] -= trace / 3.0
        # Define the last component from the first two after projection.  In
        # addition to removing the contact term, this makes the rank-2 trace
        # exactly zero rather than merely zero to accumulation precision.
        hss[2, 2] = -(hss[0, 0] + hss[1, 1])
    return hss


def compute_ssc_integrals(mol, mo_coeff=None, traceless=True):
    r"""Compute Cartesian electron spin--spin coupling integrals.

    The returned tensor represents

    .. math::

        d^{ab}_{pqrs} = \iint \phi_p^*(1)\phi_q(1)
        \frac{r_{12}^2\delta_{ab}-3r_{12,a}r_{12,b}}{r_{12}^5}
        \phi_r^*(2)\phi_s(2)\,d1\,d2.

    The implementation follows :mod:`pyscf.prop.zfs.uhf`: derivatives on both
    AOs of each product are assembled from ``int2e_ip1ip2``.  Integral shell
    quartets are generated individually and immediately transformed to the
    requested orbital basis, so a ``(9, nao, nao, nao, nao)`` AO tensor is
    never stored.  Electron-pair interchange symmetry,

    .. math:: d^{ab}_{pqrs} = d^{ba}_{rspq},

    reduces the number of shell-quartet evaluations by almost a factor of two.

    Args:
        mol : :class:`pyscf.gto.Mole`
            Molecule supplying the AO basis and libcint environment.
        mo_coeff : ndarray, optional
            Orbital coefficients with shape ``(nao, norb)``.  If omitted, an
            identity transformation is used and AO integrals are returned.
        traceless : bool, optional
            Project out the isotropic two-electron contact contribution.  The
            default ``True`` returns the symmetric traceless dipolar tensor
            required by a rank-2 spherical SSC Hamiltonian.

    Returns:
        ndarray
            Cartesian SSC integrals with shape
            ``(3, 3, norb, norb, norb, norb)``.  No fine-structure constant,
            electron g-factor, spin normalization, or electron-pair counting
            prefactor is included.
    """
    nao = mol.nao_nr()
    if mo_coeff is None:
        mo_coeff = np.eye(nao)
    else:
        mo_coeff = np.asarray(mo_coeff)

    if mo_coeff.ndim != 2 or mo_coeff.shape[0] != nao:
        raise ValueError(
            f"mo_coeff must have shape ({nao}, norb); got {mo_coeff.shape}")
    if not (np.issubdtype(mo_coeff.dtype, np.number)):
        raise TypeError("mo_coeff must contain numeric values")

    norb = mo_coeff.shape[1]
    dtype = np.result_type(mo_coeff.dtype, np.float64)
    hss = np.zeros((3, 3) + (norb,) * 4, dtype=dtype)

    ao_loc = mol.ao_loc_nr()
    mo_shells = [
        mo_coeff[ao_loc[ish]:ao_loc[ish + 1]]
        for ish in range(mol.nbas)
    ]

    # Ordered shell pairs are required because int2e_ip1ip2 differentiates
    # only the first member of each pair.  The symmetric MO factors account
    # for the derivative on the second member.  Between the two ordered shell
    # pairs, (ij|kl) = (kl|ij) with the Cartesian derivative axes exchanged,
    # so only the lower triangle of shell-pair space is evaluated.
    shell_pairs = [
        (ish, jsh)
        for ish in range(mol.nbas)
        for jsh in range(mol.nbas)
    ]
    for pair_ij, (ish, jsh) in enumerate(shell_pairs):
        mo_ij = _symmetrized_mo_pair(mo_shells[ish], mo_shells[jsh])
        if not np.any(mo_ij):
            continue

        for pair_kl in range(pair_ij + 1):
            ksh, lsh = shell_pairs[pair_kl]
            mo_kl = _symmetrized_mo_pair(
                mo_shells[ksh], mo_shells[lsh])
            if not np.any(mo_kl):
                continue

            eri = mol.intor_by_shell(
                'int2e_ip1ip2', (ish, jsh, ksh, lsh))
            eri = eri.reshape((3, 3) + eri.shape[1:])
            block = np.einsum(
                'xyijkl,ijpq,klrs->xypqrs',
                eri, mo_ij, mo_kl, optimize=True)
            hss += block

            if pair_ij != pair_kl:
                # Restore the omitted (kl|ij) shell quartet.  The derivative
                # direction and the two orbital pairs are exchanged together.
                hss += block.transpose(1, 0, 4, 5, 2, 3)

    # Enforcing the exact Cartesian symmetry also removes the small
    # shell-order-dependent roundoff left by accumulation.
    return _project_cartesian_rank2(hss, traceless)


def compute_ssc_integrals_ri(mol, mo_coeff=None, auxbasis=None, lindep=None,
                             traceless=True):
    r"""Approximate spin--spin coupling integrals using RI fitting.

    This implements Eqs. (20)--(22) of Ganyushin *et al.*, J. Chem. Phys.
    **132**, 144111 (2010), DOI: 10.1063/1.3367718,

    .. math::

        d^{ab}_{\mu\nu\rho\sigma} \simeq
        g_{\mu\nu}^{T} V^{-1} V^{ab} V^{-1} g_{\rho\sigma}.

    Here ``g`` contains the ordinary three-center Coulomb integrals
    ``(P|mu nu)``, ``V`` is the auxiliary Coulomb metric ``(P|Q)``, and
    ``Vab`` is the two-center SSC metric ``(P|h_ab|Q)``.  The three quantities
    are evaluated with PySCF's ``int3c2e``, ``int2c2e``, and
    ``int2c2e_ip1ip2`` kernels, respectively.

    PySCF's packed three-center Coulomb factors are consumed in blocks and
    transformed directly to the requested orbital space.  Thus, when
    ``mo_coeff`` contains a compact active space, a four-center AO tensor is
    never constructed.  Depending on ``mol.max_memory``, PySCF may keep its
    packed DF factors in memory or place them in temporary out-of-core storage.

    Args:
        mol : :class:`pyscf.gto.Mole`
            Molecule supplying the primary AO basis.
        mo_coeff : ndarray, optional
            Orbital coefficients with shape ``(nao, norb)``.  If omitted, an
            identity transformation is used and AO integrals are returned.
        auxbasis : optional
            Auxiliary basis specification accepted by
            :func:`pyscf.df.addons.make_auxmol`.  If omitted, PySCF selects a
            Coulomb-fitting basis for ``mol``.
        lindep : float, optional
            Absolute eigenvalue cutoff for the Coulomb metric.  The PySCF DF
            default, ``pyscf.df.incore.LINEAR_DEP_THR``, is used if omitted.
        traceless : bool, optional
            Remove the isotropic contact contribution and return the symmetric
            traceless rank-2 tensor.  Default is ``True``.

    Returns:
        ndarray
            RI SSC integrals with shape
            ``(3, 3, norb, norb, norb, norb)``.  As in
            :func:`compute_ssc_integrals`, no physical prefactor is included.
    """
    if lindep is None:
        lindep = df.incore.LINEAR_DEP_THR
    if (not np.isscalar(lindep)
            or not np.issubdtype(np.asarray(lindep).dtype, np.number)
            or not np.isfinite(lindep) or lindep < 0):
        raise ValueError("lindep must be a non-negative scalar")

    auxmol = df.addons.make_auxmol(mol, auxbasis)
    nao = mol.nao_nr()
    naux = auxmol.nao_nr()

    if mo_coeff is None:
        mo_coeff = np.eye(nao)
    else:
        mo_coeff = np.asarray(mo_coeff)
    if mo_coeff.ndim != 2 or mo_coeff.shape[0] != nao:
        raise ValueError(
            f"mo_coeff must have shape ({nao}, norb); got {mo_coeff.shape}")
    if not np.issubdtype(mo_coeff.dtype, np.number):
        raise TypeError("mo_coeff must contain numeric values")
    norb = mo_coeff.shape[1]

    metric = auxmol.intor('int2c2e', hermi=1)
    ssc_metric = auxmol.intor('int2c2e_ip1ip2', comp=9)
    ssc_metric = ssc_metric.reshape(3, 3, naux, naux)

    try:
        metric_factor = scipy.linalg.cholesky(
            metric, lower=True, check_finite=False)
    except scipy.linalg.LinAlgError:
        # This fallback is normally needed only for a linearly dependent
        # custom auxiliary basis.  The standard fitting bases follow the
        # Cholesky path below, which lets PySCF stream the AO-pair index.
        eigenvalues, eigenvectors = scipy.linalg.eigh(
            metric, check_finite=False)
        retained = eigenvalues > lindep
        if not np.any(retained):
            raise np.linalg.LinAlgError(
                "all auxiliary Coulomb-metric eigenvalues were removed")
        metric_inv_sqrt = (
            eigenvectors[:, retained] / np.sqrt(eigenvalues[retained]))
        three_center = df.incore.aux_e2(
            mol, auxmol, intor='int3c2e', aosym='s1')
        three_center = np.einsum(
            'ijP,ip,jq->Ppq', three_center, mo_coeff.conj(), mo_coeff,
            optimize=True)
        fitted_products = np.einsum(
            'uP,upq->Ppq', metric_inv_sqrt.conj(), three_center,
            optimize=True)
        ssc_metric = np.einsum(
            'uP,abuv,vQ->abPQ', metric_inv_sqrt.conj(), ssc_metric,
            metric_inv_sqrt, optimize=True)
    else:
        # PySCF's standard DF builder produces L^-1 (P|mu nu), with
        # V = L L^T.  Transform these packed AO-pair factors to the active
        # orbitals one auxiliary block at a time.
        density_fitter = df.DF(mol, auxbasis=auxbasis)
        density_fitter.build()
        nfit = density_fitter.get_naoaux()
        if nfit != naux:
            raise RuntimeError(
                "DF auxiliary dimension is inconsistent with its metric")
        dtype = np.result_type(mo_coeff.dtype, np.float64)
        fitted_products = np.empty(
            (nfit, norb, norb), dtype=dtype)
        offset = 0
        orbital_slice = (0, norb, 0, norb)
        for block in density_fitter.loop():
            block_size = block.shape[0]
            if np.iscomplexobj(mo_coeff):
                unpacked = lib.unpack_tril(block)
                transformed = np.einsum(
                    'Pij,ip,jq->Ppq', unpacked, mo_coeff.conj(),
                    mo_coeff, optimize=True)
            else:
                transformed = ao2mo._ao2mo.nr_e2(
                    block, mo_coeff, orbital_slice, aosym='s2',
                    mosym='s1')
                transformed = transformed.reshape(block_size, norb, norb)
            fitted_products[offset:offset + block_size] = transformed
            offset += block_size

        # L^-1 V_ss L^-T in the same auxiliary coordinates as the streamed
        # three-center factors.  Reshaping treats the Cartesian components as
        # independent right-hand sides for the triangular solves.
        ssc_metric = scipy.linalg.solve_triangular(
            metric_factor, ssc_metric.reshape(-1, naux).T,
            lower=True, check_finite=False).T.reshape(3, 3, naux, naux)
        ssc_metric = scipy.linalg.solve_triangular(
            metric_factor,
            ssc_metric.transpose(0, 1, 3, 2).reshape(-1, naux).T,
            lower=True, check_finite=False).T.reshape(
                3, 3, naux, naux).transpose(0, 1, 3, 2)

    hss = np.einsum(
        'Pij,abPQ,Qkl->abijkl', fitted_products, ssc_metric,
        fitted_products, optimize=True)
    return _project_cartesian_rank2(hss, traceless)


def cartesian_to_spherical(hss):
    r"""Transform a Cartesian SSC tensor to rank-2 spherical components.

    Condon--Shortley phases are used and the result is ordered as
    ``q = (-2, -1, 0, +1, +2)``.  Only the symmetric traceless part of the
    leading Cartesian ``(3, 3)`` axes contributes, so this function also
    removes any isotropic contact term present in its input.

    Args:
        hss : ndarray
            Tensor whose first two axes are the Cartesian directions ``x,y,z``.

    Returns:
        ndarray
            Five complex spherical components.  Its shape is
            ``(5,) + hss.shape[2:]``.
    """
    hss = np.asarray(hss)
    if hss.ndim < 2 or hss.shape[:2] != (3, 3):
        raise ValueError(
            "Cartesian SSC integrals must have leading shape (3, 3)")
    if not np.issubdtype(hss.dtype, np.number):
        raise TypeError("Cartesian SSC integrals must contain numeric values")

    dtype = np.result_type(hss.dtype, np.complex128)
    spherical = np.empty((5,) + hss.shape[2:], dtype=dtype)

    xx, yy, zz = hss[0, 0], hss[1, 1], hss[2, 2]
    xy = 0.5 * (hss[0, 1] + hss[1, 0])
    xz = 0.5 * (hss[0, 2] + hss[2, 0])
    yz = 0.5 * (hss[1, 2] + hss[2, 1])

    spherical[0] = 0.5 * (xx - yy - 2j * xy)       # q = -2
    spherical[1] = xz - 1j * yz                    # q = -1
    spherical[2] = (2.0 * zz - xx - yy) / np.sqrt(6.0)
    spherical[3] = -(xz + 1j * yz)                 # q = +1
    spherical[4] = 0.5 * (xx - yy + 2j * xy)       # q = +2

    return spherical
