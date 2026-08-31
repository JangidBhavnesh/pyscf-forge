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

# Author: Bhavnesh Jangid

"""Exact and RI spin--spin coupling integral builders."""

import numpy as np
import scipy.linalg

from pyscf import ao2mo, df, lib
from pyscf.lib import logger

# Convention used.
SPHERICAL_COMPONENTS = (-2, -1, 0, 1, 2)

def compute_ssc_integrals(mol):
    r"""Compute Cartesian spin--spin derivative integrals in the AO basis.

    The returned tensor is the symmetrized Cartesian second-derivative tensor
    assembled from ``int2e_ip1ip2``.  Its traceless dipolar part is

    .. math::

        d^{ab}_{\mu\nu\rho\sigma}
        = h^{ab}_{\mu\nu\rho\sigma}
        - \frac{\delta_{ab}}{3}
          \sum_c h^{cc}_{\mu\nu\rho\sigma}.

    This projection is intentionally deferred to the spin-Hamiltonian fitting
    or Cartesian-to-spherical transformation, where it can be applied after
    orbital transformations and density contractions.

    Derivatives on both AOs of each product are assembled shell by shell from
    ``int2e_ip1ip2``.  Electron-pair interchange symmetry,

    .. math:: d^{ab}_{\mu\nu\rho\sigma}
              = d^{ba}_{\rho\sigma\mu\nu},

    reduces the number of shell-quartet evaluations by nearly a factor of two.

    If the full tensor fits within ``mol.max_memory`` (or
    ``mol.incore_anyway`` is set), an :class:`numpy.ndarray` is returned.
    Otherwise, the tensor is stored under ``lib.param.TMPDIR`` and an
    :class:`h5py.Dataset` is returned.  The temporary file remains alive for
    as long as the returned dataset and is removed with it.

    Args:
        mol : :class:`pyscf.gto.Mole`
            Molecule supplying the AO basis and libcint environment.

    Returns:
        ndarray or h5py.Dataset
            Cartesian AO SSC integrals with shape
            ``(3, 3, nao, nao, nao, nao)``.  No fine-structure constant,
            electron g-factor, spin normalization, or electron-pair counting
            prefactor is included.
    """
    nao = mol.nao_nr()
    shape = (3, 3) + (nao,) * 4
    dtype = np.dtype(np.float64)
    required_mb = np.prod(shape, dtype=np.int64) * dtype.itemsize / 1e6
    current_mb = lib.current_memory()[0]

    if (current_mb + required_mb <= mol.max_memory):
        logger.debug(
            mol, 'allocating %.1f MB AO SSC tensor in memory', required_mb)
        hss = np.zeros(shape, dtype=dtype)
    else:
        logger.debug(
            mol, 'allocating %.1f MB AO SSC tensor in temporary storage',
            required_mb)
        tmpfile = lib.H5TmpFile()
        hss = tmpfile.create_dataset(
            'hss', shape, dtype=dtype, chunks=True)
        # A Dataset does not otherwise keep the H5TmpFile Python object alive.
        # Retaining it here keeps the file open and lets H5TmpFile remove it
        # when the returned dataset is released.
        hss._ssc_tmpfile = tmpfile

    ao_loc = mol.ao_loc_nr()
    shell_pairs = [
        (ish, jsh)
        for ish in range(mol.nbas)
        for jsh in range(mol.nbas)
    ]

    for pair_ij, (ish, jsh) in enumerate(shell_pairs):
        si = slice(ao_loc[ish], ao_loc[ish + 1])
        sj = slice(ao_loc[jsh], ao_loc[jsh + 1])

        for pair_kl in range(pair_ij + 1):
            ksh, lsh = shell_pairs[pair_kl]
            sk = slice(ao_loc[ksh], ao_loc[ksh + 1])
            sl = slice(ao_loc[lsh], ao_loc[lsh + 1])

            block = mol.intor_by_shell(
                'int2e_ip1ip2', (ish, jsh, ksh, lsh))
            block = block.reshape((3, 3) + block.shape[1:])

            # The spin Hamiltonian uses the symmetric Cartesian tensor.  Its
            # isotropic trace is retained here and projected out downstream.
            block = 0.5 * (block + block.swapaxes(0, 1))

            # Restore derivatives on the second AO of each product.
            hss[:, :, si, sj, sk, sl] += block
            hss[:, :, sj, si, sk, sl] += block.transpose(
                0, 1, 3, 2, 4, 5)
            hss[:, :, si, sj, sl, sk] += block.transpose(
                0, 1, 2, 3, 5, 4)
            hss[:, :, sj, si, sl, sk] += block.transpose(
                0, 1, 3, 2, 5, 4)

            if pair_ij != pair_kl:
                # Restore the omitted electron-pair-exchanged shell quartet.
                block = block.transpose(0, 1, 4, 5, 2, 3)
                hss[:, :, sk, sl, si, sj] += block
                hss[:, :, sl, sk, si, sj] += block.transpose(
                    0, 1, 3, 2, 4, 5)
                hss[:, :, sk, sl, sj, si] += block.transpose(
                    0, 1, 2, 3, 5, 4)
                hss[:, :, sl, sk, sj, si] += block.transpose(
                    0, 1, 3, 2, 5, 4)

    return hss


def compute_ssc_integrals_mo(mol, mo_coeff):
    r"""Compute Cartesian spin--spin integrals in an active MO space.

    ``mo_coeff`` defines only the orbitals that are retained.  The AO shell
    blocks are transformed directly into this space, so the full
    ``(3, 3, nao, nao, nao, nao)`` tensor is never constructed.

    As in :func:`compute_ssc_integrals`, the symmetric Cartesian tensor keeps
    its isotropic trace.  The traceless projection is deferred to the
    spin-Hamiltonian fitting or Cartesian-to-spherical transformation.

    Args:
        mol : :class:`pyscf.gto.Mole`
            Molecule supplying the AO basis and libcint environment.
        mo_coeff : ndarray
            Real active-orbital coefficients with shape ``(nao, nact)``.

    Returns:
        ndarray
            Cartesian active-space SSC integrals with shape
            ``(3, 3, nact, nact, nact, nact)``.
    """
    nao = mol.nao_nr()
    mo_coeff = np.asarray(mo_coeff)
    if mo_coeff.ndim != 2 or mo_coeff.shape[0] != nao:
        raise ValueError(
            f'mo_coeff must have shape ({nao}, nact); got {mo_coeff.shape}')
    if not np.issubdtype(mo_coeff.dtype, np.number):
        raise TypeError('mo_coeff must contain numeric values')
    if np.iscomplexobj(mo_coeff):
        raise TypeError('mo_coeff must be real')

    nact = mo_coeff.shape[1]
    dtype = np.result_type(mo_coeff.dtype, np.float64)
    hss = np.zeros((3, 3) + (nact,) * 4, dtype=dtype)

    ao_loc = mol.ao_loc_nr()
    mo_shells = [
        mo_coeff[ao_loc[ish]:ao_loc[ish + 1]]
        for ish in range(mol.nbas)
    ]
    shell_pairs = [
        (ish, jsh)
        for ish in range(mol.nbas)
        for jsh in range(mol.nbas)
    ]

    for pair_ij, (ish, jsh) in enumerate(shell_pairs):
        mo_i = mo_shells[ish]
        mo_j = mo_shells[jsh]
        mo_ij = (
            np.einsum('ip,jq->ijpq', mo_i, mo_j)
            + np.einsum('iq,jp->ijpq', mo_i, mo_j))
        if not np.any(mo_ij):
            continue

        for pair_kl in range(pair_ij + 1):
            ksh, lsh = shell_pairs[pair_kl]
            mo_k = mo_shells[ksh]
            mo_l = mo_shells[lsh]
            mo_kl = (
                np.einsum('kr,ls->klrs', mo_k, mo_l)
                + np.einsum('ks,lr->klrs', mo_k, mo_l))
            if not np.any(mo_kl):
                continue

            eri = mol.intor_by_shell(
                'int2e_ip1ip2', (ish, jsh, ksh, lsh))
            eri = eri.reshape((3, 3) + eri.shape[1:])
            # Symmetrize the small AO shell block before transforming it.  If
            # this were done afterward, an additional active-space tensor of
            # shape (3, 3, nact, nact, nact, nact) would be allocated.
            eri = 0.5 * (eri + eri.swapaxes(0, 1))

            # Contract first the AO pair that leaves the smaller intermediate.
            # Explicit tensordot calls avoid recomputing an einsum path for
            # every shell quartet and map both contractions to matrix products.
            # Equivalent einsum implementation:
            # block = np.einsum(
            #     'abijkl,ijpq,klrs->abpqrs',
            #     eri, mo_ij, mo_kl, optimize=True)
            nij = eri.shape[2] * eri.shape[3]
            nkl = eri.shape[4] * eri.shape[5]
            if nkl <= nij:
                tmp = np.tensordot(
                    eri, mo_ij, axes=((2, 3), (0, 1)))
                block = np.tensordot(
                    tmp, mo_kl, axes=((2, 3), (0, 1)))
            else:
                tmp = np.tensordot(
                    eri, mo_kl, axes=((4, 5), (0, 1)))
                block = np.tensordot(
                    tmp, mo_ij, axes=((2, 3), (0, 1)))
                block = block.transpose(0, 1, 4, 5, 2, 3)
            hss += block

            if pair_ij != pair_kl:
                hss += block.transpose(0, 1, 4, 5, 2, 3)

    return hss


def compute_ssc_integrals_ri(mol, auxbasis=None):
    r"""Compute RI spin--spin coupling integrals in the AO basis.
    Based on JCP, 132, 144111 (2010), DOI: 10.1063/1.3367718.

    .. math::

        d^{ab}_{\mu\nu\rho\sigma} \simeq
        g_{\mu\nu}^{T} V^{-1} V^{ab} V^{-1} g_{\rho\sigma}

    Here ``g`` contains the ordinary three-center Coulomb integrals
    ``(P|mu nu)``, ``V`` is the auxiliary Coulomb metric ``(P|Q)``, and
    ``Vab`` is the two-center SSC metric ``(P|h_ab|Q)``.  The three quantities
    are evaluated with PySCF's ``int3c2e``, ``int2c2e``, and
    ``int2c2e_ip1ip2`` kernels, respectively.

    PySCF's metric-orthogonalized three-center Coulomb factors are retained in
    their native packed lower-triangular AO-pair form.  This function returns
    the compact two- and three-center factors and never constructs the
    four-center AO tensor.

    The two-center Cartesian tensor retains its isotropic trace; projection is
    deferred to the spin-Hamiltonian fitting.

    Args:
        mol : :class:`pyscf.gto.Mole`
            Molecule supplying the primary AO basis.
        auxbasis : optional
            Auxiliary basis specification accepted by
            :func:`pyscf.df.addons.make_auxmol`.  If omitted, PySCF selects a
            Coulomb-fitting basis for ``mol``.
    Returns:
        two_center : ndarray
            Coulomb-metric-transformed two-center SSC integrals with shape
            ``(3, 3, nfit, nfit)``.
        three_center : ndarray
            Metric-orthogonalized three-center Coulomb integrals with the
            lower-triangular AO pair ``i >= j`` packed in the last dimension.
            Its shape is ``(nfit, nao * (nao + 1) // 2)``.

        Notes:
            If explicitly required, the full four-center AO tensor can be
            reconstructed from the returned factors as

            .. code-block:: python

                three_center_full = lib.unpack_tril(three_center)
                hss = np.einsum(
                    'Pij,abPQ,Qkl->abijkl',
                    three_center_full, two_center, three_center_full,
                    optimize=True)
    """
    density_fitter = df.DF(mol, auxbasis=auxbasis)
    density_fitter.build()
    auxmol = density_fitter.auxmol
    nao = mol.nao_nr()
    naux = auxmol.nao_nr()
    nfit = density_fitter.get_naoaux()

    metric = auxmol.intor('int2c2e', hermi=1)
    ssc_metric = auxmol.intor('int2c2e_ip1ip2', comp=9)
    ssc_metric = ssc_metric.reshape(3, 3, naux, naux)

    try:
        metric_factor = scipy.linalg.cholesky(
            metric, lower=True, check_finite=False)
    except scipy.linalg.LinAlgError:
        # Match the reduced auxiliary coordinates used internally by df.DF
        # when Cholesky fails for a linearly dependent fitting basis.
        metric_transform = df.incore._eig_decompose(auxmol, metric)
        if metric_transform.shape[0] != nfit:
            raise RuntimeError(
                'DF metric decomposition is inconsistent with its '
                'three-center factors')
        two_center = np.einsum(
            'Pu,abuv,Qv->abPQ', metric_transform, ssc_metric,
            metric_transform.conj(), optimize=True)
    else:
        if nfit != naux:
            raise RuntimeError(
                "DF auxiliary dimension is inconsistent with its metric")
        # L^-1 V_ss L^-T, in the same coordinates as df.DF's factors.
        two_center = scipy.linalg.solve_triangular(
            metric_factor, ssc_metric.reshape(-1, naux).T,
            lower=True, check_finite=False).T.reshape(3, 3, naux, naux)
        two_center = scipy.linalg.solve_triangular(
            metric_factor,
            two_center.transpose(0, 1, 3, 2).reshape(-1, naux).T,
            lower=True, check_finite=False).T.reshape(
                3, 3, naux, naux).transpose(0, 1, 3, 2)

    # Retain PySCF's native packed lower-triangular AO-pair representation,
    # regardless of whether DF selected in-core or out-of-core storage.
    nao_pair = nao * (nao + 1) // 2
    three_center = np.empty((nfit, nao_pair))
    offset = 0
    for block in density_fitter.loop():
        block_size = block.shape[0]
        three_center[offset:offset + block_size] = block
        offset += block_size
    if offset != nfit:
        raise RuntimeError('DF returned an inconsistent auxiliary dimension')

    two_center = 0.5 * (two_center + two_center.swapaxes(0, 1))
    return two_center, three_center


def compute_ssc_integrals_ri_mo(mol, mo_coeff, auxbasis=None):
    r"""Compute RI spin--spin integrals in a real active-orbital space.

    The packed, metric-orthogonalized three-center factors are transformed
    directly with PySCF's AO2MO infrastructure.  Only the active-space tensor
    is constructed; no four-center AO tensor is formed.

    Args:
        mol : :class:`pyscf.gto.Mole`
            Molecule supplying the AO basis.
        mo_coeff : ndarray
            Real active-orbital coefficients with shape ``(nao, nact)``.
        auxbasis : optional
            Auxiliary basis specification accepted by
            :func:`pyscf.df.addons.make_auxmol`.

    Returns:
        ndarray
            Cartesian active-space SSC integrals with shape
            ``(3, 3, nact, nact, nact, nact)``.  The isotropic trace is
            retained and no physical prefactor is included.
    """
    nao = mol.nao_nr()
    mo_coeff = np.asarray(mo_coeff)
    if mo_coeff.ndim != 2 or mo_coeff.shape[0] != nao:
        raise ValueError(
            f'mo_coeff must have shape ({nao}, nact); got {mo_coeff.shape}')
    if not np.issubdtype(mo_coeff.dtype, np.number):
        raise TypeError('mo_coeff must contain numeric values')
    if np.iscomplexobj(mo_coeff):
        raise TypeError('mo_coeff must be real')

    nact = mo_coeff.shape[1]
    two_center, three_center = compute_ssc_integrals_ri(
        mol, auxbasis=auxbasis)
    three_center_mo = ao2mo._ao2mo.nr_e2(
        three_center, mo_coeff, (0, nact, 0, nact),
        aosym='s2', mosym='s1')
    three_center_mo = three_center_mo.reshape(-1, nact, nact)

    # Equivalent einsum implementation:
    # hss = np.einsum(
    #     'Ppq,abPQ,Qrs->abpqrs', three_center_mo, two_center,
    #     three_center_mo, optimize=True)
    tmp = np.tensordot(
        two_center, three_center_mo, axes=((3,), (0,)))
    # tensordot returns (p, q, a, b, r, s); expose Cartesian axes first.
    hss_pqabrs = np.tensordot(
        three_center_mo, tmp, axes=((0,), (2,)))
    return hss_pqabrs.transpose(2, 3, 0, 1, 4, 5)


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


def get_ssc_integrals(mol, mo_coeff, auxbasis=None, *, use_df=False):
    r"""Build the spherical spin--spin Hamiltonian in an active space.

    The underlying exact and density-fitted builders form the Cartesian
    active-space integrals.  This wrapper converts their symmetric traceless
    part to the five rank-2 spherical components ordered as
    ``q = (-2, -1, 0, +1, +2)``.

    Args:
        mol : :class:`pyscf.gto.Mole`
            Molecule supplying the AO basis and libcint environment.
        mo_coeff : ndarray
            Real active-orbital coefficients with shape ``(nao, nact)``.
        auxbasis : optional
            Auxiliary basis passed to :func:`compute_ssc_integrals_ri` when
            density fitting is selected explicitly or by the memory check.
        use_df : bool, optional
            Use the density-fitting approximation.  If ``False``, the exact
            active-space builder is used when its estimated working set fits
            in the available ``mol.max_memory``; otherwise this wrapper warns
            and switches to density fitting automatically.

    Returns:
        ndarray
            Spherical active-space Hamiltonian integrals with shape
            ``(5, nact, nact, nact, nact)``.
    """
    if not isinstance(use_df, (bool, np.bool_)):
        raise TypeError('use_df must be boolean')

    mo_array = np.asarray(mo_coeff)
    nao = mol.nao_nr()
    if mo_array.ndim != 2 or mo_array.shape[0] != nao:
        raise ValueError(
            f'mo_coeff must have shape ({nao}, nact); got {mo_array.shape}')

    if not use_df:
        nact = mo_array.shape[1]
        nact2 = nact * nact
        nact4 = nact2 * nact2
        itemsize = np.dtype(
            np.result_type(mo_array.dtype, np.float64)).itemsize

        # The direct path holds hss and one transformed shell block together.
        # Its largest two-index intermediate is bounded by the largest AO
        # shell-pair size.  Cartesian-to-spherical conversion subsequently
        # holds both the Cartesian and complex spherical tensors.
        ao_loc = mol.ao_loc_nr()
        max_shell = int(np.max(np.diff(ao_loc), initial=0))
        cartesian_bytes = 9 * nact4 * itemsize
        shell_tmp_bytes = 9 * max_shell**2 * nact2 * itemsize
        direct_bytes = 2 * cartesian_bytes + shell_tmp_bytes
        spherical_bytes = 5 * nact4 * np.dtype(np.complex128).itemsize
        required_mb = max(
            direct_bytes, cartesian_bytes + spherical_bytes) / 1e6
        current_mb = lib.current_memory()[0]
        available_mb = max(float(mol.max_memory) - current_mb, 0.0)

        if required_mb > available_mb:
            logger.warn(
                mol,
                'Insufficient memory for direct active-space SSC integrals '
                '(estimated %.1f MB required, %.1f MB available); switching '
                'to density fitting', required_mb, available_mb)
            use_df = True

    if use_df:
        cartesian = compute_ssc_integrals_ri_mo(
            mol, mo_array, auxbasis=auxbasis)
    else:
        cartesian = compute_ssc_integrals_mo(mol, mo_array)

    return cartesian_to_spherical(cartesian)
