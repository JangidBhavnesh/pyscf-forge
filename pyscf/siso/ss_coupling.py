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

"""Spin-density intermediates for electron spin--spin coupling."""

from numbers import Integral

import numpy as np
from sympy import S as sympy_S
from sympy.physics.quantum.cg import CG

from pyscf import df, fci
from pyscf.data import nist
from pyscf.siso.sscint import cartesian_to_spherical, get_ssc_integrals


# The electron magnetic moment is -g_e/2 in atomic units.  The square removes
# the sign convention used for G_ELECTRON.
SSC_PHYSICAL_PREFACTOR = nist.ALPHA**2 * nist.G_ELECTRON**2 / 4.0


def _unpack_nelec(nelec, norb):
    """Validate and return ``(neleca, nelecb)``."""
    if isinstance(nelec, (bool, np.bool_)):
        raise TypeError("nelec must be an integer or an (alpha, beta) pair")
    if isinstance(nelec, Integral):
        nelecb = int(nelec) // 2
        neleca = int(nelec) - nelecb
    else:
        try:
            neleca, nelecb = nelec
        except (TypeError, ValueError) as err:
            raise TypeError(
                "nelec must be an integer or an (alpha, beta) pair") from err
        if (isinstance(neleca, (bool, np.bool_))
                or isinstance(nelecb, (bool, np.bool_))
                or not isinstance(neleca, Integral)
                or not isinstance(nelecb, Integral)):
            raise TypeError("alpha and beta electron counts must be integers")
        neleca, nelecb = int(neleca), int(nelecb)
    if not (0 <= neleca <= norb and 0 <= nelecb <= norb):
        raise ValueError(
            f"electron counts {(neleca, nelecb)} are invalid for {norb} "
            "orbitals")
    return neleca, nelecb


def _transition_dm2s(cibra, ciket, norb, nelec, link_index):
    """Return spin-resolved 2-TDMs, including for complex CI vectors."""
    if not (np.iscomplexobj(cibra) or np.iscomplexobj(ciket)):
        cibra = np.asarray(cibra, dtype=np.float64, order='C')
        ciket = np.asarray(ciket, dtype=np.float64, order='C')
        return fci.direct_spin1.trans_rdm12s(
            cibra, ciket, norb, nelec, link_index=link_index,
            reorder=True)[1]

    # PySCF's determinant RDM driver is real-valued.  Recover its sesquilinear
    # complex extension from four real transition-density evaluations:
    # <br+i*bi|O|kr+i*ki> = brOkr + biOki + i(brOki-biOkr).
    br = np.asarray(cibra.real, dtype=np.float64, order='C')
    bi = np.asarray(cibra.imag, dtype=np.float64, order='C')
    kr = np.asarray(ciket.real, dtype=np.float64, order='C')
    ki = np.asarray(ciket.imag, dtype=np.float64, order='C')

    def real_dm2s(bra, ket):
        return fci.direct_spin1.trans_rdm12s(
            bra, ket, norb, nelec, link_index=link_index,
            reorder=True)[1]

    rr = real_dm2s(br, kr)
    ii = real_dm2s(bi, ki)
    ri = real_dm2s(br, ki)
    ir = real_dm2s(bi, kr)
    return tuple(
        rr_block + ii_block + 1j * (ri_block - ir_block)
        for rr_block, ii_block, ri_block, ir_block in zip(rr, ii, ri, ir)
    )


def _reshape_ci_vector(ci, norb, nelec, name, link_index=None):
    """Validate and reshape a determinant CI vector."""
    ci = np.asarray(ci)
    if not np.issubdtype(ci.dtype, np.number):
        raise TypeError(f"{name} must contain numeric values")
    if link_index is None:
        shape = (
            fci.cistring.num_strings(norb, nelec[0]),
            fci.cistring.num_strings(norb, nelec[1]),
        )
    else:
        try:
            shape = (link_index[0].shape[0], link_index[1].shape[0])
        except (AttributeError, IndexError, TypeError) as err:
            raise ValueError(
                "link_index must contain alpha and beta link tables") from err
    expected_size = shape[0] * shape[1]
    if ci.size != expected_size:
        raise ValueError(
            f"{name} must contain {expected_size} coefficients for "
            f"determinant shape {shape}")
    ci = ci.reshape(shape)
    if not np.all(np.isfinite(ci)):
        raise ValueError(f"{name} must contain only finite coefficients")
    return ci


def _apply_ci_operator(operator, ci, *args):
    """Apply a PySCF CI operator without losing complex coefficients."""
    if np.iscomplexobj(ci):
        return (operator(ci.real, *args)
                + 1j * operator(ci.imag, *args))
    return operator(ci, *args)


def _spin_lower_once(ci, norb, nelec):
    r"""Apply :math:`S_- = \sum_p b_p^\dagger a_p` to a CI vector."""
    lowered_nelec = (nelec[0] - 1, nelec[1] + 1)
    shape = (
        fci.cistring.num_strings(norb, lowered_nelec[0]),
        fci.cistring.num_strings(norb, lowered_nelec[1]),
    )
    lowered = np.zeros(shape, dtype=np.result_type(ci.dtype, np.float64))
    intermediate_nelec = (nelec[0] - 1, nelec[1])
    for orbital in range(norb):
        intermediate = _apply_ci_operator(
            fci.addons.des_a, ci, norb, nelec, orbital)
        lowered += _apply_ci_operator(
            fci.addons.cre_b, intermediate, norb,
            intermediate_nelec, orbital)
    return lowered


def _lower_highest_weight(ci, norb, nelec, two_s, target_two_m):
    """Generate a normalized ``|S,M>`` CI vector from ``|S,S>``."""
    result = ci
    current_nelec = nelec
    current_two_m = two_s
    while current_two_m > target_two_m:
        # S_-|S,M> = sqrt((S+M)(S-M+1)) |S,M-1>.
        coefficient = 0.5 * np.sqrt(
            (two_s + current_two_m)
            * (two_s - current_two_m + 2))
        result = _spin_lower_once(result, norb, current_nelec) / coefficient
        current_nelec = (current_nelec[0] - 1, current_nelec[1] + 1)
        current_two_m -= 2
    return result, current_nelec


def _validate_two_s(two_s, name):
    if (isinstance(two_s, (bool, np.bool_))
            or not isinstance(two_s, Integral)):
        raise TypeError(f"{name} must be a non-negative integer")
    two_s = int(two_s)
    if two_s < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return two_s


def _rank2_allowed(two_s_bra, two_s_ket):
    """Return whether two spins satisfy the rank-2 triangle rule."""
    return (abs(two_s_bra - two_s_ket) <= 4
            and two_s_bra + two_s_ket >= 4
            and (two_s_bra - two_s_ket) % 2 == 0)


def make_quintet_density_q0(cibra, ciket, norb, nelec,
                            link_index=None, *, two_s_bra=None,
                            two_s_ket=None):
    r"""Build the rank-2, ``q=0`` spin transition density.

    Without ``two_s_bra`` and ``two_s_ket``, the CI vectors must occupy the
    same :math:`(N_\alpha,N_\beta)` determinant space and this function
    evaluates

    .. math::

        Q^0_{pqrs} = \frac{1}{4\sqrt{6}}\left\{
        E_{pq}\delta_{sr} - S^z_{ps}S^z_{rq}
        + \frac{1}{2}\left(S^z_{pq}S^z_{rs}-E_{pq}E_{rs}\right)
        \right\},

    between ``cibra`` and ``ciket``.  Here
    :math:`E_{pq}=a_p^\dagger a_q+b_p^\dagger b_q` and
    :math:`S^z_{pq}=a_p^\dagger a_q-b_p^\dagger b_q`.

    With PySCF's normal-ordered convention

    .. math::

        \Gamma^{\sigma\tau}_{pqrs} =
        \langle a^\dagger_{p\sigma}a^\dagger_{r\tau}
        a_{s\tau}a_{q\sigma}\rangle,

    the one-particle terms cancel.  The expression used here is therefore

    .. math::

        Q^0_{pqrs}=\frac{1}{4\sqrt{6}}\left[
        -\Gamma^{aa}_{psrq}-\Gamma^{bb}_{psrq}
        +\Gamma^{ab}_{psrq}+\Gamma^{ba}_{psrq}
        -\Gamma^{ab}_{pqrs}-\Gamma^{ba}_{pqrs}\right].

    When both spin labels are supplied, ``cibra`` and ``ciket`` are instead
    interpreted as highest-weight, spin-pure states :math:`|aSS\rangle` in
    their respective determinant spaces.  Normalized spin lowering brings
    both states to the common projection
    :math:`M=\min(S_\mathrm{bra},S_\mathrm{ket})`, where the ``q=0`` matrix
    element can be evaluated.  This supports every spin pair satisfying the
    rank-2 selection rule :math:`|S_\mathrm{bra}-S_\mathrm{ket}|\leq 2`.
    Pairs that also fail the lower triangle condition
    :math:`S_\mathrm{bra}+S_\mathrm{ket}\geq 2` return a zero density.

    This routine trusts the supplied spin labels; it does not explicitly test
    either input for spin purity.

    Args:
        cibra : ndarray
            Bra CI coefficients in PySCF alpha-string/beta-string ordering.
            With spin labels, this is the highest-weight bra vector.
        ciket : ndarray
            Ket CI coefficients.  With spin labels, this is the
            highest-weight ket vector and may have a different shape from the
            bra.
        norb : int
            Number of active spatial orbitals.
        nelec : int or pair of int
            Active alpha and beta electron counts.  A scalar uses PySCF's
            default lowest-``|M_S|`` partition.  When spin labels are given,
            only the total electron count is used and the highest-weight
            partitions are inferred from the two spins.
        link_index : pair of ndarray, optional
            Alpha and beta link tables forwarded to PySCF's transition-RDM
            driver.  For a mixed-spin pair these must describe the common-M
            determinant space.
        two_s_bra : int, optional
            Twice the total spin of the highest-weight bra CI vector.
        two_s_ket : int, optional
            Twice the total spin of the highest-weight ket CI vector.  Either
            both spin labels or neither must be supplied.

    Returns:
        ndarray
            The transition density
            :math:`\langle\mathrm{cibra}|Q^0_{pqrs}|\mathrm{ciket}\rangle`
            with shape ``(norb, norb, norb, norb)``.
    """
    if (isinstance(norb, (bool, np.bool_))
            or not isinstance(norb, Integral)):
        raise TypeError("norb must be a positive integer")
    norb = int(norb)
    if norb <= 0:
        raise ValueError("norb must be a positive integer")
    input_nelec = _unpack_nelec(nelec, norb)
    if (two_s_bra is None) != (two_s_ket is None):
        raise ValueError(
            "two_s_bra and two_s_ket must be supplied together")

    if two_s_bra is None:
        common_nelec = input_nelec
        cibra = _reshape_ci_vector(
            cibra, norb, common_nelec, "cibra", link_index)
        ciket = _reshape_ci_vector(
            ciket, norb, common_nelec, "ciket", link_index)
    else:
        two_s_bra = _validate_two_s(two_s_bra, "two_s_bra")
        two_s_ket = _validate_two_s(two_s_ket, "two_s_ket")
        if abs(two_s_bra - two_s_ket) > 4:
            raise ValueError(
                "a rank-2 density requires |S_bra - S_ket| <= 2")
        if (two_s_bra - two_s_ket) % 2:
            raise ValueError(
                "bra and ket spins are incompatible with one electron count")

        nelectron = sum(input_nelec)

        def highest_weight_nelec(two_s, name):
            if (nelectron + two_s) % 2:
                raise ValueError(
                    f"{name} is incompatible with {nelectron} electrons")
            highest = ((nelectron + two_s) // 2,
                       (nelectron - two_s) // 2)
            return _unpack_nelec(highest, norb)

        bra_nelec = highest_weight_nelec(two_s_bra, "two_s_bra")
        ket_nelec = highest_weight_nelec(two_s_ket, "two_s_ket")
        cibra = _reshape_ci_vector(
            cibra, norb, bra_nelec, "cibra")
        ciket = _reshape_ci_vector(
            ciket, norb, ket_nelec, "ciket")

        if not _rank2_allowed(two_s_bra, two_s_ket):
            dtype = np.result_type(cibra.dtype, ciket.dtype, np.float64)
            return np.zeros((norb,) * 4, dtype=dtype)

        common_two_m = min(two_s_bra, two_s_ket)
        cibra, common_nelec = _lower_highest_weight(
            cibra, norb, bra_nelec, two_s_bra, common_two_m)
        ciket, ket_common_nelec = _lower_highest_weight(
            ciket, norb, ket_nelec, two_s_ket, common_two_m)
        if common_nelec != ket_common_nelec:
            raise RuntimeError("spin lowering produced inconsistent spaces")
        if link_index is not None:
            cibra = _reshape_ci_vector(
                cibra, norb, common_nelec, "lowered cibra", link_index)
            ciket = _reshape_ci_vector(
                ciket, norb, common_nelec, "lowered ciket", link_index)

    dm2aa, dm2ab, dm2ba, dm2bb = _transition_dm2s(
        cibra, ciket, norb, common_nelec, link_index)

    # transpose(0, 3, 2, 1)[p,q,r,s] = dm2[p,s,r,q]
    psrq = (0, 3, 2, 1)
    q0 = (-dm2aa.transpose(psrq) - dm2bb.transpose(psrq)
          + dm2ab.transpose(psrq) + dm2ba.transpose(psrq)
          - dm2ab - dm2ba)
    q0 *= 1.0 / (4.0 * np.sqrt(6.0))
    return q0


def rank2_cg_coefficients(two_s, two_s_ket=None):
    r"""Return a rank-2 Clebsch--Gordan assembly tensor for two spins.

    The output converts contractions with spatial spherical components
    :math:`D^q` into matrix elements ordered by increasing spin projection,
    :math:`M=-S,-S+1,\ldots,S`:

    .. math::

        C^q_{MM'} = A_{S_bS_k} (-1)^{-q}
        \langle S_k M';2,-q|S_b M\rangle,

    where

    .. math::

        A_{S_bS_k} =
        \langle S_k M_0;2,0|S_b M_0\rangle^{-1},

    and :math:`M_0=\min(S_b,S_k)`.  Thus the returned coefficients
    multiply the ``q=0`` density evaluated after both highest-weight states
    have been lowered to their common :math:`M_0`.  Omitting ``two_s_ket``
    retains the original same-spin interface.

    Args:
        two_s : int
            Twice the bra total spin.
        two_s_ket : int, optional
            Twice the ket total spin.  Defaults to ``two_s``.

    Returns:
        ndarray
            Tensor with shape ``(5, two_s + 1, two_s_ket + 1)`` and
            spherical ordering ``q=(-2,-1,0,1,2)``.  It is zero when the
            rank-2 triangle rule is not satisfied.
    """
    two_s = _validate_two_s(two_s, "two_s")
    if two_s_ket is None:
        two_s_ket = two_s
    two_s_ket = _validate_two_s(two_s_ket, "two_s_ket")

    bra_multiplicity = two_s + 1
    ket_multiplicity = two_s_ket + 1
    coefficients = np.zeros((5, bra_multiplicity, ket_multiplicity))
    if not _rank2_allowed(two_s, two_s_ket):
        return coefficients

    bra_spin = sympy_S(two_s) / 2
    ket_spin = sympy_S(two_s_ket) / 2
    bra_ms_values = [
        -bra_spin + index for index in range(bra_multiplicity)]
    ket_ms_values = [
        -ket_spin + index for index in range(ket_multiplicity)]
    common_m = sympy_S(min(two_s, two_s_ket)) / 2
    anchor = CG(
        ket_spin, common_m, 2, 0, bra_spin, common_m).doit()
    normalization = 1.0 / float(anchor)

    for iq, q in enumerate((-2, -1, 0, 1, 2)):
        tensor_component = -q
        phase = (-1.0) ** tensor_component
        for im, ms_bra in enumerate(bra_ms_values):
            for jm, ms_ket in enumerate(ket_ms_values):
                coefficients[iq, im, jm] = (
                    normalization * phase
                    * float(CG(
                        ket_spin, ms_ket, 2, tensor_component,
                        bra_spin, ms_bra).doit()))
    return coefficients


def contract_ssc_integrals_q0(ssc_spherical, q0):
    r"""Contract five spherical SSC integral tensors with ``Q0`` densities.

    Args:
        ssc_spherical : ndarray
            Active-space SSC integrals with shape
            ``(5, norb, norb, norb, norb)`` in ``q=(-2,-1,0,1,2)`` order.
        q0 : ndarray
            One transition density with shape ``(norb,)*4`` or a collection
            with shape ``(..., norb, norb, norb, norb)``.

    Returns:
        ndarray
            The five contractions with shape ``q0.shape[:-4] + (5,)``.
    """
    ssc_spherical = np.asarray(ssc_spherical)
    q0 = np.asarray(q0)
    if ssc_spherical.ndim != 5 or ssc_spherical.shape[0] != 5:
        raise ValueError(
            "ssc_spherical must have shape (5, norb, norb, norb, norb)")
    if len(set(ssc_spherical.shape[1:])) != 1:
        raise ValueError("all four SSC orbital dimensions must be equal")
    norb = ssc_spherical.shape[1]
    if q0.ndim < 4 or q0.shape[-4:] != (norb,) * 4:
        raise ValueError(
            f"q0 must have trailing orbital shape {(norb,) * 4}")
    if not np.issubdtype(ssc_spherical.dtype, np.number):
        raise TypeError("ssc_spherical must contain numeric values")
    if not np.issubdtype(q0.dtype, np.number):
        raise TypeError("q0 must contain numeric values")
    return np.einsum(
        'mpqrs,...pqrs->...m', ssc_spherical, q0, optimize=True)


def assemble_ssc_hamiltonian_block(reduced_ssc, two_s, prefactor=1.0, *,
                                   two_s_ket=None):
    r"""Expand an SSC root-pair block over all ``M_S`` components.

    Args:
        reduced_ssc : ndarray
            Five spherical contractions for each root pair, with shape
            ``(nroots_bra, nroots_ket, 5)``.
        two_s : int
            Twice the bra total spin.
        prefactor : scalar, optional
            Overall multiplier.  The default ``1`` returns the raw spatial and
            spin contraction without electromagnetic constants.
        two_s_ket : int, optional
            Twice the ket total spin.  Defaults to ``two_s``.

    Returns:
        ndarray
            SSC block in root-major, increasing-``M_S`` ordering, with shape
            ``(nroots_bra*(two_s+1),
            nroots_ket*(two_s_ket+1))``.
    """
    reduced_ssc = np.asarray(reduced_ssc)
    if reduced_ssc.ndim != 3 or reduced_ssc.shape[2] != 5:
        raise ValueError(
            "reduced_ssc must have shape (nroots_bra, nroots_ket, 5)")
    if not np.issubdtype(reduced_ssc.dtype, np.number):
        raise TypeError("reduced_ssc must contain numeric values")
    if not np.isscalar(prefactor) or not np.issubdtype(
            np.asarray(prefactor).dtype, np.number):
        raise TypeError("prefactor must be a numeric scalar")

    coefficients = rank2_cg_coefficients(two_s, two_s_ket)
    nroots_bra, nroots_ket = reduced_ssc.shape[:2]
    bra_multiplicity, ket_multiplicity = coefficients.shape[1:]
    block = np.einsum(
        'ijq,qmn->imjn', reduced_ssc, coefficients, optimize=True)
    return prefactor * block.reshape(
        nroots_bra * bra_multiplicity,
        nroots_ket * ket_multiplicity)


def compute_ssc_hamiltonian(siso, ssc_integrals=None, *, use_df=True,
                            auxbasis=None,
                            prefactor=SSC_PHYSICAL_PREFACTOR):
    r"""Build SSC blocks in a SISO model-state basis.

    The driver evaluates :math:`Q^0` between a pair of highest-weight CI
    states after lowering them to a common :math:`M_S`, then uses rank-2
    Clebsch--Gordan coefficients to assemble all spin projections.  It
    includes every spin pair allowed by
    :math:`|S_\mathrm{bra}-S_\mathrm{ket}|\leq 2`; blocks that fail the full
    rank-2 triangle rule are zero.

    Args:
        siso : :class:`pyscf.siso.siso.SISO`
            Initialized SISO object containing the model space and CASSCF CI
            vectors.
        ssc_integrals : ndarray, optional
            Precomputed active-space integrals.  Accepted shapes are
            ``(5,ncas,ncas,ncas,ncas)`` for spherical components or
            ``(3,3,ncas,ncas,ncas,ncas)`` for Cartesian components.  If
            omitted, integrals are computed from the molecule.
        use_df : bool, optional
            Use RI SSC integrals when integrals are not supplied.  Default is
            ``True``.  ``False`` selects the full shell-wise integrals.
        auxbasis : optional
            Auxiliary basis passed to the RI integral builder.  When omitted,
            PySCF's standard correlation-fitting basis is selected.
        prefactor : scalar, optional
            Overall physical multiplier.  The default is
            :math:`g_e^2\alpha^2/4`; use ``1`` for an unscaled contraction.

    Returns:
        ndarray
            Hermitian SSC Hamiltonian in the same root-major and
            increasing-``M_S`` ordering used by SISO.
    """
    if not isinstance(use_df, (bool, np.bool_)):
        raise TypeError("use_df must be boolean")
    mc = siso.mc
    ncas = int(mc.ncas)

    if ssc_integrals is None:
        mo_cas = mc.mo_coeff[:, mc.ncore:mc.ncore + ncas]
        if use_df and auxbasis is None:
            auxbasis = df.addons.make_auxbasis(
                mc._scf.mol, mp2fit=True)
        spherical = get_ssc_integrals(
            mc._scf.mol, mo_cas, auxbasis=auxbasis, use_df=use_df)
    else:
        ssc_integrals = np.asarray(ssc_integrals)
        if ssc_integrals.shape == (3, 3) + (ncas,) * 4:
            spherical = cartesian_to_spherical(ssc_integrals)
        elif ssc_integrals.shape == (5,) + (ncas,) * 4:
            spherical = ssc_integrals
        else:
            raise ValueError(
                "ssc_integrals must contain five spherical or 3x3 Cartesian "
                "components in the active orbital space")

    ci_by_spin = getattr(siso.imds, 'c', None)
    if ci_by_spin is None:
        # Local import avoids a module cycle while reusing SISO's established
        # grouping of roots from multiple symmetry sectors.
        from pyscf.siso.siso import assemble_civecs
        ci_by_spin = assemble_civecs(siso)

    nelectron = int(sum(mc.nelecas))
    spins = [int(two_s) for two_s in siso.twoslst]
    nspin = len(spins)
    q0_pairs = {}
    reduced_pairs = {}
    blocks = [[None] * nspin for _ in range(nspin)]
    dtype = np.result_type(spherical.dtype, prefactor, np.complex128)

    for ibra_spin, two_s_bra in enumerate(spins):
        bra_civecs = np.asarray(ci_by_spin[ibra_spin])
        nroots_bra = bra_civecs.shape[0]
        for iket_spin, two_s_ket in enumerate(spins):
            ket_civecs = np.asarray(ci_by_spin[iket_spin])
            nroots_ket = ket_civecs.shape[0]
            block_shape = (
                nroots_bra * (two_s_bra + 1),
                nroots_ket * (two_s_ket + 1),
            )

            if not _rank2_allowed(two_s_bra, two_s_ket):
                blocks[ibra_spin][iket_spin] = np.zeros(
                    block_shape, dtype=dtype)
                continue

            pair_key = (two_s_bra, two_s_ket)
            q0 = np.empty(
                (nroots_bra, nroots_ket) + (ncas,) * 4,
                dtype=np.result_type(
                    bra_civecs.dtype, ket_civecs.dtype, np.float64))
            for ibra in range(nroots_bra):
                for iket in range(nroots_ket):
                    q0[ibra, iket] = make_quintet_density_q0(
                        bra_civecs[ibra], ket_civecs[iket], ncas,
                        nelectron, two_s_bra=two_s_bra,
                        two_s_ket=two_s_ket)
            reduced = contract_ssc_integrals_q0(spherical, q0)
            block = assemble_ssc_hamiltonian_block(
                reduced, two_s_bra, prefactor=prefactor,
                two_s_ket=two_s_ket)
            q0_pairs[pair_key] = q0
            reduced_pairs[pair_key] = reduced
            blocks[ibra_spin][iket_spin] = block

    hssc = np.block(blocks)

    deviation = np.max(np.abs(hssc - hssc.conj().T), initial=0.0)
    if not np.allclose(hssc, hssc.conj().T, atol=1e-10, rtol=1e-8):
        raise ValueError(
            "assembled SSC Hamiltonian is not Hermitian; maximum deviation "
            f"is {deviation:.3e}")
    hssc = 0.5 * (hssc + hssc.conj().T)

    siso.imds.ssc_integrals = spherical
    # Preserve the original diagonal-by-spin attributes while exposing every
    # allowed mixed-spin pair through explicit dictionaries.
    siso.imds.q0 = [q0_pairs.get((two_s, two_s)) for two_s in spins]
    siso.imds.ssc_reduced = [
        reduced_pairs.get((two_s, two_s)) for two_s in spins]
    siso.imds.q0_pairs = q0_pairs
    siso.imds.ssc_reduced_pairs = reduced_pairs
    siso.imds.hssc = hssc
    return hssc
