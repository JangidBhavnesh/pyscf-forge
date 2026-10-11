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

r"""
Electron spin-spin coupling (SSC) Hamiltonian in the SI-SO model-state basis.

The matrix elements between the model states :math:`|aSM\rangle` follow from

    1. the rank-2 (:math:`q=0`) spin transition density
       :math:`Q^0_{pqrs}` between CI vectors, evaluated at a common
       :math:`M_S`,
    2. its contraction with the spherical SSC integrals,
    3. the expansion over :math:`M_S` with rank-2 Clebsch-Gordan coefficients.

The Hamiltonian has dimension ``(nstates, nstates)`` with
:math:`n_\mathrm{states}=\sum_i n_{\mathrm{roots},i}(2S_i+1)`, in the same
root-major, increasing-:math:`M_S` ordering as the SI-SO SOC Hamiltonian.
"""


from operator import index

import numpy as np
from sympy import Rational
from sympy.physics.wigner import clebsch_gordan

from pyscf import fci, lib
from pyscf.data import nist
from pyscf.fci.addons import _unpack_nelec
from pyscf.siso.siso import assemble_civecs, contract_spin_minus
from pyscf.siso.sscint import cartesian_to_spherical, get_ssc_integrals

# Author: Bhavnesh Jangid


logger = lib.logger

# The electron magnetic moment is -g_e/2 in atomic units.  The square removes
# the sign convention used for G_ELECTRON.
SSC_PHYSICAL_PREFACTOR = nist.ALPHA**2 * nist.G_ELECTRON**2 / 4.0

# Spherical rank-2 components, q = -2, -1, 0, +1, +2.
_QS = (-2, -1, 0, 1, 2)


def _validate_twos(two_s):
    """Return a nonnegative integer ``2S``; reject fractional spin labels.

    NumPy integer scalars are accepted through the integer-index protocol.
    Half-integer physical spins are represented by odd integer labels.
    """
    try:
        two_s = index(two_s)
    except TypeError as err:
        raise ValueError('twice-spin labels must be nonnegative integers') from err
    if two_s < 0:
        raise ValueError('twice-spin labels must be nonnegative integers')
    return two_s


def _highest_weight_nelec(norb, nelectron, two_s):
    """Partition electrons at ``M = S`` using PySCF's FCI convention.

    Check parity and orbital capacity before calling ``_unpack_nelec``:
    its integer division alone would silently truncate an invalid spin.
    Returns the alpha/beta electron counts, each between zero and ``norb``.
    """
    if two_s > nelectron or (nelectron - two_s) % 2:
        raise ValueError('spin and electron count have incompatible parity or size')
    nelec = _unpack_nelec(nelectron, two_s)
    if any(n < 0 or n > norb for n in nelec):
        raise ValueError('highest-weight electron counts exceed orbital capacity')
    return nelec


def _rank2_cg(two_j, two_m, q, two_J):
    r"""Return :math:`\langle j m;2q|J,m+q\rangle` using SymPy.

    SymPy's ``clebsch_gordan`` uses the Condon--Shortley convention and
    takes angular momenta first, followed by their projections. Rational
    arithmetic represents half-integer spins exactly; only the final
    coefficient is converted to a float for NumPy contractions.

    Args:
        two_j: int
            Twice the ket angular momentum j, nonnegative.
        two_m: int
            Twice its projection m, with the same parity as two_j.
        q: int
            Rank-2 tensor projection, between -2 and +2 (not doubled).
        two_J: int
            Twice the bra angular momentum J, nonnegative.

    Returns:
        float
            Clebsch--Gordan coefficient for bra projection M = m + q.
            Forbidden couplings or projections outside either multiplet
            give zero. NumPy integer scalar arguments are accepted.

    Notes:
        Callers supply valid integer spin labels and projections. The
        public assembly routine validates the spin labels and checks the
        rank-2 selection rule before invoking this helper.
    """
    j = Rational(int(two_j), 2)
    m = Rational(int(two_m), 2)
    J = Rational(int(two_J), 2)
    return float(clebsch_gordan(j, 2, J, m, int(q), m + int(q)))


def _rank2_allowed(two_s_bra, two_s_ket):
    r"""
    Rank-2 selection rule
    :math:`|S_\mathrm{bra}-S_\mathrm{ket}|\leq 2\leq S_\mathrm{bra}+S_\mathrm{ket}`,
    with a spin difference compatible with a fixed electron count.
    """
    return (abs(two_s_bra - two_s_ket) <= 4
            and two_s_bra + two_s_ket >= 4
            and (two_s_bra - two_s_ket) % 2 == 0)

def _transition_dm2s(cibra, ciket, norb, nelec):
    r"""
    Spin-resolved transition 2-RDMs

    .. math::

        \Gamma^{\sigma\tau}_{pqrs} = \langle\mathrm{bra}|
        a^\dagger_{p\sigma}a^\dagger_{r\tau}a_{s\tau}a_{q\sigma}
        |\mathrm{ket}\rangle.

    Args:
        cibra, ciket:
            Real CI vectors of shape (nstra, nstrb) in the same determinant
            space. Complex CI vectors are not supported.
        norb:
            Number of active orbitals.
        nelec:
            (neleca, nelecb) of the determinant space.
    Returns:
        (dm2aa, dm2ab, dm2ba, dm2bb): tuple of ndarray (norb,)*4
            Normal-ordered tensors, using PySCF's ``reorder=True`` convention.

    Notes:
        Reuses PySCF's real-valued transition-RDM driver directly. Inputs
        are stored as contiguous float64 arrays for its C interface.
    """
    cibra, ciket = np.asarray(cibra), np.asarray(ciket)
    if np.iscomplexobj(cibra) or np.iscomplexobj(ciket):
        raise ValueError('SSC transition densities require real CI vectors')
    return fci.direct_spin1.trans_rdm12s(
        np.asarray(cibra, dtype=np.float64, order='C'),
        np.asarray(ciket, dtype=np.float64, order='C'),
        norb, nelec, reorder=True)[1]



def _lower_spin_projection(ci, norb, nelec, two_s, target_two_m):
    r"""
    Lower the CI vector :math:`|S,S\rangle` to the normalized
    :math:`|S,M\rangle` with :math:`S_-`,

    .. math::

        S_-|S,M\rangle = \sqrt{(S+M)(S-M+1)}\,|S,M-1\rangle.

    Args:
        ci:
            Real CI vector of the Ms = S component.
        norb:
            Number of active orbitals.
        nelec:
            (neleca, nelecb) of ci.
        two_s:
            Twice the total spin S.
        target_two_m:
            Twice the target projection M.
    Returns:
        ci: ndarray
            Real lowered vector, with the input sign and norm preserved
            for a spin-pure highest-weight input.
        nelec: tuple of int
            (neleca, nelecb) of the lowered determinant space.

    Notes:
        The existing SISO C contraction implements ``sum_p b†_p a_p`` with
        PySCF's alpha/beta fermionic phase. Its first integral component is
        set to the identity; the other two are zero. Ladder normalization is
        applied after every step. Its complex output buffer is converted
        to real coefficients because the input and lowering operator are
        real. No spin-purity test is performed.
    """
    two_s = _validate_twos(two_s)
    target_two_m = index(target_two_m)
    if (abs(target_two_m) > two_s
            or (two_s - target_two_m) % 2):
        raise ValueError('target_two_m must be a projection of spin two_s/2')
    if nelec[0] - nelec[1] != two_s:
        raise ValueError('ci must be a highest-weight (M = S) vector')
    orbs = list(range(norb))
    # S_- = sum_p b+_p a_p through SISO's spin-minus contraction with a unit
    # one-electron operator in the first spherical component.
    h1e = np.zeros((3, norb, norb))
    h1e[0] = np.eye(norb)

    result = np.asarray(ci)
    if np.iscomplexobj(result):
        raise ValueError('SSC spin lowering requires a real CI vector')
    result = np.asarray(result, dtype=np.float64)
    current_nelec = nelec
    current_two_m = two_s
    while current_two_m > target_two_m:
        coefficient = 0.5 * np.sqrt(
            (two_s + current_two_m)
            * (two_s - current_two_m + 2))
        link_index = [
            fci.cistring.gen_cre_str_index(orbs, current_nelec[0] - 1),
            fci.cistring.gen_des_str_index(orbs, current_nelec[1] + 1)]
        result = contract_spin_minus(
            h1e, result[None], link_index)[0, 0].real / coefficient
        current_nelec = (current_nelec[0] - 1, current_nelec[1] + 1)
        current_two_m -= 2
    return result, current_nelec


def make_quintet_density_q0(cibra, ciket, norb, nelec, *, two_s_bra=None,
                            two_s_ket=None):
    r"""
    Rank-2, :math:`q=0` spin transition density
    :math:`\langle\mathrm{cibra}|Q^0_{pqrs}|\mathrm{ciket}\rangle`.

    With PySCF's normal-ordered convention
    :math:`\Gamma^{\sigma\tau}_{pqrs}=\langle a^\dagger_{p\sigma}
    a^\dagger_{r\tau}a_{s\tau}a_{q\sigma}\rangle` the one-particle terms
    cancel and

    .. math::

        Q^0_{pqrs}=\frac{1}{4\sqrt{6}}\left[
        -\Gamma^{\alpha\alpha}_{psrq}-\Gamma^{\beta\beta}_{psrq}
        +\Gamma^{\alpha\beta}_{psrq}+\Gamma^{\beta\alpha}_{psrq}
        -\Gamma^{\alpha\beta}_{pqrs}-\Gamma^{\beta\alpha}_{pqrs}\right].

    Without spin labels both CI vectors are in the same
    :math:`(N_\alpha,N_\beta)` space. With both labels they are the
    :math:`M_S=S` components of states of spin :math:`S_\mathrm{bra}` and
    :math:`S_\mathrm{ket}`, which are lowered to the common
    :math:`M=\min(S_\mathrm{bra},S_\mathrm{ket})` before evaluating the
    :math:`q=0` element. Pairs with :math:`|S_\mathrm{bra}-S_\mathrm{ket}|>2`
    or :math:`S_\mathrm{bra}+S_\mathrm{ket}<2` give zero. The spin labels
    are checked for parity and orbital capacity; spin purity and normalization
    remain the caller's responsibility. "Quintet" denotes tensor rank two,
    and does not restrict either electronic state to spin two.

    Args:
        cibra, ciket:
            Real CI vectors in PySCF alpha-string/beta-string ordering.
            With spin labels they may have different shapes. Complex CI
            vectors are not supported.
        norb:
            Number of active orbitals.
        nelec:
            Electron count (int or pair). With spin labels only the total is
            used.
        two_s_bra, two_s_ket:
            Twice the total spin of the bra and ket, given together.
    Returns:
        q0: ndarray of shape (norb, norb, norb, norb)
            Real, dimensionless spin density. No magnetic prefactor or
            orbital integrals are included.

    Raises:
        ValueError: If only one spin label is supplied, a label is not a
            nonnegative integer, the determinant-space sizes are invalid,
            or either CI vector is complex.
    """
    def reshape(ci, nelec):
        return np.asarray(ci).reshape(
            fci.cistring.num_strings(norb, nelec[0]),
            fci.cistring.num_strings(norb, nelec[1]))

    if np.iscomplexobj(cibra) or np.iscomplexobj(ciket):
        raise ValueError('SSC quintet densities require real CI vectors')
    if (two_s_bra is None) != (two_s_ket is None):
        raise ValueError('two_s_bra and two_s_ket must be supplied together')
    if two_s_bra is None:
        common_nelec = _unpack_nelec(nelec)
        cibra = reshape(cibra, common_nelec)
        ciket = reshape(ciket, common_nelec)
    else:
        two_s_bra = _validate_twos(two_s_bra)
        two_s_ket = _validate_twos(two_s_ket)
        nelectron = int(np.sum(nelec))
        bra_nelec = _highest_weight_nelec(norb, nelectron, two_s_bra)
        ket_nelec = _highest_weight_nelec(norb, nelectron, two_s_ket)
        cibra = reshape(cibra, bra_nelec)
        ciket = reshape(ciket, ket_nelec)
        if not _rank2_allowed(two_s_bra, two_s_ket):
            return np.zeros((norb,) * 4)

        common_two_m = min(two_s_bra, two_s_ket)
        cibra, common_nelec = _lower_spin_projection(
            cibra, norb, bra_nelec, two_s_bra,
            common_two_m)
        ciket, _ = _lower_spin_projection(
            ciket, norb, ket_nelec, two_s_ket,
            common_two_m)

    dm2aa, dm2ab, dm2ba, dm2bb = _transition_dm2s(
        cibra, ciket, norb, common_nelec)

    # transpose(0, 3, 2, 1)[p,q,r,s] = dm2[p,s,r,q]
    psrq = (0, 3, 2, 1)
    q0 = (-dm2aa.transpose(psrq) - dm2bb.transpose(psrq)
          + dm2ab.transpose(psrq) + dm2ba.transpose(psrq)
          - dm2ab - dm2ba)
    q0 *= 1.0 / (4.0 * np.sqrt(6.0))
    return q0


def rank2_cg_coefficients(two_s, two_s_ket=None):
    r"""
    Rank-2 Clebsch-Gordan assembly tensor for a pair of spins.

    .. math::

        C^q_{MM'} = A\,(-1)^{-q}\langle S_k M';2,-q|S_b M\rangle,
        \qquad
        A=\langle S_k M_0;2,0|S_b M_0\rangle^{-1},

    with :math:`M_0=\min(S_b,S_k)` and :math:`M,M'` in increasing order. It
    multiplies the :math:`q=0` density evaluated at the common :math:`M_0`,
    see :func:`make_quintet_density_q0`.

    Args:
        two_s:
            Twice the bra total spin.
        two_s_ket:
            Twice the ket total spin. Defaults to two_s.
    Returns:
        coefficients: ndarray of shape (5, two_s + 1, two_s_ket + 1)
            Ordered as q = (-2, -1, 0, 1, 2); zero if the rank-2 selection
            rule is not satisfied. The only nonzero entries obey
            ``M_bra = M_ket - q``. The common-M, q=0 coefficient is one.

    Raises:
        ValueError: If either twice-spin label is not a nonnegative integer.
    """
    two_s = _validate_twos(two_s)
    two_s_ket = two_s if two_s_ket is None else _validate_twos(two_s_ket)
    coefficients = np.zeros((5, two_s + 1, two_s_ket + 1))
    if not _rank2_allowed(two_s, two_s_ket):
        return coefficients

    common_two_m = min(two_s, two_s_ket)
    anchor = _rank2_cg(two_s_ket, common_two_m, 0, two_s)
    for iq, q in enumerate(_QS):
        for jm, two_m_ket in enumerate(range(-two_s_ket, two_s_ket + 1, 2)):
            two_m_bra = two_m_ket - 2 * q
            if abs(two_m_bra) <= two_s:
                im = (two_m_bra + two_s) // 2
                coefficients[iq, im, jm] = (
                    (-1.0) ** q
                    * _rank2_cg(two_s_ket, two_m_ket, -q, two_s) / anchor)

    return coefficients


def contract_ssc_integrals_q0(ssc_spherical, q0):
    r"""
    Contract the five spherical SSC integrals with :math:`Q^0` densities.

    The contraction is bilinear: neither operand is conjugated, since the
    density already includes bra conjugation. Spherical phases are applied
    later by :func:`rank2_cg_coefficients`. Leading density axes can contain
    root pairs or other batches. No physical prefactor is applied here.

    Args:
        ssc_spherical:
            Integrals of shape (5, norb, norb, norb, norb), :math:`q=-2..2`.
        q0:
            Densities of shape (..., norb, norb, norb, norb).
    Returns:
        reduced: ndarray of shape q0.shape[:-4] + (5,)
    """
    ssc_spherical, q0 = np.asarray(ssc_spherical), np.asarray(q0)
    if (ssc_spherical.ndim != 5 or ssc_spherical.shape[0] != 5
            or q0.ndim < 4 or ssc_spherical.shape[1:] != q0.shape[-4:]):
        raise ValueError('expected spherical integrals (5, norb, norb, norb, norb) '
                         'and compatible q0 (..., norb, norb, norb, norb)')
    return np.einsum(
        'mpqrs,...pqrs->...m', ssc_spherical, q0, optimize=True)


def assemble_ssc_hamiltonian_block(reduced_ssc, two_s, prefactor=1.0, *,
                                   two_s_ket=None):
    """
    Expand an SSC root-pair block over all Ms components.

    Contract the last, spherical axis with :func:`rank2_cg_coefficients`.
    The intermediate axes are ``(bra_root, bra_M, ket_root, ket_M)`` before
    flattening, matching SISO's model-state ordering. This routine retains
    complex matrix elements and does not enforce Hermiticity on a block.

    Args:
        reduced_ssc:
            Spherical contractions of shape (nroots_bra, nroots_ket, 5).
        two_s:
            Twice the bra total spin.
        prefactor:
            Overall multiplier; 1 returns the raw contraction. Use
            ``SSC_PHYSICAL_PREFACTOR`` with unscaled atomic-unit integrals
            to obtain energies in Hartree.
        two_s_ket:
            Twice the ket total spin. Defaults to two_s.
    Returns:
        block: ndarray of shape
            (nroots_bra*(two_s+1), nroots_ket*(two_s_ket+1)), root-major with
            increasing Ms.
    """
    reduced_ssc = np.asarray(reduced_ssc)
    if reduced_ssc.ndim != 3 or reduced_ssc.shape[-1] != 5:
        raise ValueError('reduced_ssc must have shape (nroots_bra, nroots_ket, 5)')
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
    r"""
    SSC Hamiltonian in the SI-SO model-state basis.

    All spin pairs with |S_bra - S_ket| <= 2 are included; blocks excluded by
    the rank-2 selection rule are zero. The intermediates are stored in
    siso.imds (ssc_integrals, q0, ssc_reduced, q0_pairs, ssc_reduced_pairs,
    hssc).

    Args:
        siso:
            SISO instance or compatible object with ``mc``, ``twoslst`` and
            ``imds``. ``mc`` supplies ``ncas`` and ``nelecas``; integral
            generation additionally requires ``mo_coeff``, ``ncore`` and
            ``_scf.mol``. ``imds.c`` holds a list of highest-weight root
            arrays, one per entry of ``twoslst``. If missing or None, the
            existing SISO ``assemble_civecs`` builds it from ``mc.ci`` and
            ``siso.statelis``. CI states must be real, normalized and
            spin-pure; the spherical integrals and Hamiltonian may be complex.
        ssc_integrals:
            Active-space integrals, spherical (5, ncas, ncas, ncas, ncas) or
            Cartesian (3, 3, ncas, ncas, ncas, ncas). Computed if None.
            Inputs must use the unscaled atomic-unit convention of
            ``sscint``; spherical order is q = (-2, -1, 0, +1, +2).
            Set prefactor=1 when supplying already scaled integrals.
        use_df:
            Use RI integrals if ssc_integrals is None. False uses the exact
            integrals, falling back to RI if they do not fit in memory.
        auxbasis:
            RI auxiliary basis. If None, sscint.make_ssc_auxbasis is used.
        prefactor:
            Overall multiplier, by default :math:`g_e^2\alpha^2/4`.
    Returns:
        hssc: ndarray of shape (nstates, nstates)
            Hermitian SSC Hamiltonian in Hartree with the default prefactor
            and unscaled atomic-unit integrals. Spins follow ``twoslst``;
            each spin sector is root-major with increasing M.

    Notes:
        This standalone driver does not enable SSC in ``SISO.kernel`` or
        add spin-free energies or SOC. Each call recomputes the SSC densities
        and overwrites the SSC fields of ``imds``; they are stored results,
        not a cache consulted by later calls. ``q0_pairs`` and
        ``ssc_reduced_pairs`` map allowed ``(2S_bra, 2S_ket)`` pairs to arrays;
        ``q0`` and ``ssc_reduced`` are their spin-diagonal lists, with None
        for forbidden pairs. ``ssc_integrals`` and ``hssc`` hold the input
        spherical tensor and final Hamiltonian.

        Storing all root-pair densities requires memory proportional to
        ``sum(nroots_bra*nroots_ket) * ncas**4`` over allowed spin pairs.
        A non-Hermitian assembled matrix emits a warning and is replaced by
        its Hermitian part, including for externally supplied integrals.
    """
    mc = siso.mc
    ncas = int(mc.ncas)
    nelectron = int(np.sum(mc.nelecas))
    twoslst = siso.twoslst

    if ssc_integrals is None:
        mo_cas = mc.mo_coeff[:, mc.ncore:mc.ncore + ncas]
        spherical = get_ssc_integrals(
            mc._scf.mol, mo_cas, auxbasis=auxbasis, use_df=use_df)
    else:
        spherical = np.asarray(ssc_integrals)
        if spherical.ndim == 6:
            spherical = cartesian_to_spherical(spherical)
    if spherical.shape != (5,) + (ncas,) * 4:
        raise ValueError('SSC integrals must have shape (5, ncas, ncas, ncas, ncas) '
                         'or (3, 3, ncas, ncas, ncas, ncas)')

    imds = siso.imds
    civecs = getattr(imds, 'c', None)
    if civecs is None:
        civecs = assemble_civecs(siso)

    nspin = len(twoslst)
    q0_pairs = {}
    reduced_pairs = {}
    h_col = [[None] * nspin for _ in range(nspin)]

    spin_pairs = ((i, j, s1, s2) for i, s1 in enumerate(twoslst)
                  for j, s2 in enumerate(twoslst))
    for i, j, s1, s2 in spin_pairs:
        civecbra, civecket = civecs[i], civecs[j]
        nbra, nket = civecbra.shape[0], civecket.shape[0]

        if not _rank2_allowed(s1, s2):  # No connection
            h_col[i][j] = np.zeros((nbra * (s1 + 1), nket * (s2 + 1)),
                                   dtype=np.complex128)
            continue

        logger.debug(siso, 'SSC block 2S_bra = %d, 2S_ket = %d', s1, s2)
        q0 = np.empty((nbra, nket) + (ncas,) * 4)
        for ibra in range(nbra):
            for iket in range(nket):
                q0[ibra, iket] = make_quintet_density_q0(
                    civecbra[ibra], civecket[iket], ncas, nelectron,
                    two_s_bra=s1, two_s_ket=s2)

        reduced = contract_ssc_integrals_q0(spherical, q0)
        h_col[i][j] = assemble_ssc_hamiltonian_block(
            reduced, s1, prefactor=prefactor, two_s_ket=s2)
        q0_pairs[(s1, s2)] = q0
        reduced_pairs[(s1, s2)] = reduced

    hssc = np.block(h_col)

    hssc_deviation = np.max(np.abs(hssc - hssc.conj().T))
    if not np.allclose(hssc, hssc.conj().T):
        logger.warn(siso, 'SSC Hamiltonian is not Hermitian; max deviation: '
                    '%.3e. Symmetrizing.', hssc_deviation)
    hssc = 0.5 * (hssc + hssc.conj().T)

    imds.ssc_integrals = spherical
    # Diagonal-by-spin lists plus dictionaries for every allowed spin pair.
    imds.q0 = [q0_pairs.get((s, s)) for s in twoslst]
    imds.ssc_reduced = [reduced_pairs.get((s, s)) for s in twoslst]
    imds.q0_pairs = q0_pairs
    imds.ssc_reduced_pairs = reduced_pairs
    imds.hssc = hssc
    return hssc
