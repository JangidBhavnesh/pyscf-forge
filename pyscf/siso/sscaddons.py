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

"""Effective-spin analysis of SISO spin--spin Hamiltonians."""

from numbers import Integral

import numpy as np

from pyscf.data import nist
from pyscf.lib import logger


_SUPPORTED_MULTIPLICITIES = (3, 4, 5, 7)
_MIN_PROJECTION_SINGULAR_VALUE = 0.9


def triplet_zfs_parameters(levels):
    r"""Extract conventional ``D`` and ``E`` from three spin-one levels.

    The levels are matched to the eigenvalues of

    .. math::

        H_\mathrm{ZFS} = D[S_z^2-S(S+1)/3]
        + E(S_x^2-S_y^2), \qquad S=1,

    using the conventional principal-axis choice ``|D| >= 3 E``.
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
    """Select three SI eigenstates by overlap with a spin-free triplet root."""
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


def _validate_requests(mysiso, mltp, nroots):
    """Validate multiplicities and return requested root counts."""
    if mltp is None:
        mltp = []
    if isinstance(mltp, (str, bytes)):
        raise TypeError("mltp must be a sequence of multiplicities")
    try:
        multiplicities = list(mltp)
    except TypeError as err:
        raise TypeError(
            "mltp must be a sequence of multiplicities") from err

    statelis = np.asarray(mysiso.statelis)
    modelspace = getattr(mysiso, 'modelspace', None)
    if modelspace is None:
        modelspace_multiplicities = {
            index + 1 for index, count in enumerate(statelis) if count > 0}
    else:
        try:
            modelspace_multiplicities = {
                int(state[1]) for state in modelspace}
        except (IndexError, TypeError, ValueError) as err:
            raise ValueError(
                "mysiso.modelspace contains an invalid entry") from err
    if not multiplicities:
        multiplicities = [
            mult for mult in _SUPPORTED_MULTIPLICITIES
            if mult - 1 < statelis.size and statelis[mult - 1] > 0
        ]
    if not multiplicities:
        raise ValueError(
            "the SISO model space contains no supported spin multiplets "
            "(multiplicities 3, 4, 5, or 7)")

    validated_multiplicities = []
    for mult in multiplicities:
        if (isinstance(mult, (bool, np.bool_))
                or not isinstance(mult, Integral)):
            raise TypeError("multiplicities in mltp must be integers")
        mult = int(mult)
        if mult not in _SUPPORTED_MULTIPLICITIES:
            raise ValueError(
                "mltp supports only multiplicities 3, 4, 5, and 7")
        if mult in validated_multiplicities:
            raise ValueError("mltp must not contain duplicate multiplicities")
        if (mult not in modelspace_multiplicities
                or mult - 1 >= statelis.size
                or statelis[mult - 1] == 0):
            raise ValueError(
                f"multiplicity {mult} is not part of the SISO model space")
        validated_multiplicities.append(mult)

    if nroots is None:
        nroots = []
    if isinstance(nroots, (str, bytes)):
        raise TypeError("nroots must be a sequence of root counts")
    if isinstance(nroots, Integral) and not isinstance(
            nroots, (bool, np.bool_)):
        nroots = [int(nroots)]
    else:
        try:
            nroots = list(nroots)
        except TypeError as err:
            raise TypeError(
                "nroots must be a sequence of root counts") from err

    if not nroots:
        root_counts = [1] * len(validated_multiplicities)
    else:
        if len(nroots) != len(validated_multiplicities):
            raise ValueError("nroots must contain one count for each mltp entry")
        root_counts = []
        for mult, count in zip(validated_multiplicities, nroots):
            if (isinstance(count, (bool, np.bool_))
                    or not isinstance(count, Integral)):
                raise TypeError("root counts in nroots must be integers")
            count = int(count)
            available = int(statelis[mult - 1])
            if count < 1 or count > available:
                raise ValueError(
                    f"nroots requests {count} multiplicity-{mult} roots; "
                    f"the model space contains {available}")
            root_counts.append(count)

    return validated_multiplicities, root_counts


def _spin_free_hamiltonian(mysiso):
    """Return the diagonal spin-free Hamiltonian in SISO ordering."""
    energies = np.concatenate([
        np.repeat(np.asarray(mysiso.imds.e[index]), int(two_s) + 1)
        for index, two_s in enumerate(mysiso.twoslst)
    ])
    return np.diag(energies)


def _component_hamiltonians(mysiso):
    """Construct the SOC, SSC, and combined state-interaction matrices."""
    imds = mysiso.imds
    if (getattr(imds, 'e', None) is None
            or getattr(imds, 'd', None) is None
            or getattr(imds, 'c', None) is None):
        mysiso.build_imds()

    spin_free = getattr(imds, 'hspinfree', None)
    if spin_free is None:
        spin_free = _spin_free_hamiltonian(mysiso)
        imds.hspinfree = spin_free

    soc = getattr(imds, 'hsoc', None)
    if soc is None:
        soc_interaction = getattr(imds, 'hsoc_interaction', None)
        if soc_interaction is None:
            soc_interaction = mysiso.compute_soc_hamiltonian()
            imds.hsoc_interaction = soc_interaction
        soc = spin_free + soc_interaction
        imds.hsoc = soc
    components = {'soc': 0.5 * (soc + soc.conj().T)}

    if getattr(mysiso, 'ssc', False):
        hssc = getattr(mysiso.imds, 'hssc', None)
        if hssc is None:
            hssc = mysiso.compute_ssc_hamiltonian()
            mysiso.imds.hssc = hssc
        hssc = np.asarray(hssc)
        if hssc.shape != spin_free.shape:
            raise ValueError(
                "SSC Hamiltonian shape is inconsistent with the SISO model "
                f"space: {hssc.shape} != {spin_free.shape}")
        ssc = getattr(imds, 'hssc_only', None)
        if ssc is None:
            ssc = spin_free + hssc
            imds.hssc_only = ssc
        total = getattr(imds, 'htotal', None)
        if total is None:
            total = soc + hssc
            imds.htotal = total
        components['ssc'] = 0.5 * (ssc + ssc.conj().T)
        components['total'] = 0.5 * (total + total.conj().T)
    else:
        components['total'] = components['soc']

    return components


def _root_reference_indices(mysiso):
    """Map ``(multiplicity, root)`` to its spin-projection indices."""
    offsets = {}
    offset = 0
    for two_s in mysiso.twoslst:
        two_s = int(two_s)
        mult = two_s + 1
        count = int(mysiso.statelis[two_s])
        for root in range(count):
            start = offset + root * mult
            offsets[mult, root] = np.arange(start, start + mult)
        offset += count * mult
    return offsets


def _effective_multiplet_hamiltonian(hamiltonian, reference_indices):
    r"""Project selected eigenstates into a reference ``|S,M_S>`` space.

    The unitary polar factor of the reference/eigenstate overlap produces the
    canonical Hermitian effective Hamiltonian.  Unlike a spectrum-only fit,
    it retains the eigenvector information required to distinguish ``D`` and
    ``E`` for a quartet.
    """
    hamiltonian = np.asarray(hamiltonian)
    if (hamiltonian.ndim != 2
            or hamiltonian.shape[0] != hamiltonian.shape[1]):
        raise ValueError("the state-interaction Hamiltonian must be square")
    if not np.issubdtype(hamiltonian.dtype, np.number):
        raise TypeError("the state-interaction Hamiltonian must be numeric")
    if not np.all(np.isfinite(hamiltonian)):
        raise ValueError("the state-interaction Hamiltonian must be finite")

    hamiltonian = 0.5 * (hamiltonian + hamiltonian.conj().T)
    dimension = hamiltonian.shape[0]
    origin = np.trace(hamiltonian).real / dimension
    energies, vectors = np.linalg.eigh(
        hamiltonian - origin * np.eye(dimension))

    reference_indices = np.asarray(reference_indices, dtype=int)
    multiplicity = reference_indices.size
    weights = np.sum(
        np.abs(vectors[reference_indices, :])**2, axis=0)
    selected = np.argsort(weights)[-multiplicity:]
    selected = selected[np.argsort(energies[selected])]

    overlap = vectors[np.ix_(reference_indices, selected)]
    left, singular_values, right_h = np.linalg.svd(overlap)
    polar = left @ right_h
    effective = polar @ np.diag(energies[selected]) @ polar.conj().T
    effective += origin * np.eye(multiplicity)
    effective = 0.5 * (effective + effective.conj().T)

    return effective, {
        'selected_states': selected,
        'selected_energies': energies[selected] + origin,
        'reference_weights': weights[selected],
        'overlap_singular_values': singular_values,
    }


def _fit_rank2_hamiltonian(hamiltonian, spin):
    r"""Fit ``H = E0 + S.D.S`` with a real symmetric traceless ``D``."""
    # Reuse the spin matrices used by the SINGLE_ANISO interface.  Their
    # increasing-M_S ordering matches the SISO Hamiltonian blocks.
    from pyscf.siso.anisoaddons import spin_operators

    sx, sy, sz = spin_operators(spin)
    dimension = sx.shape[0]
    hamiltonian = np.asarray(hamiltonian)
    if hamiltonian.shape != (dimension, dimension):
        raise ValueError(
            f"the spin-{spin:g} Hamiltonian must have shape "
            f"({dimension}, {dimension})")

    identity = np.eye(dimension, dtype=np.complex128)
    operators = [
        identity,
        sx @ sx - sz @ sz,
        sy @ sy - sz @ sz,
        sx @ sy + sy @ sx,
        sx @ sz + sz @ sx,
        sy @ sz + sz @ sy,
    ]
    design = np.column_stack([operator.ravel() for operator in operators])
    real_design = np.vstack((design.real, design.imag))
    target = np.concatenate((hamiltonian.real.ravel(),
                             hamiltonian.imag.ravel()))
    coefficients = np.linalg.lstsq(real_design, target, rcond=None)[0]
    fitted = sum(
        coefficient * operator
        for coefficient, operator in zip(coefficients, operators))

    dxx, dyy, dxy, dxz, dyz = coefficients[1:]
    d_tensor = np.asarray([
        [dxx, dxy, dxz],
        [dxy, dyy, dyz],
        [dxz, dyz, -dxx - dyy],
    ])

    principal_values, principal_axes = np.linalg.eigh(d_tensor)
    iz = int(np.argmax(np.abs(principal_values)))
    transverse = [axis for axis in range(3) if axis != iz]
    if principal_values[transverse[0]] >= principal_values[transverse[1]]:
        ix, iy = transverse
    else:
        iy, ix = transverse
    order = [ix, iy, iz]
    principal_values = principal_values[order]
    principal_axes = principal_axes[:, order]
    if np.linalg.det(principal_axes) < 0:
        principal_axes[:, 0] *= -1

    d_value = 1.5 * principal_values[2]
    e_value = 0.5 * (principal_values[0] - principal_values[1])
    residual = np.linalg.norm(hamiltonian - fitted)
    anisotropic_norm = np.linalg.norm(
        hamiltonian - coefficients[0] * identity)
    relative_residual = (
        residual / anisotropic_norm if anisotropic_norm > 1e-15 else 0.0)

    return {
        'D': float(d_value),
        'E': float(e_value),
        'D_tensor': d_tensor,
        'principal_values': principal_values,
        'principal_axes': principal_axes,
        'energy_offset': float(coefficients[0]),
        'fitted_hamiltonian': fitted,
        'residual': float(residual),
        'relative_residual': float(relative_residual),
    }


def compute_D_and_E(mysiso, mltp=(), nroots=None):
    r"""Compute effective ``D`` and ``E`` for selected SISO roots.

    Args:
        mysiso : :class:`pyscf.siso.siso.SISO`
            SISO object containing a built model space.  If its intermediates
            have not yet been built, this function builds them.
        mltp : sequence of int, optional
            Multiplicities to analyze.  Supported values are ``3`` (triplet),
            ``4`` (quartet), ``5`` (quintet), and ``7`` (septet, ``S=3``).
            If omitted or empty, all supported multiplicities present in the
            model space are used.
        nroots : sequence of int, optional
            Number of roots to analyze for each requested multiplicity,
            starting from its first model-space root.  If omitted or empty,
            only the first root of every requested multiplicity is analyzed.

    Returns:
        list of dict
            One entry per requested root.  Each entry contains ``total`` and
            ``soc`` fits and, when ``mysiso.ssc`` is true, an ``ssc`` fit.
            ``D`` and ``E`` are returned in Hartree; :func:`finalize` prints
            them in ``cm^-1``.

    Notes:
        The SOC and SSC component Hamiltonians each include the diagonal
        spin-free energies.  This retains the correct energy denominators
        when other model-space roots are downfolded into the selected
        multiplet.  The reported components are independent effective-spin
        fits; their principal ``D`` and ``E`` values need not add when their
        principal axes differ.
    """
    multiplicities, root_counts = _validate_requests(
        mysiso, mltp, nroots)
    components = _component_hamiltonians(mysiso)
    reference_indices = _root_reference_indices(mysiso)

    results = []
    for mult, count in zip(multiplicities, root_counts):
        spin = 0.5 * (mult - 1)
        for root in range(count):
            indices = reference_indices[mult, root]
            result = {
                'multiplicity': mult,
                'spin': spin,
                'root': root,
                'reference_indices': indices,
            }
            for name, hamiltonian in components.items():
                effective, projection = _effective_multiplet_hamiltonian(
                    hamiltonian, indices)
                fit = _fit_rank2_hamiltonian(effective, spin)
                fit.update(projection)
                minimum_overlap = float(np.min(
                    projection['overlap_singular_values']))
                fit['minimum_overlap_singular_value'] = minimum_overlap
                fit['projection_reliable'] = (
                    minimum_overlap >= _MIN_PROJECTION_SINGULAR_VALUE)
                result[name] = fit

                if not fit['projection_reliable']:
                    logger.warn(
                        mysiso,
                        'Multiplicity-%d root %d does not define an isolated '
                        '%s effective-spin manifold; minimum projection '
                        'singular value %.3f is below %.3f',
                        mult, root, name, minimum_overlap,
                        _MIN_PROJECTION_SINGULAR_VALUE)
                if fit['relative_residual'] > 1e-6:
                    logger.warn(
                        mysiso,
                        'Multiplicity-%d root %d %s Hamiltonian is not well '
                        'described by a rank-2 spin Hamiltonian; relative '
                        'fit residual %.3e', mult, root, name,
                        fit['relative_residual'])
            results.append(result)

    mysiso.d_and_e = results
    finalize(mysiso, results)
    return results


def finalize(mysiso, results=None):
    """Print fitted total, SOC, and optional SSC ``D`` and ``E`` values."""
    if results is None:
        results = getattr(mysiso, 'd_and_e', None)
    if results is None:
        raise ValueError("compute_D_and_E must be called before finalize")

    # SISO historically inherited ``verbose=0`` from StreamObject.  Use the
    # attached multiconfigurational calculation as the authoritative output
    # stream so this report follows the user's molecular verbosity setting.
    log = logger.Logger(mysiso.mc.stdout, mysiso.mc.verbose)
    include_ssc = any('ssc' in result for result in results)

    log.note(' ')
    log.note('******** Effective spin-Hamiltonian D and E (cm^-1) ********')
    conversion = nist.HARTREE2WAVENUMBER

    def formatted(parameter, component):
        if not component['projection_reliable']:
            return f"{'N/A':>13}"
        return f"{component[parameter] * conversion:13.6f}"

    multiplicities = []
    for result in results:
        mult = result['multiplicity']
        if mult not in multiplicities:
            multiplicities.append(mult)

    for mult in multiplicities:
        heading = f'Multiplicity (2S+1): {mult}'
        log.note(' ')
        log.note('%s', heading)
        log.note('%s', '-' * len(heading))
        if include_ssc:
            log.note(
                ' Root no. |      D(total)      E(total) |'
                '        D(SOC)        E(SOC) |'
                '        D(SSC)        E(SSC)')
        else:
            log.note(
                ' Root no. |      D(total)      E(total) |'
                '        D(SOC)        E(SOC)')

        for result in results:
            if result['multiplicity'] != mult:
                continue
            total = result['total']
            soc = result['soc']
            line = (
                f" {result['root']:8d} |"
                f" {formatted('D', total)}"
                f" {formatted('E', total)} |"
                f" {formatted('D', soc)}"
                f" {formatted('E', soc)}")
            if include_ssc:
                ssc = result['ssc']
                line += (
                    f" | {formatted('D', ssc)}"
                    f" {formatted('E', ssc)}")
            log.note('%s', line)
    log.note(' ')
    return results
