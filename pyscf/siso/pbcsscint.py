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

"""Gamma-point periodic spin--spin coupling integrals in an AO/RI form.

The implementation follows ``PBC_SSC_ROADMAP.md`` and fixes the
Fourier-transform, derivative-sign, and ``G = 0`` conventions explicitly.
"""

from numbers import Real

import numpy as np
import scipy.linalg

from pyscf import lib
from pyscf.lib import logger
from pyscf.pbc import df as pbcdf
from pyscf.pbc import gto as pbcgto
from pyscf.pbc.df import aft, ft_ao
from pyscf.pbc.df.gdf_builder import _CCGDFBuilder
from pyscf.pbc.df.rsdf_builder import _RSGDFBuilder


_GAMMA = np.zeros(3)
_GAMMA_IMAG_TOL = 1e-10


class _MetricCaptureMixin:
    """Capture the exact metric decomposition consumed by a GDF build."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ssc_metric_decompositions = []

    def decompose_j2c(self, metric):
        metric = np.asarray(metric)
        # PySCF normally tries Cholesky first and only falls back to an
        # eigendecomposition when Cholesky raises.  A numerically singular
        # matrix can occasionally pass Cholesky and retain arbitrary tiny
        # directions.  Inspect the lowest eigenvalue so that the advertised
        # linear-dependence threshold is deterministic.
        lowest_eigenvalue = scipy.linalg.eigvalsh(
            metric,
            subset_by_index=(0, 0),
            check_finite=False,
        )[0]
        if lowest_eigenvalue <= self.linear_dep_threshold:
            with np.errstate(divide="ignore", invalid="ignore"):
                decomposition = self.eigenvalue_decomposed_metric(metric)
        else:
            decomposition = super().decompose_j2c(metric)
        factor, negative_factor, tag = decomposition
        captured = (
            np.asarray(factor).copy(),
            None
            if negative_factor is None
            else np.asarray(negative_factor).copy(),
            tag,
        )
        self.ssc_metric_decompositions.append(captured)
        return decomposition


class _MetricCaptureRSGDFBuilder(_MetricCaptureMixin, _RSGDFBuilder):
    """Range-separated GDF builder that exposes its metric coordinates."""


class _MetricCaptureCCGDFBuilder(_MetricCaptureMixin, _CCGDFBuilder):
    """Compensated-charge GDF builder that exposes its metric coordinates."""


def _validate_gamma_cell(cell):
    """Validate the cell assumptions of the first periodic SSC milestone."""
    if not isinstance(cell, pbcgto.Cell):
        raise TypeError("cell must be a pyscf.pbc.gto.Cell")
    if not getattr(cell, "_built", False):
        raise ValueError("cell must be built before computing SSC integrals")
    if cell.dimension != 3:
        raise NotImplementedError(
            "periodic SSC integrals currently support only 3D cells"
        )


def _validate_mesh(mesh, default):
    """Return a positive three-integer reciprocal mesh."""
    if mesh is None:
        mesh = default
    mesh_array = np.asarray(mesh)
    if mesh_array.shape != (3,):
        raise ValueError("mesh must contain three positive integers")
    if not np.issubdtype(mesh_array.dtype, np.integer):
        raise TypeError("mesh must contain three positive integers")
    if np.any(mesh_array <= 0):
        raise ValueError("mesh must contain three positive integers")
    return np.asarray(mesh_array, dtype=np.int32)


def _select_ssc_mesh(cell, auxcell, mesh=None):
    """Select a derivative-safe reciprocal mesh for the auxiliary metric."""
    if mesh is not None:
        return _validate_mesh(mesh, cell.mesh)

    # The W kernel contains G_a G_b in addition to the Coulomb kernel.  Base
    # the cutoff on the auxiliary functions themselves rather than assuming
    # that the primary-AO mesh is adequate.  PySCF's four-center Coulomb
    # estimate is conservative here: for equal Gaussian exponents its pair
    # densities decay more slowly in reciprocal space than P(G)^* Q(G).
    ke_cutoff = aft.estimate_ke_cutoff(auxcell, precision=cell.precision)
    auxiliary_mesh = np.asarray(cell.cutoff_to_mesh(ke_cutoff), dtype=np.int32)
    selected_mesh = np.maximum(
        _validate_mesh(cell.mesh, cell.mesh), auxiliary_mesh
    )
    selected_mesh = np.asarray(
        cell.symmetrize_mesh(selected_mesh), dtype=np.int32
    )
    selected_mesh = _validate_mesh(selected_mesh, cell.mesh)
    logger.debug(
        cell,
        "Gamma SSC automatic mesh %s from auxiliary cutoff %.6g and cell "
        "mesh %s",
        selected_mesh,
        ke_cutoff,
        cell.mesh,
    )
    return selected_mesh


def _real_if_gamma(array, name, tol=_GAMMA_IMAG_TOL):
    """Return the real part of a Gamma tensor after checking its residual."""
    array = np.asarray(array)
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{name} must contain numeric values")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    if not np.iscomplexobj(array):
        return array

    scale = max(float(np.max(np.abs(array.real), initial=0.0)), 1.0)
    imag_max = float(np.max(np.abs(array.imag), initial=0.0))
    if imag_max > tol * scale:
        raise ValueError(
            f"{name} is not real at Gamma: maximum imaginary residual "
            f"{imag_max:.3e} exceeds {tol * scale:.3e}"
        )
    return array.real


def _make_gamma_gdf(cell, auxbasis=None, *, mesh=None):
    """Create a Gamma-only GDF object and build its auxiliary cell."""
    _validate_gamma_cell(cell)
    initial_mesh = _validate_mesh(mesh, cell.mesh)

    density_fitter = pbcdf.GDF(cell, kpts=_GAMMA.reshape(1, 3))
    density_fitter.auxbasis = auxbasis
    density_fitter.mesh = initial_mesh
    density_fitter.build(with_j3c=False)
    density_fitter.mesh = _select_ssc_mesh(
        cell, density_fitter.auxcell, mesh=mesh
    )

    kpts = np.asarray(density_fitter.kpts).reshape(-1, 3)
    if kpts.shape != (1, 3) or not np.allclose(
        kpts[0], _GAMMA, atol=1e-12, rtol=0.0
    ):
        raise RuntimeError("GDF builder did not preserve the Gamma point")
    return density_fitter


def _make_metric_capture_builder(density_fitter):
    """Mirror ``GDF._make_j3c`` while capturing its metric decomposition."""
    cell = density_fitter.cell
    auxcell = density_fitter.auxcell
    kpts = np.asarray(density_fitter.kpts).reshape(-1, 3)
    if density_fitter._prefer_ccdf or cell.omega > 0:
        builder = _MetricCaptureCCGDFBuilder(cell, auxcell, kpts)
        builder.eta = density_fitter.eta
    else:
        builder = _MetricCaptureRSGDFBuilder(cell, auxcell, kpts)
    builder.mesh = density_fitter.mesh
    builder.linear_dep_threshold = density_fitter.linear_dep_threshold
    return builder


def _metric_transform_from_decomposition(decomposition, naux):
    """Convert a captured PySCF GDF decomposition to ``T`` with ``T^H T=V+``."""
    factor, negative_factor, tag = decomposition
    if negative_factor is not None:
        raise NotImplementedError(
            "negative periodic DF metric sectors are not supported for SSC"
        )
    if tag == "CD":
        if factor.shape != (naux, naux):
            raise RuntimeError(
                "GDF Cholesky metric factor has an unexpected shape"
            )
        transform = scipy.linalg.solve_triangular(
            factor,
            np.eye(naux, dtype=factor.dtype),
            lower=True,
            check_finite=False,
        )
    elif tag == "ED":
        if factor.ndim != 2 or factor.shape[1] != naux:
            raise RuntimeError(
                "GDF eigen metric factor has an unexpected shape"
            )
        # PySCF's eigen path already returns diag(w^-1/2) @ eigenvectors^H.
        transform = factor
    else:
        raise RuntimeError(f"unknown GDF metric decomposition tag {tag!r}")
    return np.asarray(transform)


def _get_metric_transform(density_fitter):
    """Build Gamma CDERIs and return their exact auxiliary metric transform."""
    cell = density_fitter.cell
    _validate_gamma_cell(cell)
    auxcell = density_fitter.auxcell
    if auxcell is None:
        raise ValueError("density_fitter must be built before use")
    kpts = np.asarray(density_fitter.kpts).reshape(-1, 3)
    if kpts.shape != (1, 3) or not np.allclose(
        kpts[0], _GAMMA, atol=1e-12, rtol=0.0
    ):
        raise NotImplementedError(
            "periodic SSC integrals currently support one Gamma point only"
        )

    cderi_target = density_fitter._cderi_to_save
    if isinstance(cderi_target, str):
        cderi_file = cderi_target
    else:
        cderi_file = cderi_target.name

    builder = _make_metric_capture_builder(density_fitter)
    builder.make_j3c(
        cderi_file,
        aosym="s2",
        j_only=True,
        dataname=density_fitter._dataname,
    )
    if len(builder.ssc_metric_decompositions) != 1:
        raise RuntimeError(
            "Gamma GDF build did not produce exactly one metric decomposition"
        )

    density_fitter._cderi = cderi_file
    density_fitter._j_only = True
    transform = _metric_transform_from_decomposition(
        builder.ssc_metric_decompositions[0], auxcell.nao_nr()
    )
    logger.debug(
        cell,
        "Gamma GDF metric transform: %d auxiliary functions, %d retained",
        auxcell.nao_nr(),
        transform.shape[0],
    )
    return transform


def _collect_gamma_three_center(density_fitter):
    """Collect positive-metric packed Gamma CDERIs from a completed GDF build."""
    cell = density_fitter.cell
    nao = cell.nao_nr()
    nao_pair = nao * (nao + 1) // 2
    blocks = []
    max_memory = max(
        float(density_fitter.max_memory) - lib.current_memory()[0], 1.0
    )
    for real, imag, sign in density_fitter.sr_loop(
        kpti_kptj=np.zeros((2, 3)),
        max_memory=max_memory,
        compact=True,
    ):
        if sign != 1:
            raise NotImplementedError(
                "negative periodic DF metric sectors are not supported for SSC"
            )
        block = np.asarray(real) + 1j * np.asarray(imag)
        if block.ndim != 2 or block.shape[1] != nao_pair:
            raise RuntimeError(
                "Gamma GDF three-center block has an unexpected shape"
            )
        blocks.append(
            np.asarray(
                _real_if_gamma(block, "Gamma three-center factors"),
                dtype=np.float64,
            )
        )
    if not blocks:
        raise RuntimeError("Gamma GDF build returned no three-center factors")
    return np.concatenate(blocks, axis=0)


def _reciprocal_block_size(density_fitter, naux, ngrids):
    """Choose a conservative block size for auxiliary Fourier transforms."""
    current_mb = lib.current_memory()[0]
    available_mb = max(float(density_fitter.max_memory) - current_mb, 1.0)
    # Hold the complex auxiliary transform and one weighted copy.  The six
    # naux-by-naux accumulators are allocated independently of this block.
    bytes_per_grid = max(2 * np.dtype(np.complex128).itemsize * naux, 1)
    block_size = int(available_mb * 0.4e6 / bytes_per_grid)
    return max(1, min(block_size, ngrids, 200000))


def _build_aux_ssc_metric(density_fitter, *, mesh=None):
    r"""Build the 3D Gamma auxiliary SSC metric in reciprocal space.

    If ``P(G) = integral exp(-i G.r) P(r) dr``, integration by parts gives

    ``FT[d_a P](G) = i G_a P(G)``.

    The bra derivative is conjugated, so the two factors multiply to
    ``(-i G_a) * (i G_b) = +G_a G_b``.  Consequently no extra minus sign is
    applied below.  ``ft_ao.ft_ao(...).T`` is indexed as ``[P,G]`` and
    ``weighted_coulG`` already includes both the Coulomb kernel and reciprocal
    quadrature weights.  The contraction is therefore

    ``W[a,b,P,Q] = sum_G P(G)^* weighted_coulG(G) G_a G_b Q(G)``.

    The ``G = 0`` contribution is explicitly set to zero, defining the 3D
    tin-foil convention used by this prototype.

    Args:
        density_fitter : :class:`pyscf.pbc.df.GDF`
            A built, Gamma-only Gaussian density-fitting object.
        mesh : array_like of int, optional
            Reciprocal mesh.  Defaults to the derivative-safe mesh stored in
            ``density_fitter.mesh``.

    Returns:
        ndarray
            Real symmetric Cartesian auxiliary metric with shape
            ``(3, 3, naux, naux)``.
    """
    cell = density_fitter.cell
    _validate_gamma_cell(cell)
    kpts = np.asarray(density_fitter.kpts).reshape(-1, 3)
    if kpts.shape != (1, 3) or not np.allclose(
        kpts[0], _GAMMA, atol=1e-12, rtol=0.0
    ):
        raise NotImplementedError(
            "periodic SSC integrals currently support one Gamma point only"
        )
    auxcell = density_fitter.auxcell
    if auxcell is None:
        raise ValueError("density_fitter must be built before use")

    default_mesh = density_fitter.mesh
    if default_mesh is None:
        default_mesh = cell.mesh
    mesh = _validate_mesh(mesh, default_mesh)

    Gv, Gvbase, _ = cell.get_Gv_weights(mesh)
    ngrids = Gv.shape[0]
    naux = auxcell.nao_nr()
    weighted_coulG = np.asarray(
        aft.weighted_coulG(density_fitter, _GAMMA, False, mesh),
        dtype=np.float64,
    )
    if weighted_coulG.shape != (ngrids,):
        raise RuntimeError("weighted Coulomb kernel has an unexpected shape")

    # Do not rely on the current PySCF value of coulG[0].  G=0 is a boundary
    # condition for the dipolar lattice sum and is fixed explicitly here.
    zero_G = np.linalg.norm(Gv, axis=1) < 1e-14
    weighted_coulG = weighted_coulG.copy()
    weighted_coulG[zero_G] = 0.0

    metric = np.zeros((3, 3, naux, naux), dtype=np.complex128)
    reciprocal_vectors = cell.reciprocal_vectors()
    gxyz = lib.cartesian_prod(
        [np.arange(len(grid_axis), dtype=np.int32) for grid_axis in Gvbase]
    )
    block_size = _reciprocal_block_size(
        density_fitter, naux, ngrids
    )

    for p0, p1 in lib.prange(0, ngrids, block_size):
        auxG = ft_ao.ft_ao(
            auxcell,
            Gv[p0:p1],
            None,
            reciprocal_vectors,
            gxyz[p0:p1],
            Gvbase,
        ).T
        if auxG.shape != (naux, p1 - p0):
            raise RuntimeError(
                "auxiliary Fourier transform has an unexpected shape"
            )

        coulG_block = weighted_coulG[p0:p1]
        Gv_block = Gv[p0:p1]
        for axis_a in range(3):
            for axis_b in range(axis_a, 3):
                weights = (
                    coulG_block
                    * Gv_block[:, axis_a]
                    * Gv_block[:, axis_b]
                )
                metric[axis_a, axis_b] += lib.dot(
                    auxG.conj() * weights, auxG.T
                )

    # Each Cartesian component is Hermitian in the auxiliary indices.  The
    # scalar G_a G_b factor also makes the Cartesian tensor symmetric.  Record
    # the raw residual before removing reciprocal-grid roundoff.
    hermitian_residual = 0.0
    for axis_a in range(3):
        for axis_b in range(axis_a, 3):
            block = metric[axis_a, axis_b]
            hermitian_residual = max(
                hermitian_residual,
                float(np.max(np.abs(block - block.conj().T), initial=0.0)),
            )
            block = 0.5 * (block + block.conj().T)
            metric[axis_a, axis_b] = block
            metric[axis_b, axis_a] = block
    logger.debug(
        cell,
        "Gamma auxiliary SSC metric: mesh %s, block size %d, maximum "
        "Hermitian residual %.3e, maximum imaginary residual %.3e",
        mesh,
        block_size,
        hermitian_residual,
        float(np.max(np.abs(metric.imag), initial=0.0)),
    )
    return np.asarray(
        _real_if_gamma(metric, "reciprocal auxiliary SSC metric"),
        dtype=np.float64,
    )


def compute_ssc_integrals_ri(
    cell, auxbasis=None, *, mesh=None, linear_dep_threshold=None
):
    r"""Compute 3D Gamma-point RI SSC factors in the AO basis.

    The periodic RI approximation is represented as

    .. math::

        d^{ab}_{\mu\nu\rho\sigma}
        \simeq B^*_{P,\mu\nu} M^{ab}_{PQ} B_{Q,\rho\sigma},

    where ``B = T g``, ``M = T W T^H``, and ``T^H T = V^+``.  ``g`` is the
    periodic three-center Coulomb tensor, ``V`` is the auxiliary Coulomb
    metric, and ``W`` is the auxiliary SSC derivative metric.

    The exact metric decomposition used by PySCF to build ``B`` is captured
    and reused to transform ``W``.  This keeps both returned factors in the
    same retained auxiliary coordinates, including when an eigenvalue
    decomposition removes linear dependencies.

    Args:
        cell : :class:`pyscf.pbc.gto.Cell`
            Built three-dimensional periodic cell.
        auxbasis : optional
            Auxiliary basis accepted by :class:`pyscf.pbc.df.GDF`.
        mesh : array_like of int, optional
            Three-component reciprocal mesh used by both GDF and the SSC
            derivative metric.  If omitted, the componentwise maximum of
            ``cell.mesh`` and a mesh derived from the auxiliary-basis Coulomb
            cutoff at ``cell.precision`` is used.
        linear_dep_threshold : float, optional
            Positive eigenvalue threshold for the auxiliary Coulomb metric.
            Defaults to the value configured by :class:`pyscf.pbc.df.GDF`.

    Returns:
        two_center : ndarray
            Metric-transformed Cartesian SSC metric with shape
            ``(3, 3, nfit, nfit)``.
        three_center : ndarray
            Metric-orthogonalized Coulomb factors with packed lower-triangular
            AO pairs and shape ``(nfit, nao * (nao + 1) // 2)``.

    Notes:
        Only the 3D tin-foil Gamma convention is supported.  The Cartesian
        trace is retained and no fine-structure, electron-g-factor, spin, or
        electron-pair-counting prefactor is included.
    """
    if linear_dep_threshold is not None:
        if (
            isinstance(linear_dep_threshold, (bool, np.bool_))
            or not isinstance(linear_dep_threshold, Real)
        ):
            raise TypeError("linear_dep_threshold must be a positive real scalar")
        linear_dep_threshold = float(linear_dep_threshold)
        if not np.isfinite(linear_dep_threshold) or linear_dep_threshold <= 0:
            raise ValueError(
                "linear_dep_threshold must be a positive finite scalar"
            )

    density_fitter = _make_gamma_gdf(cell, auxbasis, mesh=mesh)
    if linear_dep_threshold is not None:
        density_fitter.linear_dep_threshold = linear_dep_threshold

    transform = _get_metric_transform(density_fitter)
    transform = np.asarray(
        _real_if_gamma(transform, "Gamma auxiliary metric transform"),
        dtype=np.float64,
    )
    three_center = _collect_gamma_three_center(density_fitter)
    if transform.shape[0] != three_center.shape[0]:
        raise RuntimeError(
            "GDF metric transform and three-center factors retain different "
            "auxiliary dimensions"
        )

    ssc_metric = _build_aux_ssc_metric(density_fitter, mesh=mesh)
    two_center = np.einsum(
        "Pu,abuv,Qv->abPQ",
        transform,
        ssc_metric,
        transform.conj(),
        optimize=True,
    )
    # Both symmetries hold analytically.  Apply them after transformation to
    # remove only floating-point contraction noise.
    cartesian_residual = float(
        np.max(
            np.abs(two_center - two_center.swapaxes(0, 1)), initial=0.0
        )
    )
    auxiliary_residual = float(
        np.max(
            np.abs(
                two_center
                - two_center.transpose(0, 1, 3, 2).conj()
            ),
            initial=0.0,
        )
    )
    two_center = 0.5 * (two_center + two_center.swapaxes(0, 1))
    two_center = 0.5 * (
        two_center + two_center.transpose(0, 1, 3, 2).conj()
    )
    logger.debug(
        cell,
        "Transformed Gamma SSC metric: maximum Cartesian residual %.3e, "
        "maximum auxiliary residual %.3e",
        cartesian_residual,
        auxiliary_residual,
    )
    two_center = np.asarray(
        _real_if_gamma(two_center, "transformed Gamma SSC metric"),
        dtype=np.float64,
    )
    return two_center, three_center


def _build_aux_ssc_metric_pbc_intor(auxcell):
    r"""Build the analytical lattice-sum metric used as a prototype check.

    ``int2c2e_ip1ip2`` differentiates the first and second auxiliary
    functions, respectively.  libcint returns the Cartesian components with
    the derivative on the first function as the slow component index, so its
    nine components reshape directly to ``(a, b, P, Q)``.

    Unlike :func:`_build_aux_ssc_metric`, this result inherits the real-space
    lattice summation convention and cutoff from ``auxcell.pbc_intor``.  Only
    its symmetric-traceless part is used to validate the reciprocal result.
    """
    _validate_gamma_cell(auxcell)
    naux = auxcell.nao_nr()
    metric = np.asarray(
        auxcell.pbc_intor(
            "int2c2e_ip1ip2", comp=9, hermi=0, kpt=_GAMMA
        )
    )
    expected_shape = (9, naux, naux)
    if metric.shape != expected_shape:
        raise RuntimeError(
            "int2c2e_ip1ip2 returned shape "
            f"{metric.shape}; expected {expected_shape}"
        )
    metric = metric.reshape(3, 3, naux, naux)
    return np.asarray(
        _real_if_gamma(metric, "analytical auxiliary SSC metric"),
        dtype=np.float64,
    )
