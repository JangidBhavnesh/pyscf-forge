# Gamma-point periodic SSC integrals in an AO/RI representation

## Goal

Implement the periodic spin--spin coupling (SSC) integral builder in
`pyscf/siso/pbcsscint.py` for a three-dimensional `pyscf.pbc.gto.Cell` at the
Gamma point.  The first implementation will construct the SSC operator in a
Gaussian auxiliary-basis resolution-of-the-identity (RI) representation while
leaving the orbital indices in the AO basis.

The public result should mirror the compact molecular contract in
`pyscf.siso.sscint.compute_ssc_integrals_ri`: a two-center SSC tensor and
metric-orthogonalized, packed three-center AO-pair factors.  It must not form a
`(3, 3, nao, nao, nao, nao)` array during normal execution.

This roadmap deliberately separates integral construction from the physical
SSC prefactor and from the Cartesian-to-spherical rank-2 transformation.  The
new builder returns the raw symmetric Cartesian tensor, including its
isotropic trace, with no factor of `alpha**2 * g_e**2 / 4`.

## Scope of the first implementation

Supported:

- three-dimensional periodic cells (`cell.dimension == 3`);
- one Gamma point only, `k = (0, 0, 0)`;
- real Gaussian AOs and real Gamma-point orbital products;
- Gaussian density fitting with a user-supplied or PySCF-selected auxiliary
  basis;
- packed lower-triangular AO pairs;
- linearly dependent auxiliary spaces through a thresholded metric
  eigendecomposition;
- in-core returned factors, with reciprocal-grid work performed in blocks.

Deferred:

- non-Gamma k-points and k-point momentum conservation;
- one- and two-dimensional Coulomb kernels;
- an exact periodic four-center SSC builder;
- direct active-MO transformation and SISO driver integration;
- out-of-core storage of the returned factors;
- magnetic sample-shape/demagnetization corrections at `G = 0`.
- primitive-cell translation or k-point covariance; the supplied supercell is
  treated directly as one Gamma-point cell.

The implementation should reject deferred cases with a precise
`NotImplementedError`; it should not silently evaluate them using a 3D Gamma
formula.

## Mathematical contract

Let `P,Q` denote auxiliary Gaussian functions and let `mu,nu,rho,sigma`
denote Gamma-point AOs.  Define

```text
g[P,mn]       = (P | mu nu)_Gamma
V[P,Q]        = (P | Q)_Gamma
W[a,b,P,Q]    = (d_a P | d_b Q)_Gamma
```

where the inner product uses the same periodic Coulomb kernel and boundary
condition throughout.  The RI approximation is

```text
D[a,b,mn,rs] = g[:,mn]^H V^+ W[a,b] V^+ g[:,rs].
```

For a retained metric transform `T` satisfying `T^H T = V^+`, define

```text
B[P,mn]       = (T g)[P,mn]
M[a,b,P,Q]    = (T W[a,b] T^H)[P,Q].
```

The public compact result is `(M, B)`, with shapes

```text
M: (3, 3, nfit, nfit)
B: (nfit, nao * (nao + 1) // 2)
```

and a tiny-system AO tensor may be reconstructed for testing as

```python
B_full = lib.unpack_tril(B)
D = np.einsum(
    "Pij,abPQ,Qkl->abijkl", B_full.conj(), M, B_full,
    optimize=True,
)
```

At Gamma, imaginary roundoff should be discarded only after it has been
checked against a documented tolerance.

### Periodic derivative metric and `G = 0`

For the 3D tin-foil convention, build the auxiliary SSC metric from reciprocal
vectors as

```text
W[a,b,P,Q] = sum_(G != 0) w(G) v_coul(G) G[a] G[b]
                            P(G)^* Q(G),
```

where `pyscf.pbc.df.aft.weighted_coulG` supplies `w(G) * v_coul(G)` and
`pyscf.pbc.df.ft_ao.ft_ao` supplies auxiliary-function Fourier transforms.
The `G = 0` contribution is explicitly zero in the first implementation.

This convention is preferable to making the production result depend
implicitly on the real-space cutoff used by
`auxcell.pbc_intor('int2c2e_ip1ip2')`.  That analytical integral is still an
important validation reference: after removal of the Cartesian trace, it
should converge to the reciprocal result.  Any non-isotropic disagreement is
a release blocker.

The trace is retained in `M`.  The existing
`sscint.cartesian_to_spherical` routine performs the symmetric-traceless
projection when the factors are eventually contracted into an AO or active-MO
tensor.

## Proposed API in `pbcsscint.py`

```python
def compute_ssc_integrals_ri(cell, auxbasis=None, *, mesh=None,
                             linear_dep_threshold=None):
    """Return `(two_center, three_center)` Gamma-point AO/RI factors."""
```

Contract details:

- `cell` must be a built 3D `pyscf.pbc.gto.Cell`.
- `auxbasis` follows `pyscf.pbc.df.GDF.auxbasis` conventions.
- `mesh=None` selects a derivative-safe mesh from the auxiliary cell and the
  requested cell precision; an explicit mesh is useful for convergence tests.
- `linear_dep_threshold=None` uses the matching `GDF` threshold.
- `two_center` is `M` above and has shape `(3, 3, nfit, nfit)`.
- `three_center` is `B` above and has shape `(nfit, nao_pair)`.
- both outputs are real `float64` arrays for the supported Gamma case.
- no physical prefactor and no traceless projection are applied.

Keep these implementation helpers private until a second use case establishes
a stable public contract:

```text
_validate_gamma_cell
_make_gamma_gdf
_get_metric_transform
_build_aux_ssc_metric
_collect_gamma_three_center
_real_if_gamma
```

Do not add the new builder to `pyscf/siso/__init__.py` until its numerical
contract and tests are stable.  During development it can be imported as
`from pyscf.siso import pbcsscint`.

## Implementation phases

### 1. Lock down conventions with a small numerical prototype

1. Build a small 3D all-electron cell with an explicit auxiliary basis.
2. Confirm that `auxcell.pbc_intor('int2c2e_ip1ip2', comp=9, kpt=gamma)`
   has the expected `(9, naux, naux)` layout and derivative sign.
3. Independently construct `W` on successively finer reciprocal meshes.
4. Compare the symmetric-traceless portions of both constructions and compare
   the large-vacuum limit with the molecular auxiliary derivative metric used
   by `sscint.compute_ssc_integrals_ri`.
5. Record the Fourier sign, normalization, and transpose conventions in code
   comments next to the contraction.

The prototype is complete only when off-diagonal Cartesian components and a
non-cubic geometry have been tested.  A one-function cubic cell is insufficient
because its traceless tensor can vanish by symmetry even with a sign or axis
error.

Status (2026-09-02): complete.  `pbcsscint.py` now contains the Gamma-cell,
mesh, and real-residual validation helpers together with independent
reciprocal and `pbc_intor` auxiliary SSC metric builders.  The prototype tests
use a triclinic cell with displaced centers, observe nonzero off-diagonal
Cartesian components, demonstrate reciprocal-mesh convergence, demonstrate
real-space-cutoff convergence of the analytical rank-2 metric, fix the
Fourier derivative sign to positive `G_a G_b`, and recover the molecular
auxiliary metric in the large-vacuum limit.  As expected, the unprojected
analytical and reciprocal tensors can differ by an isotropic boundary term;
comparisons therefore project to rank 2.

### 2. Build one consistent Gamma-point RI factorization

1. Create a Gamma-only `pyscf.pbc.df.GDF` object and build its auxiliary cell
   and packed three-center factors.
2. Obtain the raw periodic auxiliary Coulomb metric from the same GDF builder
   and use exactly the same Cholesky/eigen transform for both `B` and `M`.
3. Isolate use of PySCF's private range-separated GDF builder behind
   `_get_metric_transform`.  Add a focused test for this adapter so that an
   upstream PySCF API change fails locally and clearly.
4. On the Cholesky path, use `T = solve_triangular(L, I)` for `V = L L^H`.
5. On the eigen path, retain eigenvectors with
   `eigenvalue > linear_dep_threshold` and use
   `T = diag(eigenvalue**-0.5) @ eigenvectors.conj().T`.
6. Verify that `T.shape[0]`, the number of rows collected from `GDF.sr_loop`,
   and `nfit` are identical.  Never assume `nfit == naux`.
7. For 3D Gamma, require every `sr_loop` metric sign to be `+1`; reject a
   negative-metric block rather than dropping its sign.

The metric decomposition must be shared, not recomputed independently after
the GDF factors are built.  Degenerate retained eigenvectors may otherwise be
rotated differently, misaligning `M` and `B` even though their dimensions
match.  If the existing GDF API cannot expose the decomposition safely, the
fallback is a small local builder adapter that produces `T` and `B` in one
pass—not inference of `T` from `B`.

Status (2026-09-02): complete.  A local adapter subclasses both GDF builder
paths and captures the decomposition from the exact `decompose_j2c` call used
to write the packed three-center factors.  `compute_ssc_integrals_ri` now
returns `(M, B)` in those shared coordinates.  Tests cover the Cholesky path,
packed-factor reconstruction of `GDF.get_eri`, positive-metric enforcement,
and the eigen path with an exactly duplicated auxiliary basis.  The latter
reconstructs the same AO SSC tensor as the equivalent deduplicated basis,
directly guarding against a rotated or otherwise misaligned retained space.

### 3. Implement the reciprocal auxiliary SSC metric

1. Generate `Gv`, reciprocal-grid metadata, and weights for the selected mesh.
2. Evaluate auxiliary Fourier transforms in memory-bounded `G` blocks.
3. Accumulate only the six unique symmetric Cartesian components using
   `weighted_coulG * G[:,a] * G[:,b]`.
4. Set the `G = 0` kernel contribution to zero explicitly.
5. Restore all nine Cartesian components, enforce
   `W[a,b,P,Q] = W[b,a,Q,P].conj()`, and measure the correction before
   symmetrizing it.
6. Transform in the auxiliary indices: `M = T @ W @ T.conj().T`.
7. Check the Gamma imaginary residual and return real arrays.

Mesh selection needs a derivative-specific convergence check: the extra two
powers of `G` make a mesh adequate for the Coulomb metric potentially
inadequate for `W`.  Start from the auxiliary-cell kinetic-energy estimate and
increase it until a small representative cell meets a tolerance tied to
`cell.precision`.  Avoid encoding an unexplained fixed mesh multiplier.

Status (2026-09-02): complete.  The reciprocal builder evaluates auxiliary
Fourier transforms in memory-bounded blocks, accumulates only six Cartesian
components with the positive `G_a G_b` kernel, explicitly removes `G = 0`,
measures the raw auxiliary-Hermitian residual, restores all tensor symmetries,
checks the Gamma imaginary residual, and transforms `W` with the captured GDF
metric coordinates.  With no user mesh, it now takes the componentwise maximum
of `cell.mesh` and the mesh obtained from PySCF's auxiliary-basis four-center
Coulomb cutoff estimate at `cell.precision`, followed by cell-symmetry mesh
restoration.  A 100-times tighter precision estimate serves as the convergence
reference in the prototype test; an explicit mesh remains an exact override.

### 4. Complete validation and failure behavior

Add `pyscf/siso/test/test_pbcsscint.py` with the following groups.

Algebra and shape tests:

- expected factor shapes and `float64` dtype;
- packed AO-pair ordering agrees with `lib.pack_tril`/`lib.unpack_tril`;
- Cartesian symmetry and auxiliary Hermiticity;
- reconstructed AO pair-exchange symmetry;
- retained auxiliary dimension shrinks for a deliberately dependent basis;
- no NaN or infinite values.

Periodic numerical tests:

- reciprocal `W` is stable when the mesh is refined;
- its symmetric-traceless part agrees with the analytical
  `pbc_intor('int2c2e_ip1ip2')` result;
- the compact Coulomb factors reconstruct the Gamma GDF ERIs, independently
  checking metric-transform alignment;
- an anisotropic cell with off-diagonal tensor elements exercises axis and
  Fourier-phase conventions;
- the reconstructed AO tensor is stable under supercell-mesh refinement;
- the large-vacuum limit approaches the full molecular RI AO result using the
  same AO and auxiliary bases;
- PySCF-selected and named standard auxiliary bases produce finite factors;
- both the range-separated and compensated-charge GDF paths preserve their
  metric coordinates.

Input and scope tests:

- reject a `gto.Mole`, an unbuilt cell, and non-3D cells;
- reject malformed and nonpositive mesh specifications;
- reject any non-Gamma k-point argument if one is later exposed;
- reject unexpected negative-metric `sr_loop` blocks;
- reject a Gamma imaginary residual above tolerance instead of silently taking
  `.real`.

Expensive mesh- and vacuum-convergence tests should be marked slow if needed;
at least one small end-to-end periodic reconstruction must remain in the
normal test set.

Status (2026-09-02): complete for the supercell-only milestone.  Tests now
cover factor shapes and finiteness, packed AO ordering, all within-pair and
electron-pair AO symmetries, Cartesian and auxiliary symmetries, deterministic
linear-dependence removal, reciprocal and full-factor mesh convergence,
analytical `pbc_intor` agreement, GDF ERI reconstruction, both GDF builder
paths, explicit and PySCF-selected fitting bases, strict Gamma-real behavior,
negative-metric rejection, malformed inputs, and explicit 0D/1D/2D rejection.
The full reconstructed periodic rank-2 AO tensor agrees with its molecular RI
large-vacuum reference to 0.66% in the documented test supercell.  No
primitive-translation test or functionality is part of this scope.

### 5. Optimize without changing the contract

After correctness is established:

- choose the reciprocal block size from `cell.max_memory` and current memory;
- avoid retaining the full `auxG` grid;
- use BLAS matrix products for each of the six Cartesian accumulations;
- reuse `GDF` three-center storage rather than rebuilding it;
- log selected mesh, `naux`, `nfit`, discarded metric eigenvalues, maximum
  imaginary residual, and estimated memory at debug level;
- benchmark scaling in `naux`, `nao_pair`, and number of plane waves.

Do not optimize by constructing the full AO four-index tensor.  The compact
factor representation is the production result, not merely an intermediate.

### 6. Integrate with molecular SSC code

Once the AO/RI builder is stable:

1. Factor any boundary-condition-independent tensor utilities out of
   `sscint.py` only if this removes real duplication.
2. Add a periodic active-MO contraction that transforms packed `B` directly;
   do not reconstruct AO four-index integrals.
3. Reuse `sscint.cartesian_to_spherical` for the rank-2 conversion.
4. Teach the SISO SSC driver to dispatch on `Cell` versus `Mole` in a separate
   change with its own end-to-end test.
5. Export the periodic API from `pyscf.siso` only after naming and behavior are
   finalized.

## Acceptance criteria for the AO/RI milestone

- A documented 3D Gamma calculation returns `(M, B)` with the stated shapes.
- Normal execution scales as `O(nfit**2 + nfit * nao_pair)` in returned
  storage, never as `O(nao**4)`.
- Metric linear dependence is handled without non-finite values or a mismatch
  between `M` and `B` coordinates.
- All enforced symmetries agree within `1e-10` for the small test systems.
- Reciprocal-mesh refinement changes the symmetric-traceless reconstructed
  tensor by no more than a tolerance derived from `cell.precision`.
- The non-isotropic part agrees with the analytical periodic reference and the
  large-vacuum molecular limit to empirically justified, documented
  tolerances.
- Unsupported dimensions and k-points fail explicitly.
- Existing molecular SSC tests remain unchanged and pass.

## Main risks to keep visible

1. **Boundary condition:** the dipolar lattice sum is conditionally convergent;
   `G = 0` is a physical convention, not a numerical nuisance.
2. **Metric-coordinate alignment:** `M` and `B` must use the identical retained
   auxiliary subspace and basis orientation.
3. **Derivative mesh:** convergence of `W` is stricter than convergence of the
   ordinary Coulomb metric.
4. **Complex conventions:** Gamma results are real only after complete
   `+G/-G` cancellation; premature `.real` calls can hide phase errors.
5. **Trace/contact term:** the RI builder retains the Cartesian trace, whereas
   the spherical rank-2 conversion removes it.  Tests must compare like with
   like.
6. **Low-dimensional kernels:** 1D/2D Coulomb truncation and negative metric
   sectors require a separate derivation and must not leak into this milestone.
