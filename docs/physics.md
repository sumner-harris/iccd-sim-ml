# Continuum image model

## State and geometry

The hydrodynamic solver is axisymmetric. Its header calls the axial coordinate
`x` and radial coordinate `y`; this package renames them `z` and `r`.
Temperature and the neutral, electron, singly ionized, and doubly ionized
number densities are resampled onto a common `(r, z)` grid. A side-view ray at
image coordinate `(x, z)` is sampled along `y` with

```text
r(x, y) = sqrt(x^2 + y^2).
```

Points with `r` outside the simulated radius are vacuum. No explicit theta
dimension is needed for an axisymmetric source.

## Spectral opacity

At every wavelength, the continuum extinction coefficient is

```text
kappa_lambda = alpha_PI + alpha_IB,en + alpha_IB,ei     [m^-1].
```

All three terms are recalculated from `T`, `n0`, `ne`, `n1`, and `n2` over the
camera band. The solver's saved `alpha_*` fields describe its 248-nm heating
laser and are used only for validation. In particular, the supplied standalone
Cu export and the earlier HDF5 archive have a known missing-array-index bug in
their `alphaPICu` output; those particular columns and the shifted mesh-level
columns are flagged and ignored. This is not assumed for future exports:
corrected 248-nm values are retained as useful validation references.

Photoionization uses explicit element-specific ionization energies and energy
levels for LTE partition functions. Electron-neutral inverse bremsstrahlung
uses the declared project-wide constant `Q = 1e-50 m^5` for every element.
Electron-ion inverse
bremsstrahlung includes charge weighting `n1 + 4 n2`. Stable implementations
of the stimulated-emission factor use `-expm1(-h c / (lambda k T))`. The
legacy-notebook photoionization formula is a true absorption coefficient for
laser heating. Before it enters LTE image transfer it is multiplied by this
factor to obtain the net bound-free opacity required by detailed balance.
The uncorrected true coefficient is retained separately for comparisons with
the solver's saved 248-nm laser-absorption value.

Atomic inputs are resolved through `data/reference/catalog.json`. Both modes
use the fixed-Q electron-neutral assumption. Production strict mode refuses
photoionization when its element bundle is incomplete. Approximate mode sets
unavailable photoionization terms to zero. Its outputs carry an
`approximate_incomplete_continuum` label. See
[atomic-data.md](atomic-data.md).

## Emission and transfer

Kirchhoff's law in LTE makes the local source function the Planck function.
The package uses its photon form,

```text
B_lambda^ph(T) = 2 c / {lambda^4 [exp(h c / lambda k T) - 1]},
```

with units `photons s^-1 m^-2 sr^-1 m^-1`. Across one cell of length `ds`,

```text
tau = kappa_lambda ds
I_out = I_in exp(-tau) + B_lambda^ph(T) [1 - exp(-tau)].
```

The expression is valid in both optically thin and thick limits and is
evaluated with `expm1` for numerical stability. The background intensity is
currently zero.

The final idealized result is

```text
I_band(x,z) = integral I_lambda(x,z) R(lambda) d lambda,
```

where `R(lambda) = 1` by request. With no collection optics or sensor model,
this remains radiance, not detected photons or digital counts.

## Deliberate omissions

- bound-bound atomic lines and line broadening;
- non-LTE population kinetics;
- a finite aperture, magnification, depth of field, aberrations, and PSF;
- finite exposure/gate integration between hydrodynamic outputs;
- wavelength-dependent photocathode response;
- intensifier gain statistics, pixel sampling, read noise, dark current,
  saturation, and digitization.

These are staged extensions. Bound-bound emission is the highest-priority
physics addition because the local distilled NIST files are transition lists,
and copper visible emission is often line dominated.

## Model assumptions and numerical convergence

- The electron-ion Kramers approximation currently uses a free-free Gaunt
  factor of one.
- Finite atomic-level tables and any real-plasma continuum lowering limit the
  accuracy of LTE partition functions and photoionization.
- A Planck source for the bound-free term assumes that the supplied charge
  populations are consistent with Saha-Boltzmann equilibrium using the same
  atomic data. If they are not, bound-free emissivity should instead be
  calculated explicitly from the Milne relation/recombination model.
- The configured spatial grid, line-of-sight samples, and wavelength samples
  are quadrature choices, not camera pixels. Production datasets should be
  preceded by spatial, line-of-sight, spectral, and temperature-lookup
  convergence studies.
- For ML training, the current balanced baseline is 48 wavelengths over
  300--800 nm, a 96 x 96 image, 128 line-of-sight cells, and a 160-point
  temperature lookup. In the coupled 3006 ns Cu sweep this `r3` profile took
  about 37 seconds per CPU frame. Its integrated radiance differed by 4.7%
  and its interpolated image L2 norm by 12% from the finest tested `r4`
  profile (64 wavelengths, 128 x 128, 192 LOS). It is therefore a deliberate
  cost/quality compromise for dataset generation, not a converged reference.
- The 96 x 96 array is the saved model input. The 128 LOS cells exist only
  during ray integration and do not add a dimension to the saved video.
- Lookup queries outside the configured temperature range are clipped to the
  nearest edge. The standalone example has a few colder vacuum cells with
  zero contributing particle densities, so its 300 K floor does not change
  the result; future configurations must cover the active plasma range.
