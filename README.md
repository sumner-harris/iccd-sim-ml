# ICCD-sim-ML

Physics-first conversion of axisymmetric nanosecond laser-ablation simulations
into idealized side-view image sequences, plus leakage-aware video models for
material-property prediction.

The maintained package replaces the exploratory workflow retained in
`legacy/` and `notebooks/legacy/`. The public notebook snapshots have their
outputs cleared and machine-specific paths replaced with placeholders. The
package keeps image formation out of
`Dataset.__getitem__`: radiative transfer is run once, versioned products are
written to `data/processed/`, and model training reads those deterministic
products.

## What the simulator returns

For each axisymmetric `(r, z)` plasma state, the current simulator:

1. resamples the adaptive mesh using `T`, `n0`, `ne`, `n1`, and `n2`;
2. builds parallel side-view rays with `r = sqrt(x^2 + y^2)` without creating
   a redundant angular volume;
3. recalculates wavelength-dependent photoionization, electron-neutral
   inverse-bremsstrahlung, and electron-ion inverse-bremsstrahlung opacity;
4. solves LTE transfer cell-by-cell as
   `I_out = I_in exp(-kappa ds) + B_lambda(T) [1 - exp(-kappa ds)]`;
5. integrates photon spectral radiance from 300 to 800 nm with response 1.

Because spatial optics and sensor effects are intentionally disabled, the
primary output is **band-integrated photon radiance** in
`photons s^-1 m^-2 sr^-1`, not camera counts. Display images may be normalized
for visualization, but the saved numerical array is not.

## Install

Python 3.10 or newer is required. With `uv`:

```powershell
uv --native-tls sync --extra io --extra viz --extra dev
```

The ML modules additionally require:

```powershell
uv --native-tls sync --extra io --extra viz --extra ml --extra dev
```

## Commands

Inspect an HDF5 container without loading every AMR frame:

```powershell
uv run iccd-sim inspect-h5 "C:\path\to\Cu.h5"
```

Simulate the supplied standalone timestep:

```powershell
uv run iccd-sim timestep `
  "C:\path\to\res_3005.17_ns.dat" `
  --config configs\cu_continuum_flat_response.json `
  --atomic-reference data\reference `
  --element Cu `
  --output outputs\cu_3005ns.npz `
  --preview outputs\cu_3005ns.png `
  --validate-248
```

Simulate one complete HDF5 group. If `--simulation` is omitted, a documented
representative-condition selector is used:

```powershell
uv run iccd-sim sequence "C:\path\to\Cu_6.h5" `
  --simulation Cu_3_68 `
  --config configs\cu_continuum_flat_response.json `
  --atomic-reference data\reference `
  --output outputs\cu_sequence.npz `
  --reuse-existing
```

Use `--stride` or `--max-frames` for a faster smoke test. Frame zero is kept;
times are parsed from dataset names and need not be uniformly spaced. Named
groups are indexed directly, so the other simulations in a large container
are not scanned. `--reuse-existing` validates a fingerprint of the source
group, selected frame keys, physics configuration, package version, and atomic
data before treating the output NPZ as a cache hit.

Audit atomic-data readiness for every simulated or planned element with:

```powershell
uv run iccd-sim atomic-status --atomic-reference data\reference
```

The target catalog contains Al, As, B, Be, Bi, C, Ca, Co, Cs, Cu, Fe, Ge,
Ho, In, Mg, Mo, Na, Nb, Ni, P, Pr, Pt, Rb, Sb, Sc, Se, Si, Sm, Sr, Ta, Te,
Ti, Tm, V, W, Zn, and Zr. All 37 elements have locally cached NIST ASD I--IV
bound-level tables and ionization thresholds. Every element uses the declared fixed
`Q = 1e-50 m^5` electron-neutral model. Strict mode is the production default
and refuses incomplete element-specific photoionization inputs.
`--atomic-mode approximate` is available only for explicitly labeled
sensitivity and software tests. See [atomic-data.md](docs/atomic-data.md).

Regenerate or update the local atomic database from the live NIST site with:

```powershell
uv run python scripts\build_nist_level_database.py
```

The builder records exact query URLs, retrieval timestamps, response and file
checksums, NIST version, ionization cutoffs, and excluded-level counts. It keeps
only absolute levels below the same-stage ionization limit for partition-function
and photoionization calculations.

The maintained HDF5 reader targets the current 34-column solver export only;
older HDF5 layouts are rejected explicitly. See
[hdf5-schema.md](docs/hdf5-schema.md) for the field mapping and streaming
policy.

Run a coupled numerical-resolution sweep for a single HDF5 frame with:

```powershell
uv run python scripts\single_frame_resolution_sweep.py "C:\path\to\Cu_6.h5" `
  --simulation Cu_3_68 `
  --time-ns 3006 `
  --atomic-reference data\reference `
  --output-dir outputs\Cu_3006ns_resolution_sweep
```

The script saves each physical-radiance array, a shared-scale comparison plot,
and JSON/CSV convergence metrics. Every profile uses the same declared field
of view so differences measure numerical resolution rather than cropping.

## Recommended ML simulation profile

For offline generation of ML training videos, the recommended balanced
starting point is the measured `r3` profile:

- 48 uniformly spaced wavelengths from 300 to 800 nm;
- a 96 x 96 output image;
- 128 line-of-sight integration cells per image ray;
- 160 points in the temperature-opacity lookup table.

Use [`configs/cu_continuum_ml_balanced.json`](configs/cu_continuum_ml_balanced.json)
for the early-time Cu field of view tested here. On the 3006 ns benchmark it
required about 37 seconds per frame on CPU, while its integrated radiance was
4.7% above and its interpolated image L2 difference was 12% relative to the
finest tested `r4` result. This makes `r3` a pragmatic throughput/quality
choice for model development, not a claim of full numerical convergence.

The bundled field of view is fixed at `x = +/-15 mm` and `z = 0--32 mm`, which
contains the tested Cu plume through approximately 5 microseconds. Change and
version those bounds for a different time window, material, or camera view;
do not infer them from frame zero. Keep one physics configuration, field of
view, and physical time grid across all samples in a training dataset.

A maximum-power/maximum-spot late-time survey found that raw hot-cell support
is not a valid camera-sizing metric. Some source solutions (notably V, Sm,
Nb, and condition-dependent C/Fe cases) become slab-like or fill the solver
window rather than retaining a localized plume. The survey now caps automatic
diagnostic views at 200 mm, reports edge censoring, and excludes slab-like or
field-filling radiance when calculating typical-plume percentiles.

For plume-shaped cases, the measured 99.9%-radiance medians were approximately
`r=44 mm, z=105 mm` at 8 microseconds, `r=48 mm, z=124 mm` at 10
microseconds, and `r=61 mm, z=157 mm` at 20 microseconds. Versioned
median-plus-margin configurations are provided as
[`continuum_ml_typical_8us.json`](configs/continuum_ml_typical_8us.json),
[`continuum_ml_typical_10us.json`](configs/continuum_ml_typical_10us.json), and
[`continuum_ml_typical_20us.json`](configs/continuum_ml_typical_20us.json).
The 20-microsecond configuration does not imply complete source coverage:
eight maximum-condition simulations end before 20 microseconds, and even 10
microseconds is unavailable for Fe and Tm. Eight microseconds is the longest
common target in the current files.

The selected production profile uses the all-element 8-microsecond window at
256 x 256. Its physical pixel pitches are 0.431 mm horizontally and 0.490 mm
axially. The longer-time configurations remain lower-resolution survey
alternatives rather than production defaults:

| target | field of view | x pitch | z pitch | source coverage |
|---|---:|---:|---:|---:|
| 8 microseconds | `x=+/-55 mm`, `z=0--125 mm` | 0.431 mm | 0.490 mm | 37/37 |
| 10 microseconds | `x=+/-60 mm`, `z=0--145 mm` | 1.26 mm | 1.53 mm | 35/37 |
| 20 microseconds | `x=+/-75 mm`, `z=0--185 mm` | 1.58 mm | 1.95 mm | 29/37 |

Use the 8-microsecond, 256 x 256 profile when one canonical all-element
production dataset is required. With the default 16-frame time grid, frames
are uniformly placed from 0 through 8 microseconds. Use the longer profiles
only for explicitly coverage-filtered experiments, and retain the morphology
flags rather than silently discarding anomalous source simulations.

The 128 line-of-sight cells are internal radiative-transfer quadrature points,
not a third model dimension. The production ML tensor is `(T, 256, 256)` and
is promoted to `(C, T, 256, 256)` by the dataset. The earlier 96 x 96 r3
profile remains useful for convergence studies and inexpensive pilots.

## Joint conditional model

The optional PyTorch package also includes one jointly trained conditional
VAE for three related tasks: video generation from laser conditions and
material properties, material-property regression from video, and known-class
element classification. The public components are `JointCVAEConfig`,
`JointConditionalVAE`, `JointPrecomputedVideoDataset`, `JointLossConfig`, and
`train_joint_one_epoch`.

The current laser vector is `[laser_power_wcm, rspot]`.
`laser_power_wcm` is the historical intensity-like HDF5 field, not pulse
energy in joules. The material vector is `[cp_metal, h_vapor, kappa_metal,
laser_reflectivity, mass_density_metal, t_boil, tcrit]`. Regression and
classification never receive the true property/class targets as inputs; those
properties condition only the generator and its latent distributions.

All videos must first be aligned to a fixed physical `times_s` grid and
transformed with train-only radiance statistics. Generation produces
standardized log-radiance, which must be inverse-transformed before physical
interpretation. See
[conditional-generation.md](docs/conditional-generation.md) for the
architecture, training/API pattern, uncertainty semantics, and evaluation
limits.

## End-to-end training and reports

[`scripts/run_training_pipeline.py`](scripts/run_training_pipeline.py) is the
combined experiment entry point. It delegates to modules in
`iccd_sim_ml.pipeline` for each stage rather than implementing caching,
training, and plotting inline. In one invocation it:

1. selects a deterministic element/simulation subset;
2. validates existing NPZ cache fingerprints and builds only missing products;
3. creates a leakage-safe within-element simulation split and train-only scalers;
4. trains regression, classification, the joint CVAE, or any requested subset;
5. saves checkpoints, metrics JSON, learning curves, regression parity plots,
   classification confusion matrices, and CVAE generation-error maps.

For the lightweight three-element smoke workflow:

```bash
python scripts/run_training_pipeline.py \
  --data-dir /path/to/plasma_sim_data \
  --elements Al Cu V \
  --simulations-per-element 3 \
  --cache-dir work/cache \
  --output-dir work/reports \
  --models all \
  --epochs 2 \
  --architecture smoke \
  --device auto
```

The multi-element smoke cache backend creates a deterministic projection of
plasma temperature and charge density. It is explicitly labeled
`pipeline_smoke_test_only` and must not be interpreted as ICCD radiance. Its
purpose is to exercise the complete software and GPU workflow quickly.
Scientific training should use cached products from the validated continuum
simulator and the per-element NIST reference catalog without changing the
downstream experiment and report modules. See
[`docs/training-workflow.md`](docs/training-workflow.md) for the module
boundaries and artifact layout.

For a physical continuum-radiance cache, select the `continuum` backend. This
strict example uses Cu, but `--elements` accepts any elements registered in
`data/reference/catalog.json`:

```bash
python scripts/run_training_pipeline.py \
  --data-dir /path/to/plasma_sim_data \
  --elements Cu \
  --simulations-per-element 3 \
  --cache-backend continuum \
  --atomic-reference data/reference \
  --atomic-mode strict \
  --frames 16 --start-ns 0 --stop-ns 8000 \
  --image-width 256 --image-height 256 \
  --radial-max-mm 55 --axial-max-mm 125 \
  --cache-dir work/continuum-cache \
  --output-dir work/reports \
  --models regression \
  --architecture standard \
  --device cuda
```

The continuum backend defaults to the production 0--8 microsecond,
256 x 256, `x=+/-55 mm`, `z=0--125 mm` profile. It retains the r3 spectral and
LOS settings: 48 wavelengths and 128 LOS points. It simulates source frames
bracketing a shared physical-time grid, linearly interpolates photon radiance
onto that grid, refuses temporal extrapolation, and fingerprints the atomic
inputs and fidelity in every product.

Before committing to a full cache, run one resumable production-profile pilot
per element:

```bash
python scripts/run_production_cache_pilot.py \
  --data-dir /path/to/plasma_sim_data \
  --atomic-reference data/reference \
  --cache-dir work/production-pilot-cache \
  --output-dir work/production-pilot-report
```

The pilot keeps zero/no-ablation samples and flags them instead of discarding
them. It reports positive-radiance quantiles, exact-zero pixel and frame
fractions, plasma-state checks, I-IV level counts, sparse NIST stages, and
per-element runtime/storage extrapolations. Progress is committed after each
element, so the command is safe to resume after interruption; valid NPZ files
are cache hits.

After reviewing the pilot, cache every simulation that covers the complete
canonical 0--8 microsecond grid with the resumable full-cache runner:

```bash
python scripts/run_full_production_cache.py \
  --data-dir /path/to/plasma_sim_data \
  --atomic-reference data/reference \
  --cache-dir /path/to/plasma_sim_data_cache \
  --workers 16
```

The runner indexes every simulation group in every registered element file,
records groups excluded for missing attributes or insufficient time coverage,
and interleaves elements across workers to reduce contention on any one HDF5
file. Each sample is independently fingerprinted and written atomically. A
restart revalidates existing products, while `progress.json`,
`source_inventory.json`, and the final `manifest.json` remain in the cache
root.

## Repository map

```text
configs/                 versioned simulator settings
data/reference/          versioned per-element NIST level/ionization catalog
data/processed/          generated videos/manifests (ignored by git)
docs/                    physics, HDF5, and validation notes
legacy/                  original exploratory Python source
notebooks/legacy/        sanitized exploratory notebook snapshots
src/iccd_sim_ml/
  atomic/                levels, cross sections, continuum opacity
  io/                    .dat and streaming HDF5 readers
  imaging/               AMR resampling, ray geometry, LTE transfer
  data/                  processed-video datasets, transforms, group splits
  models/                maintained video encoders and prediction heads
  training/              training and provenance-rich checkpoints
  pipeline/              cache, experiment orchestration, and report plots
tests/                   unit and analytic physics tests
```

## Scientific scope

This is a continuum LTE proof of concept. It fixes the positive,
dimensionally invalid attenuation exponent in the exploratory code and never
uses a 248-nm laser coefficient as a broadband camera coefficient. Known-bad
legacy `alpha_PI` exports are flagged, while future corrected exports remain
available as an independent 248-nm validation reference. It does **not yet
include bound-bound line emission**, non-LTE level populations,
self-consistent spectral line shapes, a finite collection cone, PSF,
intensifier gate/gain, quantum-efficiency variation, pixelization, saturation,
or noise. Copper plume spectra can be strongly line dominated, so continuum
images should not be presented as a complete synthetic ICCD forward model.

See [physics.md](docs/physics.md), [atomic-data.md](docs/atomic-data.md),
[hdf5-schema.md](docs/hdf5-schema.md), [data-and-splits.md](docs/data-and-splits.md), and
[conditional-generation.md](docs/conditional-generation.md) before
interpreting or training on generated images.
