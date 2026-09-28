# Modular cache, training, and reporting workflow

## Entry point

`scripts/run_training_pipeline.py` is deliberately thin. It parses an
experiment request, then calls maintained modules for the individual stages.
The same modules can be imported from a notebook or a cluster scheduler.

```text
HDF5 subset discovery
        |
        v
cache.ensure_proxy_cache or
cache.ensure_continuum_cache ----> manifest.json + immutable NPZ samples
        |
        v
make_known_material_split -------> split.json
        |
        v
fit_experiment_scalers ----------> scalers.json (training IDs only)
        |
        +--> run_regression_experiment
        +--> run_classification_experiment
        +--> run_joint_cvae_experiment
                         |
                         v
                 reports + checkpoints
```

## Module boundaries

- `pipeline.cache` owns deterministic source selection, cache fingerprints,
  cache validation, cache writes, and manifest construction. A stale,
  incomplete, non-finite, shape-mismatched, or differently configured product
  is a cache miss.
- `data` owns immutable NPZ loading, grouped splits, and train-only scalers.
- `models` owns the dedicated regressor, dedicated classifier, and joint CVAE.
- `training` owns task-specific epoch loops, metrics, and self-describing
  checkpoints.
- `pipeline.experiments` assembles a model-specific run from those components.
- `pipeline.reports` owns headless plots and JSON-safe metric output.

The combined script can train `regression`, `classification`, `joint_cvae`, or
`all`. The dedicated models do not pay the computational cost of the CVAE when
generation is unnecessary.

## Production manifest mode

The full-cache manifest is the training source for production runs. Manifest
mode consumes every record by default; `--elements` is only an explicit filter
for targeted experiments. It does not reopen the raw HDF5 files or run image
formation during training.

Prepare a deterministic, within-element 70/15/15 split and fit train-only
scalers without starting a model:

```bash
python scripts/run_training_pipeline.py \
  --manifest /mnt/shared_drive/plasma_sim_data_cache/manifest.json \
  --output-dir /mnt/shared_drive/plasma_sim_training/8us-known-material-seed42 \
  --split-strategy known-material \
  --split-ratios 0.70 0.15 0.15 \
  --seed 42 \
  --prepare-only
```

Remove `--prepare-only` and select `--models regression`, `classification`,
`joint_cvae`, or `all` to train. Use `--architecture standard` for production;
the default `smoke` architecture exists only for fast integration tests.

The first invocation atomically writes `split.json` and `scalers.json` in the
output directory. Later invocations load both. A changed manifest, seed,
ratio, or grouping policy fails instead of silently changing membership. Use
`--regenerate-split` or `--refit-scalers` only when intentionally starting a
new data-preparation version. `split.json` protects complete simulation IDs,
and scaler parameters are computed from `split.train` only.

Available split strategies are:

- `known-material`: stratifies within every element, appropriate for the
  primary regression/classification/CVAE experiment;
- `element-held-out`: keeps entire elements in separate partitions for
  unseen-material regression/generation evaluation;
- `legacy`: reproduces the original global 70/30 same-material regime and has
  no test partition.

## Smoke-test cache versus scientific cache

The bundled multi-element cache backend is a software smoke-test fixture. It
projects a non-negative function of temperature and charge density along the
line of sight and labels the result
`quantity=dimensionless_plasma_state_proxy` and
`scientific_use=pipeline_smoke_test_only`. This allows real HDF5 parsing,
multi-element labels, cache misses/hits, GPU training, and reporting to be
tested before every element has the atomic inputs required by the continuum
forward model.

Do not combine proxy images with continuum-radiance images in a scientific
dataset. `--cache-backend continuum` constructs versioned radiance NPZs
directly. It uses a common physical-time grid, simulates the bracketing source
frames, interpolates in photon radiance, and refuses extrapolation. Both modes
use the documented fixed-Q electron-neutral model. Strict atomic mode is
required for production photoionization; approximate mode is labeled and is
intended only for integration and sensitivity studies. The downstream
datasets, splits, models, training loops, and reports use the same NPZ array
contract for both backends.

## Smoke-test split

With three simulations per element, the runner uses a known-material split of
two simulations for training and one for validation within every element.
Complete simulations stay intact. This is appropriate for exercising parity
and confusion-matrix code, but nine total samples are not sufficient for a
scientific performance claim and no final test set is created.

## Artifact layout

```text
output-dir/
  pipeline_summary.json
  split.json
  scalers.json
  regression/
    checkpoint.pt
    metrics.json
    learning_curves.png
    parity.png
  classification/
    checkpoint.pt
    metrics.json
    learning_curves.png
    confusion_matrix.png
  joint_cvae/
    checkpoint.pt
    metrics.json
    learning_curves.png
    parity.png
    confusion_matrix.png
    generation_error_maps.png
```

Metrics from a smoke run verify execution only. Use an untouched test split,
more simulations, production radiance, convergence-qualified image settings,
and repeated seeds before interpreting model accuracy.

## Production-cache pilot

Run `scripts/run_production_cache_pilot.py` before a full continuum cache. It
selects one deterministic interior simulation per requested element and uses
the production profile: 0--8 microseconds, 48 wavelengths, a 128 x 128 image,
`x=+/-55 mm`, `z=0--125 mm`, 128 line-of-sight cells, and a 160-point
temperature lookup. Its output includes a portable
manifest, one immutable NPZ per element, a per-element progress ledger, CSV
and JSON quality reports, and headless diagnostic plots.

An exactly zero image is retained. The report separately flags an all-zero
video, source windows with no charged plasma, and source windows whose maximum
temperature remains below the recorded boiling point. Sparse atomic coverage
is also a flag, not a filter. The default threshold is fewer than 10 canonical
levels in any required photoionization charge state; change it only by an
explicit command-line option and record that choice.

The progress ledger is written atomically after every completed element. A
restarted command validates existing cache fingerprints and reuses valid
products, making it suitable for a detached cluster or remote-host job.
