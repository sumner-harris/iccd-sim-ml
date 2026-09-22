# Modular cache, training, and reporting workflow

## Entry point

`scripts/run_training_pipeline.py` is deliberately thin. It parses an
experiment request, then calls maintained modules for the individual stages.
The same modules can be imported from a notebook or a cluster scheduler.

```text
HDF5 subset discovery
        |
        v
cache.ensure_proxy_cache --------> manifest.json + immutable NPZ samples
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
dataset. Production manifests should point to the versioned `iccd-sim`
continuum NPZs. The downstream datasets, splits, models, training loops, and
reports require the same NPZ array contract and therefore do not need to be
rewritten when a production cache backend is selected.

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
