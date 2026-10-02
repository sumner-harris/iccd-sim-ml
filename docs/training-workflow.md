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
        +--> run_material_set_experiment (Deep Sets or Set Transformer)
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

The combined script can also train `deep_set_regression` and
`set_transformer_regression`. These models consume unordered sets of
same-material simulations and predict one property vector per set. The
dedicated models do not pay the computational cost of the CVAE when generation
is unnecessary.

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
`joint_cvae`, or either set regressor to train. `all` retains the original
three single-video/joint tasks; request the set regressors explicitly because
they require at least `--set-size` simulations per element. Use
`--architecture standard` for production; the default `smoke` architecture
exists only for fast integration tests.

For unseen-material property prediction, use the element-held-out split and a
set regressor. This example initializes the shared 3D video and laser-condition
encoders from a trained single-video regressor, then fine-tunes them at one
tenth of the new set layers' learning rate:

```bash
python scripts/run_training_pipeline.py \
  --manifest /mnt/shared_drive/plasma_sim_data_cache/manifest.json \
  --output-dir /mnt/shared_drive/plasma_sim_training/material-set-seed42 \
  --split-strategy element-held-out \
  --split-file /mnt/shared_drive/plasma_sim_training/element-held-out-seed42/split.json \
  --scalers-file /mnt/shared_drive/plasma_sim_training/element-held-out-seed42/scalers.json \
  --models deep_set_regression set_transformer_regression \
  --set-size 8 \
  --set-train-bags-per-material 16 \
  --set-validation-bags-per-material 32 \
  --set-pretrained-regressor /path/to/regression/checkpoint.pt \
  --set-encoder-learning-rate-scale 0.1 \
  --architecture standard --batch-size 4 --epochs 100 \
  --early-stopping-patience 25 --device cuda
```

Every set contains only one element, has no positional order, and samples
without replacement. The primary validation metric is the macro R² over
repeated independent K-video sets. A second `material_ensemble` metric averages
those set predictions per held-out element; it is useful as a many-measurement
upper bound but is not the primary K-video result. The test elements remain
untouched until architecture selection is finished.

When initialized from a single-video checkpoint, both set models retain that
regressor's complete prediction head. They average its K predictions and add a
zero-initialized, permutation-invariant residual learned from the set. The
epoch-zero validation result is therefore exactly the simple prediction-average
baseline and is eligible for checkpoint restoration. Set training cannot erase
that baseline merely because a newly initialized aggregation head overfits.
Set `--set-encoder-learning-rate-scale 0` to freeze the complete pretrained
baseline, including its batch-normalization state, and train only the set
residual. This is the recommended first experiment when the number of distinct
training materials is small; nonzero fine-tuning is a later ablation.

For joint fine-tuning on same-material bags, use
`--set-freeze-pretrained-batchnorm` so correlated K-video batches do not replace
the mixed-material running statistics learned by the standalone regressor.
`--set-pretrained-warmup-epochs N` first trains only the new set layers for N
epochs and then unfreezes the pretrained weights at their scaled learning rate.
The optional `--set-baseline-pooling target_attention` replaces the fixed mean
of K single-video predictions with per-property convex attention weights. Its
weights are initialized uniformly, so epoch zero remains exactly the protected
mean-regressor baseline.

Omit `--set-pretrained-regressor` to train the 3D encoder, per-video regressor,
and set residual jointly from random initialization. The set objective is then
augmented by per-video property supervision, controlled by
`--set-individual-loss-weight` (default `0.2`). This preserves a useful
single-video representation while the aggregate branch learns which
condition-dependent evidence to combine across the set.

For a production run, `--early-stopping-patience N` monitors validation loss,
stops after `N` consecutive non-improving epochs, and restores the best model
and optimizer state before final metrics and checkpoint output. This applies to
regression, classification, and the joint CVAE.
`--early-stopping-min-delta` sets the minimum absolute loss improvement and
defaults to zero. Early stopping is disabled when patience is omitted.

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

`scripts/run_production_training_queue.py` executes regression,
classification, and joint-CVAE jobs sequentially for both production splits on
one GPU. It reuses the saved split/scaler artifacts and skips an existing
checkpoint unless `--rerun-completed` is supplied. Closed-set classification
on the element-held-out split is retained as a diagnostic only: its validation
classes were not observed during training, so low scores are expected and do
not measure the normal known-class classification task.

Regression parity panels report per-property coefficient of determination
(`R²`). Classification reports include accuracy, balanced accuracy, per-class
precision/recall/F1/support, and macro, weighted, and micro averages. Joint-CVAE
generation reporting includes MAE, RMSE, relative L1 error, PSNR, spatial SSIM,
Pearson correlation, and temporal-difference MAE in standardized log-radiance
space, plus physical-radiance MAE/RMSE/relative-L1 values. FID/FVD are not used
by default because natural-image feature networks are not validated for these
single-channel plasma-radiance videos.

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
  deep_set_regression/
    checkpoint.pt
    metrics.json
    learning_curves.png
    set_parity.png
    material_ensemble_parity.png
  set_transformer_regression/
    ...same artifact types...
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
