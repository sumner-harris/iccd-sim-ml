# Processed data and leakage-safe splits

Radiative transfer should run offline. Each processed sample should carry a
manifest containing the source file and group, exact frame times, simulator
configuration, atomic-data checksums, code version, units, element,
laser condition, and a stable sample ID. Training datasets should perform only
loading, declared transforms, and verification against a versioned canonical
grid. Temporal resampling belongs in a reproducible preprocessing step rather
than being hidden inside `__getitem__`.

## Processed sample contract

The `iccd-sim sequence` NPZ uses these default arrays:

- `video`: `(T,H,W)`, promoted to `(C,T,H,W)` with `C=1` by the dataset;
- `times_s`: the physical time of every frame, in seconds;
- `conditions`: `[laser_power_wcm, rspot]` in that exact order;
- `regression_targets`: `[cp_metal, h_vapor, kappa_metal,
  laser_reflectivity, mass_density_metal, t_boil, tcrit]` in that exact order;
- `class_index`: an optional integer element label.

The names and source attributes are retained in `metadata_json`. Never infer
array ordering from alphabetical order. A training manifest should also name
the element explicitly so class labels and grouped splits can be audited.

`laser_power_wcm` is the historical HDF5 field for an intensity-like laser
condition conventionally expressed in `W cm^-2`. It is not pulse energy in
joules. Converting it to pulse energy requires the spatial profile, pulse
duration and temporal profile; multiplying by spot area alone is not enough.
`rspot` is the source simulation's spot-radius parameter. Preserve both raw
field names and units in derived data even if a presentation layer uses more
readable labels.

## Canonical video grid

Ordinary PyTorch batching and the conditional decoder require one fixed
`(C,T,H,W)` shape. The spatial simulator configuration supplies a common
`H,W`, but HDF5 sequences can have different and irregular `times_s`. Define a
canonical physical time grid in seconds and version it with the dataset.
Interpolate each sequence onto that grid before batching; do not treat frame
index as elapsed time and do not silently pad by repeating endpoint frames.

The preferred policy is to retain only samples whose measured/simulated time
range covers the complete canonical grid. If partial sequences must be kept,
store an explicit validity mask and use it in every reconstruction metric and
loss. Any clipping, interpolation method, or extrapolation policy is part of
the data specification and must be identical across partitions.

For the production ML dataset, use 48 wavelengths, 256 x 256 saved images,
and 128 line-of-sight quadrature cells. The LOS count affects image formation
but is not part of the tensor shape. The resulting unbatched tensor is
`(C,T,H,W) = (1,T,256,256)`. Treat this as a versioned dataset choice: do not
mix the 96 x 96 survey products with 256 x 256 production simulations unless
an explicit, identical resampling step produces the canonical 256 x 256 grid.

The supplied balanced Cu configuration uses a fixed `x = +/-15 mm`,
`z = 0--32 mm` field of view selected for the approximately 0--5 microsecond
window. Spatial shape alone is insufficient metadata; record the physical
field of view and time grid because two same-shaped videos with different bounds
do not represent the same measurement.

For longer all-element sequences, use radiance-containment population
statistics rather than the maximum hot-cell coordinate. The current files
share coverage only through 8 microseconds; the corresponding typical-plume
configuration is `configs/continuum_ml_typical_8us.json` with 256 x 256 pixels,
`x = +/-55 mm`, and `z = 0--125 mm`. This corresponds to 0.431 mm horizontal
and 0.490 mm axial pitch. The canonical 16-frame grid spans 0--8 microseconds.
The 10- and 20-microsecond variants are
coverage-filtered alternatives. Their source-coverage counts are 35/37 and
29/37, respectively. Slab-like and field-filling simulations are retained
and flagged for review, but they do not set the typical field of view.

## Split according to the scientific claim

The correct split depends on what is being claimed:

- To test **unseen elements or materials**, group by element and keep every
  simulation, clip, augmentation, and generated derivative of an element in
  exactly one of train, validation, or test. A sample-level random split leaks
  material identity because the seven physical properties are constant within
  an element.
- To test **new laser conditions for known elements**, elements may appear in
  all partitions, but split by complete `simulation_id`. Every frame and every
  overlapping clip from a laser condition must remain in one partition.
- Never split frames from one simulation independently. Multiple processed
  versions made from the same plasma simulation belong to the same group.

Use two reported evaluation tracks for the joint model rather than mixing the
claims:

1. **Element-held-out generation/regression.** Hold out complete elements in
   an outer test fold. Select hyperparameters using only grouped inner folds
   from the remaining elements. Compare generated videos with the withheld
   simulations at matched laser conditions and properties, and report results
   per held-out element as well as in aggregate.
2. **Known-class classification.** Fix the class vocabulary and ensure every
   class is represented in training. Split complete simulations within each
   element to test new laser conditions. This evaluates closed-set element
   identification, not recognition of a never-trained class.

A conventional closed-set classifier cannot correctly assign an element class
that was absent while its output layer was trained. Open-set detection or
descriptor-based retrieval would be a separate task.

## Reproduce the original notebook split

The original notebook concatenated the filtered Group 1a and Group 1b
datasets and applied PyTorch's `random_split` to complete simulation items:
70% training and 30% validation. Its saved run had 1,543 simulations, giving
1,080 training and 463 validation examples. It had no test set and supplied no
random seed. Thus complete time sequences stayed together, but the same
elements and the same element-level material-property vectors occurred in
both partitions.

Use the named helper for a reproducible comparison with that regime:

```python
from iccd_sim_ml.data import (
    PrecomputedVideoDataset,
    make_legacy_train_validation_split,
)

split = make_legacy_train_validation_split(manifest, seed=42)
train_data = PrecomputedVideoDataset(
    manifest, sample_ids=split.train, task="regression"
)
validation_data = PrecomputedVideoDataset(
    manifest, sample_ids=split.validation, task="regression"
)
```

This retains the original global 70/30 behavior while protecting
`simulation_id`, so clips or other derivatives of one simulation cannot cross
the boundary. It allows element overlap but does not mathematically guarantee
that every element lands in both partitions. To guarantee coverage for a
known-material/new-laser-condition experiment, stratify within each element:

```python
from iccd_sim_ml.data import make_known_material_split

split = make_known_material_split(
    manifest,
    ratios=(0.7, 0.3, 0.0),
    seed=42,
)
```

The stratified version requires at least two independent simulations per
element for a train/validation split. Both versions test interpolation across
laser conditions for known materials; neither supports a claim about
generalization to unseen materials. Fit every scaler using `split.train` only.

## Fit preprocessing without leakage

Fit video, laser-condition, and material-property transforms using training
sample IDs only. Reuse the frozen transforms for validation, test, conditional
generation, and checkpoint reloads. The maintained video transform applies
`log1p` to non-negative radiance and then a global standardization; the decoder
therefore predicts standardized log-radiance, not raw camera counts.

Record all feature transforms explicitly. In particular, a logarithm may be
useful for positive laser quantities spanning orders of magnitude, but it
must not be introduced implicitly or fitted using held-out values. Constant
features, missing attributes, and out-of-range values should be reported
rather than silently replaced.

Maintain an untouched outer test set. Validation is for model selection, loss
weights, stopping, and calibration choices rather than the final performance
estimate. Save the exact split manifest and the train-only scaler state in
every checkpoint.
