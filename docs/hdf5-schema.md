# HDF5 schema and streaming policy

The historical Group 1 files contain one top-level group per laser condition.
Group attributes hold targets and simulation conditions; child datasets whose
names match

```text
^res_([0-9.+\-eE]+)_ns(?:\.dat)?$
```

are plasma frames. Diagnostic datasets such as `metal`, `laser`,
`laser_mon1`, `laser_mon2`, and `laserplasmastats` are not frames.

Frame times are parsed as floating-point nanoseconds and sorted numerically.
They are preserved as supplied, including time zero, because spacing can be
irregular. AMR makes the row count vary from frame to frame. The maintained
HDF5 contract requires exactly 34 numeric columns. The state and saved 248-nm
opacity fields occupy columns 0--19, radial velocity ``u.y`` is column 20, and
AMR ``Level`` is column 21. Columns 22--32 contain the newer collision and
transport diagnostics; column 33 is currently reserved by the solver export.
Older 20/21-column HDF5 frames are deliberately rejected rather than guessed.
The standalone ``.dat`` diagnostic reader remains separate.

The reader builds a lightweight index, then opens a context-managed file while
streaming selected timesteps. It does not retain `h5py.File` handles on a
dataset object or share them across PyTorch workers. Indexing reads HDF5 object
metadata, not every frame array. Sequence generation then reads only the
chosen group's timesteps one at a time; it never loads or simulates the full
23.8 GB container in memory.

When no group is specified for a sequence prototype, the selector chooses the
condition closest to the median in `(rspot, log(laser_power_wcm))`, using robust
median scaling, then prefers a longer sequence and finally the lexical key.
The chosen key and all group attributes are written to output metadata.

Sequence imaging requires explicit `radial_max_m` and `axial_max_m` values in
the simulator configuration. A fixed field of view is part of the dataset
definition. Inferring it from the initial frame is unsafe because the solver
domain and emitting plume expand strongly with time; doing so can turn plume
motion out of frame into a false decay in predicted radiance.
