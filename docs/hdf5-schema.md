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
irregular. AMR makes the row count vary from frame to frame. Twenty-column
legacy arrays and 21-column arrays with mesh level are both accepted.

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
