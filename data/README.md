# Data layout

Large simulation inputs and generated videos are intentionally not committed.

- `reference/cu/` contains the small Cu I--IV level tables and Cu momentum-transfer data used by the reproducible prototype.
- `processed/` is reserved for versioned, offline-generated videos and manifests. Training code should read these products rather than run radiative transfer inside `Dataset.__getitem__`.
- The supplied prototype timestep remains at `../../data/Cu_130/res_3005.17_ns.dat` relative to this repository.

The SBH OES database is located at `../../Data Exploration/The SBH OES Database`. Its CSV files are transition-line tables. They are appropriate for future bound-bound emission work but are not complete canonical energy-level tables for every charge state.
