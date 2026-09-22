# Copper atomic reference data

- `Cu_I_levels.tsv` through `Cu_IV_levels.tsv`: active bound-level tables
  generated from the live NIST ASD HTML results. `species.json` points to
  these files, and `manifest.json` records queries, cutoffs, counts, and
  checksums.
- `ionization_energies.tsv`: NIST ASD thresholds and uncertainty/reference
  metadata for Cu I--IV.
- `CuIEnergyC.txt` through `CuIVEnergyC.txt`: compact level tables (`g`, energy
  in eV) distilled from the user's earlier NIST ASD exports and retained only
  as legacy comparison artifacts.
- `MT_01_01`: retained for provenance and notebook reproduction, but no longer
  used by the image simulator. Every element, including Cu, uses the catalog's
  fixed `Q = 1e-50 m^5` electron-neutral kernel.

The Cu I-IV ionization energies are explicit in `species.json`; they are not
inferred from incomplete transition lists.
The broader local "SBH OES Database" contains spectral-line exports and is not
used as a substitute for complete partition-function level tables.

NIST ASD data can be cited with DOI `10.18434/T4W30F`. Regenerate the active
tables with `scripts/build_nist_level_database.py` when refreshing the database.
