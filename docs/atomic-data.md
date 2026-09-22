# Element-specific atomic data

## Why the first implementation was copper-only

The HDF5 files already contain the generic plasma state needed by the
continuum model: temperature and neutral, electron, singly ionized, and
doubly ionized number densities. The first implementation was nevertheless
copper-only because it constructed every opacity lookup from Cu-specific
inputs embedded directly in the code and repository:

1. Cu I--IV energy-level tables for LTE partition functions;
2. Cu I--III ionization energies for bound-free thresholds.

Electron--ion inverse bremsstrahlung is already element-independent for the
available charge states: it depends on `ne * (n1 + 4*n2)`, temperature, and
wavelength. Photoionization depends on the element's thresholds and level
structure. Reusing Cu photoionization inputs for another element would
therefore produce plausible-looking but incorrectly labeled images.

Species-specific electron--neutral momentum-transfer tables are not available
for the target set. By project decision, every element therefore uses the same
declared SI `Q = 1e-50 m^5` kernel. The stimulated-emission factor still makes
the resulting electron--neutral coefficient depend on temperature and
wavelength, but the underlying `Q` is fixed and element independent.

## Catalog and target set

`data/reference/catalog.json` records the 36 elements represented by current
HDF5 filenames plus planned As:

```text
Al As B Be Bi C Ca Co Cs Cu Fe Ge Ho In Mg Mo Na Nb Ni P Pr Pt Rb Sb
Sc Se Si Sm Sr Ta Te Ti Tm V W Zn Zr
```

All 37 elements now have local NIST ASD bundles containing levels for stages
I--IV and ionization energies for charges 0--3. There are 148 bound-level
tables containing 28,041 canonical levels after loader deduplication. Inspect
the machine-readable status with:

```bash
iccd-sim atomic-status --atomic-reference data/reference
```

Each row reports strict readiness, usable photoionization charge states, the
electron-neutral model, missing components, warnings, and an overall fidelity
label.

## Rebuilding the NIST database

The database is reproducibly generated from the live NIST ASD HTML energy-level
tables and CSV ionization-energy results:

```bash
python scripts/build_nist_level_database.py
```

The builder defaults to `target_elements` in `data/reference/catalog.json`.
For a targeted update, pass explicit symbols:

```bash
python scripts/build_nist_level_database.py --elements Cu Fe Al
```

For every element it:

1. retrieves NIST ionization energies;
2. retrieves and parses the formatted HTML levels table for stages I--IV;
3. keeps levels with an absolute numeric energy satisfying
   `0 <= E_level < E_ionization` for that stage;
4. writes normalized tab-delimited tables and a `species.json`; and
5. records exact query URLs, retrieval times, ASD version, raw-response
   SHA-256 values, normalized-file SHA-256 values, row counts, and exclusions
   in `manifest.json`.

The bound-state cutoff is important. For example, the current NIST Cu I page
contains 360 parseable levels, but only 152 are below the 7.72638 eV Cu I
ionization threshold. Including the higher autoionizing states in an LTE bound
partition function would be physically inconsistent.

“Strict ready” means that the required adjacent level tables and thresholds
are present; it does not assert that NIST has equally complete experimental
coverage for every ion. Coverage is uneven. In this snapshot Pt III and Ta
III/IV contain only a ground level, and Tm IV contains seven bound levels.
Their partition functions are therefore lower-information approximations.
Per-stage counts are recorded in the manifests and catalog provenance so this
limitation remains auditable.

## Per-element bundle contract

Register each element as `<catalog-root>/<lowercase-symbol>/species.json` and
add it to the `species` mapping in `catalog.json`. A complete descriptor has
this structure:

```json
{
  "schema_version": 1,
  "symbol": "X",
  "level_files": {
    "0": "X_I_levels.tsv",
    "1": "X_II_levels.tsv",
    "2": "X_III_levels.tsv",
    "3": "X_IV_levels.tsv"
  },
  "ionization_energy_ev_by_charge": {
    "0": 1.0,
    "1": 2.0,
    "2": 3.0,
    "3": 4.0
  },
  "provenance": {
    "nist_asd_doi": "10.18434/T4W30F",
    "manifest": "manifest.json"
  }
}
```

The example energies are placeholders, not physical values. Compact level
files contain `g energy_eV`; tab-delimited NIST level exports with `g` and
`Level (eV)` columns are also accepted.

For `n0`, `n1`, and `n2` all to contribute to photoionization, levels are
needed for charges 0--3 and thresholds for charges 0--2. The generated bundles
also retain the charge-3 threshold because it defines the bound-state cutoff
for the stage-IV table. A partial adjacent
pair can be inspected or used in approximate mode, but only its corresponding
density contributes and the limitation is reported. Strict mode requires all
three contributing charge states.

The local SBH/NIST OES CSVs are transition-line lists. They are useful for a
future bound-bound model, but the transition-derived set of levels is not
guaranteed complete enough for partition functions. Dedicated NIST level
exports are preferred. Ionization energies must remain an explicit cited
registry rather than being inferred from the largest line energy.

Authoritative inputs are retrieved from the
[NIST ASD energy-level form](https://physics.nist.gov/PhysRefData/ASD/levels_form.html)
and
[NIST ASD ionization-energy form](https://physics.nist.gov/PhysRefData/ASD/ionEnergy.html),
with the database version, query settings, retrieval date, response checksum,
and source DOI `10.18434/T4W30F` recorded in each manifest.

## Strict versus approximate execution

Strict mode is the production default. It fails before image formation when
enabled photoionization lacks level or ionization-threshold data for any of
the `n0`, `n1`, or `n2` charge states. This prevents a cache from quietly
mixing complete and incomplete bound-free models. The fixed-Q
electron--neutral assumption is used in both strict and approximate modes.

Approximate mode exists for software integration and sensitivity studies. For
an element without a bundle it:

- retains the generic electron--ion Kramers term with Gaunt factor one;
- uses the same project-wide electron--neutral `Q = 1e-50 m^5`; and
- sets unavailable photoionization contributions to zero.

Every such NPZ is labeled `approximate_incomplete_continuum`. Its missing
inputs and chosen models are included in the cache fingerprint and metadata.
Approximate products must not be mixed with strict products in a scientific
training dataset.

## Production cache integration

`run_training_pipeline.py --cache-backend continuum` runs physical LTE
continuum synthesis through the same cache, training, and reporting workflow.
It:

- resolves atomic inputs independently for every element;
- builds one opacity lookup per element;
- simulates only source frames needed to bracket the canonical times;
- linearly interpolates photon radiance onto the common time grid;
- refuses temporal extrapolation;
- fingerprints the simulator configuration and atomic files; and
- reuses a cache only when shape, times, source identity, physics settings,
  and atomic fingerprints still match.

The continuum model remains idealized. It does not yet include bound-bound
lines, non-LTE populations, optics, spectral detector response variation,
gain, noise, or saturation.
