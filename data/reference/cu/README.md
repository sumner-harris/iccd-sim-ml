# Copper atomic reference data

- `CuIEnergyC.txt` through `CuIVEnergyC.txt`: compact level tables (`g`, energy
  in eV) distilled from the user's NIST ASD exports.
- `MT_01_01`: electron-neutral momentum-transfer cross section used by the
  original laser-absorption notebook.

The Cu I-III ionization energies are explicit constants in
`atomic/species.py`; they are not inferred from incomplete transition lists.
The broader local "SBH OES Database" contains spectral-line exports and is not
used as a substitute for complete partition-function level tables.

Source provenance should be refreshed before publication. NIST ASD data can be
cited with DOI `10.18434/T4W30F`; the local distilled files do not embed an ASD
version or original query settings.
