"""Load canonical energy levels used for LTE populations and photoionization."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from iccd_sim_ml.constants import BOLTZMANN_EV_K

FloatArray = NDArray[np.float64]


def _numeric_token(value: str) -> float:
    token = value.strip()
    token = token.replace("&dagger;", "").replace("?", "")
    token = token.strip("[]()")
    token = re.sub(r"[+*]$", "", token)
    if not token or re.search(r"\+[xy]$", token, flags=re.IGNORECASE):
        raise ValueError(f"Energy value {value!r} is not an absolute numeric level")
    return float(token)


@dataclass(frozen=True)
class EnergyLevels:
    """Sorted, deduplicated levels for one charge state."""

    energy_ev: FloatArray
    degeneracy: FloatArray
    source: Path

    def __post_init__(self) -> None:
        if self.energy_ev.ndim != 1 or self.degeneracy.ndim != 1:
            raise ValueError("Energy and degeneracy arrays must be one-dimensional")
        if self.energy_ev.shape != self.degeneracy.shape or self.energy_ev.size == 0:
            raise ValueError("Energy and degeneracy arrays must have equal non-zero lengths")
        if np.any(self.energy_ev < 0) or np.any(self.degeneracy <= 0):
            raise ValueError("Energy levels must be non-negative and degeneracies positive")
        if np.any(np.diff(self.energy_ev) < 0):
            raise ValueError("Energy levels must be sorted")

    def partition_function(self, temperature_K: FloatArray | float) -> FloatArray:
        temperature = np.asarray(temperature_K, dtype=np.float64)
        safe_temperature = np.maximum(temperature, np.finfo(np.float64).tiny)
        exponent = -self.energy_ev.reshape((-1,) + (1,) * temperature.ndim) / (
            BOLTZMANN_EV_K * safe_temperature
        )
        weights = self.degeneracy.reshape((-1,) + (1,) * temperature.ndim)
        result = np.sum(weights * np.exp(np.clip(exponent, -745.0, 0.0)), axis=0)
        return np.where(temperature > 0, result, 0.0)


def _deduplicate(energy: list[float], degeneracy: list[float], source: Path) -> EnergyLevels:
    grouped: dict[float, set[float]] = {}
    for level, weight in zip(energy, degeneracy, strict=True):
        key = round(float(level), 10)
        grouped.setdefault(key, set()).add(float(weight))
    # Distinct atomic levels may be exactly energy-degenerate. Their
    # statistical weights add in the partition function; repeated identical
    # (energy, g) records are still counted only once.
    ordered = sorted((level, sum(weights)) for level, weights in grouped.items())
    return EnergyLevels(
        energy_ev=np.asarray([item[0] for item in ordered], dtype=np.float64),
        degeneracy=np.asarray([item[1] for item in ordered], dtype=np.float64),
        source=source,
    )


def load_energy_levels(path: str | Path) -> EnergyLevels:
    """Load either the compact two-column or NIST tabular level export.

    Compact files are expected to contain ``g  energy_eV``. NIST exports must
    contain the columns ``g`` and ``Level (eV)``.
    """

    source = Path(path).expanduser().resolve()
    with source.open("r", encoding="utf-8-sig", errors="replace") as stream:
        first_line = stream.readline()

    energy: list[float] = []
    degeneracy: list[float] = []
    if "Level (eV)" in first_line and "g" in first_line:
        with source.open("r", encoding="utf-8-sig", errors="replace", newline="") as stream:
            reader = csv.DictReader(stream, delimiter="\t")
            for row in reader:
                try:
                    level = _numeric_token(row.get("Level (eV)", ""))
                    weight = _numeric_token(row.get("g", ""))
                except (TypeError, ValueError):
                    continue
                energy.append(level)
                degeneracy.append(weight)
    else:
        with source.open("r", encoding="utf-8-sig", errors="replace") as stream:
            for line in stream:
                pieces = line.split()
                if len(pieces) < 2:
                    continue
                try:
                    weight = _numeric_token(pieces[0])
                    level = _numeric_token(pieces[1])
                except ValueError:
                    continue
                energy.append(level)
                degeneracy.append(weight)

    if not energy:
        raise ValueError(f"No usable energy levels were found in {source}")
    return _deduplicate(energy, degeneracy, source)
