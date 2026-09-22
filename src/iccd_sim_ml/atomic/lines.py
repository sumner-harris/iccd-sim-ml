"""Parser for the distilled SBH/NIST optical-emission line CSV files."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path


def _optional_float(value: str | None) -> tuple[float | None, str | None]:
    if value is None:
        return None, None
    raw = value.strip()
    if not raw or raw.lower() == "nan":
        return None, None
    qualifier = None
    if raw[-1:] in {"+", "*"}:
        qualifier, raw = raw[-1], raw[:-1]
    raw = raw.strip("[]()")
    if re.search(r"\+[xy]$", raw, flags=re.IGNORECASE):
        return None, "unknown_offset"
    raw = raw.replace("?", "").replace("&dagger;", "")
    try:
        return float(raw), qualifier
    except ValueError:
        return None, "unparsed"


@dataclass(frozen=True)
class AtomicLine:
    element: str
    spectrum_number: int
    charge_state: int
    wavelength_vacuum_nm: float
    transition_probability_s: float
    lower_energy_ev: float | None
    upper_energy_ev: float | None
    lower_degeneracy: float | None
    upper_degeneracy: float | None
    transition_type: str
    wavelength_qualifier: str | None = None


def load_oes_lines(
    path: str | Path,
    *,
    charge_states: set[int] | None = None,
    wavelength_range_nm: tuple[float, float] | None = None,
) -> list[AtomicLine]:
    """Load quantitative line fields while preserving wavelength qualifiers."""

    source = Path(path).expanduser().resolve()
    inferred_element = source.name.split()[0]
    result: list[AtomicLine] = []
    with source.open("r", encoding="utf-8-sig", errors="replace", newline="") as stream:
        for row in csv.DictReader(stream):
            element = (row.get("element") or inferred_element).strip()
            try:
                spectrum = int(float(row.get("sp_num") or 1))
                a_value = float(row.get("Aki(s^-1)") or "nan")
            except ValueError:
                continue
            charge = spectrum - 1
            if charge_states is not None and charge not in charge_states:
                continue
            ritz, qualifier = _optional_float(row.get("ritz_wl_vac(nm)"))
            observed, observed_qualifier = _optional_float(row.get("obs_wl_vac(nm)"))
            wavelength = ritz if ritz is not None else observed
            qualifier = qualifier if ritz is not None else observed_qualifier
            if wavelength is None or not (a_value > 0):
                continue
            if wavelength_range_nm is not None:
                low, high = wavelength_range_nm
                if wavelength < low or wavelength > high:
                    continue
            lower, _ = _optional_float(row.get("Ei(eV)"))
            upper, _ = _optional_float(row.get("Ek(eV)"))
            lower_g, _ = _optional_float(row.get("g_i"))
            upper_g, _ = _optional_float(row.get("g_k"))
            transition_type = (row.get("Type") or "").strip()
            if not transition_type or transition_type.lower() == "nan":
                transition_type = "E1"
            result.append(
                AtomicLine(
                    element=element,
                    spectrum_number=spectrum,
                    charge_state=charge,
                    wavelength_vacuum_nm=wavelength,
                    transition_probability_s=a_value,
                    lower_energy_ev=lower,
                    upper_energy_ev=upper,
                    lower_degeneracy=lower_g,
                    upper_degeneracy=upper_g,
                    transition_type=transition_type,
                    wavelength_qualifier=qualifier,
                )
            )
    return result
