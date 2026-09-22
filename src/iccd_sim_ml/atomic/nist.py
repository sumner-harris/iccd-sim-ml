"""Retrieve and normalize atomic level data from the NIST ASD website."""

from __future__ import annotations

import csv
import hashlib
import html as html_module
import io
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Final
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

LEVELS_ENDPOINT: Final = "https://physics.nist.gov/cgi-bin/ASD/energy1.pl"
IONIZATION_ENDPOINT: Final = "https://physics.nist.gov/cgi-bin/ASD/ie.pl"
NIST_ASD_DOI: Final = "10.18434/T4W30F"
ROMAN_STAGE: Final = {0: "I", 1: "II", 2: "III", 3: "IV"}
LEVEL_COLUMNS: Final = (
    "Configuration",
    "Term",
    "J",
    "g",
    "Level (eV)",
    "Uncertainty (eV)",
    "Lande",
    "Leading percentages",
    "Reference",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _numeric_token(value: str) -> float | None:
    token = value.strip().replace("−", "-")
    if not token or re.search(r"\+[xy]$", token, flags=re.IGNORECASE):
        return None
    token = token.replace("†", "").replace("‡", "").replace("?", "")
    token = token.strip("[]()")
    token = re.sub(r"[+*]$", "", token).strip()
    try:
        return float(token)
    except ValueError:
        return None


def _clean_cell(parts: list[str]) -> str:
    return " ".join("".join(parts).replace("\xa0", " ").split())


class _LevelsHTMLParser(HTMLParser):
    """Extract NIST ``bsl`` level rows while preserving simple exponents."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[str]] = []
        self._in_row = False
        self._in_cell = False
        self._row_class = ""
        self._row: list[str] = []
        self._cell: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag == "tr":
            self._in_row = True
            self._row_class = attributes.get("class") or ""
            self._row = []
        elif self._in_row and tag in {"td", "th"}:
            self._in_cell = True
            self._cell = []
        elif self._in_cell and tag == "br":
            self._cell.append(" ")
        elif self._in_cell and tag == "sup":
            self._cell.append("^")
        elif self._in_cell and tag == "sub":
            self._cell.append("_")

    def handle_data(self, data: str) -> None:
        if self._in_cell:
            self._cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag in {"td", "th"} and self._in_cell:
            self._row.append(_clean_cell(self._cell))
            self._in_cell = False
        elif tag == "tr" and self._in_row:
            if "bsl" in self._row_class.split():
                self.rows.append(self._row)
            self._in_row = False


@dataclass(frozen=True)
class NISTLevel:
    configuration: str
    term: str
    j: str
    degeneracy: float
    energy_text_ev: str
    uncertainty_text_ev: str
    lande: str
    leading_percentages: str
    reference: str

    @property
    def energy_ev(self) -> float | None:
        return _numeric_token(self.energy_text_ev)

    def is_bound(self, ionization_energy_ev: float) -> bool:
        energy = self.energy_ev
        return energy is not None and 0.0 <= energy < ionization_energy_ev

    def to_tsv_row(self) -> dict[str, str | float]:
        return {
            "Configuration": self.configuration,
            "Term": self.term,
            "J": self.j,
            "g": self.degeneracy,
            "Level (eV)": self.energy_text_ev,
            "Uncertainty (eV)": self.uncertainty_text_ev,
            "Lande": self.lande,
            "Leading percentages": self.leading_percentages,
            "Reference": self.reference,
        }


@dataclass(frozen=True)
class NISTLevelsResponse:
    spectrum: str
    query_url: str
    retrieved_at_utc: str
    response_sha256: str
    database_version: str | None
    reported_level_count: int | None
    levels: tuple[NISTLevel, ...]


@dataclass(frozen=True)
class NISTIonizationEnergy:
    spectrum: str
    charge: int
    energy_ev: float
    prefix: str
    suffix: str
    uncertainty_ev: str
    references: str


@dataclass(frozen=True)
class NISTIonizationResponse:
    symbol: str
    query_url: str
    retrieved_at_utc: str
    response_sha256: str
    energies: tuple[NISTIonizationEnergy, ...]


def level_query_url(symbol: str, charge: int) -> str:
    if charge not in ROMAN_STAGE:
        raise ValueError("NIST level database builder supports charge states 0 through 3")
    parameters = {
        "de": "0",
        "spectrum": f"{symbol} {ROMAN_STAGE[charge]}",
        "units": "1",
        "format": "0",
        "output": "0",
        "page_size": "5000",
        "multiplet_ordered": "1",
        "conf_out": "on",
        "term_out": "on",
        "level_out": "on",
        "unc_out": "1",
        "j_out": "on",
        "g_out": "on",
        "lande_out": "on",
        "perc_out": "on",
        "biblio": "on",
        "splitting": "1",
    }
    return f"{LEVELS_ENDPOINT}?{urlencode(parameters)}"


def ionization_query_url(symbol: str) -> str:
    parameters = {
        "spectra": symbol,
        "units": "1",
        "format": "2",
        "order": "0",
        "sp_name_out": "on",
        "ion_charge_out": "on",
        "e_out": "0",
        "unc_out": "on",
        "biblio": "on",
    }
    return f"{IONIZATION_ENDPOINT}?{urlencode(parameters)}"


def parse_levels_html(
    html: str,
    *,
    spectrum: str,
    query_url: str = "",
    retrieved_at_utc: str | None = None,
) -> NISTLevelsResponse:
    """Parse one formatted NIST ASD energy-level result page."""

    if "NIST ASD : Input Error" in html or "Error Message:" in html:
        raise ValueError(f"NIST returned an input error for {spectrum}")
    parser = _LevelsHTMLParser()
    parser.feed(html)
    readable = html_module.unescape(html)
    lande_available = "landé factors are not available" not in readable.lower()

    levels: list[NISTLevel] = []
    configuration = ""
    term = ""
    for row in parser.rows:
        # NIST omits unused leading-percentage cells, so physical rows have a
        # variable width. The first seven fields and final reference field are
        # stable; Landé availability is declared once above the table.
        if len(row) < 8:
            continue
        degeneracy = _numeric_token(row[3])
        if degeneracy is None or degeneracy <= 0.0 or not row[4].strip():
            continue
        if row[0]:
            configuration = row[0]
            term = row[1]
        elif row[1]:
            term = row[1]
        lande_index = 7 if lande_available else None
        leading_start = 8 if lande_available else 7
        leading = " | ".join(value for value in row[leading_start:-1] if value)
        levels.append(
            NISTLevel(
                configuration=configuration,
                term=term,
                j=row[2],
                degeneracy=degeneracy,
                energy_text_ev=row[4],
                uncertainty_text_ev=row[5],
                lande=row[lande_index] if lande_index is not None else "",
                leading_percentages=leading,
                reference=row[-1],
            )
        )

    if not levels:
        raise ValueError(f"No energy levels could be parsed for {spectrum}")
    version_match = re.search(r"ver\.\s*([0-9]+(?:\.[0-9]+)+)", readable)
    count_match = re.search(r"([0-9,]+)\s+Levels Found", readable, flags=re.IGNORECASE)
    return NISTLevelsResponse(
        spectrum=spectrum,
        query_url=query_url,
        retrieved_at_utc=retrieved_at_utc or _utc_now(),
        response_sha256=hashlib.sha256(html.encode("latin-1", errors="replace")).hexdigest(),
        database_version=version_match.group(1) if version_match else None,
        reported_level_count=(int(count_match.group(1).replace(",", "")) if count_match else None),
        levels=tuple(levels),
    )


def _unwrap_nist_csv(value: str | None) -> str:
    token = (value or "").strip()
    if token.startswith('="') and token.endswith('"'):
        return token[2:-1]
    return token


def parse_ionization_csv(
    text: str,
    *,
    symbol: str,
    query_url: str = "",
    retrieved_at_utc: str | None = None,
) -> NISTIonizationResponse:
    """Parse the NIST ASD CSV ionization-energy response for one element."""

    energies: list[NISTIonizationEnergy] = []
    for row in csv.DictReader(io.StringIO(text)):
        spectrum = _unwrap_nist_csv(row.get("Sp. Name"))
        charge_text = _unwrap_nist_csv(row.get("Ion Charge")).lstrip("+")
        energy_text = _unwrap_nist_csv(row.get("Ionization Energy (eV)"))
        if not spectrum.startswith(f"{symbol} "):
            continue
        try:
            charge = int(charge_text)
            energy_ev = float(energy_text)
        except ValueError:
            continue
        energies.append(
            NISTIonizationEnergy(
                spectrum=spectrum,
                charge=charge,
                energy_ev=energy_ev,
                prefix=_unwrap_nist_csv(row.get("Prefix")),
                suffix=_unwrap_nist_csv(row.get("Suffix")),
                uncertainty_ev=_unwrap_nist_csv(row.get("Uncertainty (eV)")),
                references=_unwrap_nist_csv(row.get("References")),
            )
        )
    if not energies:
        raise ValueError(f"No ionization energies could be parsed for {symbol}")
    return NISTIonizationResponse(
        symbol=symbol,
        query_url=query_url,
        retrieved_at_utc=retrieved_at_utc or _utc_now(),
        response_sha256=hashlib.sha256(text.encode("latin-1", errors="replace")).hexdigest(),
        energies=tuple(energies),
    )


def _download_text(
    url: str,
    *,
    timeout: float,
    retries: int,
    user_agent: str,
) -> str:
    request = Request(url, headers={"User-Agent": user_agent})
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            with urlopen(request, timeout=timeout) as response:
                return response.read().decode("latin-1")
        except (HTTPError, URLError, TimeoutError) as error:
            last_error = error
            if attempt < retries:
                time.sleep(2**attempt)
    raise RuntimeError(f"Could not retrieve {url}") from last_error


def fetch_levels(
    symbol: str,
    charge: int,
    *,
    timeout: float = 120.0,
    retries: int = 3,
    user_agent: str = "plume-ai/0.1 NIST-ASD atomic-data retrieval",
) -> NISTLevelsResponse:
    url = level_query_url(symbol, charge)
    retrieved = _utc_now()
    html = _download_text(url, timeout=timeout, retries=retries, user_agent=user_agent)
    return parse_levels_html(
        html,
        spectrum=f"{symbol} {ROMAN_STAGE[charge]}",
        query_url=url,
        retrieved_at_utc=retrieved,
    )


def fetch_ionization_energies(
    symbol: str,
    *,
    timeout: float = 120.0,
    retries: int = 3,
    user_agent: str = "plume-ai/0.1 NIST-ASD atomic-data retrieval",
) -> NISTIonizationResponse:
    url = ionization_query_url(symbol)
    retrieved = _utc_now()
    text = _download_text(url, timeout=timeout, retries=retries, user_agent=user_agent)
    return parse_ionization_csv(
        text,
        symbol=symbol,
        query_url=url,
        retrieved_at_utc=retrieved,
    )


def write_level_table(levels: tuple[NISTLevel, ...], path: str | Path) -> None:
    """Write normalized NIST levels in the tabular format used by the physics loader."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=LEVEL_COLUMNS, delimiter="\t")
        writer.writeheader()
        writer.writerows(level.to_tsv_row() for level in levels)


def ionization_metadata(energy: NISTIonizationEnergy) -> dict[str, str | int | float]:
    return asdict(energy)


__all__ = [
    "IONIZATION_ENDPOINT",
    "LEVELS_ENDPOINT",
    "NIST_ASD_DOI",
    "NISTIonizationEnergy",
    "NISTIonizationResponse",
    "NISTLevel",
    "NISTLevelsResponse",
    "ROMAN_STAGE",
    "fetch_ionization_energies",
    "fetch_levels",
    "ionization_metadata",
    "ionization_query_url",
    "level_query_url",
    "parse_ionization_csv",
    "parse_levels_html",
    "write_level_table",
]
