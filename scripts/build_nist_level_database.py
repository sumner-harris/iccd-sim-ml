"""Build element-specific bound-level bundles from the live NIST ASD website."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from iccd_sim_ml.atomic.catalog import normalize_element_symbol  # noqa: E402
from iccd_sim_ml.atomic.nist import (  # noqa: E402
    NIST_ASD_DOI,
    ROMAN_STAGE,
    fetch_ionization_energies,
    fetch_levels,
    ionization_metadata,
    write_level_table,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_ionization_table(path: Path, rows: list[dict[str, str | int | float]]) -> None:
    import csv

    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "spectrum",
                "charge",
                "energy_ev",
                "prefix",
                "suffix",
                "uncertainty_ev",
                "references",
            ),
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)


def build_element(
    symbol: str,
    output_root: Path,
    *,
    delay_seconds: float,
    timeout_seconds: float,
) -> dict[str, Any]:
    destination = output_root / symbol.lower()
    destination.mkdir(parents=True, exist_ok=True)

    ionization = fetch_ionization_energies(symbol, timeout=timeout_seconds)
    by_charge = {row.charge: row for row in ionization.energies}
    missing_thresholds = [charge for charge in ROMAN_STAGE if charge not in by_charge]
    if missing_thresholds:
        raise RuntimeError(f"{symbol} lacks NIST thresholds for charges {missing_thresholds}")

    ionization_rows = [ionization_metadata(by_charge[charge]) for charge in ROMAN_STAGE]
    ionization_path = destination / "ionization_energies.tsv"
    _write_ionization_table(ionization_path, ionization_rows)

    level_files: dict[str, str] = {}
    stage_metadata: dict[str, Any] = {}
    versions: set[str] = set()
    for charge, roman in ROMAN_STAGE.items():
        if delay_seconds:
            time.sleep(delay_seconds)
        response = fetch_levels(symbol, charge, timeout=timeout_seconds)
        if response.database_version:
            versions.add(response.database_version)
        threshold = by_charge[charge].energy_ev
        bound = tuple(level for level in response.levels if level.is_bound(threshold))
        nonabsolute = sum(level.energy_ev is None for level in response.levels)
        above_threshold = sum(
            level.energy_ev is not None and level.energy_ev >= threshold
            for level in response.levels
        )
        if not bound:
            raise RuntimeError(f"{response.spectrum} has no usable bound levels")
        minimum_energy = min(level.energy_ev for level in bound if level.energy_ev is not None)
        if abs(minimum_energy) > 1e-9:
            raise RuntimeError(f"{response.spectrum} has no usable zero-energy ground level")
        filename = f"{symbol}_{roman}_levels.tsv"
        path = destination / filename
        write_level_table(bound, path)
        level_files[str(charge)] = filename
        stage_metadata[str(charge)] = {
            "spectrum": response.spectrum,
            "query_url": response.query_url,
            "retrieved_at_utc": response.retrieved_at_utc,
            "response_sha256": f"sha256:{response.response_sha256}",
            "reported_levels": response.reported_level_count,
            "parsed_levels": len(response.levels),
            "bound_levels_written": len(bound),
            "excluded_nonabsolute_levels": nonabsolute,
            "excluded_at_or_above_threshold": above_threshold,
            "ionization_cutoff_ev": threshold,
            "file": filename,
            "file_sha256": f"sha256:{_sha256(path)}",
        }

    species = {
        "schema_version": 1,
        "symbol": symbol,
        "level_files": level_files,
        "ionization_energy_ev_by_charge": {
            str(charge): by_charge[charge].energy_ev for charge in ROMAN_STAGE
        },
        "provenance": {
            "description": (
                "Bound atomic energy levels and ionization thresholds retrieved from "
                "the live NIST Atomic Spectra Database"
            ),
            "nist_asd_doi": NIST_ASD_DOI,
            "nist_asd_versions": sorted(versions),
            "level_filter": "absolute numeric levels with 0 <= E < same-stage ionization energy",
            "manifest": "manifest.json",
        },
    }
    species_path = destination / "species.json"
    _write_json(species_path, species)

    manifest = {
        "schema_version": 1,
        "symbol": symbol,
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "source": "NIST Atomic Spectra Database",
        "nist_asd_doi": NIST_ASD_DOI,
        "nist_asd_versions": sorted(versions),
        "ionization_query": {
            "query_url": ionization.query_url,
            "retrieved_at_utc": ionization.retrieved_at_utc,
            "response_sha256": f"sha256:{ionization.response_sha256}",
            "values": ionization_rows,
            "file": ionization_path.name,
            "file_sha256": f"sha256:{_sha256(ionization_path)}",
        },
        "levels": stage_metadata,
        "species_sha256": f"sha256:{_sha256(species_path)}",
    }
    _write_json(destination / "manifest.json", manifest)
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPOSITORY_ROOT / "data" / "reference",
        help="Atomic reference catalog root (default: repository data/reference)",
    )
    parser.add_argument(
        "--elements",
        nargs="+",
        help="Element symbols; defaults to target_elements in catalog.json",
    )
    parser.add_argument(
        "--delay-seconds",
        type=float,
        default=0.35,
        help="Polite delay before each level request (default: 0.35)",
    )
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    output_root = args.output_root.expanduser().resolve()
    catalog_path = output_root / "catalog.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    elements = args.elements or catalog.get("target_elements", [])
    symbols = tuple(dict.fromkeys(normalize_element_symbol(value) for value in elements))
    if not symbols:
        raise ValueError("No elements were requested")

    completed: dict[str, str] = {}
    summary: dict[str, Any] = {}
    for index, symbol in enumerate(symbols, start=1):
        print(f"[{index}/{len(symbols)}] Retrieving {symbol} I-IV", flush=True)
        manifest = build_element(
            symbol,
            output_root,
            delay_seconds=max(args.delay_seconds, 0.0),
            timeout_seconds=args.timeout_seconds,
        )
        completed[symbol] = f"{symbol.lower()}/species.json"
        summary[symbol] = {
            charge: details["bound_levels_written"]
            for charge, details in manifest["levels"].items()
        }

    species = dict(catalog.get("species", {}))
    species.update(completed)
    catalog["species"] = dict(sorted(species.items()))
    provenance = catalog.setdefault("provenance", {})
    previous_database = provenance.get("level_database", {})
    all_counts = dict(previous_database.get("bound_level_counts", {}))
    all_counts.update(summary)
    provenance["level_database"] = {
        "source": "NIST Atomic Spectra Database",
        "doi": NIST_ASD_DOI,
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "builder": "scripts/build_nist_level_database.py",
        "bound_level_counts": dict(sorted(all_counts.items())),
    }
    _write_json(catalog_path, catalog)
    print(f"Generated complete NIST bundles for {len(completed)} elements in {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
