"""Cache one r3 continuum video per element and produce a quality/scale report."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from iccd_sim_ml.atomic import AtomicDataCatalog  # noqa: E402
from iccd_sim_ml.data import DatasetManifest  # noqa: E402
from iccd_sim_ml.imaging import ImagingConfig  # noqa: E402
from iccd_sim_ml.io import get_h5_simulation, list_h5_simulations  # noqa: E402
from iccd_sim_ml.pipeline import (  # noqa: E402
    ContinuumCacheConfig,
    discover_balanced_subset,
    ensure_continuum_cache,
)
from iccd_sim_ml.pipeline.quality import (  # noqa: E402
    sparse_level_stages,
    summarize_plasma_window,
    summarize_radiance,
)

SPECTROSCOPIC_STAGES = ("I", "II", "III", "IV")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _stage_label(charge: int) -> str:
    if 0 <= charge < len(SPECTROSCOPIC_STAGES):
        return SPECTROSCOPIC_STAGES[charge]
    return f"charge_{charge}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--atomic-reference", type=Path, required=True)
    parser.add_argument("--elements", nargs="+")
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--start-ns", type=float, default=0.0)
    parser.add_argument("--stop-ns", type=float, default=5000.0)
    parser.add_argument("--sparse-level-threshold", type=int, default=10)
    return parser.parse_args()


def _inventory(data_dir: Path, elements: tuple[str, ...]) -> dict[str, int]:
    result: dict[str, int] = {}
    for element in elements:
        matches = sorted(data_dir.glob(f"{element}_*.h5"))
        if len(matches) != 1:
            raise ValueError(f"Expected one {element}_*.h5 file, found {matches}")
        simulations = list_h5_simulations(matches[0])
        result[element] = sum(
            len(simulation.timestep_keys) >= 2
            and str(simulation.attrs.get("element", element)) == element
            for simulation in simulations
        )
    return result


def _plots(rows: list[dict[str, Any]], output_dir: Path, *, sparse_level_threshold: int) -> None:
    symbols = [row["element"] for row in rows]
    peaks = np.asarray([row["radiance"]["maximum"] for row in rows], dtype=np.float64)
    frame_fraction = np.asarray(
        [row["radiance"]["non_emissive_frame_fraction"] for row in rows], dtype=np.float64
    )
    level_minimum = np.asarray(
        [min(row["atomic_level_counts"].values()) for row in rows], dtype=np.float64
    )
    positions = np.arange(len(rows))

    figure, axis = plt.subplots(figsize=(14, 5), constrained_layout=True)
    visible = np.where(peaks > 0.0, peaks, np.nan)
    axis.bar(positions, visible)
    axis.set_yscale("log")
    axis.set_xticks(positions, symbols, rotation=90)
    axis.set(ylabel="peak photon radiance", title="r3 pilot peak radiance by element")
    axis.grid(axis="y", alpha=0.25)
    figure.savefig(output_dir / "peak_radiance_by_element.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(14, 5), constrained_layout=True)
    axis.bar(positions, frame_fraction)
    axis.set_xticks(positions, symbols, rotation=90)
    axis.set(
        ylabel="fraction of exactly zero frames",
        ylim=(0.0, 1.0),
        title="r3 pilot non-emissive frame fraction",
    )
    axis.grid(axis="y", alpha=0.25)
    figure.savefig(output_dir / "non_emissive_frame_fraction.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(14, 5), constrained_layout=True)
    axis.bar(positions, level_minimum)
    axis.axhline(
        sparse_level_threshold,
        color="red",
        linestyle="--",
        linewidth=1.2,
        label=f"{sparse_level_threshold}-level flag",
    )
    axis.set_xticks(positions, symbols, rotation=90)
    axis.set(ylabel="minimum I-IV level count", title="NIST bound-level coverage")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.savefig(output_dir / "atomic_level_coverage.png", dpi=180)
    plt.close(figure)


def main() -> int:
    args = _parse_args()
    cache_dir = args.cache_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    catalog = AtomicDataCatalog(args.atomic_reference)
    elements = tuple(args.elements or catalog.target_elements)
    imaging = ImagingConfig(
        wavelength_min_nm=300.0,
        wavelength_max_nm=800.0,
        wavelength_points=48,
        radial_points=96,
        axial_points=96,
        line_of_sight_points=128,
        temperature_table_points=160,
        radial_max_m=0.015,
        axial_max_m=0.032,
    )
    cache_config = ContinuumCacheConfig(
        imaging=imaging,
        frame_times_s=tuple(
            np.linspace(args.start_ns, args.stop_ns, args.frames, dtype=np.float64) * 1.0e-9
        ),
        atomic_mode="strict",
    )
    print(f"Discovering one interior simulation for {len(elements)} elements...", flush=True)
    samples = discover_balanced_subset(
        args.data_dir,
        elements=elements,
        simulations_per_element=1,
        minimum_frames=2,
    )

    progress_path = output_dir / "pilot_progress.json"
    progress: dict[str, Any] = {}
    if progress_path.is_file():
        progress = json.loads(progress_path.read_text(encoding="utf-8")).get("elements", {})
    records = []
    run_started = time.perf_counter()
    for index, sample in enumerate(samples, start=1):
        started = time.perf_counter()
        result = ensure_continuum_cache((sample,), cache_dir, cache_config, args.atomic_reference)
        elapsed = time.perf_counter() - started
        records.extend(result.manifest.records)
        cache_path = result.manifest.resolve_path(result.manifest.records[0])
        previous = progress.get(sample.element, {})
        generation_seconds = elapsed if result.misses else previous.get("generation_seconds")
        progress[sample.element] = {
            "sample_id": sample.sample_id,
            "simulation_key": sample.simulation_key,
            "cache_status": "miss" if result.misses else "hit",
            "last_call_seconds": elapsed,
            "generation_seconds": generation_seconds,
            "cache_bytes": cache_path.stat().st_size,
            "completed_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        }
        _write_json(
            progress_path,
            {"schema_version": 1, "config": asdict(cache_config), "elements": progress},
        )
        print(
            f"[{index}/{len(samples)}] {sample.element}: "
            f"{'generated' if result.misses else 'reused'} in {elapsed:.1f} s",
            flush=True,
        )
    manifest_path = DatasetManifest(tuple(records)).save(cache_dir / "manifest.json")
    manifest = DatasetManifest.load(manifest_path)

    rows: list[dict[str, Any]] = []
    for sample, record in zip(samples, records, strict=True):
        with np.load(manifest.resolve_path(record), allow_pickle=False) as product:
            video = np.asarray(product["video"], dtype=np.float64)
            metadata = json.loads(str(product["metadata_json"].item()))
        cached_symbol = metadata.get("atomic_data", {}).get("symbol")
        request_symbol = metadata.get("cache_request", {}).get("atomic_data", {}).get("symbol")
        if cached_symbol != sample.element or request_symbol != sample.element:
            raise ValueError(
                f"Atomic-data identity mismatch for {sample.sample_id}: "
                f"metadata={cached_symbol!r}, request={request_symbol!r}, "
                f"expected={sample.element!r}"
            )
        source_indices = tuple(int(value) for value in metadata["source_frame_indices"])
        simulation = get_h5_simulation(sample.source, sample.simulation_key)
        boiling = sample.attrs.get("t_boil")
        plasma = summarize_plasma_window(
            simulation.iter_timesteps(source_indices),
            boiling_temperature_K=None if boiling is None else float(boiling),
        )
        reference = catalog.load(sample.element, mode="strict")
        level_counts = {
            _stage_label(charge): int(levels.energy_ev.size)
            for charge, levels in sorted(reference.species.levels_by_charge.items())
        }
        sparse = sparse_level_stages(
            dict(reference.species.levels_by_charge),
            minimum_levels=args.sparse_level_threshold,
        )
        radiance = summarize_radiance(video)
        flags = []
        if radiance.all_zero_radiance:
            flags.append("all_zero_radiance")
        if plasma.no_charged_plasma_in_window:
            flags.append("no_charged_plasma_in_window")
        if plasma.below_boiling_temperature_in_window:
            flags.append("below_boiling_temperature_in_window")
        if sparse:
            flags.append("sparse_nist_bound_levels")
        rows.append(
            {
                "element": sample.element,
                "sample_id": sample.sample_id,
                "simulation_key": sample.simulation_key,
                "laser_power_wcm": sample.attrs.get("laser_power_wcm"),
                "rspot_m": sample.attrs.get("rspot"),
                "radiance": radiance.to_dict(),
                "plasma_window": plasma.to_dict(),
                "atomic_level_counts": level_counts,
                "atomic_identity_verified": True,
                "sparse_atomic_stages": {_stage_label(key): value for key, value in sparse.items()},
                "flags": flags,
                "cache_bytes": progress[sample.element]["cache_bytes"],
                "generation_seconds": progress[sample.element]["generation_seconds"],
            }
        )

    inventory = _inventory(args.data_dir.expanduser().resolve(), elements)
    known_times = {
        element: float(details["generation_seconds"])
        for element, details in progress.items()
        if details.get("generation_seconds") is not None
    }
    if not known_times:
        raise RuntimeError("No measured generation times are available for runtime estimation")
    fallback_seconds = float(np.median(tuple(known_times.values())))
    estimated_seconds = sum(
        known_times.get(element, fallback_seconds) * inventory[element] for element in elements
    )
    estimated_bytes = sum(
        progress[element]["cache_bytes"] * inventory[element] for element in elements
    )
    summary = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "configuration": asdict(cache_config),
        "pilot_elements": len(elements),
        "pilot_cache_bytes": sum(row["cache_bytes"] for row in rows),
        "pilot_wall_seconds_this_run": time.perf_counter() - run_started,
        "non_emissive": {
            "all_zero_videos": [
                row["element"] for row in rows if row["radiance"]["all_zero_radiance"]
            ],
            "no_charged_plasma": [
                row["element"]
                for row in rows
                if row["plasma_window"]["no_charged_plasma_in_window"]
            ],
            "below_boiling_temperature": [
                row["element"]
                for row in rows
                if row["plasma_window"]["below_boiling_temperature_in_window"]
            ],
        },
        "sparse_atomic_coverage": {
            row["element"]: row["sparse_atomic_stages"]
            for row in rows
            if row["sparse_atomic_stages"]
        },
        "available_simulations_by_element": inventory,
        "available_simulations_total": sum(inventory.values()),
        "full_cache_estimate": {
            "method": (
                "per-element pilot generation time and compressed bytes multiplied by "
                "valid simulation count"
            ),
            "serial_seconds": estimated_seconds,
            "serial_days": estimated_seconds / 86400.0,
            "bytes": estimated_bytes,
            "gibibytes": estimated_bytes / 1024**3,
            "directly_measured_timing_elements": sorted(known_times),
            "median_fallback_seconds": fallback_seconds,
        },
        "samples": rows,
    }
    _write_json(output_dir / "pilot_report.json", summary)
    with (output_dir / "pilot_samples.csv").open("w", encoding="utf-8", newline="") as stream:
        fieldnames = (
            "element",
            "sample_id",
            "generation_seconds",
            "cache_bytes",
            "peak_radiance",
            "mean_radiance",
            "positive_pixel_fraction",
            "non_emissive_frame_fraction",
            "maximum_temperature_K",
            "boiling_temperature_K",
            "maximum_ne_m3",
            "flags",
            "sparse_atomic_stages",
        )
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "element": row["element"],
                    "sample_id": row["sample_id"],
                    "generation_seconds": row["generation_seconds"],
                    "cache_bytes": row["cache_bytes"],
                    "peak_radiance": row["radiance"]["maximum"],
                    "mean_radiance": row["radiance"]["mean"],
                    "positive_pixel_fraction": row["radiance"]["positive_pixel_fraction"],
                    "non_emissive_frame_fraction": row["radiance"]["non_emissive_frame_fraction"],
                    "maximum_temperature_K": row["plasma_window"]["maximum_temperature_K"],
                    "boiling_temperature_K": row["plasma_window"]["boiling_temperature_K"],
                    "maximum_ne_m3": row["plasma_window"]["maximum_ne_m3"],
                    "flags": ";".join(row["flags"]),
                    "sparse_atomic_stages": json.dumps(row["sparse_atomic_stages"], sort_keys=True),
                }
            )
    _plots(rows, output_dir, sparse_level_threshold=args.sparse_level_threshold)
    print(
        json.dumps(
            {
                key: summary[key]
                for key in (
                    "pilot_elements",
                    "pilot_cache_bytes",
                    "non_emissive",
                    "sparse_atomic_coverage",
                    "available_simulations_total",
                    "full_cache_estimate",
                )
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
