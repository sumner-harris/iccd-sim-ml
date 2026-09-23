"""Build the complete resumable continuum cache for every eligible simulation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from itertools import zip_longest
from pathlib import Path
from typing import Any

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from iccd_sim_ml.atomic import AtomicDataCatalog  # noqa: E402
from iccd_sim_ml.data import DatasetManifest, SampleRecord  # noqa: E402
from iccd_sim_ml.imaging import ImagingConfig  # noqa: E402
from iccd_sim_ml.io import list_h5_simulations  # noqa: E402
from iccd_sim_ml.pipeline import ContinuumCacheConfig, ensure_continuum_cache  # noqa: E402
from iccd_sim_ml.pipeline.cache import (  # noqa: E402
    CONDITION_NAMES,
    TARGET_NAMES,
    SourceSample,
)

DEFAULT_PRODUCTION_IMAGING_CONFIG = REPOSITORY_ROOT / "configs" / "continuum_ml_typical_8us.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--atomic-reference", type=Path, required=True)
    parser.add_argument("--imaging-config", type=Path, default=DEFAULT_PRODUCTION_IMAGING_CONFIG)
    parser.add_argument("--elements", nargs="+")
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--start-ns", type=float, default=0.0)
    parser.add_argument("--stop-ns", type=float, default=8000.0)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--inventory-only", action="store_true")
    return parser.parse_args()


def _safe_sample_id(element: str, source: Path, simulation_key: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", simulation_key).strip("_.-") or "simulation"
    digest = hashlib.sha256(f"{source.name}::{simulation_key}".encode()).hexdigest()[:12]
    return f"{element}__{slug}__{digest}"


def _inventory(
    data_dir: Path,
    elements: tuple[str, ...],
    *,
    start_s: float,
    stop_s: float,
) -> tuple[tuple[SourceSample, ...], dict[str, Any]]:
    class_to_index = {element: index for index, element in enumerate(sorted(elements))}
    samples_by_element: dict[str, list[SourceSample]] = {element: [] for element in elements}
    report: dict[str, Any] = {}
    required_attrs = (*CONDITION_NAMES, *TARGET_NAMES)
    for element in elements:
        matches = sorted(data_dir.glob(f"{element}_*.h5"))
        if len(matches) != 1:
            raise ValueError(f"Expected one {element}_*.h5 file, found {matches}")
        source = matches[0].absolute()
        simulations = list_h5_simulations(source)
        exclusions: dict[str, list[str]] = {}
        warning_count = 0
        for simulation in simulations:
            reasons: list[str] = []
            warning_count += len(simulation.index_warnings)
            declared_element = str(simulation.attrs.get("element", element))
            if declared_element != element:
                reasons.append(f"declared_element={declared_element!r}")
            missing = [name for name in required_attrs if name not in simulation.attrs]
            if missing:
                reasons.append("missing_attrs=" + ",".join(missing))
            if len(simulation.times_s) < 2:
                reasons.append("fewer_than_two_frames")
            elif simulation.times_s[0] > start_s or simulation.times_s[-1] < stop_s:
                reasons.append(
                    f"insufficient_time_coverage={simulation.times_s[0] * 1e6:g}--"
                    f"{simulation.times_s[-1] * 1e6:g}_us"
                )
            if reasons:
                exclusions[simulation.key] = reasons
                continue
            samples_by_element[element].append(
                SourceSample(
                    sample_id=_safe_sample_id(element, source, simulation.key),
                    element=element,
                    class_index=class_to_index[element],
                    source=source,
                    simulation_key=simulation.key,
                    attrs=simulation.attrs,
                    timestep_keys=simulation.timestep_keys,
                )
            )
        samples_by_element[element].sort(key=lambda item: item.simulation_key)
        report[element] = {
            "source": str(source),
            "indexed_simulations": len(simulations),
            "eligible_simulations": len(samples_by_element[element]),
            "excluded_simulations": len(exclusions),
            "index_warning_count": warning_count,
            "exclusions": exclusions,
        }

    # Round-robin submission avoids sending all workers to the same large HDF5
    # file at once and gives each element early progress in a restarted run.
    interleaved = tuple(
        sample
        for group in zip_longest(*(samples_by_element[element] for element in elements))
        for sample in group
        if sample is not None
    )
    if not interleaved:
        raise RuntimeError("No simulation covers the requested production time grid")
    return interleaved, report


def _cache_one(
    sample: SourceSample,
    cache_root: Path,
    config: ContinuumCacheConfig,
    atomic_reference: Path,
) -> dict[str, Any]:
    sample_dir = cache_root / "samples" / sample.element / sample.sample_id
    started = time.perf_counter()
    result = ensure_continuum_cache((sample,), sample_dir, config, atomic_reference)
    elapsed = time.perf_counter() - started
    source_record = result.manifest.records[0]
    product = result.manifest.resolve_path(source_record)
    record = SampleRecord(
        sample_id=source_record.sample_id,
        path=product,
        element=source_record.element,
        simulation_id=source_record.simulation_id,
        family_id=source_record.family_id,
        metadata={
            **source_record.metadata,
            "source_file": str(sample.source),
            "source_group": sample.simulation_key,
        },
    )
    return {
        "record": record,
        "cache_status": "miss" if result.misses else "hit",
        "elapsed_seconds": elapsed,
        "cache_bytes": product.stat().st_size,
    }


def _load_existing_record(cache_root: Path, detail: dict[str, Any]) -> SampleRecord | None:
    relative = detail.get("path")
    if not isinstance(relative, str):
        return None
    product = cache_root / relative
    if not product.is_file():
        return None
    return SampleRecord(
        sample_id=str(detail["sample_id"]),
        path=product,
        element=str(detail["element"]),
        simulation_id=str(detail["simulation_id"]),
        family_id=str(detail["element"]),
        metadata=dict(detail.get("metadata", {})),
    )


def main() -> int:
    args = _parse_args()
    if args.workers < 1 or args.frames < 2:
        raise ValueError("--workers must be positive and --frames must be at least two")
    if not 0.0 <= args.start_ns < args.stop_ns:
        raise ValueError("Require 0 <= --start-ns < --stop-ns")
    data_dir = args.data_dir.expanduser().resolve()
    cache_root = args.cache_dir.expanduser().resolve()
    atomic_reference = args.atomic_reference.expanduser().resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    catalog = AtomicDataCatalog(atomic_reference)
    elements = tuple(args.elements or catalog.target_elements)
    imaging = ImagingConfig.from_json(args.imaging_config)
    config = ContinuumCacheConfig(
        imaging=imaging,
        frame_times_s=tuple(
            np.linspace(args.start_ns, args.stop_ns, args.frames, dtype=np.float64) * 1.0e-9
        ),
        atomic_mode="strict",
    )
    started_at = _utc_now()
    print(f"Indexing all simulations for {len(elements)} elements...", flush=True)
    samples, inventory = _inventory(
        data_dir,
        elements,
        start_s=args.start_ns * 1.0e-9,
        stop_s=args.stop_ns * 1.0e-9,
    )
    inventory_payload = {
        "schema_version": 1,
        "generated_at_utc": _utc_now(),
        "configuration": asdict(config),
        "eligible_total": len(samples),
        "excluded_total": sum(row["excluded_simulations"] for row in inventory.values()),
        "elements": inventory,
    }
    _write_json(cache_root / "source_inventory.json", inventory_payload)
    print(
        f"Eligible: {len(samples)}; excluded for schema/time coverage: "
        f"{inventory_payload['excluded_total']}",
        flush=True,
    )
    if args.inventory_only:
        return 0

    progress_path = cache_root / "progress.json"
    previous: dict[str, Any] = {}
    if progress_path.is_file():
        try:
            previous = json.loads(progress_path.read_text(encoding="utf-8")).get("samples", {})
        except (OSError, json.JSONDecodeError):
            previous = {}
    progress = dict(previous)
    records: dict[str, SampleRecord] = {}
    for sample_id, detail in progress.items():
        if detail.get("status") == "complete":
            record = _load_existing_record(cache_root, detail)
            if record is not None:
                records[sample_id] = record

    def persist(state: str) -> None:
        complete = sum(item.get("status") == "complete" for item in progress.values())
        failed = sum(item.get("status") == "failed" for item in progress.values())
        _write_json(
            progress_path,
            {
                "schema_version": 1,
                "state": state,
                "started_at_utc": started_at,
                "updated_at_utc": _utc_now(),
                "configuration": asdict(config),
                "eligible_total": len(samples),
                "complete": complete,
                "failed": failed,
                "pending": len(samples) - complete - failed,
                "workers": args.workers,
                "samples": progress,
            },
        )

    persist("running")
    completed_this_run = 0
    failures_this_run = 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(_cache_one, sample, cache_root, config, atomic_reference): sample
            for sample in samples
        }
        for future in as_completed(futures):
            sample = futures[future]
            try:
                result = future.result()
            except Exception as exc:  # continue to preserve all independent results
                failures_this_run += 1
                progress[sample.sample_id] = {
                    "status": "failed",
                    "sample_id": sample.sample_id,
                    "element": sample.element,
                    "simulation_id": f"{sample.source.name}::{sample.simulation_key}",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "updated_at_utc": _utc_now(),
                }
                print(
                    f"FAILED {sample.sample_id}: {type(exc).__name__}: {exc}",
                    flush=True,
                )
            else:
                completed_this_run += 1
                record = result["record"]
                records[sample.sample_id] = record
                progress[sample.sample_id] = {
                    "status": "complete",
                    "sample_id": sample.sample_id,
                    "element": sample.element,
                    "simulation_id": record.simulation_id,
                    "path": str(record.path.relative_to(cache_root)),
                    "metadata": dict(record.metadata),
                    "cache_status": result["cache_status"],
                    "elapsed_seconds": result["elapsed_seconds"],
                    "cache_bytes": result["cache_bytes"],
                    "updated_at_utc": _utc_now(),
                }
                print(
                    f"[{len(records)}/{len(samples)}] {sample.sample_id}: "
                    f"{result['cache_status']} in {result['elapsed_seconds']:.1f} s",
                    flush=True,
                )
            persist("running")

    ordered_records = tuple(
        records[sample.sample_id] for sample in samples if sample.sample_id in records
    )
    DatasetManifest(ordered_records).save(cache_root / "manifest.json")
    failed = {key: value for key, value in progress.items() if value.get("status") == "failed"}
    report = {
        "schema_version": 1,
        "completed_at_utc": _utc_now(),
        "configuration": asdict(config),
        "workers": args.workers,
        "eligible_total": len(samples),
        "cached_total": len(ordered_records),
        "failed_total": len(failed),
        "completed_this_run": completed_this_run,
        "failures_this_run": failures_this_run,
        "cache_bytes": sum(
            int(value.get("cache_bytes", 0))
            for value in progress.values()
            if value.get("status") == "complete"
        ),
        "manifest": str(cache_root / "manifest.json"),
        "inventory": str(cache_root / "source_inventory.json"),
        "failed_samples": failed,
    }
    _write_json(cache_root / "run_report.json", report)
    persist("complete" if not failed else "complete_with_failures")
    print(json.dumps({key: report[key] for key in report if key != "failed_samples"}, indent=2))
    return 0 if not failed else 2


if __name__ == "__main__":
    raise SystemExit(main())
