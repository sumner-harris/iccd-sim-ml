"""Survey late-time plume fields of view using maximum laser conditions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from iccd_sim_ml.atomic import AtomicDataCatalog  # noqa: E402
from iccd_sim_ml.imaging import ImagingConfig, simulate_continuum_image  # noqa: E402
from iccd_sim_ml.imaging.grid import infer_dyadic_domain_max  # noqa: E402
from iccd_sim_ml.io import get_h5_simulation, list_h5_simulations  # noqa: E402
from iccd_sim_ml.pipeline import (  # noqa: E402
    bracketing_frame_indices,
    select_maximum_condition,
    summarize_radiance_extent,
)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--atomic-reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--elements", nargs="+")
    parser.add_argument("--time-us", type=float, default=20.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--survey-radial-mm", type=float)
    parser.add_argument("--survey-axial-mm", type=float)
    parser.add_argument("--margin-fraction", type=float, default=0.10)
    parser.add_argument("--round-up-mm", type=float, default=5.0)
    parser.add_argument("--inspect-only", action="store_true")
    return parser.parse_args()


def _active_extent(timesteps: list[Any]) -> dict[str, float | None]:
    charged_r = []
    charged_z = []
    hot_r = []
    hot_z = []
    maximum_temperature = 0.0
    maximum_charged_density = 0.0
    above_lookup_cells = 0
    total_cells = 0
    for timestep in timesteps:
        charged = timestep.ne_m3 + timestep.n1_m3 + timestep.n2_m3
        charged_maximum = float(np.max(charged))
        maximum_charged_density = max(maximum_charged_density, charged_maximum)
        maximum_temperature = max(maximum_temperature, float(np.max(timestep.temperature_K)))
        total_cells += timestep.size
        above_lookup_cells += int(np.count_nonzero(timestep.temperature_K > 100_000.0))
        if charged_maximum > 0.0:
            active = charged >= charged_maximum * 1.0e-8
            charged_r.extend(timestep.r_m[active].tolist())
            charged_z.extend(timestep.z_m[active].tolist())
        hot = timestep.temperature_K >= 1_000.0
        hot_r.extend(timestep.r_m[hot].tolist())
        hot_z.extend(timestep.z_m[hot].tolist())

    def maximum_or_none(values: list[float]) -> float | None:
        return None if not values else float(max(values))

    return {
        "charged_relative_threshold": 1.0e-8,
        "charged_radial_max_m": maximum_or_none(charged_r),
        "charged_axial_max_m": maximum_or_none(charged_z),
        "hot_threshold_K": 1_000.0,
        "hot_radial_max_m": maximum_or_none(hot_r),
        "hot_axial_max_m": maximum_or_none(hot_z),
        "maximum_temperature_K": maximum_temperature,
        "maximum_charged_density_m3": maximum_charged_density,
        "fraction_cells_above_100000_K": (
            0.0 if total_cells == 0 else above_lookup_cells / total_cells
        ),
    }


def _inspect_element(data_dir: Path, element: str, target_time_s: float) -> dict[str, Any]:
    matches = sorted(data_dir.glob(f"{element}_*.h5"))
    if len(matches) != 1:
        raise ValueError(f"Expected one {element}_*.h5 file, found {matches}")
    simulations = list_h5_simulations(matches[0])
    selected = select_maximum_condition(simulations)
    conditions = []
    for item in simulations:
        try:
            power = float(item.attrs["laser_power_wcm"])
            spot = float(item.attrs["rspot"])
        except (KeyError, TypeError, ValueError):
            continue
        if np.isfinite(power) and power > 0.0 and np.isfinite(spot) and spot > 0.0:
            conditions.append((power, spot))
    all_powers = [item[0] for item in conditions]
    all_spots = [item[1] for item in conditions]
    record: dict[str, Any] = {
        "element": element,
        "source": str(selected.source),
        "simulation_key": selected.key,
        "laser_power_wcm": float(selected.attrs["laser_power_wcm"]),
        "rspot_m": float(selected.attrs["rspot"]),
        "maximum_laser_power_wcm_in_file": max(all_powers),
        "maximum_rspot_m_in_file": max(all_spots),
        "joint_maximum_condition": (
            float(selected.attrs["laser_power_wcm"]) == max(all_powers)
            and float(selected.attrs["rspot"]) == max(all_spots)
        ),
        "frame_count": len(selected.timestep_keys),
        "source_start_us": float(selected.times_s[0] * 1.0e6),
        "source_stop_us": float(selected.times_s[-1] * 1.0e6),
        "target_time_us": target_time_s * 1.0e6,
        "target_covered": bool(selected.times_s[0] <= target_time_s <= selected.times_s[-1]),
        "index_warnings": list(selected.index_warnings),
    }
    if not record["target_covered"]:
        return record
    lower, upper, weight = bracketing_frame_indices(selected.times_s, target_time_s)
    indices = [lower] if lower == upper else [lower, upper]
    timesteps = list(selected.iter_timesteps(indices))
    radial_domains = [infer_dyadic_domain_max(item.r_m) for item in timesteps]
    axial_domains = [infer_dyadic_domain_max(item.z_m) for item in timesteps]
    record.update(
        {
            "lower_index": lower,
            "upper_index": upper,
            "upper_interpolation_weight": weight,
            "source_frame_times_us": [float(selected.times_s[index] * 1.0e6) for index in indices],
            "source_timestep_keys": [selected.timestep_keys[index] for index in indices],
            "common_radial_domain_m": float(min(radial_domains)),
            "common_axial_domain_m": float(min(axial_domains)),
            "source_state": _active_extent(timesteps),
        }
    )
    return record


def _request_fingerprint(request: dict[str, Any]) -> str:
    payload = json.dumps(request, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _render_element(
    record: dict[str, Any],
    output_dir: Path,
    atomic_reference_root: Path,
    config: ImagingConfig,
) -> dict[str, Any]:
    element = record["element"]
    simulation = get_h5_simulation(record["source"], record["simulation_key"])
    reference = AtomicDataCatalog(atomic_reference_root).load(element, mode="strict")
    source_stat = Path(record["source"]).stat()
    request = {
        "schema": 1,
        "element": element,
        "source": record["source"],
        "source_size": source_stat.st_size,
        "source_mtime_ns": source_stat.st_mtime_ns,
        "simulation_key": record["simulation_key"],
        "source_timestep_keys": record["source_timestep_keys"],
        "source_frame_times_us": record["source_frame_times_us"],
        "target_time_us": record["target_time_us"],
        "upper_interpolation_weight": record["upper_interpolation_weight"],
        "imaging_config": config.to_dict(),
        "atomic_data": reference.fingerprint(),
    }
    fingerprint = _request_fingerprint(request)
    destination = output_dir / "frames" / f"{element}_{record['target_time_us']:g}us_r3.npz"
    if destination.is_file():
        try:
            with np.load(destination, allow_pickle=False) as product:
                metadata = json.loads(str(product["metadata_json"].item()))
                valid = (
                    metadata.get("request_fingerprint") == fingerprint
                    and product["image_photon_radiance"].shape
                    == (config.radial_points, config.axial_points)
                    and np.isfinite(product["image_photon_radiance"]).all()
                    and np.all(product["image_photon_radiance"] >= 0.0)
                )
                if valid:
                    return {
                        "element": element,
                        "path": str(destination),
                        "cache_status": "hit",
                        "runtime_s": metadata.get("runtime_s"),
                    }
        except (OSError, KeyError, ValueError, json.JSONDecodeError):
            pass

    started = time.perf_counter()
    direct = reference.build_opacity_model()
    lookup = direct.build_lookup(
        np.geomspace(
            config.temperature_table_min_K,
            config.temperature_table_max_K,
            config.temperature_table_points,
        ),
        config.wavelengths_m,
    )
    indices = [record["lower_index"]]
    if record["upper_index"] != record["lower_index"]:
        indices.append(record["upper_index"])
    timesteps = list(simulation.iter_timesteps(indices))
    rendered = [simulate_continuum_image(item, config, lookup) for item in timesteps]
    if len(rendered) == 1:
        image = rendered[0].image_photon_radiance
    else:
        weight = float(record["upper_interpolation_weight"])
        image = (
            rendered[0].image_photon_radiance * (1.0 - weight)
            + rendered[1].image_photon_radiance * weight
        )
    runtime = time.perf_counter() - started
    metadata = {
        "request_fingerprint": fingerprint,
        "request": request,
        "runtime_s": runtime,
        "units": "photons s^-1 m^-2 sr^-1",
        "temporal_interpolation": "linear_in_photon_radiance",
        "source_quality_flags": [list(item.quality_flags) for item in timesteps],
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream,
                image_photon_radiance=np.asarray(image, dtype=np.float32),
                x_m=rendered[0].x_m,
                z_m=rendered[0].z_m,
                metadata_json=json.dumps(metadata, sort_keys=True),
            )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "element": element,
        "path": str(destination),
        "cache_status": "miss",
        "runtime_s": runtime,
    }


def _round_up_m(value_m: float, increment_mm: float) -> float:
    return math.ceil(value_m * 1.0e3 / increment_mm) * increment_mm * 1.0e-3


def _plot_montage(rows: list[dict[str, Any]], output: Path, target_time_us: float) -> None:
    columns = 7
    row_count = math.ceil(len(rows) / columns)
    figure, axes = plt.subplots(
        row_count,
        columns,
        figsize=(2.65 * columns, 2.45 * row_count),
        constrained_layout=True,
        squeeze=False,
    )
    for axis, row in zip(axes.ravel(), rows, strict=False):
        image = row["image"]
        positive = image[image > 0.0]
        if positive.size:
            peak = float(np.max(positive))
            lower = max(float(np.quantile(positive, 0.01)), peak * 1.0e-8)
            if lower >= peak:
                lower = peak * 0.1
            norm = LogNorm(vmin=lower, vmax=peak)
        else:
            norm = None
        axis.imshow(
            image.T,
            origin="lower",
            aspect="auto",
            extent=(row["x_m"][0] * 1.0e3, row["x_m"][-1] * 1.0e3, 0.0, row["z_m"][-1] * 1.0e3),
            cmap="inferno",
            norm=norm,
        )
        axis.set_title(row["element"])
        axis.tick_params(labelsize=7)
    for axis in axes.ravel()[len(rows) :]:
        axis.axis("off")
    figure.suptitle(
        f"Maximum-power / maximum-spot continuum images at {target_time_us:g} µs\n"
        "independent logarithmic scale per element",
        fontsize=15,
    )
    figure.supxlabel("x / mm")
    figure.supylabel("z / mm")
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _plot_extents(rows: list[dict[str, Any]], output: Path, target_time_us: float) -> None:
    elements = [row["element"] for row in rows]
    radial = [
        (row["radiance_extent"]["radial_containment_m"]["99.9%"] or 0.0) * 1.0e3 for row in rows
    ]
    axial = [
        (row["radiance_extent"]["axial_containment_m"]["99.9%"] or 0.0) * 1.0e3 for row in rows
    ]
    position = np.arange(len(rows))
    width = 0.42
    figure, axis = plt.subplots(figsize=(15, 5.5), constrained_layout=True)
    axis.bar(position - width / 2, radial, width, label="radial 99.9% containment")
    axis.bar(position + width / 2, axial, width, label="axial 99.9% containment")
    axis.set_xticks(position, elements, rotation=90)
    axis.set_ylabel("extent / mm")
    axis.set_title(f"Continuum-radiance containment at {target_time_us:g} µs")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = (
        "element",
        "simulation_key",
        "laser_power_wcm",
        "rspot_m",
        "runtime_s",
        "peak",
        "radial_99_9_mm",
        "axial_99_9_mm",
        "radial_support_1e_6_mm",
        "axial_support_1e_6_mm",
        "radial_edge_fraction",
        "axial_edge_fraction",
        "maximum_temperature_K",
        "fraction_cells_above_100000_K",
    )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            extent = row["radiance_extent"]
            state = row["source_state"]
            writer.writerow(
                {
                    "element": row["element"],
                    "simulation_key": row["simulation_key"],
                    "laser_power_wcm": row["laser_power_wcm"],
                    "rspot_m": row["rspot_m"],
                    "runtime_s": row["runtime_s"],
                    "peak": extent["peak"],
                    "radial_99_9_mm": (
                        None
                        if extent["radial_containment_m"]["99.9%"] is None
                        else extent["radial_containment_m"]["99.9%"] * 1.0e3
                    ),
                    "axial_99_9_mm": (
                        None
                        if extent["axial_containment_m"]["99.9%"] is None
                        else extent["axial_containment_m"]["99.9%"] * 1.0e3
                    ),
                    "radial_support_1e_6_mm": (
                        None
                        if extent["radial_relative_support_m"]["1e-06"] is None
                        else extent["radial_relative_support_m"]["1e-06"] * 1.0e3
                    ),
                    "axial_support_1e_6_mm": (
                        None
                        if extent["axial_relative_support_m"]["1e-06"] is None
                        else extent["axial_relative_support_m"]["1e-06"] * 1.0e3
                    ),
                    "radial_edge_fraction": extent["radial_edge_fraction"],
                    "axial_edge_fraction": extent["axial_outer_edge_fraction"],
                    "maximum_temperature_K": state["maximum_temperature_K"],
                    "fraction_cells_above_100000_K": state["fraction_cells_above_100000_K"],
                }
            )


def main() -> int:
    args = _parse_args()
    if args.time_us <= 0.0 or args.workers < 1:
        raise ValueError("--time-us and --workers must be positive")
    if not 0.0 <= args.margin_fraction < 1.0 or args.round_up_mm <= 0.0:
        raise ValueError("Margin must be in [0,1), and round-up increment must be positive")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    catalog = AtomicDataCatalog(args.atomic_reference)
    elements = tuple(args.elements or catalog.target_elements)
    target_time_s = args.time_us * 1.0e-6
    data_dir = args.data_dir.expanduser().absolute()

    print(f"Inspecting maximum laser conditions for {len(elements)} elements...", flush=True)
    inspected_by_element: dict[str, dict[str, Any]] = {}
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        future_to_element = {
            executor.submit(_inspect_element, data_dir, element, target_time_s): element
            for element in elements
        }
        for completed, future in enumerate(as_completed(future_to_element), start=1):
            element = future_to_element[future]
            inspected_by_element[element] = future.result()
            print(f"[{completed}/{len(elements)}] indexed {element}", flush=True)
    inspected = [inspected_by_element[element] for element in elements]
    inspection = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "target_time_us": args.time_us,
        "elements": inspected,
    }
    _write_json(output_dir / "fov_inspection.json", inspection)
    uncovered = [row["element"] for row in inspected if not row["target_covered"]]
    if uncovered:
        print(f"Target time is not covered for: {', '.join(uncovered)}", flush=True)
        return 2
    if args.inspect_only:
        return 0

    common_radial_domain = min(row["common_radial_domain_m"] for row in inspected)
    common_axial_domain = min(row["common_axial_domain_m"] for row in inspected)
    radial_max = (
        common_radial_domain if args.survey_radial_mm is None else args.survey_radial_mm * 1.0e-3
    )
    axial_max = (
        common_axial_domain if args.survey_axial_mm is None else args.survey_axial_mm * 1.0e-3
    )
    if radial_max > common_radial_domain or axial_max > common_axial_domain:
        raise ValueError(
            "Survey FOV exceeds at least one source domain: requested "
            f"r={radial_max:g} m, z={axial_max:g} m; common domain is "
            f"r={common_radial_domain:g} m, z={common_axial_domain:g} m"
        )
    config = ImagingConfig(
        wavelength_min_nm=300.0,
        wavelength_max_nm=800.0,
        wavelength_points=48,
        radial_points=96,
        axial_points=96,
        line_of_sight_points=128,
        temperature_table_min_K=300.0,
        temperature_table_max_K=100_000.0,
        temperature_table_points=160,
        radial_max_m=radial_max,
        axial_max_m=axial_max,
    )
    print(
        f"Rendering r3 on x=±{radial_max * 1e3:g} mm, z=0..{axial_max * 1e3:g} mm...",
        flush=True,
    )
    rendered_by_element: dict[str, dict[str, Any]] = {}
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        future_to_element = {
            executor.submit(
                _render_element,
                row,
                output_dir,
                args.atomic_reference.expanduser().resolve(),
                config,
            ): row["element"]
            for row in inspected
        }
        for completed, future in enumerate(as_completed(future_to_element), start=1):
            element = future_to_element[future]
            rendered_by_element[element] = future.result()
            print(
                f"[{completed}/{len(elements)}] {element}: "
                f"{rendered_by_element[element]['cache_status']}",
                flush=True,
            )

    rows: list[dict[str, Any]] = []
    for inspected_row in inspected:
        rendered = rendered_by_element[inspected_row["element"]]
        with np.load(rendered["path"], allow_pickle=False) as product:
            image = np.asarray(product["image_photon_radiance"], dtype=np.float64)
            x_m = np.asarray(product["x_m"], dtype=np.float64)
            z_m = np.asarray(product["z_m"], dtype=np.float64)
            metadata = json.loads(str(product["metadata_json"].item()))
        if metadata["request"]["atomic_data"]["symbol"] != inspected_row["element"]:
            raise ValueError(f"Atomic identity mismatch for {inspected_row['element']}")
        extent = summarize_radiance_extent(image, x_m, z_m)
        rows.append(
            {
                **inspected_row,
                "runtime_s": rendered["runtime_s"],
                "cache_status": rendered["cache_status"],
                "radiance_extent": extent.to_dict(),
                "image": image,
                "x_m": x_m,
                "z_m": z_m,
            }
        )

    maximum_radial = max(
        max(
            row["radiance_extent"]["radial_containment_m"]["99.9%"] or 0.0,
            row["radiance_extent"]["radial_relative_support_m"]["1e-06"] or 0.0,
        )
        for row in rows
    )
    maximum_axial = max(
        max(
            row["radiance_extent"]["axial_containment_m"]["99.9%"] or 0.0,
            row["radiance_extent"]["axial_relative_support_m"]["1e-06"] or 0.0,
        )
        for row in rows
    )
    recommended_radial = min(
        _round_up_m(maximum_radial * (1.0 + args.margin_fraction), args.round_up_mm),
        radial_max,
    )
    recommended_axial = min(
        _round_up_m(maximum_axial * (1.0 + args.margin_fraction), args.round_up_mm),
        axial_max,
    )
    radial_pitch = 2.0 * recommended_radial / (config.radial_points - 1)
    axial_pitch = recommended_axial / (config.axial_points - 1)
    baseline_radial_pitch = 2.0 * 0.015 / (config.radial_points - 1)
    baseline_axial_pitch = 0.032 / (config.axial_points - 1)
    output_rows = [
        {key: value for key, value in row.items() if key not in {"image", "x_m", "z_m"}}
        for row in rows
    ]
    report = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "target_time_us": args.time_us,
        "selection": "maximum laser_power_wcm, then maximum rspot at that power",
        "configuration": config.to_dict(),
        "survey_field_of_view_mm": {
            "r_max": radial_max * 1.0e3,
            "x_min": -radial_max * 1.0e3,
            "x_max": radial_max * 1.0e3,
            "z_min": 0.0,
            "z_max": axial_max * 1.0e3,
        },
        "recommended_common_field_of_view_mm": {
            "r_max": recommended_radial * 1.0e3,
            "x_min": -recommended_radial * 1.0e3,
            "x_max": recommended_radial * 1.0e3,
            "z_min": 0.0,
            "z_max": recommended_axial * 1.0e3,
            "margin_fraction": args.margin_fraction,
            "round_up_mm": args.round_up_mm,
        },
        "r3_pixel_pitch_mm": {
            "x": radial_pitch * 1.0e3,
            "z": axial_pitch * 1.0e3,
            "x_ratio_vs_0_5us_baseline": radial_pitch / baseline_radial_pitch,
            "z_ratio_vs_0_5us_baseline": axial_pitch / baseline_axial_pitch,
        },
        "maximum_edge_fractions": {
            "radial": max(row["radiance_extent"]["radial_edge_fraction"] or 0.0 for row in rows),
            "axial_outer": max(
                row["radiance_extent"]["axial_outer_edge_fraction"] or 0.0 for row in rows
            ),
        },
        "elements": output_rows,
    }
    _write_json(output_dir / "fov_survey_report.json", report)
    _write_csv(output_dir / "fov_survey_elements.csv", output_rows)
    _plot_montage(rows, output_dir / "fov_survey_montage.png", args.time_us)
    _plot_extents(rows, output_dir / "fov_survey_extents.png", args.time_us)
    print(json.dumps({key: report[key] for key in report if key != "elements"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
