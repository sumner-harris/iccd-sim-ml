"""Run a coupled wavelength/image/line-of-sight convergence sweep."""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from scipy.interpolate import RegularGridInterpolator

from iccd_sim_ml.atomic import ContinuumOpacityModel, load_copper_species
from iccd_sim_ml.imaging import ImagingConfig, simulate_continuum_image
from iccd_sim_ml.io import get_h5_simulation


@dataclass(frozen=True)
class Resolution:
    name: str
    wavelengths: int
    image_points: int
    line_of_sight_points: int


PROFILES = (
    Resolution("r0", 12, 32, 32),
    Resolution("r1", 24, 48, 64),
    Resolution("r2", 36, 64, 96),
    Resolution("r3", 48, 96, 128),
    Resolution("r4", 64, 128, 192),
)


def _integrated_image(image: np.ndarray, x_m: np.ndarray, z_m: np.ndarray) -> float:
    integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(integrate(integrate(image, z_m, axis=1), x_m, axis=0))


def _onto_reference(
    image: np.ndarray,
    x_m: np.ndarray,
    z_m: np.ndarray,
    reference_x_m: np.ndarray,
    reference_z_m: np.ndarray,
) -> np.ndarray:
    interpolator = RegularGridInterpolator((x_m, z_m), image, bounds_error=True)
    x_grid, z_grid = np.meshgrid(reference_x_m, reference_z_m, indexing="ij")
    return interpolator(np.column_stack((x_grid.ravel(), z_grid.ravel()))).reshape(x_grid.shape)


def _plot(results: list[dict[str, Any]], destination: Path, time_ns: float) -> None:
    maximum = max(float(np.max(item["image"])) for item in results)
    norm = LogNorm(vmin=maximum * 1.0e-12, vmax=maximum, clip=True)
    figure, axes = plt.subplots(
        2,
        len(results),
        figsize=(3.25 * len(results), 6.4),
        constrained_layout=True,
        squeeze=False,
    )
    shared_image = None
    for column, item in enumerate(results):
        extent = (
            item["x_m"][0] * 1e3,
            item["x_m"][-1] * 1e3,
            item["z_m"][0] * 1e3,
            item["z_m"][-1] * 1e3,
        )
        shared_image = axes[0, column].imshow(
            np.maximum(item["image"], maximum * 1.0e-12).T,
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap="inferno",
            norm=norm,
        )
        peak = float(np.max(item["image"]))
        axes[1, column].imshow(
            (item["image"] / peak).T,
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap="inferno",
            vmin=0.0,
            vmax=1.0,
        )
        profile = item["profile"]
        title = (
            f"{profile.name}: {profile.wavelengths} wavelengths\n"
            f"{profile.image_points}x{profile.image_points}, "
            f"{profile.line_of_sight_points} LOS"
        )
        axes[0, column].set_title(title, fontsize=9)
        axes[1, column].set_title(
            f"normalized; peak={peak:.3e}\n{item['runtime_s']:.1f} s", fontsize=9
        )
        for row in range(2):
            axes[row, column].set_xlabel("x (mm)")
            axes[row, column].set_ylabel("z (mm)")
    if shared_image is not None:
        colorbar = figure.colorbar(shared_image, ax=axes[0].tolist(), shrink=0.82, pad=0.01)
        colorbar.set_label("photon radiance (photons s$^{-1}$ m$^{-2}$ sr$^{-1}$)")
    figure.suptitle(
        f"Cu plume at {time_ns:g} ns: numerical-resolution sweep\n"
        "top: shared absolute log scale; bottom: independently normalized morphology"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=190)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("h5", type=Path)
    parser.add_argument("--simulation", required=True)
    parser.add_argument("--time-ns", type=float, required=True)
    parser.add_argument("--atomic-reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--radial-max-mm", type=float, default=15.0)
    parser.add_argument("--axial-max-mm", type=float, default=32.0)
    args = parser.parse_args()

    simulation = get_h5_simulation(args.h5, args.simulation)
    times_ns = simulation.times_s * 1.0e9
    index = int(np.argmin(np.abs(times_ns - args.time_ns)))
    if not np.isclose(times_ns[index], args.time_ns, rtol=0.0, atol=1.0e-6):
        raise ValueError(
            f"Requested {args.time_ns:g} ns; nearest available frame is {times_ns[index]:g} ns"
        )
    timestep = next(simulation.iter_timesteps([index]))

    reference = args.atomic_reference.expanduser().resolve()
    species = load_copper_species(reference)
    direct = ContinuumOpacityModel.from_momentum_transfer_file(species, reference / "MT_01_01")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []

    for profile in PROFILES:
        config = ImagingConfig(
            wavelength_min_nm=300.0,
            wavelength_max_nm=800.0,
            wavelength_points=profile.wavelengths,
            radial_points=profile.image_points,
            axial_points=profile.image_points,
            line_of_sight_points=profile.line_of_sight_points,
            radial_max_m=args.radial_max_mm * 1.0e-3,
            axial_max_m=args.axial_max_mm * 1.0e-3,
            temperature_table_min_K=300.0,
            temperature_table_max_K=100_000.0,
            temperature_table_points=160,
        )
        print(
            f"{profile.name}: {profile.wavelengths} wavelengths, "
            f"{profile.image_points}x{profile.image_points}, "
            f"{profile.line_of_sight_points} LOS",
            flush=True,
        )
        started = time.perf_counter()
        temperature_grid = np.geomspace(
            config.temperature_table_min_K,
            config.temperature_table_max_K,
            config.temperature_table_points,
        )
        lookup = direct.build_lookup(temperature_grid, config.wavelengths_m)
        simulated = simulate_continuum_image(timestep, config, lookup)
        runtime_s = time.perf_counter() - started
        image = simulated.image_photon_radiance
        metadata = {
            "source": str(simulation.source),
            "simulation_key": simulation.key,
            "source_timestep_key": simulation.timestep_keys[index],
            "time_ns": float(times_ns[index]),
            "profile": asdict(profile),
            "config": config.to_dict(),
            "runtime_s": runtime_s,
            "units": "photons s^-1 m^-2 sr^-1",
        }
        np.savez_compressed(
            output_dir / f"Cu_3006ns_{profile.name}.npz",
            image_photon_radiance=image,
            x_m=simulated.x_m,
            z_m=simulated.z_m,
            wavelengths_m=config.wavelengths_m,
            metadata_json=json.dumps(metadata, sort_keys=True),
        )
        results.append(
            {
                "profile": profile,
                "image": image,
                "x_m": simulated.x_m,
                "z_m": simulated.z_m,
                "runtime_s": runtime_s,
                "integrated_radiance_area": _integrated_image(image, simulated.x_m, simulated.z_m),
            }
        )
        print(f"  completed in {runtime_s:.2f} s; peak={np.max(image):.6e}", flush=True)

    reference_result = results[-1]
    reference_image = reference_result["image"]
    reference_norm = float(np.linalg.norm(reference_image))
    reference_integral = reference_result["integrated_radiance_area"]
    bright = reference_image > float(np.max(reference_image)) * 1.0e-8
    summary: list[dict[str, Any]] = []
    for item in results:
        on_reference = _onto_reference(
            item["image"],
            item["x_m"],
            item["z_m"],
            reference_result["x_m"],
            reference_result["z_m"],
        )
        difference = on_reference - reference_image
        log_difference = np.log10(np.maximum(on_reference[bright], 1.0e-300)) - np.log10(
            reference_image[bright]
        )
        summary.append(
            {
                **asdict(item["profile"]),
                "runtime_s": item["runtime_s"],
                "peak_photon_radiance": float(np.max(item["image"])),
                "integrated_radiance_area": item["integrated_radiance_area"],
                "relative_integrated_difference_vs_r4": (
                    item["integrated_radiance_area"] / reference_integral - 1.0
                ),
                "relative_l2_image_error_vs_r4": float(np.linalg.norm(difference) / reference_norm),
                "bright_region_log10_rmse_vs_r4": float(np.sqrt(np.mean(log_difference**2))),
            }
        )
    payload = {
        "source": str(simulation.source),
        "simulation_key": simulation.key,
        "source_timestep_key": simulation.timestep_keys[index],
        "time_ns": float(times_ns[index]),
        "field_of_view_mm": {
            "x_min": -args.radial_max_mm,
            "x_max": args.radial_max_mm,
            "z_min": 0.0,
            "z_max": args.axial_max_mm,
        },
        "reference_profile": PROFILES[-1].name,
        "profiles": summary,
    }
    (output_dir / "resolution_sweep_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    with (output_dir / "resolution_sweep_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    _plot(results, output_dir / "Cu_3006ns_resolution_sweep.png", float(times_ns[index]))
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
