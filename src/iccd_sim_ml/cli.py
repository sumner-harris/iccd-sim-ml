"""Command-line entry points for inspection and offline image generation."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from iccd_sim_ml import __version__
from iccd_sim_ml.atomic import ContinuumOpacityModel, load_copper_species
from iccd_sim_ml.atomic.opacity import photoionization_effective_cross_sections_m2
from iccd_sim_ml.imaging import (
    ImageSimulation,
    ImagingConfig,
    SequenceSimulation,
    simulate_continuum_image,
    simulate_continuum_sequence,
)
from iccd_sim_ml.io import (
    get_h5_simulation,
    list_h5_simulations,
    load_plasma_timestep,
    select_representative_simulation,
)


def _atomic_models(config: ImagingConfig, reference: Path):
    species = load_copper_species(reference)
    direct = ContinuumOpacityModel.from_momentum_transfer_file(species, reference / "MT_01_01")
    temperature_grid = np.geomspace(
        config.temperature_table_min_K,
        config.temperature_table_max_K,
        config.temperature_table_points,
    )
    lookup = direct.build_lookup(temperature_grid, config.wavelengths_m)
    return direct, lookup


def _base_metadata(config: ImagingConfig, reference: Path) -> dict[str, Any]:
    manifest_path = reference / "manifest.json"
    atomic_manifest = None
    if manifest_path.exists():
        atomic_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "package_version": __version__,
        "quantity": "band_integrated_photon_radiance",
        "units": "photons s^-1 m^-2 sr^-1",
        "detector_spectral_response": "1 at all wavelengths",
        "spatial_optics_applied": False,
        "sensor_effects_applied": False,
        "config": config.to_dict(),
        "atomic_reference": str(reference.resolve()),
        "atomic_manifest": atomic_manifest,
    }


def _relative_summary(calculated: np.ndarray, stored: np.ndarray) -> dict[str, float] | None:
    mask = np.isfinite(stored) & (stored > 0) & np.isfinite(calculated)
    if not np.any(mask):
        return None
    ratio = calculated[mask] / stored[mask]
    relative = np.abs(calculated[mask] - stored[mask]) / stored[mask]
    return {
        "points": int(mask.sum()),
        "median_calculated_to_stored_ratio": float(np.median(ratio)),
        "median_absolute_relative_error": float(np.median(relative)),
        "p95_absolute_relative_error": float(np.percentile(relative, 95.0)),
    }


def _diagnose_248(timestep, direct_model) -> dict[str, Any]:
    result = direct_model.components(
        248e-9,
        timestep.temperature_K,
        timestep.n0_m3,
        timestep.ne_m3,
        timestep.n1_m3,
        timestep.n2_m3,
    )
    pi_reference_usable = "corrupt_alpha_pi_and_mesh_level_export" not in timestep.quality_flags
    photoionization = None
    if pi_reference_usable:
        cross_sections = photoionization_effective_cross_sections_m2(
            direct_model.species,
            248e-9,
            timestep.temperature_K,
            chunk_size=min(direct_model.chunk_size, 8_192),
        )
        densities = {
            0: timestep.n0_m3,
            1: timestep.n1_m3,
            2: timestep.n2_m3,
        }
        true_absorption = np.zeros_like(timestep.temperature_K, dtype=np.float64)
        for charge, cross_section in cross_sections.items():
            density = densities.get(charge)
            if density is not None:
                true_absorption += cross_section * density
        photoionization = _relative_summary(
            true_absorption,
            timestep.alpha_pi_248_m1,
        )
    return {
        "electron_neutral": _relative_summary(
            result.electron_neutral_m1, timestep.alpha_ib_en_248_m1
        ),
        "electron_ion": _relative_summary(result.electron_ion_m1, timestep.alpha_ib_ei_248_m1),
        "photoionization": photoionization,
        "photoionization_reference_usable": pi_reference_usable,
        "photoionization_comparison_kind": "true_absorption_before_lte_net_correction",
    }


def _save_single(path: Path, result: ImageSimulation, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        image_photon_radiance=result.image_photon_radiance,
        spectral_photon_radiance=result.transfer.spectral_photon_radiance,
        optical_depth=result.transfer.optical_depth,
        wavelengths_m=result.transfer.wavelengths_m,
        x_m=result.x_m,
        z_m=result.z_m,
        time_s=np.nan if result.time_s is None else result.time_s,
        metadata_json=json.dumps(metadata, sort_keys=True),
    )


def _save_sequence(path: Path, result: SequenceSimulation, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, Any] = {
        "video": result.images_photon_radiance,
        "times_s": result.times_s,
        "x_m": result.x_m,
        "z_m": result.z_m,
        "metadata_json": json.dumps(metadata, sort_keys=True),
    }
    attrs = metadata.get("simulation_attrs", {})
    condition_names = ("laser_power_wcm", "rspot")
    target_names = (
        "cp_metal",
        "h_vapor",
        "kappa_metal",
        "laser_reflectivity",
        "mass_density_metal",
        "t_boil",
        "tcrit",
    )
    if all(name in attrs for name in condition_names):
        arrays["conditions"] = np.asarray([attrs[name] for name in condition_names])
        metadata["condition_names"] = condition_names
    if all(name in attrs for name in target_names):
        arrays["regression_targets"] = np.asarray([attrs[name] for name in target_names])
        metadata["regression_target_names"] = target_names
    arrays["metadata_json"] = json.dumps(metadata, sort_keys=True)
    np.savez_compressed(path, **arrays)


def _display_scale(image: np.ndarray) -> np.ndarray:
    finite = np.where(np.isfinite(image) & (image > 0), image, 0.0)
    positive = finite[finite > 0]
    if positive.size == 0:
        return finite
    scale = float(np.percentile(positive, 90.0))
    if not scale > 0:
        scale = float(np.max(positive))
    display = np.arcsinh(finite / scale)
    maximum = float(np.max(display))
    return display / maximum if maximum > 0 else display


def _preview_single(path: Path, result: ImageSimulation) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("Preview output requires `pip install iccd-sim-ml[viz]`") from exc
    figure, axis = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    axis.imshow(
        _display_scale(result.image_photon_radiance).T,
        origin="lower",
        aspect="auto",
        extent=(
            result.x_m[0] * 1e3,
            result.x_m[-1] * 1e3,
            result.z_m[0] * 1e3,
            result.z_m[-1] * 1e3,
        ),
        cmap="inferno",
        vmin=0,
        vmax=1,
    )
    axis.set(xlabel="transverse x (mm)", ylabel="axial z (mm)")
    axis.set_title("Continuum photon radiance (normalized preview only)")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _preview_sequence(path: Path, result: SequenceSimulation) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("Preview output requires `pip install iccd-sim-ml[viz]`") from exc
    count = min(12, result.images_photon_radiance.shape[0])
    indices = np.unique(
        np.linspace(0, result.images_photon_radiance.shape[0] - 1, count).astype(int)
    )
    figure, axes = plt.subplots(3, 4, figsize=(13, 8), constrained_layout=True)
    for axis in axes.ravel():
        axis.set_visible(False)
    for axis, index in zip(axes.ravel(), indices, strict=False):
        axis.set_visible(True)
        axis.imshow(
            _display_scale(result.images_photon_radiance[index]).T,
            origin="lower",
            aspect="auto",
            extent=(
                result.x_m[0] * 1e3,
                result.x_m[-1] * 1e3,
                result.z_m[0] * 1e3,
                result.z_m[-1] * 1e3,
            ),
            cmap="inferno",
            vmin=0,
            vmax=1,
        )
        time_ns = result.times_s[index] * 1e9
        axis.set_title(f"frame {index}, {time_ns:.4g} ns")
        axis.set(xlabel="x (mm)", ylabel="z (mm)")
    figure.suptitle("Continuum sequence (each frame normalized for display only)")
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _add_simulation_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--atomic-reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--preview",
        type=Path,
        help="Optional normalized PNG preview; raw radiance remains unchanged in the NPZ",
    )


def _inspect_h5(args: argparse.Namespace) -> int:
    simulations = list_h5_simulations(args.path)
    payload = {
        "path": str(args.path.resolve()),
        "simulation_count": len(simulations),
        "simulations": [
            {
                "key": item.key,
                "frame_count": len(item.timestep_keys),
                "first_time_ns": float(item.times_s[0] * 1e9),
                "last_time_ns": float(item.times_s[-1] * 1e9),
                "attrs": item.attrs,
                "index_warnings": item.index_warnings,
            }
            for item in simulations
        ],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _simulate_timestep(args: argparse.Namespace) -> int:
    config = ImagingConfig.from_json(args.config)
    reference = args.atomic_reference.resolve()
    timestep = load_plasma_timestep(args.path)
    direct, lookup = _atomic_models(config, reference)
    print(f"Simulating {timestep.source.name} ({timestep.size:,} AMR cells)...", flush=True)
    result = simulate_continuum_image(timestep, config, lookup)
    metadata = _base_metadata(config, reference)
    metadata.update(
        {
            "source": str(timestep.source),
            "source_key": timestep.source_key,
            "quality_flags": timestep.quality_flags,
        }
    )
    if args.validate_248:
        print("Comparing recalculated and saved 248-nm opacity...", flush=True)
        metadata["validation_248_nm"] = _diagnose_248(timestep, direct)
    _save_single(args.output, result, metadata)
    if args.preview is not None:
        _preview_single(args.preview, result)
    print(
        f"Saved {args.output} with image range "
        f"[{result.image_photon_radiance.min():.6g}, {result.image_photon_radiance.max():.6g}]",
        flush=True,
    )
    return 0


def _simulate_sequence(args: argparse.Namespace) -> int:
    config = ImagingConfig.from_json(args.config)
    reference = args.atomic_reference.resolve()
    simulations = list_h5_simulations(args.path)
    if args.simulation is None:
        simulation = select_representative_simulation(simulations)
    else:
        simulation = get_h5_simulation(args.path, args.simulation)
    indices = list(range(0, len(simulation.timestep_keys), args.stride))
    if args.max_frames is not None:
        indices = indices[: args.max_frames]
    if not indices:
        raise ValueError("Frame selection is empty")
    _, lookup = _atomic_models(config, reference)
    total = len(indices)
    print(
        f"Simulating {total} frame(s) from {simulation.key!r}; "
        f"rspot={simulation.attrs.get('rspot')}, "
        f"laser_power_wcm={simulation.attrs.get('laser_power_wcm')}",
        flush=True,
    )

    def progress(index: int, frame: ImageSimulation) -> None:
        time_ns = np.nan if frame.time_s is None else frame.time_s * 1e9
        print(f"  [{index + 1}/{total}] {time_ns:.6g} ns", flush=True)

    result = simulate_continuum_sequence(
        simulation.iter_timesteps(indices), config, lookup, progress=progress
    )
    metadata = _base_metadata(config, reference)
    metadata.update(
        {
            "source": str(simulation.source),
            "simulation_key": simulation.key,
            "simulation_attrs": simulation.attrs,
            "source_timestep_keys": [simulation.timestep_keys[index] for index in indices],
            "source_index_warnings": simulation.index_warnings,
            "quality_flags_by_frame": result.quality_flags_by_frame,
            "frame_stride": args.stride,
        }
    )
    _save_sequence(args.output, result, metadata)
    if args.preview is not None:
        _preview_sequence(args.preview, result)
    print(f"Saved {result.images_photon_radiance.shape[0]} frames to {args.output}", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="iccd-sim", description="Idealized LTE continuum images from plasma simulations"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect-h5", help="Index an HDF5 container")
    inspect_parser.add_argument("path", type=Path)
    inspect_parser.set_defaults(function=_inspect_h5)

    timestep_parser = subparsers.add_parser("timestep", help="Simulate one .dat frame")
    timestep_parser.add_argument("path", type=Path)
    _add_simulation_arguments(timestep_parser)
    timestep_parser.add_argument(
        "--validate-248", action="store_true", help="Compare saved and recomputed laser opacity"
    )
    timestep_parser.set_defaults(function=_simulate_timestep)

    sequence_parser = subparsers.add_parser("sequence", help="Simulate one HDF5 group")
    sequence_parser.add_argument("path", type=Path)
    _add_simulation_arguments(sequence_parser)
    sequence_parser.add_argument("--simulation", help="Exact HDF5 group key")
    sequence_parser.add_argument("--stride", type=int, default=1)
    sequence_parser.add_argument("--max-frames", type=int)
    sequence_parser.set_defaults(function=_simulate_sequence)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "stride", 1) < 1:
        parser.error("--stride must be positive")
    if getattr(args, "max_frames", None) is not None and args.max_frames < 1:
        parser.error("--max-frames must be positive")
    return int(args.function(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
