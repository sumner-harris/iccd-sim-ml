"""Render comparable late-time continuum sequences for selected elements."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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
from iccd_sim_ml.io import list_h5_simulations  # noqa: E402
from iccd_sim_ml.pipeline import bracketing_frame_indices, select_maximum_condition  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--atomic-reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--elements", nargs="+", required=True)
    parser.add_argument("--times-us", nargs="+", type=float, required=True)
    parser.add_argument("--radial-mm", type=float, default=15.0)
    parser.add_argument("--axial-mm", type=float, default=32.0)
    return parser.parse_args()


def _render_element(
    data_dir: Path,
    atomic_reference: Path,
    element: str,
    target_times_s: np.ndarray,
    config: ImagingConfig,
) -> tuple[np.ndarray, dict[str, object]]:
    matches = sorted(data_dir.glob(f"{element}_*.h5"))
    if len(matches) != 1:
        raise ValueError(f"Expected one {element}_*.h5 file, found {matches}")
    simulation = select_maximum_condition(list_h5_simulations(matches[0]))
    if target_times_s[0] < simulation.times_s[0] or target_times_s[-1] > simulation.times_s[-1]:
        raise ValueError(
            f"{element} covers {simulation.times_s[0] * 1e6:g}--"
            f"{simulation.times_s[-1] * 1e6:g} us, not all requested times"
        )

    brackets = [bracketing_frame_indices(simulation.times_s, value) for value in target_times_s]
    indices = sorted({index for lower, upper, _ in brackets for index in (lower, upper)})
    timesteps = dict(zip(indices, simulation.iter_timesteps(indices), strict=True))
    atomic = AtomicDataCatalog(atomic_reference).load(element, mode="strict")
    temperature_grid = np.geomspace(
        config.temperature_table_min_K,
        config.temperature_table_max_K,
        config.temperature_table_points,
    )
    wavelength_grid_m = (
        np.linspace(
            config.wavelength_min_nm,
            config.wavelength_max_nm,
            config.wavelength_points,
        )
        * 1.0e-9
    )
    opacity = atomic.build_opacity_model().build_lookup(temperature_grid, wavelength_grid_m)
    rendered = {
        index: simulate_continuum_image(timesteps[index], config, opacity).image_photon_radiance
        for index in indices
    }
    frames = []
    sources = []
    for lower, upper, weight in brackets:
        image = (
            rendered[lower]
            if lower == upper
            else (1.0 - weight) * rendered[lower] + weight * rendered[upper]
        )
        frames.append(image)
        sources.append(
            {
                "lower": simulation.timestep_keys[lower],
                "upper": simulation.timestep_keys[upper],
                "upper_weight": weight,
            }
        )
    metadata: dict[str, object] = {
        "element": element,
        "source": str(matches[0]),
        "simulation_key": simulation.key,
        "laser_power_wcm": float(simulation.attrs["laser_power_wcm"]),
        "rspot_m": float(simulation.attrs["rspot"]),
        "sources": sources,
        "atomic_data": atomic.fingerprint(),
    }
    return np.stack(frames), metadata


def main() -> int:
    args = _parse_args()
    if any(value < 0.0 for value in args.times_us):
        raise ValueError("--times-us must be non-negative")
    target_times_us = np.asarray(sorted(set(args.times_us)), dtype=np.float64)
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
        radial_max_m=args.radial_mm * 1.0e-3,
        axial_max_m=args.axial_mm * 1.0e-3,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for element in args.elements:
        print(f"Rendering {element}...", flush=True)
        frames, metadata = _render_element(
            args.data_dir,
            args.atomic_reference,
            element,
            target_times_us * 1.0e-6,
            config,
        )
        rows.append((element, frames, metadata))
        np.savez_compressed(
            args.output_dir / f"{element}_sequence.npz",
            video_photon_radiance=frames,
            times_s=target_times_us * 1.0e-6,
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        )

    figure, axes = plt.subplots(
        len(rows),
        target_times_us.size,
        figsize=(2.5 * target_times_us.size, 2.6 * len(rows)),
        squeeze=False,
        constrained_layout=True,
    )
    for row_index, (element, frames, metadata) in enumerate(rows):
        positive = frames[frames > 0.0]
        normalization = None
        if positive.size:
            normalization = LogNorm(
                vmax=float(np.max(positive)),
                vmin=max(float(np.max(positive)) * 1.0e-6, float(np.min(positive))),
            )
        for column_index, target_us in enumerate(target_times_us):
            axis = axes[row_index, column_index]
            axis.imshow(
                frames[column_index].T,
                origin="lower",
                aspect="auto",
                extent=(-args.radial_mm, args.radial_mm, 0.0, args.axial_mm),
                cmap="inferno",
                norm=normalization,
            )
            axis.set_title(f"{element}, {target_us:g} us")
            axis.set_xlabel("x / mm")
            if column_index == 0:
                axis.set_ylabel("z / mm")
        axes[row_index, 0].text(
            0.02,
            0.98,
            f"{metadata['laser_power_wcm']:.2g} W/cm2\nrspot={metadata['rspot_m'] * 1e3:g} mm",
            transform=axes[row_index, 0].transAxes,
            va="top",
            color="white",
            fontsize=8,
        )
    figure.suptitle(
        "Maximum-power / maximum-spot continuum sequence\nshared log scale within each element"
    )
    figure.savefig(args.output_dir / "plume_sequence_comparison.png", dpi=180)
    plt.close(figure)
    (args.output_dir / "comparison_metadata.json").write_text(
        json.dumps(
            {
                "times_us": target_times_us.tolist(),
                "configuration": config.to_dict(),
                "elements": [metadata for _, _, metadata in rows],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
