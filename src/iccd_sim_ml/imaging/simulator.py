"""High-level single-frame and sequence simulation orchestration."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from iccd_sim_ml.io import PlasmaTimestep

from .config import ImagingConfig
from .grid import AxisymmetricGrid, build_parallel_side_view, resample_timestep
from .transfer import OpacityModel, SelectedOpacityModel, TransferResult, solve_lte_continuum

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ImageSimulation:
    time_s: float | None
    x_m: FloatArray
    z_m: FloatArray
    transfer: TransferResult
    quality_flags: tuple[str, ...]

    @property
    def image_photon_radiance(self) -> FloatArray:
        return self.transfer.image_photon_radiance


@dataclass(frozen=True)
class SequenceSimulation:
    images_photon_radiance: FloatArray
    times_s: FloatArray
    x_m: FloatArray
    z_m: FloatArray
    quality_flags_by_frame: tuple[tuple[str, ...], ...]


def simulate_continuum_image(
    timestep: PlasmaTimestep,
    config: ImagingConfig,
    opacity_model: OpacityModel,
    *,
    grid: AxisymmetricGrid | None = None,
) -> ImageSimulation:
    """Create one idealized side-on continuum photon-radiance image."""

    axisymmetric = grid or resample_timestep(
        timestep,
        radial_points=config.radial_points,
        axial_points=config.axial_points,
        radial_max_m=config.radial_max_m,
        axial_max_m=config.axial_max_m,
    )
    state = build_parallel_side_view(
        axisymmetric, x_points=config.radial_points, y_points=config.line_of_sight_points
    )
    selected_opacity = SelectedOpacityModel(
        opacity_model,
        include_photoionization=config.include_photoionization,
        include_electron_neutral=config.include_electron_neutral_inverse_bremsstrahlung,
        include_electron_ion=config.include_electron_ion_inverse_bremsstrahlung,
    )
    transfer = solve_lte_continuum(state, config.wavelengths_m, selected_opacity)
    return ImageSimulation(
        time_s=timestep.time_s,
        x_m=state.x_m,
        z_m=state.z_m,
        transfer=transfer,
        quality_flags=timestep.quality_flags,
    )


def simulate_continuum_sequence(
    timesteps: Iterable[PlasmaTimestep],
    config: ImagingConfig,
    opacity_model: OpacityModel,
    *,
    progress: Callable[[int, ImageSimulation], None] | None = None,
) -> SequenceSimulation:
    """Simulate a sequence on one explicitly declared, fixed spatial grid."""

    if config.radial_max_m is None or config.axial_max_m is None:
        raise ValueError(
            "Sequence simulation requires explicit radial_max_m and axial_max_m; "
            "inferring them from frame zero can crop an expanding plume"
        )

    iterator = iter(timesteps)
    try:
        first = next(iterator)
    except StopIteration as exc:
        raise ValueError("At least one timestep is required") from exc
    first_grid = resample_timestep(
        first,
        radial_points=config.radial_points,
        axial_points=config.axial_points,
        radial_max_m=config.radial_max_m,
        axial_max_m=config.axial_max_m,
    )
    first_frame = simulate_continuum_image(first, config, opacity_model, grid=first_grid)
    images = [first_frame.image_photon_radiance]
    times = [np.nan if first_frame.time_s is None else first_frame.time_s]
    flags = [first_frame.quality_flags]
    if progress is not None:
        progress(0, first_frame)
    radial_max = config.radial_max_m
    axial_max = config.axial_max_m
    for timestep in iterator:
        grid = resample_timestep(
            timestep,
            radial_points=config.radial_points,
            axial_points=config.axial_points,
            radial_max_m=radial_max,
            axial_max_m=axial_max,
        )
        frame = simulate_continuum_image(timestep, config, opacity_model, grid=grid)
        images.append(frame.image_photon_radiance)
        times.append(np.nan if frame.time_s is None else frame.time_s)
        flags.append(frame.quality_flags)
        if progress is not None:
            progress(len(images) - 1, frame)
    return SequenceSimulation(
        images_photon_radiance=np.stack(images),
        times_s=np.asarray(times, dtype=np.float64),
        x_m=first_frame.x_m,
        z_m=first_frame.z_m,
        quality_flags_by_frame=tuple(flags),
    )
