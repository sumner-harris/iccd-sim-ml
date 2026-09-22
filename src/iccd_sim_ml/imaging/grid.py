"""Resampling and ray geometry for axisymmetric solver state."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator, RegularGridInterpolator
from scipy.spatial import Delaunay

from iccd_sim_ml.io import PlasmaTimestep

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class AxisymmetricGrid:
    r_m: FloatArray
    z_m: FloatArray
    temperature_K: FloatArray
    n0_m3: FloatArray
    ne_m3: FloatArray
    n1_m3: FloatArray
    n2_m3: FloatArray
    alpha_ib_en_248_m1: FloatArray
    alpha_ib_ei_248_m1: FloatArray

    def fields(self) -> dict[str, FloatArray]:
        return {
            "temperature_K": self.temperature_K,
            "n0_m3": self.n0_m3,
            "ne_m3": self.ne_m3,
            "n1_m3": self.n1_m3,
            "n2_m3": self.n2_m3,
        }


@dataclass(frozen=True)
class SideViewState:
    x_m: FloatArray
    y_m: FloatArray
    z_m: FloatArray
    path_length_m: float
    temperature_K: FloatArray
    n0_m3: FloatArray
    ne_m3: FloatArray
    n1_m3: FloatArray
    n2_m3: FloatArray

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.temperature_K.shape


def infer_dyadic_domain_max(coordinate_m: FloatArray) -> float:
    """Infer the outer Basilisk domain edge from dyadic AMR cell centres."""

    values = np.asarray(coordinate_m, dtype=np.float64)
    positive = values[np.isfinite(values) & (values > 0)]
    if positive.size == 0:
        raise ValueError("Cannot infer a positive domain from the coordinates")
    smallest_center = float(np.min(positive))
    maximum_center = float(np.max(positive))
    ratio = maximum_center / smallest_center
    candidate = smallest_center * 2.0 ** np.ceil(np.log2(max(ratio, 1.0)))
    if candidate >= maximum_center and candidate <= 1.25 * maximum_center:
        return float(candidate)
    unique = np.unique(positive)
    step = np.min(np.diff(unique)) if unique.size > 1 else smallest_center * 2.0
    return maximum_center + 0.5 * float(step)


def _interpolate_fields(
    points: FloatArray, targets: FloatArray, fields: dict[str, FloatArray]
) -> dict[str, FloatArray]:
    triangulation = Delaunay(points)
    nearest = None
    result: dict[str, FloatArray] = {}
    for name, values in fields.items():
        interpolated = np.asarray(
            LinearNDInterpolator(triangulation, values, fill_value=np.nan)(targets),
            dtype=np.float64,
        )
        missing = ~np.isfinite(interpolated)
        if np.any(missing):
            nearest = NearestNDInterpolator(points, values)
            interpolated[missing] = nearest(targets[missing])
        result[name] = np.maximum(interpolated, 0.0)
    return result


def resample_timestep(
    timestep: PlasmaTimestep,
    *,
    radial_points: int,
    axial_points: int,
    radial_max_m: float | None = None,
    axial_max_m: float | None = None,
) -> AxisymmetricGrid:
    """Resample adaptive cell centres onto a common r-z grid.

    Nearest interpolation is used only to extend the linear interpolant from
    outermost cell centres to the inferred physical domain edges. Vacuum is
    imposed later for cylindrical radius beyond ``radial_max_m``.
    """

    r_max = infer_dyadic_domain_max(timestep.r_m) if radial_max_m is None else radial_max_m
    z_max = infer_dyadic_domain_max(timestep.z_m) if axial_max_m is None else axial_max_m
    r_axis = np.linspace(0.0, r_max, radial_points, dtype=np.float64)
    z_axis = np.linspace(0.0, z_max, axial_points, dtype=np.float64)
    radial_grid, axial_grid = np.meshgrid(r_axis, z_axis, indexing="ij")
    targets = np.column_stack((radial_grid.ravel(), axial_grid.ravel()))
    points = np.column_stack((timestep.r_m, timestep.z_m))
    values = _interpolate_fields(
        points,
        targets,
        {
            "temperature_K": timestep.temperature_K,
            "n0_m3": timestep.n0_m3,
            "ne_m3": timestep.ne_m3,
            "n1_m3": timestep.n1_m3,
            "n2_m3": timestep.n2_m3,
            "alpha_ib_en_248_m1": timestep.alpha_ib_en_248_m1,
            "alpha_ib_ei_248_m1": timestep.alpha_ib_ei_248_m1,
        },
    )
    shaped = {name: item.reshape(radial_points, axial_points) for name, item in values.items()}
    return AxisymmetricGrid(r_m=r_axis, z_m=z_axis, **shaped)


def build_parallel_side_view(
    grid: AxisymmetricGrid, *, x_points: int, y_points: int
) -> SideViewState:
    """Sample an axisymmetric state along parallel side-view rays."""

    radius = float(grid.r_m[-1])
    x_axis = np.linspace(-radius, radius, x_points, dtype=np.float64)
    path_length = 2.0 * radius / y_points
    y_axis = np.linspace(
        -radius + 0.5 * path_length,
        radius - 0.5 * path_length,
        y_points,
        dtype=np.float64,
    )
    x_grid, z_grid = np.meshgrid(x_axis, grid.z_m, indexing="ij")
    volumes = {
        name: np.empty((y_points, x_points, grid.z_m.size), dtype=np.float64)
        for name in grid.fields()
    }
    interpolators = {
        name: RegularGridInterpolator(
            (grid.r_m, grid.z_m), values, bounds_error=False, fill_value=0.0
        )
        for name, values in grid.fields().items()
    }
    for index, y_value in enumerate(y_axis):
        radial = np.sqrt(x_grid**2 + y_value**2)
        query = np.column_stack((radial.ravel(), z_grid.ravel()))
        for name, interpolator in interpolators.items():
            volumes[name][index] = np.maximum(
                np.asarray(interpolator(query)).reshape(x_grid.shape), 0.0
            )
    return SideViewState(
        x_m=x_axis,
        y_m=y_axis,
        z_m=grid.z_m,
        path_length_m=path_length,
        **volumes,
    )
