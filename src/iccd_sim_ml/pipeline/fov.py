"""Selection and radiance-containment utilities for field-of-view surveys."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True)
class RadianceExtent:
    total_pixel_sum: float
    peak: float
    radial_containment_m: dict[str, float | None]
    axial_containment_m: dict[str, float | None]
    radial_relative_support_m: dict[str, float | None]
    axial_relative_support_m: dict[str, float | None]
    radial_edge_fraction: float | None
    axial_outer_edge_fraction: float | None
    nonzero_pixel_fraction: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RadianceMorphology:
    """Simple, auditable screening of a rendered late-time image."""

    classification: str
    eligible_for_typical_fov: bool
    radial_edge_cropped: bool
    axial_edge_cropped: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def select_maximum_condition(simulations: Sequence[Any]) -> Any:
    """Select the joint maximum laser power and spot radius deterministically.

    The simulation grid is expected to contain the joint maximum. If it does
    not, the highest power is selected first and the largest spot available at
    that power is selected second. Duplicate conditions prefer later time
    coverage, then more frames, then the lexical group key.
    """

    usable: list[tuple[Any, float, float]] = []
    for simulation in simulations:
        try:
            power = float(simulation.attrs["laser_power_wcm"])
            radius = float(simulation.attrs["rspot"])
        except (KeyError, TypeError, ValueError):
            continue
        if np.isfinite(power) and power > 0.0 and np.isfinite(radius) and radius > 0.0:
            usable.append((simulation, power, radius))
    if not usable:
        raise ValueError("No simulation has finite positive laser_power_wcm and rspot attributes")
    maximum_power = max(item[1] for item in usable)
    at_maximum_power = [item for item in usable if item[1] == maximum_power]
    maximum_radius = max(item[2] for item in at_maximum_power)
    candidates = [item[0] for item in at_maximum_power if item[2] == maximum_radius]
    return sorted(
        candidates,
        key=lambda item: (
            -float(item.times_s[-1]),
            -len(item.timestep_keys),
            item.key,
        ),
    )[0]


def bracketing_frame_indices(times_s: ArrayLike, target_time_s: float) -> tuple[int, int, float]:
    """Return lower/upper source indices and the upper-frame interpolation weight."""

    times = np.asarray(times_s, dtype=np.float64)
    if times.ndim != 1 or times.size == 0 or not np.isfinite(times).all():
        raise ValueError("times_s must be a finite non-empty one-dimensional array")
    if times.size > 1 and np.any(np.diff(times) <= 0.0):
        raise ValueError("times_s must be strictly increasing")
    target = float(target_time_s)
    if not np.isfinite(target) or target < times[0] or target > times[-1]:
        raise ValueError(
            f"Target {target:.9g} s is outside source coverage [{times[0]:.9g}, {times[-1]:.9g}] s"
        )
    upper = int(np.searchsorted(times, target, side="left"))
    if upper == times.size:
        upper = times.size - 1
    if times[upper] == target:
        return upper, upper, 0.0
    lower = max(upper - 1, 0)
    weight = (target - times[lower]) / (times[upper] - times[lower])
    return lower, upper, float(weight)


def summarize_radiance_extent(
    image: ArrayLike,
    x_m: ArrayLike,
    z_m: ArrayLike,
    *,
    containment: tuple[float, ...] = (0.99, 0.999, 0.9999),
    relative_thresholds: tuple[float, ...] = (1.0e-2, 1.0e-4, 1.0e-6),
    edge_pixels: int = 2,
) -> RadianceExtent:
    """Measure symmetric radial and positive-axial support in a side-view image."""

    values = np.asarray(image, dtype=np.float64)
    x = np.asarray(x_m, dtype=np.float64)
    z = np.asarray(z_m, dtype=np.float64)
    if values.shape != (x.size, z.size) or values.size == 0:
        raise ValueError("image shape must match non-empty x_m and z_m axes")
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError("image must be finite and non-negative")
    if not np.isfinite(x).all() or not np.isfinite(z).all():
        raise ValueError("coordinate axes must be finite")
    if edge_pixels < 1 or edge_pixels * 2 > x.size or edge_pixels > z.size:
        raise ValueError("edge_pixels is incompatible with the image shape")
    for fraction in containment:
        if not 0.0 < fraction <= 1.0:
            raise ValueError("containment fractions must lie in (0, 1]")
    for threshold in relative_thresholds:
        if not 0.0 < threshold <= 1.0:
            raise ValueError("relative thresholds must lie in (0, 1]")

    total = float(np.sum(values))
    peak = float(np.max(values))

    def labels(items: tuple[float, ...]) -> dict[str, float | None]:
        return {f"{100.0 * item:g}%": None for item in items}

    radial_containment = labels(containment)
    axial_containment = labels(containment)
    radial_support = {f"{value:g}": None for value in relative_thresholds}
    axial_support = {f"{value:g}": None for value in relative_thresholds}
    if total > 0.0:
        radial_weights = np.sum(values, axis=1)
        radial_order = np.argsort(np.abs(x), kind="stable")
        radial_cumulative = np.cumsum(radial_weights[radial_order]) / total
        axial_weights = np.sum(values, axis=0)
        axial_order = np.argsort(z, kind="stable")
        axial_cumulative = np.cumsum(axial_weights[axial_order]) / total
        for fraction in containment:
            radial_index = min(int(np.searchsorted(radial_cumulative, fraction)), x.size - 1)
            axial_index = min(int(np.searchsorted(axial_cumulative, fraction)), z.size - 1)
            radial_containment[f"{100.0 * fraction:g}%"] = float(abs(x[radial_order[radial_index]]))
            axial_containment[f"{100.0 * fraction:g}%"] = float(z[axial_order[axial_index]])
        for threshold in relative_thresholds:
            active = values >= peak * threshold
            radial_support[f"{threshold:g}"] = float(np.max(np.abs(x[np.any(active, axis=1)])))
            axial_support[f"{threshold:g}"] = float(np.max(z[np.any(active, axis=0)]))
        radial_edge = float((np.sum(values[:edge_pixels]) + np.sum(values[-edge_pixels:])) / total)
        axial_edge = float(np.sum(values[:, -edge_pixels:]) / total)
    else:
        radial_edge = None
        axial_edge = None
    return RadianceExtent(
        total_pixel_sum=total,
        peak=peak,
        radial_containment_m=radial_containment,
        axial_containment_m=axial_containment,
        radial_relative_support_m=radial_support,
        axial_relative_support_m=axial_support,
        radial_edge_fraction=radial_edge,
        axial_outer_edge_fraction=axial_edge,
        nonzero_pixel_fraction=float(np.count_nonzero(values > 0.0) / values.size),
    )


def assess_radiance_morphology(
    extent: RadianceExtent,
    *,
    slab_aspect_ratio: float = 1.5,
    field_filling_fraction: float = 0.9,
    field_filling_edge_fraction: float = 1.0e-2,
    cropped_edge_fraction: float = 1.0e-3,
) -> RadianceMorphology:
    """Flag slab-like or field-filling emission without calling it a plume.

    This intentionally uses rendered radiance rather than hot-cell or charged-
    density support. Edge-cropping flags are reported separately: a large but
    otherwise plume-shaped image remains useful for robust population
    statistics, while its measured extent is understood to be a lower bound.
    """

    if slab_aspect_ratio <= 1.0:
        raise ValueError("slab_aspect_ratio must exceed one")
    for name, value in (
        ("field_filling_fraction", field_filling_fraction),
        ("field_filling_edge_fraction", field_filling_edge_fraction),
        ("cropped_edge_fraction", cropped_edge_fraction),
    ):
        if not 0.0 < value < 1.0:
            raise ValueError(f"{name} must lie in (0, 1)")

    radial_edge = extent.radial_edge_fraction or 0.0
    axial_edge = extent.axial_outer_edge_fraction or 0.0
    radial_cropped = radial_edge >= cropped_edge_fraction
    axial_cropped = axial_edge >= cropped_edge_fraction
    reasons: list[str] = []
    if extent.total_pixel_sum <= 0.0 or extent.peak <= 0.0:
        classification = "non_emissive"
        reasons.append("zero integrated radiance")
    else:
        radial = extent.radial_containment_m.get("99.9%")
        axial = extent.axial_containment_m.get("99.9%")
        field_filling = (
            extent.nonzero_pixel_fraction >= field_filling_fraction
            and max(radial_edge, axial_edge) >= field_filling_edge_fraction
        )
        slab_like = (
            radial is not None
            and axial is not None
            and axial > 0.0
            and radial > slab_aspect_ratio * axial
        )
        if slab_like:
            classification = "slab_like"
            reasons.append("radial 99.9% extent is much larger than axial extent")
        elif field_filling:
            classification = "field_filling"
            reasons.append("radiance fills the diagnostic window and reaches an edge")
        else:
            classification = "plume_like"
    if radial_cropped:
        reasons.append("radial extent is censored by the survey boundary")
    if axial_cropped:
        reasons.append("axial extent is censored by the survey boundary")
    return RadianceMorphology(
        classification=classification,
        eligible_for_typical_fov=classification == "plume_like",
        radial_edge_cropped=radial_cropped,
        axial_edge_cropped=axial_cropped,
        reasons=tuple(reasons),
    )


__all__ = [
    "RadianceMorphology",
    "RadianceExtent",
    "assess_radiance_morphology",
    "bracketing_frame_indices",
    "select_maximum_condition",
    "summarize_radiance_extent",
]
