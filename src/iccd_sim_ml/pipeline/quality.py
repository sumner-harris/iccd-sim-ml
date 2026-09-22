"""Quality summaries for cached continuum-radiance videos."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RadianceQuality:
    minimum: float
    maximum: float
    mean: float
    positive_pixel_fraction: float
    zero_pixel_fraction: float
    non_emissive_frame_fraction: float
    positive_quantiles: dict[str, float | None]
    all_zero_radiance: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PlasmaWindowQuality:
    maximum_temperature_K: float
    maximum_ne_m3: float
    maximum_n1_m3: float
    maximum_n2_m3: float
    boiling_temperature_K: float | None
    below_boiling_temperature_in_window: bool | None
    no_charged_plasma_in_window: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def summarize_radiance(video: np.ndarray) -> RadianceQuality:
    """Summarize one non-negative cached video without inventing a detector threshold."""

    values = np.asarray(video, dtype=np.float64)
    if values.ndim != 3 or values.size == 0:
        raise ValueError("Cached radiance must have shape (T,H,W)")
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError("Cached radiance must be finite and non-negative")
    positive = values[values > 0.0]
    frame_maximum = np.max(values, axis=(1, 2))
    quantiles: dict[str, float | None]
    if positive.size:
        quantiles = {
            name: float(value)
            for name, value in zip(
                ("p01", "p10", "p50", "p90", "p99"),
                np.quantile(positive, (0.01, 0.10, 0.50, 0.90, 0.99)),
                strict=True,
            )
        }
    else:
        quantiles = {name: None for name in ("p01", "p10", "p50", "p90", "p99")}
    return RadianceQuality(
        minimum=float(np.min(values)),
        maximum=float(np.max(values)),
        mean=float(np.mean(values)),
        positive_pixel_fraction=float(np.count_nonzero(values > 0.0) / values.size),
        zero_pixel_fraction=float(np.count_nonzero(values == 0.0) / values.size),
        non_emissive_frame_fraction=float(np.count_nonzero(frame_maximum == 0.0) / values.shape[0]),
        positive_quantiles=quantiles,
        all_zero_radiance=not bool(positive.size),
    )


def summarize_plasma_window(
    timesteps: Iterable[Any],
    *,
    boiling_temperature_K: float | None,
) -> PlasmaWindowQuality:
    """Summarize plasma state over the source frames used by temporal interpolation."""

    maximum_temperature = 0.0
    maximum_ne = 0.0
    maximum_n1 = 0.0
    maximum_n2 = 0.0
    count = 0
    for timestep in timesteps:
        maximum_temperature = max(maximum_temperature, float(np.max(timestep.temperature_K)))
        maximum_ne = max(maximum_ne, float(np.max(timestep.ne_m3)))
        maximum_n1 = max(maximum_n1, float(np.max(timestep.n1_m3)))
        maximum_n2 = max(maximum_n2, float(np.max(timestep.n2_m3)))
        count += 1
    if count == 0:
        raise ValueError("At least one plasma timestep is required")
    boiling = None if boiling_temperature_K is None else float(boiling_temperature_K)
    below_boiling = None if boiling is None else maximum_temperature < boiling
    return PlasmaWindowQuality(
        maximum_temperature_K=maximum_temperature,
        maximum_ne_m3=maximum_ne,
        maximum_n1_m3=maximum_n1,
        maximum_n2_m3=maximum_n2,
        boiling_temperature_K=boiling,
        below_boiling_temperature_in_window=below_boiling,
        no_charged_plasma_in_window=(maximum_ne <= 0.0 and maximum_n1 <= 0.0 and maximum_n2 <= 0.0),
    )


def sparse_level_stages(
    levels_by_charge: dict[int, Any],
    *,
    minimum_levels: int = 10,
) -> dict[int, int]:
    """Return charge states whose canonical bound-level count is below a declared floor."""

    if minimum_levels < 1:
        raise ValueError("minimum_levels must be positive")
    return {
        int(charge): int(levels.energy_ev.size)
        for charge, levels in sorted(levels_by_charge.items())
        if int(levels.energy_ev.size) < minimum_levels
    }


__all__ = [
    "PlasmaWindowQuality",
    "RadianceQuality",
    "sparse_level_stages",
    "summarize_plasma_window",
    "summarize_radiance",
]
