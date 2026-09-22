"""Configuration for continuum image synthesis."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ImagingConfig:
    wavelength_min_nm: float = 300.0
    wavelength_max_nm: float = 800.0
    wavelength_points: int = 48
    radial_points: int = 64
    axial_points: int = 64
    line_of_sight_points: int = 96
    temperature_table_min_K: float = 300.0
    temperature_table_max_K: float = 100_000.0
    temperature_table_points: int = 160
    detector_response: str = "flat"
    include_photoionization: bool = True
    include_electron_neutral_inverse_bremsstrahlung: bool = True
    include_electron_ion_inverse_bremsstrahlung: bool = True
    apply_spatial_optics: bool = False
    apply_sensor_effects: bool = False
    radial_max_m: float | None = None
    axial_max_m: float | None = None

    def __post_init__(self) -> None:
        if not (0 < self.wavelength_min_nm < self.wavelength_max_nm):
            raise ValueError("Wavelength bounds must be positive and increasing")
        for name in (
            "wavelength_points",
            "radial_points",
            "axial_points",
            "line_of_sight_points",
            "temperature_table_points",
        ):
            if getattr(self, name) < 2:
                raise ValueError(f"{name} must be at least two")
        if not (0 < self.temperature_table_min_K < self.temperature_table_max_K):
            raise ValueError("Temperature lookup bounds must be positive and increasing")
        if self.detector_response != "flat":
            raise ValueError("Only the requested flat spectral response is currently supported")
        if self.apply_spatial_optics or self.apply_sensor_effects:
            raise NotImplementedError(
                "Spatial optics and sensor effects are deliberately disabled in this model stage"
            )
        for name in ("radial_max_m", "axial_max_m"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive when supplied")

    @property
    def wavelengths_m(self) -> np.ndarray:
        return np.linspace(
            self.wavelength_min_nm * 1e-9,
            self.wavelength_max_nm * 1e-9,
            self.wavelength_points,
            dtype=np.float64,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> ImagingConfig:
        known = set(cls.__dataclass_fields__)
        unknown = sorted(set(values) - known)
        if unknown:
            raise ValueError(f"Unknown imaging configuration fields: {', '.join(unknown)}")
        return cls(**dict(values))

    @classmethod
    def from_json(cls, path: str | Path) -> ImagingConfig:
        source = Path(path).expanduser().resolve()
        with source.open("r", encoding="utf-8") as stream:
            return cls.from_mapping(json.load(stream))
