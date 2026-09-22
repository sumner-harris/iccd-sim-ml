"""LTE continuum radiative transfer along parallel rays."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from .blackbody import planck_photon_radiance_lambda
from .detector import flat_spectral_response
from .grid import SideViewState

FloatArray = NDArray[np.float64]


class OpacityResult(Protocol):
    electron_neutral_m1: FloatArray
    electron_ion_m1: FloatArray
    photoionization_m1: FloatArray

    @property
    def total_m1(self) -> FloatArray: ...


class OpacityModel(Protocol):
    def components(
        self,
        wavelength_m: float,
        temperature_K: FloatArray,
        n0_m3: FloatArray,
        ne_m3: FloatArray,
        n1_m3: FloatArray,
        n2_m3: FloatArray,
    ) -> OpacityResult: ...


@dataclass(frozen=True)
class _SelectedComponents:
    electron_neutral_m1: FloatArray
    electron_ion_m1: FloatArray
    photoionization_m1: FloatArray

    @property
    def total_m1(self) -> FloatArray:
        return self.electron_neutral_m1 + self.electron_ion_m1 + self.photoionization_m1


@dataclass(frozen=True)
class SelectedOpacityModel:
    """Apply explicit component switches while retaining one opacity API."""

    base: OpacityModel
    include_photoionization: bool = True
    include_electron_neutral: bool = True
    include_electron_ion: bool = True

    def components(
        self,
        wavelength_m: float,
        temperature_K: FloatArray,
        n0_m3: FloatArray,
        ne_m3: FloatArray,
        n1_m3: FloatArray,
        n2_m3: FloatArray,
    ) -> _SelectedComponents:
        result = self.base.components(wavelength_m, temperature_K, n0_m3, ne_m3, n1_m3, n2_m3)
        zero = np.zeros_like(result.total_m1)
        return _SelectedComponents(
            result.electron_neutral_m1 if self.include_electron_neutral else zero,
            result.electron_ion_m1 if self.include_electron_ion else zero,
            result.photoionization_m1 if self.include_photoionization else zero,
        )


@dataclass(frozen=True)
class TransferResult:
    image_photon_radiance: FloatArray
    spectral_photon_radiance: FloatArray
    optical_depth: FloatArray
    wavelengths_m: FloatArray


def formal_solution_step(
    incoming: FloatArray, source: FloatArray, opacity_m1: FloatArray, path_length_m: float
) -> FloatArray:
    """Exact constant-cell formal solution with stable optically-thin behavior."""

    optical_depth = np.maximum(opacity_m1, 0.0) * path_length_m
    absorbed_fraction = -np.expm1(-np.minimum(optical_depth, 745.0))
    transmission = np.exp(-np.minimum(optical_depth, 745.0))
    return incoming * transmission + source * absorbed_fraction


def solve_lte_continuum(
    state: SideViewState, wavelengths_m: FloatArray, opacity_model: OpacityModel
) -> TransferResult:
    """Solve side-on LTE transfer and integrate the flat-response photon band."""

    wavelengths = np.asarray(wavelengths_m, dtype=np.float64)
    if wavelengths.ndim != 1 or wavelengths.size < 2 or np.any(np.diff(wavelengths) <= 0):
        raise ValueError("Wavelengths must be a one-dimensional increasing grid")
    spectral = np.empty((wavelengths.size, state.x_m.size, state.z_m.size), dtype=np.float64)
    optical_depth = np.empty_like(spectral)
    for wavelength_index, wavelength in enumerate(wavelengths):
        opacity = opacity_model.components(
            float(wavelength),
            state.temperature_K,
            state.n0_m3,
            state.ne_m3,
            state.n1_m3,
            state.n2_m3,
        ).total_m1
        if not np.isfinite(opacity).all() or np.any(opacity < 0):
            raise FloatingPointError("Opacity model returned negative or non-finite values")
        intensity = np.zeros((state.x_m.size, state.z_m.size), dtype=np.float64)
        for y_index in range(state.y_m.size):
            source = planck_photon_radiance_lambda(wavelength, state.temperature_K[y_index])
            intensity = formal_solution_step(
                intensity, source, opacity[y_index], state.path_length_m
            )
        spectral[wavelength_index] = intensity
        optical_depth[wavelength_index] = np.sum(opacity, axis=0) * state.path_length_m
    response = flat_spectral_response(wavelengths)
    weighted = spectral * response[:, None, None]
    if hasattr(np, "trapezoid"):
        image = np.trapezoid(weighted, wavelengths, axis=0)
    else:  # NumPy < 2.0
        image = np.trapz(weighted, wavelengths, axis=0)
    return TransferResult(
        image_photon_radiance=image,
        spectral_photon_radiance=spectral,
        optical_depth=optical_depth,
        wavelengths_m=wavelengths,
    )
