"""Stable photon form of Planck's spectral radiance."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from iccd_sim_ml.constants import BOLTZMANN_J_K, PLANCK_J_S, SPEED_OF_LIGHT_M_S


def planck_photon_radiance_lambda(
    wavelength_m: float | ArrayLike, temperature_K: ArrayLike
) -> NDArray[np.float64]:
    """Return photon spectral radiance in photons s^-1 m^-2 sr^-1 m^-1."""

    wavelength = np.asarray(wavelength_m, dtype=np.float64)
    temperature = np.asarray(temperature_K, dtype=np.float64)
    if np.any(wavelength <= 0):
        raise ValueError("Wavelengths must be positive")
    wavelength, temperature = np.broadcast_arrays(wavelength, temperature)
    exponent = np.full(wavelength.shape, np.inf, dtype=np.float64)
    np.divide(
        PLANCK_J_S * SPEED_OF_LIGHT_M_S,
        wavelength * BOLTZMANN_J_K * temperature,
        out=exponent,
        where=temperature > 0,
    )
    denominator = np.expm1(np.minimum(exponent, 700.0))
    radiance = (2.0 * SPEED_OF_LIGHT_M_S / wavelength**4) / denominator
    return np.where(temperature > 0, radiance, 0.0)
