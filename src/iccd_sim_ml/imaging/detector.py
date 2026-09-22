"""Spectral detector responses.

The current proof of concept intentionally applies no spatial optics, quantum
efficiency roll-off, gain, digitization, saturation, or noise.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def flat_spectral_response(wavelength_m: ArrayLike) -> NDArray[np.float64]:
    """Return the user-requested unit response at every wavelength."""

    wavelength = np.asarray(wavelength_m, dtype=np.float64)
    if np.any(wavelength <= 0):
        raise ValueError("Wavelengths must be positive")
    return np.ones_like(wavelength)
