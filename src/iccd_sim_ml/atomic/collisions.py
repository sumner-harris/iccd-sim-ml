"""Optional electron--neutral momentum-transfer reference calculations.

These utilities reproduce the energy-dependent calculation explored with the
Cu ``MT_01_01`` table. The production image catalog instead uses the declared
project-wide fixed-Q assumption for every element. The Maxwellian integral
here is retained for notebook reproduction and sensitivity studies.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from iccd_sim_ml.constants import (
    BOHR_RADIUS_M,
    BOLTZMANN_J_K,
    COULOMB_CONSTANT,
    ELECTRON_MASS_KG,
    ELEMENTARY_CHARGE_C,
    PLANCK_J_S,
    SPEED_OF_LIGHT_M_S,
)

FloatArray = NDArray[np.float64]


def _finite_positive(name: str, value: ArrayLike) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if np.any(~np.isfinite(array)) or np.any(array <= 0.0):
        raise ValueError(f"{name} must contain only finite positive values")
    return array


def _finite_nonnegative(name: str, value: ArrayLike) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if np.any(~np.isfinite(array)) or np.any(array < 0.0):
        raise ValueError(f"{name} must contain only finite non-negative values")
    return array


@dataclass(frozen=True)
class MomentumTransferTable:
    """Electron--neutral momentum-transfer cross section versus energy.

    Parameters are stored in SI-compatible units: electron energy in eV and
    cross section in square metres.  ``outside="edge"`` reproduces the
    nearest-edge behaviour of the exploratory notebook outside the measured
    energy range.  ``outside="zero"`` is available for conservative studies.
    """

    energy_ev: FloatArray
    cross_section_m2: FloatArray
    source: Path | None = None
    outside: Literal["edge", "zero"] = "edge"

    def __post_init__(self) -> None:
        energy = np.asarray(self.energy_ev, dtype=np.float64).copy()
        cross_section = np.asarray(self.cross_section_m2, dtype=np.float64).copy()
        if energy.ndim != 1 or cross_section.ndim != 1 or energy.size < 2:
            raise ValueError("Momentum-transfer arrays must be one-dimensional with >= 2 rows")
        if energy.shape != cross_section.shape:
            raise ValueError("Energy and momentum-transfer arrays must have equal lengths")
        if np.any(~np.isfinite(energy)) or np.any(energy <= 0.0):
            raise ValueError("Momentum-transfer energies must be finite and positive")
        if np.any(~np.isfinite(cross_section)) or np.any(cross_section < 0.0):
            raise ValueError("Momentum-transfer cross sections must be finite and non-negative")
        if np.any(np.diff(energy) <= 0.0):
            raise ValueError("Momentum-transfer energies must be strictly increasing")
        if self.outside not in {"edge", "zero"}:
            raise ValueError("outside must be either 'edge' or 'zero'")
        energy.setflags(write=False)
        cross_section.setflags(write=False)
        object.__setattr__(self, "energy_ev", energy)
        object.__setattr__(self, "cross_section_m2", cross_section)

    def evaluate(self, energy_ev: ArrayLike) -> FloatArray:
        """Linearly interpolate the momentum-transfer cross section in energy."""

        query = _finite_nonnegative("energy_ev", energy_ev)
        if self.outside == "zero":
            left = right = 0.0
        else:
            left = float(self.cross_section_m2[0])
            right = float(self.cross_section_m2[-1])
        values = np.interp(
            query.ravel(),
            self.energy_ev,
            self.cross_section_m2,
            left=left,
            right=right,
        )
        return values.reshape(query.shape)


def load_momentum_transfer_table(
    path: str | Path,
    *,
    cross_section_unit: Literal["bohr2", "m2"] = "bohr2",
    outside: Literal["edge", "zero"] = "edge",
) -> MomentumTransferTable:
    """Load a two-column ``energy_eV, sigma_MT`` whitespace table.

    The supplied Cu table stores its second column in :math:`a_0^2`; use
    ``cross_section_unit="m2"`` only for an already-SI table.
    """

    source = Path(path).expanduser().resolve()
    data = np.loadtxt(source, dtype=np.float64, comments="#", skiprows=1, usecols=(0, 1))
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[0] < 2 or data.shape[1] != 2:
        raise ValueError(f"Expected at least two momentum-transfer rows in {source}")
    order = np.argsort(data[:, 0])
    energy = data[order, 0]
    cross_section = data[order, 1]
    if cross_section_unit == "bohr2":
        cross_section = cross_section * BOHR_RADIUS_M**2
    elif cross_section_unit != "m2":
        raise ValueError("cross_section_unit must be either 'bohr2' or 'm2'")
    return MomentumTransferTable(energy, cross_section, source=source, outside=outside)


@lru_cache(maxsize=16)
def _laguerre_nodes_weights(order: int) -> tuple[FloatArray, FloatArray]:
    if not 8 <= order <= 128:
        raise ValueError("quadrature_order must be between 8 and 128")
    # Golub--Welsch generalized Gauss--Laguerre rule for the Maxwellian
    # weight x^(1/2) exp(-x).  Building the small symmetric Jacobi matrix
    # avoids requiring scipy.special.roots_genlaguerre at runtime.
    alpha = 0.5
    indices = np.arange(order, dtype=np.float64)
    jacobi = np.diag(2.0 * indices + 1.0 + alpha)
    off_diagonal_index = np.arange(1, order, dtype=np.float64)
    off_diagonal = np.sqrt(off_diagonal_index * (off_diagonal_index + alpha))
    jacobi += np.diag(off_diagonal, 1) + np.diag(off_diagonal, -1)
    nodes, eigenvectors = np.linalg.eigh(jacobi)
    weights = math.gamma(alpha + 1.0) * eigenvectors[0, :] ** 2
    return nodes, weights


def maxwellian_electron_neutral_kernel_m5(
    temperature_K: ArrayLike,
    wavelength_m: ArrayLike,
    momentum_transfer: MomentumTransferTable,
    *,
    quadrature_order: int = 48,
    chunk_size: int = 65_536,
) -> FloatArray:
    r"""Return the Maxwellian-averaged electron--neutral kernel ``Q`` in m^5.

    The implemented notebook model is

    .. math::

       Q(T,\nu) = \int_0^\infty K_a(E,\nu) f_E(E,T)\,dE,

    .. math::

       K_a = \frac{2 e^2 k_e}{3\pi m_e c\nu^2}
       \sqrt{\frac{2(E+h\nu)}{m_e}}
       \frac{E+h\nu}{h\nu}\,\sigma_{MT}(E+h\nu).

    The stimulated-emission correction is intentionally not included here;
    it is applied once by the opacity model.
    """

    temperature = _finite_nonnegative("temperature_K", temperature_K)
    wavelength = _finite_positive("wavelength_m", wavelength_m)
    temperature, wavelength = np.broadcast_arrays(temperature, wavelength)
    result = np.zeros(temperature.shape, dtype=np.float64)
    active = temperature.ravel() > 0.0
    if not np.any(active):
        return result
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")

    nodes, weights = _laguerre_nodes_weights(quadrature_order)
    quadrature_weights = (2.0 / np.sqrt(np.pi)) * weights
    flat_temperature = temperature.ravel()
    flat_wavelength = wavelength.ravel()
    flat_result = result.ravel()
    active_indices = np.flatnonzero(active)

    for start in range(0, active_indices.size, chunk_size):
        indices = active_indices[start : start + chunk_size]
        temp = flat_temperature[indices, None]
        wave = flat_wavelength[indices, None]
        photon_energy_j = PLANCK_J_S * SPEED_OF_LIGHT_M_S / wave
        initial_energy_j = BOLTZMANN_J_K * temp * nodes[None, :]
        final_energy_j = initial_energy_j + photon_energy_j
        sigma_m2 = momentum_transfer.evaluate(final_energy_j / ELEMENTARY_CHARGE_C)
        frequency_hz = SPEED_OF_LIGHT_M_S / wave
        prefactor = (
            2.0
            * ELEMENTARY_CHARGE_C**2
            * COULOMB_CONSTANT
            / (3.0 * np.pi * ELECTRON_MASS_KG * SPEED_OF_LIGHT_M_S * frequency_hz**2)
        )
        kernel = (
            prefactor
            * np.sqrt(2.0 * final_energy_j / ELECTRON_MASS_KG)
            * (final_energy_j / photon_energy_j)
            * sigma_m2
        )
        averaged = np.sum(kernel * quadrature_weights[None, :], axis=1)
        flat_result[indices] = np.maximum(averaged, 0.0)

    if np.any(~np.isfinite(result)):
        raise FloatingPointError("Electron-neutral Maxwellian average produced non-finite values")
    return result


__all__ = [
    "MomentumTransferTable",
    "load_momentum_transfer_table",
    "maxwellian_electron_neutral_kernel_m5",
]
