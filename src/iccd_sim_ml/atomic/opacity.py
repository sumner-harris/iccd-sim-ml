"""Wavelength-dependent LTE continuum opacity for an atomic plasma.

The model follows the equations in ``Laser-Plasma Absorption.ipynb`` while
using SI units throughout.  It includes electron--neutral and electron--ion
inverse bremsstrahlung and the notebook's level-resolved Kramers-like
photoionization approximation.  Bound--bound lines are deliberately separate
from this continuum model.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from iccd_sim_ml.constants import (
    BOLTZMANN_EV_K,
    BOLTZMANN_J_K,
    COULOMB_CONSTANT,
    ELECTRON_MASS_KG,
    ELEMENTARY_CHARGE_C,
    PLANCK_J_S,
    SPEED_OF_LIGHT_M_S,
)

from .collisions import (
    MomentumTransferTable,
    load_momentum_transfer_table,
    maxwellian_electron_neutral_kernel_m5,
)
from .levels import EnergyLevels
from .species import AtomicSpecies

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


def _checked_nonnegative(name: str, value: ArrayLike) -> FloatArray:
    result = np.maximum(np.asarray(value, dtype=np.float64), 0.0)
    if np.any(~np.isfinite(result)):
        raise FloatingPointError(f"{name} produced non-finite values")
    return result


def stimulated_emission_factor(
    wavelength_m: ArrayLike,
    temperature_K: ArrayLike,
) -> FloatArray:
    r"""Return ``1 - exp(-h c / (lambda k_B T))`` stably.

    ``-expm1(-x)`` retains precision in the infrared/high-temperature limit,
    where direct subtraction would lose significant digits.  Zero-temperature
    cells are assigned zero opacity because the plasma formulas are not valid
    there and this convention keeps empty/background cells benign.
    """

    wavelength = _finite_positive("wavelength_m", wavelength_m)
    temperature = _finite_nonnegative("temperature_K", temperature_K)
    wavelength, temperature = np.broadcast_arrays(wavelength, temperature)
    factor = np.zeros(wavelength.shape, dtype=np.float64)
    active = temperature > 0.0
    exponent = np.empty(wavelength.shape, dtype=np.float64)
    exponent[active] = (
        PLANCK_J_S * SPEED_OF_LIGHT_M_S / (wavelength[active] * BOLTZMANN_J_K * temperature[active])
    )
    factor[active] = -np.expm1(-exponent[active])
    return np.clip(factor, 0.0, 1.0)


def electron_neutral_coefficient_m5(
    wavelength_m: ArrayLike,
    temperature_K: ArrayLike,
    momentum_transfer: MomentumTransferTable,
    *,
    quadrature_order: int = 48,
    chunk_size: int = 65_536,
) -> FloatArray:
    r"""Return ``alpha_en / (n_e n_0)`` in m^5."""

    kernel = maxwellian_electron_neutral_kernel_m5(
        temperature_K,
        wavelength_m,
        momentum_transfer,
        quadrature_order=quadrature_order,
        chunk_size=chunk_size,
    )
    coefficient = stimulated_emission_factor(wavelength_m, temperature_K) * kernel
    return _checked_nonnegative("electron-neutral coefficient", coefficient)


def electron_ion_coefficient_m5(
    wavelength_m: ArrayLike,
    temperature_K: ArrayLike,
) -> FloatArray:
    r"""Return ``alpha_ei / (n_e sum(z^2 n_z))`` in m^5.

    This is the SI form of the cgs expression used in the notebook.  The
    factor ``k_e^3`` converts the three Gaussian-CGS factors of ``e^2``.
    """

    wavelength = _finite_positive("wavelength_m", wavelength_m)
    temperature = _finite_nonnegative("temperature_K", temperature_K)
    wavelength, temperature = np.broadcast_arrays(wavelength, temperature)
    coefficient = np.zeros(wavelength.shape, dtype=np.float64)
    active = temperature > 0.0
    if np.any(active):
        wave = wavelength[active]
        temp = temperature[active]
        coefficient[active] = (
            stimulated_emission_factor(wave, temp)
            * 4.0
            * ELEMENTARY_CHARGE_C**6
            * COULOMB_CONSTANT**3
            * wave**3
            / (3.0 * PLANCK_J_S * SPEED_OF_LIGHT_M_S**4 * ELECTRON_MASS_KG)
            * np.sqrt(2.0 * np.pi / (3.0 * ELECTRON_MASS_KG * BOLTZMANN_J_K * temp))
        )
    return _checked_nonnegative("electron-ion coefficient", coefficient)


def _partition_function(levels: EnergyLevels, temperature_K: FloatArray) -> FloatArray:
    """Evaluate a partition function without allocating over unrelated cells."""

    energy = levels.energy_ev - levels.energy_ev[0]
    exponent = -energy[:, None] / (BOLTZMANN_EV_K * temperature_K[None, :])
    return np.sum(
        levels.degeneracy[:, None] * np.exp(np.clip(exponent, -745.0, 0.0)),
        axis=0,
    )


def photoionization_effective_cross_sections_m2(
    species: AtomicSpecies,
    wavelength_m: ArrayLike,
    temperature_K: ArrayLike,
    *,
    chunk_size: int = 8_192,
) -> dict[int, FloatArray]:
    r"""Return LTE-averaged photoionization cross section for each charge state.

    For level ``j`` of charge state ``z``, the notebook approximation is

    .. math::

       \sigma_{j,z} = \frac{32\pi^2(z+1)^2e^6k_e^3}
       {3\sqrt{3}h^4c\nu^3}\frac{U_{z+1}}{g_{j,z}}\Delta E_j.

    It is multiplied by the LTE level fraction and included only when
    ``E_j >= chi_z - h nu``.  The level degeneracy cancels analytically in
    this product, improving numerical stability and reducing work.
    """

    wavelength = _finite_positive("wavelength_m", wavelength_m)
    temperature = _finite_nonnegative("temperature_K", temperature_K)
    wavelength, temperature = np.broadcast_arrays(wavelength, temperature)
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")

    flat_wave = wavelength.ravel()
    flat_temperature = temperature.ravel()
    active_indices = np.flatnonzero(flat_temperature > 0.0)
    result: dict[int, FloatArray] = {}
    photon_energy_ev = PLANCK_J_S * SPEED_OF_LIGHT_M_S / ELEMENTARY_CHARGE_C / flat_wave

    for charge in species.photoionization_charge_states():
        levels = species.levels_by_charge[charge]
        next_levels = species.levels_by_charge[charge + 1]
        ionization_energy_ev = species.ionization_energy_ev_by_charge[charge]
        energy_ev = levels.energy_ev - levels.energy_ev[0]
        spacing_j = np.diff(energy_ev, prepend=energy_ev[0]) * ELEMENTARY_CHARGE_C
        sigma = np.zeros(flat_wave.shape, dtype=np.float64)

        for start in range(0, active_indices.size, chunk_size):
            indices = active_indices[start : start + chunk_size]
            temp = flat_temperature[indices]
            wave = flat_wave[indices]
            current_partition = _partition_function(levels, temp)
            next_partition = _partition_function(next_levels, temp)
            boltzmann = np.exp(
                np.clip(
                    -energy_ev[:, None] / (BOLTZMANN_EV_K * temp[None, :]),
                    -745.0,
                    0.0,
                )
            )
            threshold_ev = ionization_energy_ev - photon_energy_ev[indices]
            eligible = energy_ev[:, None] >= threshold_ev[None, :]
            weighted_spacing_j = np.sum(
                spacing_j[:, None] * boltzmann * eligible,
                axis=0,
            )
            frequency_hz = SPEED_OF_LIGHT_M_S / wave
            prefactor = (
                32.0
                * np.pi**2
                * (charge + 1) ** 2
                * ELEMENTARY_CHARGE_C**6
                * COULOMB_CONSTANT**3
                / (3.0 * np.sqrt(3.0) * PLANCK_J_S**4 * SPEED_OF_LIGHT_M_S * frequency_hz**3)
            )
            sigma[indices] = prefactor * (next_partition / current_partition) * weighted_spacing_j

        result[charge] = _checked_nonnegative(
            f"photoionization cross section for charge {charge}",
            sigma.reshape(wavelength.shape),
        )
    return result


@dataclass(frozen=True)
class OpacityComponents:
    """Continuum absorption components, each in inverse metres."""

    electron_neutral_m1: FloatArray
    electron_ion_m1: FloatArray
    photoionization_m1: FloatArray

    def __post_init__(self) -> None:
        arrays = np.broadcast_arrays(
            np.asarray(self.electron_neutral_m1, dtype=np.float64),
            np.asarray(self.electron_ion_m1, dtype=np.float64),
            np.asarray(self.photoionization_m1, dtype=np.float64),
        )
        names = ("electron_neutral_m1", "electron_ion_m1", "photoionization_m1")
        for name, array in zip(names, arrays, strict=True):
            checked = _checked_nonnegative(name, array)
            checked.setflags(write=False)
            object.__setattr__(self, name, checked)

    @property
    def total_m1(self) -> FloatArray:
        return _checked_nonnegative(
            "total continuum opacity",
            self.electron_neutral_m1 + self.electron_ion_m1 + self.photoionization_m1,
        )


def _broadcast_state(
    wavelength_m: ArrayLike,
    temperature_K: ArrayLike,
    n0_m3: ArrayLike,
    ne_m3: ArrayLike,
    n1_m3: ArrayLike,
    n2_m3: ArrayLike,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    values = np.broadcast_arrays(
        _finite_positive("wavelength_m", wavelength_m),
        _finite_nonnegative("temperature_K", temperature_K),
        _finite_nonnegative("n0_m3", n0_m3),
        _finite_nonnegative("ne_m3", ne_m3),
        _finite_nonnegative("n1_m3", n1_m3),
        _finite_nonnegative("n2_m3", n2_m3),
    )
    return (
        np.asarray(values[0], dtype=np.float64),
        np.asarray(values[1], dtype=np.float64),
        np.asarray(values[2], dtype=np.float64),
        np.asarray(values[3], dtype=np.float64),
        np.asarray(values[4], dtype=np.float64),
        np.asarray(values[5], dtype=np.float64),
    )


@dataclass(frozen=True)
class ContinuumOpacityModel:
    """Direct wavelength-dependent continuum-opacity calculator."""

    species: AtomicSpecies
    momentum_transfer: MomentumTransferTable | None = None
    electron_neutral_constant_m5: float | None = None
    quadrature_order: int = 48
    chunk_size: int = 65_536

    def __post_init__(self) -> None:
        if not 8 <= self.quadrature_order <= 128:
            raise ValueError("quadrature_order must be between 8 and 128")
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        if self.momentum_transfer is not None and self.electron_neutral_constant_m5 is not None:
            raise ValueError(
                "Choose either a momentum-transfer table or a constant electron-neutral kernel"
            )
        if self.electron_neutral_constant_m5 is not None and (
            not np.isfinite(self.electron_neutral_constant_m5)
            or self.electron_neutral_constant_m5 < 0.0
        ):
            raise ValueError("electron_neutral_constant_m5 must be finite and non-negative")

    @classmethod
    def from_momentum_transfer_file(
        cls,
        species: AtomicSpecies,
        path: str | Path,
        *,
        cross_section_unit: Literal["bohr2", "m2"] = "bohr2",
        quadrature_order: int = 48,
        chunk_size: int = 65_536,
    ) -> ContinuumOpacityModel:
        table = load_momentum_transfer_table(
            path,
            cross_section_unit=cross_section_unit,
        )
        return cls(
            species=species,
            momentum_transfer=table,
            quadrature_order=quadrature_order,
            chunk_size=chunk_size,
        )

    @classmethod
    def from_constant_electron_neutral(
        cls,
        species: AtomicSpecies,
        coefficient_m5: float,
        *,
        chunk_size: int = 65_536,
    ) -> ContinuumOpacityModel:
        """Build an explicitly approximate constant-Q electron-neutral model."""

        return cls(
            species=species,
            electron_neutral_constant_m5=coefficient_m5,
            chunk_size=chunk_size,
        )

    def _electron_neutral_coefficient(
        self,
        wavelength_m: ArrayLike,
        temperature_K: ArrayLike,
    ) -> FloatArray:
        if self.momentum_transfer is not None:
            return electron_neutral_coefficient_m5(
                wavelength_m,
                temperature_K,
                self.momentum_transfer,
                quadrature_order=self.quadrature_order,
                chunk_size=self.chunk_size,
            )
        wavelength, temperature = np.broadcast_arrays(
            _finite_positive("wavelength_m", wavelength_m),
            _finite_nonnegative("temperature_K", temperature_K),
        )
        if self.electron_neutral_constant_m5 is None:
            return np.zeros(wavelength.shape, dtype=np.float64)
        return _checked_nonnegative(
            "constant electron-neutral coefficient",
            stimulated_emission_factor(wavelength, temperature) * self.electron_neutral_constant_m5,
        )

    def components(
        self,
        wavelength_m: ArrayLike,
        temperature_K: ArrayLike,
        n0_m3: ArrayLike,
        ne_m3: ArrayLike,
        n1_m3: ArrayLike,
        n2_m3: ArrayLike,
    ) -> OpacityComponents:
        """Calculate all continuum opacity components in m^-1."""

        wave, temp, n0, ne, n1, n2 = _broadcast_state(
            wavelength_m,
            temperature_K,
            n0_m3,
            ne_m3,
            n1_m3,
            n2_m3,
        )
        alpha_en = self._electron_neutral_coefficient(wave, temp) * (ne * n0)
        alpha_ei = electron_ion_coefficient_m5(wave, temp) * ne * (n1 + 4.0 * n2)
        effective_cross_sections = photoionization_effective_cross_sections_m2(
            self.species,
            wave,
            temp,
            chunk_size=min(self.chunk_size, 8_192),
        )
        densities: Mapping[int, FloatArray] = {0: n0, 1: n1, 2: n2}
        alpha_pi_absorption = np.zeros(wave.shape, dtype=np.float64)
        for charge, sigma in effective_cross_sections.items():
            density = densities.get(charge)
            if density is not None:
                alpha_pi_absorption = alpha_pi_absorption + sigma * density
        # The legacy-notebook expression is the true photoabsorption
        # coefficient used for laser heating.  Kirchhoff transfer with a
        # Planck source requires the net LTE opacity after stimulated
        # recombination is included through detailed balance.
        alpha_pi_net = stimulated_emission_factor(wave, temp) * alpha_pi_absorption
        return OpacityComponents(
            _checked_nonnegative("electron-neutral opacity", alpha_en),
            _checked_nonnegative("electron-ion opacity", alpha_ei),
            _checked_nonnegative("net photoionization opacity", alpha_pi_net),
        )

    def build_lookup(
        self,
        temperature_grid_K: ArrayLike,
        wavelength_grid_m: ArrayLike,
    ) -> ContinuumOpacityLookup:
        """Precompute density-independent coefficients on a regular 2-D grid."""

        temperature = _finite_positive("temperature_grid_K", temperature_grid_K)
        wavelength = _finite_positive("wavelength_grid_m", wavelength_grid_m)
        if temperature.ndim != 1 or wavelength.ndim != 1:
            raise ValueError("Lookup temperature and wavelength grids must be one-dimensional")
        if temperature.size < 2 or wavelength.size < 2:
            raise ValueError("Lookup grids must each contain at least two points")
        if np.any(np.diff(temperature) <= 0.0) or np.any(np.diff(wavelength) <= 0.0):
            raise ValueError("Lookup grids must be strictly increasing")
        temp_mesh = temperature[:, None]
        wave_mesh = wavelength[None, :]
        en = self._electron_neutral_coefficient(wave_mesh, temp_mesh)
        ei = electron_ion_coefficient_m5(wave_mesh, temp_mesh)
        pi = photoionization_effective_cross_sections_m2(
            self.species,
            wave_mesh,
            temp_mesh,
            chunk_size=min(self.chunk_size, 8_192),
        )
        return ContinuumOpacityLookup(temperature, wavelength, en, ei, pi)


def _bilinear_interpolate(
    temperature_grid: FloatArray,
    wavelength_grid: FloatArray,
    values: FloatArray,
    temperature_K: FloatArray,
    wavelength_m: FloatArray,
) -> FloatArray:
    temperature, wavelength = np.broadcast_arrays(temperature_K, wavelength_m)
    temp = np.clip(temperature, temperature_grid[0], temperature_grid[-1])
    wave = np.clip(wavelength, wavelength_grid[0], wavelength_grid[-1])
    ti = np.searchsorted(temperature_grid, temp, side="right") - 1
    wi = np.searchsorted(wavelength_grid, wave, side="right") - 1
    ti = np.clip(ti, 0, temperature_grid.size - 2)
    wi = np.clip(wi, 0, wavelength_grid.size - 2)
    t0 = temperature_grid[ti]
    t1 = temperature_grid[ti + 1]
    w0 = wavelength_grid[wi]
    w1 = wavelength_grid[wi + 1]
    tf = (temp - t0) / (t1 - t0)
    wf = (wave - w0) / (w1 - w0)
    interpolated = (
        values[ti, wi] * (1.0 - tf) * (1.0 - wf)
        + values[ti + 1, wi] * tf * (1.0 - wf)
        + values[ti, wi + 1] * (1.0 - tf) * wf
        + values[ti + 1, wi + 1] * tf * wf
    )
    return _checked_nonnegative("interpolated opacity coefficient", interpolated)


@dataclass(frozen=True)
class ContinuumOpacityLookup:
    """Fast bilinear lookup for density-independent continuum coefficients.

    Queries outside the precomputed domain are clipped to its nearest edge;
    production configurations should therefore span the complete plasma state
    and camera-response range.
    """

    temperature_grid_K: FloatArray
    wavelength_grid_m: FloatArray
    electron_neutral_coefficient_m5: FloatArray
    electron_ion_coefficient_m5: FloatArray
    photoionization_cross_section_m2: Mapping[int, FloatArray]

    def __post_init__(self) -> None:
        temperature = _finite_positive("temperature_grid_K", self.temperature_grid_K).copy()
        wavelength = _finite_positive("wavelength_grid_m", self.wavelength_grid_m).copy()
        if temperature.ndim != 1 or wavelength.ndim != 1:
            raise ValueError("Lookup grids must be one-dimensional")
        if temperature.size < 2 or wavelength.size < 2:
            raise ValueError("Lookup grids must each contain at least two points")
        if np.any(np.diff(temperature) <= 0.0) or np.any(np.diff(wavelength) <= 0.0):
            raise ValueError("Lookup grids must be strictly increasing")
        expected = (temperature.size, wavelength.size)
        en = _checked_nonnegative(
            "electron-neutral lookup", self.electron_neutral_coefficient_m5
        ).copy()
        ei = _checked_nonnegative("electron-ion lookup", self.electron_ion_coefficient_m5).copy()
        if en.shape != expected or ei.shape != expected:
            raise ValueError(f"Opacity coefficient tables must have shape {expected}")
        photo: dict[int, FloatArray] = {}
        for charge, table in self.photoionization_cross_section_m2.items():
            checked = _checked_nonnegative(
                f"photoionization lookup for charge {charge}", table
            ).copy()
            if checked.shape != expected:
                raise ValueError(f"Photoionization table must have shape {expected}")
            checked.setflags(write=False)
            photo[int(charge)] = checked
        for array in (temperature, wavelength, en, ei):
            array.setflags(write=False)
        object.__setattr__(self, "temperature_grid_K", temperature)
        object.__setattr__(self, "wavelength_grid_m", wavelength)
        object.__setattr__(self, "electron_neutral_coefficient_m5", en)
        object.__setattr__(self, "electron_ion_coefficient_m5", ei)
        object.__setattr__(self, "photoionization_cross_section_m2", photo)

    def components(
        self,
        wavelength_m: ArrayLike,
        temperature_K: ArrayLike,
        n0_m3: ArrayLike,
        ne_m3: ArrayLike,
        n1_m3: ArrayLike,
        n2_m3: ArrayLike,
    ) -> OpacityComponents:
        wave, temp, n0, ne, n1, n2 = _broadcast_state(
            wavelength_m,
            temperature_K,
            n0_m3,
            ne_m3,
            n1_m3,
            n2_m3,
        )
        en_coefficient = _bilinear_interpolate(
            self.temperature_grid_K,
            self.wavelength_grid_m,
            self.electron_neutral_coefficient_m5,
            temp,
            wave,
        )
        ei_coefficient = _bilinear_interpolate(
            self.temperature_grid_K,
            self.wavelength_grid_m,
            self.electron_ion_coefficient_m5,
            temp,
            wave,
        )
        densities: Mapping[int, FloatArray] = {0: n0, 1: n1, 2: n2}
        alpha_pi_absorption = np.zeros(wave.shape, dtype=np.float64)
        for charge, table in self.photoionization_cross_section_m2.items():
            density = densities.get(charge)
            if density is None:
                continue
            sigma = _bilinear_interpolate(
                self.temperature_grid_K,
                self.wavelength_grid_m,
                table,
                temp,
                wave,
            )
            alpha_pi_absorption = alpha_pi_absorption + sigma * density
        active = temp > 0.0
        alpha_pi_net = stimulated_emission_factor(wave, temp) * alpha_pi_absorption
        return OpacityComponents(
            _checked_nonnegative(
                "electron-neutral opacity", np.where(active, en_coefficient * ne * n0, 0.0)
            ),
            _checked_nonnegative(
                "electron-ion opacity",
                np.where(active, ei_coefficient * ne * (n1 + 4.0 * n2), 0.0),
            ),
            _checked_nonnegative(
                "net photoionization opacity", np.where(active, alpha_pi_net, 0.0)
            ),
        )


__all__ = [
    "ContinuumOpacityLookup",
    "ContinuumOpacityModel",
    "OpacityComponents",
    "electron_ion_coefficient_m5",
    "electron_neutral_coefficient_m5",
    "photoionization_effective_cross_sections_m2",
    "stimulated_emission_factor",
]
