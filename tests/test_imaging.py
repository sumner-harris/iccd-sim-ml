from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from iccd_sim_ml.imaging import ImagingConfig, simulate_continuum_sequence
from iccd_sim_ml.imaging.blackbody import planck_photon_radiance_lambda
from iccd_sim_ml.imaging.grid import SideViewState, infer_dyadic_domain_max, resample_timestep
from iccd_sim_ml.imaging.transfer import formal_solution_step, solve_lte_continuum
from iccd_sim_ml.io import plasma_timestep_from_array


@dataclass(frozen=True)
class _Opacity:
    total_m1: np.ndarray


class _ConstantOpacity:
    def __init__(self, value: float) -> None:
        self.value = value

    def components(self, wavelength_m, temperature_K, n0_m3, ne_m3, n1_m3, n2_m3):
        del wavelength_m, n0_m3, ne_m3, n1_m3, n2_m3
        return _Opacity(np.full_like(temperature_K, self.value))


def test_photon_planck_is_finite_and_increases_with_temperature() -> None:
    radiance = planck_photon_radiance_lambda(500e-9, np.asarray([0.0, 1000.0, 5000.0]))
    assert np.isfinite(radiance).all()
    assert radiance[0] == 0.0
    assert np.all(np.diff(radiance) > 0)


def test_formal_solution_matches_uniform_slab() -> None:
    source = np.asarray([[7.0]])
    result = formal_solution_step(np.zeros((1, 1)), source, np.asarray([[2.0]]), 0.25)
    assert np.allclose(result, source * (1.0 - np.exp(-0.5)))


def test_uniform_multicell_transfer_matches_analytic_solution() -> None:
    shape = (5, 2, 3)
    temperature = np.full(shape, 4000.0)
    zeros = np.zeros(shape)
    state = SideViewState(
        x_m=np.linspace(-1.0, 1.0, 2),
        y_m=np.linspace(-0.8, 0.8, 5),
        z_m=np.linspace(0.0, 1.0, 3),
        path_length_m=0.4,
        temperature_K=temperature,
        n0_m3=zeros,
        ne_m3=zeros,
        n1_m3=zeros,
        n2_m3=zeros,
    )
    wavelengths = np.asarray([500e-9, 600e-9])
    result = solve_lte_continuum(state, wavelengths, _ConstantOpacity(0.3))
    expected = planck_photon_radiance_lambda(500e-9, 4000.0) * (1.0 - np.exp(-0.6))
    assert np.allclose(result.spectral_photon_radiance[0], expected)
    assert np.allclose(result.optical_depth, 0.6)


def test_domain_inference_recovers_dyadic_edge() -> None:
    minimum_center = 0.05 / 1024.0
    coordinates = np.asarray([minimum_center, 500 * minimum_center, 1008 * minimum_center])
    assert np.isclose(infer_dyadic_domain_max(coordinates), 0.05)


def test_resampling_larger_fov_imposes_vacuum_outside_source_domain() -> None:
    raw = np.zeros((4, 34), dtype=np.float64)
    raw[:, 0] = [0.025, 0.075, 0.025, 0.075]
    raw[:, 1] = [0.025, 0.025, 0.075, 0.075]
    raw[:, 4] = 5_000.0
    raw[:, 9:13] = 1.0e18
    timestep = plasma_timestep_from_array(raw, source="synthetic.h5", mesh_level_index=21)
    grid = resample_timestep(
        timestep,
        radial_points=3,
        axial_points=3,
        radial_max_m=0.2,
        axial_max_m=0.2,
    )
    assert np.all(grid.temperature_K[-1, :] == 0.0)
    assert np.all(grid.temperature_K[:, -1] == 0.0)
    assert grid.temperature_K[0, 0] > 0.0


def test_sequence_requires_explicit_fixed_field_of_view() -> None:
    with pytest.raises(ValueError, match="explicit radial_max_m and axial_max_m"):
        simulate_continuum_sequence([], ImagingConfig(), _ConstantOpacity(0.3))
