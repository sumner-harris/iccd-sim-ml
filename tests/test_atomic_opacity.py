from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from iccd_sim_ml.atomic.catalog import (
    FIXED_ELECTRON_NEUTRAL_Q_CM5,
    FIXED_ELECTRON_NEUTRAL_Q_M5,
    AtomicDataCatalog,
    AtomicDataUnavailableError,
)
from iccd_sim_ml.atomic.collisions import (
    MomentumTransferTable,
    load_momentum_transfer_table,
    maxwellian_electron_neutral_kernel_m5,
)
from iccd_sim_ml.atomic.levels import EnergyLevels, _deduplicate
from iccd_sim_ml.atomic.opacity import (
    ContinuumOpacityModel,
    electron_ion_coefficient_m5,
    photoionization_effective_cross_sections_m2,
    stimulated_emission_factor,
)
from iccd_sim_ml.atomic.species import AtomicSpecies
from iccd_sim_ml.constants import (
    BOLTZMANN_EV_K,
    COULOMB_CONSTANT,
    ELEMENTARY_CHARGE_C,
    PLANCK_J_S,
    SPEED_OF_LIGHT_M_S,
)


def _levels(energy_ev: list[float], degeneracy: list[float], name: str) -> EnergyLevels:
    return EnergyLevels(
        np.asarray(energy_ev, dtype=np.float64),
        np.asarray(degeneracy, dtype=np.float64),
        Path(name),
    )


def _synthetic_species() -> AtomicSpecies:
    return AtomicSpecies(
        symbol="Cu",
        levels_by_charge={
            0: _levels([0.0, 3.0, 5.0, 8.0], [2.0, 4.0, 6.0, 8.0], "CuI"),
            1: _levels([0.0, 2.0, 4.0], [1.0, 3.0, 5.0], "CuII"),
            2: _levels([0.0, 1.0, 3.0], [2.0, 4.0, 6.0], "CuIII"),
            3: _levels([0.0, 1.0], [1.0, 3.0], "CuIV"),
        },
        ionization_energy_ev_by_charge={0: 10.0, 1: 20.0, 2: 36.0},
    )


def _constant_momentum_transfer() -> MomentumTransferTable:
    return MomentumTransferTable(
        np.asarray([1.0e-6, 1.0e5]),
        np.asarray([2.5e-19, 2.5e-19]),
    )


class MomentumTransferTests(unittest.TestCase):
    def test_loader_converts_bohr_squared_and_interpolates(self) -> None:
        path = Path(__file__).parent / "fixtures" / "mt_small.txt"
        table = load_momentum_transfer_table(path)
        expected = 4.0 * (5.291_772_109_03e-11) ** 2
        self.assertAlmostEqual(float(table.evaluate(2.0)), expected, delta=expected * 1e-14)

    def test_maxwellian_kernel_matches_constant_cross_section_reference(self) -> None:
        value = maxwellian_electron_neutral_kernel_m5(
            20_000.0,
            248.0e-9,
            _constant_momentum_transfer(),
            quadrature_order=32,
        )
        # Independently converged dimensionless Maxwellian integral for sigma=2.5e-19 m2.
        self.assertAlmostEqual(float(value) / 7.809_148_754_087_157e-50, 1.0, places=11)


class EnergyLevelTests(unittest.TestCase):
    def test_exactly_degenerate_distinct_levels_add_statistical_weights(self) -> None:
        levels = _deduplicate([0.0, 1.0, 1.0, 1.0], [2.0, 3.0, 5.0, 5.0], Path("levels"))
        np.testing.assert_allclose(levels.energy_ev, [0.0, 1.0])
        np.testing.assert_allclose(levels.degeneracy, [2.0, 8.0])


class OpacityFormulaTests(unittest.TestCase):
    def test_stimulated_emission_factor_is_stable_in_small_x_limit(self) -> None:
        wavelength = 1.0
        temperature = 1.0e12
        value = float(stimulated_emission_factor(wavelength, temperature))
        x = 6.626_070_15e-34 * SPEED_OF_LIGHT_M_S / (wavelength * 1.380_649e-23 * temperature)
        self.assertGreater(value, 0.0)
        self.assertAlmostEqual(value / x, 1.0, places=12)
        self.assertEqual(float(stimulated_emission_factor(wavelength, 0.0)), 0.0)

    def test_electron_ion_si_formula_reproduces_notebook_cgs_value(self) -> None:
        coefficient = float(electron_ion_coefficient_m5(248.0e-9, 20_000.0))
        alpha = coefficient * 1.0e25 * (1.0e24 + 4.0 * 1.0e23)
        self.assertAlmostEqual(alpha, 1.955_460_611_050_009_2, places=12)

    def test_photoionization_uses_photon_threshold_and_lte_populations(self) -> None:
        species = _synthetic_species()
        temperature = 10_000.0
        wavelength = 248.0e-9
        calculated = float(
            photoionization_effective_cross_sections_m2(
                species,
                wavelength,
                temperature,
            )[0]
        )

        current = species.levels_by_charge[0]
        following = species.levels_by_charge[1]
        u0 = np.sum(
            current.degeneracy * np.exp(-current.energy_ev / (BOLTZMANN_EV_K * temperature))
        )
        u1 = np.sum(
            following.degeneracy * np.exp(-following.energy_ev / (BOLTZMANN_EV_K * temperature))
        )
        photon_ev = PLANCK_J_S * SPEED_OF_LIGHT_M_S / ELEMENTARY_CHARGE_C / wavelength
        eligible = current.energy_ev >= 10.0 - photon_ev
        spacing_j = np.diff(current.energy_ev, prepend=current.energy_ev[0]) * ELEMENTARY_CHARGE_C
        weighted_spacing = np.sum(
            spacing_j * np.exp(-current.energy_ev / (BOLTZMANN_EV_K * temperature)) * eligible
        )
        frequency = SPEED_OF_LIGHT_M_S / wavelength
        prefactor = (
            32.0
            * np.pi**2
            * ELEMENTARY_CHARGE_C**6
            * COULOMB_CONSTANT**3
            / (3.0 * np.sqrt(3.0) * PLANCK_J_S**4 * SPEED_OF_LIGHT_M_S * frequency**3)
        )
        expected = prefactor * (u1 / u0) * weighted_spacing
        self.assertAlmostEqual(calculated / expected, 1.0, places=13)

        below_threshold = photoionization_effective_cross_sections_m2(
            species,
            2.0e-6,
            temperature,
        )[0]
        self.assertEqual(float(below_threshold), 0.0)


class ContinuumOpacityModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = ContinuumOpacityModel(
            _synthetic_species(),
            _constant_momentum_transfer(),
            quadrature_order=24,
        )

    def test_components_are_finite_nonnegative_and_sum_to_total(self) -> None:
        components = self.model.components(
            np.asarray([248.0e-9, 500.0e-9]),
            np.asarray([10_000.0, 20_000.0]),
            1.0e24,
            1.0e23,
            2.0e22,
            3.0e21,
        )
        self.assertEqual(components.total_m1.shape, (2,))
        self.assertTrue(np.all(np.isfinite(components.total_m1)))
        self.assertTrue(np.all(components.total_m1 >= 0.0))
        np.testing.assert_allclose(
            components.total_m1,
            components.electron_neutral_m1
            + components.electron_ion_m1
            + components.photoionization_m1,
            rtol=0.0,
            atol=0.0,
        )

    def test_zero_densities_give_zero_opacity(self) -> None:
        components = self.model.components(500.0e-9, 10_000.0, 0.0, 0.0, 0.0, 0.0)
        self.assertEqual(float(components.total_m1), 0.0)

    def test_photoionization_component_is_net_lte_opacity(self) -> None:
        wavelength = 500.0e-9
        temperature = 20_000.0
        n0 = 1.0e24
        components = self.model.components(
            wavelength,
            temperature,
            n0,
            0.0,
            0.0,
            0.0,
        )
        true_cross_section = photoionization_effective_cross_sections_m2(
            self.model.species,
            wavelength,
            temperature,
        )[0]
        expected = true_cross_section * n0 * stimulated_emission_factor(wavelength, temperature)
        np.testing.assert_allclose(components.photoionization_m1, expected, rtol=2e-14)

    def test_lookup_matches_direct_model_at_grid_nodes(self) -> None:
        temperatures = np.asarray([8_000.0, 16_000.0, 30_000.0])
        wavelengths = np.asarray([248.0e-9, 500.0e-9, 800.0e-9])
        lookup = self.model.build_lookup(temperatures, wavelengths)
        for temperature in temperatures:
            for wavelength in wavelengths:
                direct = self.model.components(
                    wavelength,
                    temperature,
                    1.0e24,
                    1.0e23,
                    2.0e22,
                    3.0e21,
                )
                tabulated = lookup.components(
                    wavelength,
                    temperature,
                    1.0e24,
                    1.0e23,
                    2.0e22,
                    3.0e21,
                )
                np.testing.assert_allclose(tabulated.total_m1, direct.total_m1, rtol=2e-14)

        zero_temperature = lookup.components(
            500.0e-9,
            0.0,
            1.0e24,
            1.0e23,
            2.0e22,
            3.0e21,
        )
        self.assertEqual(float(zero_temperature.total_m1), 0.0)

    def test_invalid_physical_inputs_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.model.components(500.0e-9, 10_000.0, -1.0, 1.0, 1.0, 1.0)
        with self.assertRaises(ValueError):
            self.model.components(0.0, 10_000.0, 1.0, 1.0, 1.0, 1.0)

    def test_fixed_electron_neutral_kernel_matches_si_formula(self) -> None:
        model = ContinuumOpacityModel.from_constant_electron_neutral(
            AtomicSpecies("Fe", {}, {}),
            FIXED_ELECTRON_NEUTRAL_Q_M5,
        )
        wavelength = 500.0e-9
        temperature = 20_000.0
        ne = 2.0e23
        n0 = 3.0e24
        result = model.components(wavelength, temperature, n0, ne, 0.0, 0.0)
        expected = (
            float(stimulated_emission_factor(wavelength, temperature))
            * FIXED_ELECTRON_NEUTRAL_Q_M5
            * ne
            * n0
        )
        self.assertAlmostEqual(float(result.electron_neutral_m1) / expected, 1.0, places=14)
        self.assertEqual(float(result.photoionization_m1), 0.0)


class AtomicCatalogTests(unittest.TestCase):
    def test_bundled_copper_reference_is_strict_ready(self) -> None:
        root = Path(__file__).parents[1] / "data" / "reference"
        reference = AtomicDataCatalog(root).load("cu", mode="strict")
        self.assertEqual(reference.species.symbol, "Cu")
        self.assertEqual(reference.status.fidelity, "fixed_Q_continuum")
        self.assertEqual(reference.status.photoionization_charge_states, (0, 1, 2))
        self.assertEqual(reference.status.missing_components, ())
        self.assertIsNone(reference.momentum_transfer)
        self.assertEqual(
            reference.electron_neutral_constant_m5,
            FIXED_ELECTRON_NEUTRAL_Q_M5,
        )
        self.assertEqual(reference.status.electron_neutral_model, "project_fixed_Q")
        fingerprint = reference.fingerprint()
        self.assertIn("species.json", fingerprint["files"])
        self.assertNotIn("MT_01_01", fingerprint["files"])
        self.assertEqual(
            fingerprint["electron_neutral_fixed_Q"],
            {
                "Q_cm5": FIXED_ELECTRON_NEUTRAL_Q_CM5,
                "Q_m5": FIXED_ELECTRON_NEUTRAL_Q_M5,
            },
        )

    def test_missing_element_fails_strict_and_is_explicit_in_approximate_mode(self) -> None:
        root = Path(__file__).parents[1] / "data" / "reference"
        catalog = AtomicDataCatalog(root)
        with self.assertRaises(AtomicDataUnavailableError):
            catalog.load("Fe", mode="strict")
        reference = catalog.load("Fe", mode="approximate")
        self.assertEqual(reference.species.symbol, "Fe")
        self.assertEqual(reference.status.fidelity, "approximate_incomplete_continuum")
        self.assertEqual(
            reference.status.electron_neutral_model,
            "project_fixed_Q",
        )
        self.assertIn(
            "photoionization_charge_states:0,1,2",
            reference.status.missing_components,
        )
        self.assertEqual(reference.build_opacity_model().species.symbol, "Fe")

    def test_fixed_q_does_not_require_an_element_bundle_when_photoionization_is_off(self) -> None:
        root = Path(__file__).parents[1] / "data" / "reference"
        reference = AtomicDataCatalog(root).load(
            "As",
            mode="strict",
            require_photoionization=False,
        )
        self.assertEqual(reference.status.fidelity, "fixed_Q_continuum")
        self.assertEqual(reference.status.missing_components, ())
        self.assertEqual(reference.status.electron_neutral_model, "project_fixed_Q")


if __name__ == "__main__":
    unittest.main()
