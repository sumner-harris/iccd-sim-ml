"""Material-specific atomic data bundles."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .levels import EnergyLevels, load_energy_levels


@dataclass(frozen=True)
class AtomicSpecies:
    symbol: str
    levels_by_charge: dict[int, EnergyLevels]
    ionization_energy_ev_by_charge: dict[int, float]

    def photoionization_charge_states(self) -> tuple[int, ...]:
        return tuple(
            charge
            for charge in sorted(self.ionization_energy_ev_by_charge)
            if charge in self.levels_by_charge and charge + 1 in self.levels_by_charge
        )


def load_copper_species(reference_directory: str | Path) -> AtomicSpecies:
    """Load the bundled Cu I--IV compact level tables.

    Ionization energies are in eV for Cu I, Cu II, and Cu III. They are kept
    in an explicit registry rather than inferred from incomplete line lists.
    """

    root = Path(reference_directory).expanduser().resolve()
    levels = {
        0: load_energy_levels(root / "CuIEnergyC.txt"),
        1: load_energy_levels(root / "CuIIEnergyC.txt"),
        2: load_energy_levels(root / "CuIIIEnergyC.txt"),
        3: load_energy_levels(root / "CuIVEnergyC.txt"),
    }
    return AtomicSpecies(
        symbol="Cu",
        levels_by_charge=levels,
        ionization_energy_ev_by_charge={0: 7.72638, 1: 20.29239, 2: 36.841},
    )
