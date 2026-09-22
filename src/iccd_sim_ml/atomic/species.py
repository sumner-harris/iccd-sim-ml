"""Material-specific atomic species definitions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from .levels import EnergyLevels, load_energy_levels


@dataclass(frozen=True)
class AtomicSpecies:
    symbol: str
    levels_by_charge: Mapping[int, EnergyLevels]
    ionization_energy_ev_by_charge: Mapping[int, float]

    def __post_init__(self) -> None:
        symbol = self.symbol.strip()
        if not symbol or not symbol[0].isupper() or (len(symbol) > 1 and not symbol[1:].islower()):
            raise ValueError("Atomic symbol must use canonical capitalization, such as 'Cu'")
        levels = {int(charge): value for charge, value in self.levels_by_charge.items()}
        energies = {
            int(charge): float(value)
            for charge, value in self.ionization_energy_ev_by_charge.items()
        }
        if any(charge < 0 for charge in levels) or any(charge < 0 for charge in energies):
            raise ValueError("Atomic charge states must be non-negative")
        if any(not value > 0.0 for value in energies.values()):
            raise ValueError("Ionization energies must be positive")
        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "levels_by_charge", MappingProxyType(levels))
        object.__setattr__(
            self,
            "ionization_energy_ev_by_charge",
            MappingProxyType(energies),
        )

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
