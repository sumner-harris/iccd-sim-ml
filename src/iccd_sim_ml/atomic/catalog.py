"""Element-aware atomic reference catalog with explicit fidelity reporting."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from .collisions import MomentumTransferTable
from .levels import load_energy_levels
from .species import AtomicSpecies

AtomicMode = Literal["strict", "approximate"]

# Project-wide electron--neutral inverse-bremsstrahlung assumption. The source
# convention is cgs; the opacity implementation converts it to SI exactly once.
FIXED_ELECTRON_NEUTRAL_Q_CM5 = 1.0e-40
CM5_TO_M5 = 1.0e-10
FIXED_ELECTRON_NEUTRAL_Q_M5 = FIXED_ELECTRON_NEUTRAL_Q_CM5 * CM5_TO_M5
# Kept as a public alias for callers written before the fixed-Q policy was
# adopted. Both names refer to the same active model constant.
LEGACY_CONSTANT_ELECTRON_NEUTRAL_Q_M5 = FIXED_ELECTRON_NEUTRAL_Q_M5
REQUIRED_PHOTOIONIZATION_CHARGE_STATES = (0, 1, 2)


class AtomicDataUnavailableError(ValueError):
    """Raised when requested continuum components lack element data."""


def normalize_element_symbol(value: str) -> str:
    symbol = value.strip().title()
    if not symbol or len(symbol) > 2 or not symbol.isalpha():
        raise ValueError(f"Invalid element symbol {value!r}")
    return symbol


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class AtomicDataStatus:
    symbol: str
    mode: AtomicMode
    descriptor_available: bool
    electron_ion_model: str
    electron_neutral_model: str
    photoionization_charge_states: tuple[int, ...]
    missing_components: tuple[str, ...]
    warnings: tuple[str, ...]
    fidelity: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AtomicReference:
    """Resolved element data and the approximations selected for one run."""

    species: AtomicSpecies
    status: AtomicDataStatus
    descriptor: dict[str, Any]
    descriptor_path: Path | None
    referenced_files: tuple[Path, ...]
    momentum_transfer: MomentumTransferTable | None
    electron_neutral_constant_m5: float | None

    def build_opacity_model(self, *, quadrature_order: int = 48, chunk_size: int = 65_536):
        from .opacity import ContinuumOpacityModel

        return ContinuumOpacityModel(
            species=self.species,
            momentum_transfer=self.momentum_transfer,
            electron_neutral_constant_m5=self.electron_neutral_constant_m5,
            quadrature_order=quadrature_order,
            chunk_size=chunk_size,
        )

    def fingerprint(self) -> dict[str, Any]:
        files: dict[str, str] = {}
        for path in self.referenced_files:
            files[path.name] = f"sha256:{_sha256(path)}"
        if self.descriptor_path is not None:
            files[self.descriptor_path.name] = f"sha256:{_sha256(self.descriptor_path)}"
        electron_neutral = None
        if self.electron_neutral_constant_m5 is not None:
            electron_neutral = {
                "Q_cm5": self.electron_neutral_constant_m5 / CM5_TO_M5,
                "Q_m5": self.electron_neutral_constant_m5,
            }
        return {
            "symbol": self.species.symbol,
            "status": self.status.to_dict(),
            "electron_neutral_fixed_Q": electron_neutral,
            "descriptor": self.descriptor,
            "files": files,
        }

    def to_metadata(self) -> dict[str, Any]:
        return self.fingerprint()


class AtomicDataCatalog:
    """Resolve per-element atomic bundles from a versioned reference root.

    The root may be the catalog directory containing ``catalog.json`` or one
    element directory containing ``species.json``. Every element uses the
    project-wide fixed electron-neutral kernel. Strict mode fails whenever
    enabled photoionization lacks required element data. Approximate mode sets
    unavailable photoionization charge states to zero and records that choice.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        self.catalog_path = self.root / "catalog.json"
        self.catalog = self._read_json(self.catalog_path) if self.catalog_path.is_file() else {}
        fixed_q_cm5 = float(
            self.catalog.get("electron_neutral_fixed_Q_cm5", FIXED_ELECTRON_NEUTRAL_Q_CM5)
        )
        if not 0.0 < fixed_q_cm5 < float("inf"):
            raise ValueError("electron_neutral_fixed_Q_cm5 must be finite and positive")
        self.electron_neutral_fixed_q_cm5 = fixed_q_cm5
        self.electron_neutral_fixed_q_m5 = fixed_q_cm5 * CM5_TO_M5

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as stream:
            value = json.load(stream)
        if not isinstance(value, dict):
            raise ValueError(f"Expected a JSON object in {path}")
        return value

    @property
    def target_elements(self) -> tuple[str, ...]:
        raw = self.catalog.get("target_elements", ())
        return tuple(normalize_element_symbol(str(value)) for value in raw)

    def _descriptor_path(self, symbol: str) -> Path | None:
        direct = self.root / "species.json"
        if direct.is_file():
            return direct
        configured = self.catalog.get("species", {}).get(symbol)
        if configured is not None:
            candidate = self.root / str(configured)
            return candidate if candidate.is_file() else None
        candidate = self.root / symbol.lower() / "species.json"
        return candidate if candidate.is_file() else None

    def load(
        self,
        element: str,
        *,
        mode: AtomicMode = "strict",
        require_electron_neutral: bool = True,
        require_photoionization: bool = True,
    ) -> AtomicReference:
        symbol = normalize_element_symbol(element)
        if mode not in {"strict", "approximate"}:
            raise ValueError("mode must be 'strict' or 'approximate'")
        descriptor_path = self._descriptor_path(symbol)
        if descriptor_path is None:
            if mode == "strict" and require_photoionization:
                raise AtomicDataUnavailableError(
                    f"No atomic species bundle is registered for {symbol}. Add species.json, "
                    "level tables, and ionization energies, or use approximate mode for a "
                    "labeled run with missing photoionization."
                )
            return self._approximate_reference(
                symbol,
                mode,
                require_electron_neutral=require_electron_neutral,
                require_photoionization=require_photoionization,
            )

        descriptor = self._read_json(descriptor_path)
        declared = normalize_element_symbol(str(descriptor.get("symbol", "")))
        if declared != symbol:
            raise ValueError(
                f"Atomic descriptor {descriptor_path} declares {declared}, expected {symbol}"
            )
        base = descriptor_path.parent
        referenced: list[Path] = []
        levels = {}
        for raw_charge, filename in descriptor.get("level_files", {}).items():
            path = (base / str(filename)).resolve()
            if not path.is_file():
                raise FileNotFoundError(f"Missing level table for {symbol}: {path}")
            levels[int(raw_charge)] = load_energy_levels(path)
            referenced.append(path)
        ionization = {
            int(charge): float(value)
            for charge, value in descriptor.get("ionization_energy_ev_by_charge", {}).items()
        }
        species = AtomicSpecies(symbol, levels, ionization)

        momentum = None
        constant_q = None
        electron_neutral_model = "disabled"
        missing: list[str] = []
        warnings: list[str] = []
        if require_electron_neutral:
            constant_q = self.electron_neutral_fixed_q_m5
            electron_neutral_model = "project_fixed_Q"
            warnings.append(
                "electron-neutral inverse bremsstrahlung uses the declared project-wide "
                f"fixed Q={self.electron_neutral_fixed_q_cm5:.6g} cm^5 "
                f"({constant_q:.6g} m^5) for every element"
            )

        charges = species.photoionization_charge_states()
        missing_charges = tuple(
            charge for charge in REQUIRED_PHOTOIONIZATION_CHARGE_STATES if charge not in charges
        )
        if require_photoionization and missing_charges:
            missing.append("photoionization_charge_states:" + ",".join(map(str, missing_charges)))
            if mode == "strict":
                raise AtomicDataUnavailableError(
                    f"{symbol} lacks complete level/ionization data for photoionization "
                    f"from charge states {missing_charges}; the HDF5 state contains n0, n1, "
                    "and n2, so strict imaging requires charge states 0, 1, and 2"
                )
            warnings.append(
                "photoionization is zero for unavailable charge states "
                f"{missing_charges}; available states remain element specific"
            )

        fidelity = "fixed_Q_continuum" if not missing else "approximate_incomplete_continuum"
        status = AtomicDataStatus(
            symbol=symbol,
            mode=mode,
            descriptor_available=True,
            electron_ion_model="element_independent_kramers_gaunt_1",
            electron_neutral_model=electron_neutral_model,
            photoionization_charge_states=charges if require_photoionization else (),
            missing_components=tuple(missing),
            warnings=tuple(warnings),
            fidelity=fidelity,
        )
        return AtomicReference(
            species=species,
            status=status,
            descriptor=descriptor,
            descriptor_path=descriptor_path,
            referenced_files=tuple(referenced),
            momentum_transfer=momentum,
            electron_neutral_constant_m5=constant_q,
        )

    def _approximate_reference(
        self,
        symbol: str,
        mode: AtomicMode,
        *,
        require_electron_neutral: bool,
        require_photoionization: bool,
    ) -> AtomicReference:
        missing = []
        warnings = []
        constant_q = None
        electron_neutral_model = "disabled"
        if require_electron_neutral:
            constant_q = self.electron_neutral_fixed_q_m5
            electron_neutral_model = "project_fixed_Q"
            warnings.append(
                "electron-neutral inverse bremsstrahlung uses the declared project-wide "
                f"fixed Q={self.electron_neutral_fixed_q_cm5:.6g} cm^5 "
                f"({constant_q:.6g} m^5) for every element"
            )
        if require_photoionization:
            missing.append("photoionization_charge_states:0,1,2")
            warnings.append("photoionization is zero because no element bundle is available")
        species = AtomicSpecies(symbol, {}, {})
        status = AtomicDataStatus(
            symbol=symbol,
            mode=mode,
            descriptor_available=False,
            electron_ion_model="element_independent_kramers_gaunt_1",
            electron_neutral_model=electron_neutral_model,
            photoionization_charge_states=(),
            missing_components=tuple(missing),
            warnings=tuple(warnings),
            fidelity=("approximate_incomplete_continuum" if missing else "fixed_Q_continuum"),
        )
        return AtomicReference(
            species=species,
            status=status,
            descriptor={},
            descriptor_path=None,
            referenced_files=(),
            momentum_transfer=None,
            electron_neutral_constant_m5=constant_q,
        )


__all__ = [
    "AtomicDataCatalog",
    "AtomicDataStatus",
    "AtomicDataUnavailableError",
    "AtomicMode",
    "AtomicReference",
    "FIXED_ELECTRON_NEUTRAL_Q_CM5",
    "FIXED_ELECTRON_NEUTRAL_Q_M5",
    "LEGACY_CONSTANT_ELECTRON_NEUTRAL_Q_M5",
    "REQUIRED_PHOTOIONIZATION_CHARGE_STATES",
    "normalize_element_symbol",
]
