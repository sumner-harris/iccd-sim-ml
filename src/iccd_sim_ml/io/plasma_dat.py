"""Parser for a single adaptive-mesh ``res_*_ns.dat`` export."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .schemas import PLASMA_DAT_COLUMNS, PLASMA_DAT_REQUIRED_COLUMN_COUNT

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class PlasmaTimestep:
    """One axisymmetric solver output in SI units."""

    source: Path
    source_key: str | None
    time_s: float | None
    z_m: FloatArray
    r_m: FloatArray
    density_number_m3: FloatArray
    mass_density_kg_m3: FloatArray
    temperature_K: FloatArray
    n0_m3: FloatArray
    ne_m3: FloatArray
    n1_m3: FloatArray
    n2_m3: FloatArray
    alpha_ib_en_248_m1: FloatArray
    alpha_ib_ei_248_m1: FloatArray
    alpha_pi_248_m1: FloatArray
    mesh_level: FloatArray
    quality_flags: tuple[str, ...]
    raw: FloatArray

    @property
    def size(self) -> int:
        return int(self.z_m.size)

    def validate(self) -> None:
        lengths = {
            self.z_m.size,
            self.r_m.size,
            self.temperature_K.size,
            self.n0_m3.size,
            self.ne_m3.size,
            self.n1_m3.size,
            self.n2_m3.size,
        }
        if len(lengths) != 1:
            raise ValueError("Plasma timestep fields do not have a common length")
        if self.raw.ndim != 2 or self.raw.shape[1] < PLASMA_DAT_REQUIRED_COLUMN_COUNT:
            raise ValueError(f"Expected at least {PLASMA_DAT_REQUIRED_COLUMN_COUNT} data columns")
        if not np.isfinite(self.z_m).all() or not np.isfinite(self.r_m).all():
            raise ValueError("Coordinate columns contain non-finite values")
        if np.any(self.r_m < 0):
            raise ValueError("Axisymmetric radial coordinates must be non-negative")
        for name in ("temperature_K", "n0_m3", "ne_m3", "n1_m3", "n2_m3"):
            values = getattr(self, name)
            if not np.isfinite(values).all():
                raise ValueError(f"{name} contains non-finite values")
            if np.any(values < 0):
                raise ValueError(f"{name} contains negative values")


def _quality_flags(column: dict[str, FloatArray]) -> tuple[str, ...]:
    flags: list[str] = []
    alpha_pi = column["alpha_pi_248"]
    mesh_level = column["mesh_level"]
    if alpha_pi.size and np.ptp(alpha_pi) == 0 and np.ptp(mesh_level) == 0:
        if np.isclose(mesh_level[0], 33.0):
            flags.append("corrupt_alpha_pi_and_mesh_level_export")
    return tuple(flags)


def plasma_timestep_from_array(
    raw: FloatArray,
    *,
    source: str | Path,
    source_key: str | None = None,
    time_s: float | None = None,
    mesh_level_index: int | None = None,
) -> PlasmaTimestep:
    """Build a validated timestep from an in-memory solver array.

    Saved 248-nm opacity columns are retained for diagnostics only. The image
    simulator recomputes wavelength-dependent opacity from state variables.
    """

    values = np.asarray(raw, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < PLASMA_DAT_REQUIRED_COLUMN_COUNT:
        raise ValueError(
            f"Data have shape {values.shape}; expected "
            f"(points, >= {PLASMA_DAT_REQUIRED_COLUMN_COUNT})"
        )
    source_column_count = values.shape[1]
    if mesh_level_index is not None and not 0 <= mesh_level_index < source_column_count:
        raise ValueError(
            f"mesh_level_index {mesh_level_index} is outside the {source_column_count}-column array"
        )
    if source_column_count == PLASMA_DAT_REQUIRED_COLUMN_COUNT:
        values = np.column_stack((values, np.full(values.shape[0], np.nan)))
    column = {spec.name: values[:, spec.index] for spec in PLASMA_DAT_COLUMNS}
    if mesh_level_index is not None:
        column["mesh_level"] = values[:, mesh_level_index]
    timestep = PlasmaTimestep(
        # Avoid dereferencing mapped/network files: on Windows a read-only SMB
        # file can reject the metadata handle used by ``Path.resolve()`` even
        # when the data file itself is readable.
        source=Path(source).expanduser().absolute(),
        source_key=source_key,
        time_s=time_s,
        z_m=column["z"],
        r_m=column["r"],
        density_number_m3=column["density_number"],
        mass_density_kg_m3=column["mass_density"],
        temperature_K=column["temperature"],
        n0_m3=column["n0"],
        ne_m3=column["ne"],
        n1_m3=column["n1"],
        n2_m3=column["n2"],
        alpha_ib_en_248_m1=column["alpha_ib_en_248"],
        alpha_ib_ei_248_m1=column["alpha_ib_ei_248"],
        alpha_pi_248_m1=column["alpha_pi_248"],
        mesh_level=column["mesh_level"],
        quality_flags=_quality_flags(column),
        raw=values,
    )
    timestep.validate()
    return timestep


def _time_from_name(path: Path) -> float | None:
    match = re.search(r"res_([0-9]+(?:\.[0-9]+)?)_ns", path.name, flags=re.IGNORECASE)
    return None if match is None else float(match.group(1)) * 1e-9


def parse_header(path: str | Path) -> tuple[str, ...]:
    """Return header field names in their declared numerical order."""

    source = Path(path)
    with source.open("r", encoding="utf-8") as stream:
        header = stream.readline().strip()
    if not header.startswith("#"):
        raise ValueError(f"{source} does not start with a solver column header")
    tokens = re.findall(r"(\d+):([^\s]+)", header)
    if not tokens:
        raise ValueError(f"Could not parse column declarations from {source}")
    return tuple(name for _, name in sorted(tokens, key=lambda item: int(item[0])))


def load_plasma_timestep(path: str | Path) -> PlasmaTimestep:
    """Load the standard 21-column solver export and assign SI-aware fields."""

    source = Path(path).expanduser().absolute()
    header_names = parse_header(source)
    if len(header_names) < PLASMA_DAT_REQUIRED_COLUMN_COUNT:
        raise ValueError(
            f"Header declares {len(header_names)} columns; "
            f"expected {PLASMA_DAT_REQUIRED_COLUMN_COUNT}"
        )
    raw = np.loadtxt(source, comments="#", dtype=np.float64, ndmin=2)
    if raw.shape[1] < PLASMA_DAT_REQUIRED_COLUMN_COUNT:
        raise ValueError(
            f"Data contain {raw.shape[1]} columns; expected {PLASMA_DAT_REQUIRED_COLUMN_COUNT}"
        )

    return plasma_timestep_from_array(raw, source=source, time_s=_time_from_name(source))
