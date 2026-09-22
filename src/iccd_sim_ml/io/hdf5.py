"""Read simulation sequences from the project's HDF5 container format."""

from __future__ import annotations

import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .plasma_dat import PlasmaTimestep, plasma_timestep_from_array

H5_FRAME_COLUMN_COUNT = 34
H5_MESH_LEVEL_COLUMN = 21


def _h5py():
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("HDF5 support requires `pip install iccd-sim-ml[io]`") from exc
    return h5py


def _time_ns(name: str) -> float | None:
    match = re.fullmatch(r"res_([0-9.+\-eE]+)_ns(?:\.dat)?", name, flags=re.IGNORECASE)
    if match is None:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _json_safe(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


@dataclass(frozen=True)
class H5Simulation:
    """Lazy description of one simulation group with no retained file handle."""

    source: Path
    key: str
    attrs: dict[str, Any]
    timestep_keys: tuple[str, ...]
    times_s: np.ndarray
    index_warnings: tuple[str, ...] = ()

    def iter_timesteps(self, indices: Sequence[int] | None = None) -> Iterator[PlasmaTimestep]:
        selected = range(len(self.timestep_keys)) if indices is None else indices
        h5py = _h5py()
        with h5py.File(self.source, "r") as handle:
            group = handle[self.key]
            for index in selected:
                dataset_key = self.timestep_keys[index]
                raw = np.asarray(group[dataset_key], dtype=np.float64)
                time = self.times_s[index]
                yield plasma_timestep_from_array(
                    raw,
                    source=self.source,
                    source_key=f"{self.key}/{dataset_key}",
                    time_s=float(time) if np.isfinite(time) else None,
                    mesh_level_index=H5_MESH_LEVEL_COLUMN,
                )


def _index_group(source: Path, group_key: str, group: Any, h5py: Any) -> H5Simulation | None:
    pairs: list[tuple[float, str]] = []
    warnings: list[str] = []
    for key in group.keys():
        obj = group[key]
        time_ns = _time_ns(key)
        if isinstance(obj, h5py.Dataset) and time_ns is not None:
            if obj.ndim != 2 or obj.shape[1] != H5_FRAME_COLUMN_COUNT:
                warnings.append(
                    f"ignored {key}: expected a two-dimensional frame with exactly "
                    f"{H5_FRAME_COLUMN_COUNT} columns"
                )
            elif not np.issubdtype(obj.dtype, np.number):
                warnings.append(f"ignored {key}: frame dtype {obj.dtype} is not numeric")
            else:
                pairs.append((time_ns, key))
    pairs.sort(key=lambda item: (item[0], item[1]))
    if not pairs:
        return None
    counts: dict[float, int] = {}
    for time_ns, _ in pairs:
        counts[time_ns] = counts.get(time_ns, 0) + 1
    duplicate_times = sorted(time for time, count in counts.items() if count > 1)
    if duplicate_times:
        warnings.append(
            "duplicate frame times kept by lexical key: "
            + ", ".join(f"{time:g} ns" for time in duplicate_times)
        )
    pairs = list({time_ns: key for time_ns, key in reversed(pairs)}.items())
    pairs.sort(key=lambda item: item[0])
    times_s = np.asarray([time_ns * 1e-9 for time_ns, _ in pairs], dtype=np.float64)
    return H5Simulation(
        source=source,
        key=group_key,
        attrs={str(key): _json_safe(value) for key, value in group.attrs.items()},
        timestep_keys=tuple(key for _, key in pairs),
        times_s=times_s,
        index_warnings=tuple(warnings),
    )


def list_h5_simulations(path: str | Path) -> list[H5Simulation]:
    """Index all top-level simulation groups deterministically."""

    # ``Path.resolve()`` opens the path on Windows.  Some read-only SMB shares
    # reject that metadata handle with a byte-range lock error even though
    # HDF5 can safely open the file for reading.  ``absolute()`` gives us a
    # stable path without dereferencing the network file.
    source = Path(path).expanduser().absolute()
    h5py = _h5py()
    result: list[H5Simulation] = []
    with h5py.File(source, "r") as handle:
        for group_key in sorted(handle.keys()):
            group = handle[group_key]
            if not isinstance(group, h5py.Group):
                continue
            indexed = _index_group(source, group_key, group, h5py)
            if indexed is not None:
                result.append(indexed)
    return result


def get_h5_simulation(path: str | Path, key: str | None = None) -> H5Simulation:
    """Return a named simulation or the first deterministically sorted group."""

    if key is None:
        simulations = list_h5_simulations(path)
        if not simulations:
            raise ValueError(f"No simulation groups containing res* datasets were found in {path}")
        return simulations[0]
    source = Path(path).expanduser().absolute()
    h5py = _h5py()
    with h5py.File(source, "r") as handle:
        if key not in handle or not isinstance(handle[key], h5py.Group):
            available = ", ".join(sorted(handle.keys())[:10])
            raise KeyError(f"Simulation {key!r} not found. Available groups include: {available}")
        simulation = _index_group(source, key, handle[key], h5py)
    if simulation is None:
        raise ValueError(
            f"Simulation {key!r} contains no valid {H5_FRAME_COLUMN_COUNT}-column res* frames"
        )
    return simulation


def select_representative_simulation(simulations: Sequence[H5Simulation]) -> H5Simulation:
    """Choose a central laser condition, then prefer a longer sequence.

    Distance is measured in spot radius and log laser power after robust
    median scaling. If those attributes are unavailable, the longest sequence
    (then lexical group key) is selected deterministically.
    """

    if not simulations:
        raise ValueError("At least one simulation is required")
    usable: list[tuple[H5Simulation, float, float]] = []
    for simulation in simulations:
        try:
            radius = float(simulation.attrs["rspot"])
            power = float(simulation.attrs["laser_power_wcm"])
        except (KeyError, TypeError, ValueError):
            continue
        if np.isfinite(radius) and radius > 0 and np.isfinite(power) and power > 0:
            usable.append((simulation, radius, np.log(power)))
    if not usable:
        return sorted(simulations, key=lambda item: (-len(item.timestep_keys), item.key))[0]
    conditions = np.asarray([(radius, log_power) for _, radius, log_power in usable])
    center = np.median(conditions, axis=0)
    scale = np.median(np.abs(conditions - center), axis=0)
    scale = np.where(scale > 0, scale, 1.0)
    distance = np.linalg.norm((conditions - center) / scale, axis=1)
    ranked = sorted(
        zip(usable, distance, strict=True),
        key=lambda item: (item[1], -len(item[0][0].timestep_keys), item[0][0].key),
    )
    return ranked[0][0][0]
