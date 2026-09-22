"""Deterministic HDF5 subset discovery and offline cache construction."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from iccd_sim_ml.data import DatasetManifest, SampleRecord
from iccd_sim_ml.imaging.grid import build_parallel_side_view, resample_timestep
from iccd_sim_ml.io import get_h5_simulation

CONDITION_NAMES = ("laser_power_wcm", "rspot")
TARGET_NAMES = (
    "cp_metal",
    "h_vapor",
    "kappa_metal",
    "laser_reflectivity",
    "mass_density_metal",
    "t_boil",
    "tcrit",
)
H5_FRAME_COLUMN_COUNT = 34
_FRAME_PATTERN = re.compile(r"res_([0-9.+\-eE]+)_ns(?:\.dat)?$", re.IGNORECASE)


@dataclass(frozen=True)
class SourceSample:
    """One source simulation selected for preprocessing."""

    sample_id: str
    element: str
    class_index: int
    source: Path
    simulation_key: str
    attrs: dict[str, Any]
    timestep_keys: tuple[str, ...]


@dataclass(frozen=True)
class ProxyCacheConfig:
    """Fast state-projection backend used only for pipeline smoke tests.

    This deliberately does not claim ICCD radiance. It projects a fixed,
    non-negative function of temperature and charge density through the
    axisymmetric state so cache/dataset/training/reporting code can be tested
    across elements before element-specific atomic data are available.
    """

    frame_count: int = 8
    image_width: int = 32
    image_height: int = 32
    line_of_sight_points: int = 32
    radial_max_m: float = 0.015
    axial_max_m: float = 0.032
    electron_density_scale_m3: float = 1.0e16
    temperature_offset_K: float = 300.0
    temperature_scale_K: float = 1000.0

    def __post_init__(self) -> None:
        integer_fields = (
            self.frame_count,
            self.image_width,
            self.image_height,
            self.line_of_sight_points,
        )
        if any(value < 2 for value in integer_fields):
            raise ValueError("Cache frame/grid/LOS counts must all be at least two")
        positive = (
            self.radial_max_m,
            self.axial_max_m,
            self.electron_density_scale_m3,
            self.temperature_scale_K,
        )
        if any(not np.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("Cache physical scales and bounds must be finite and positive")


@dataclass(frozen=True)
class CacheResult:
    manifest: DatasetManifest
    hits: tuple[str, ...]
    misses: tuple[str, ...]


def _h5py():
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("HDF5 cache construction requires h5py") from exc
    return h5py


def _json_safe(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _frame_keys(group: Any, h5py: Any) -> tuple[str, ...]:
    pairs: list[tuple[float, str]] = []
    for key in group:
        match = _FRAME_PATTERN.fullmatch(key)
        obj = group[key]
        if match is None or not isinstance(obj, h5py.Dataset):
            continue
        if obj.ndim != 2 or obj.shape[1] != H5_FRAME_COLUMN_COUNT:
            continue
        pairs.append((float(match.group(1)), key))
    pairs.sort(key=lambda item: (item[0], item[1]))
    return tuple(key for _, key in pairs)


def discover_balanced_subset(
    data_dir: str | Path,
    *,
    elements: tuple[str, ...],
    simulations_per_element: int,
    minimum_frames: int,
) -> tuple[SourceSample, ...]:
    """Select deterministic interior-condition simulations for each element."""

    if simulations_per_element < 1 or minimum_frames < 1:
        raise ValueError("simulations_per_element and minimum_frames must be positive")
    root = Path(data_dir).expanduser().absolute()
    h5py = _h5py()
    class_to_index = {element: index for index, element in enumerate(sorted(elements))}
    selected: list[SourceSample] = []
    for element in elements:
        matches = sorted(root.glob(f"{element}_*.h5"))
        if len(matches) != 1:
            raise ValueError(
                f"Expected one HDF5 file matching {element}_*.h5 in {root}, found {matches}"
            )
        source = matches[0].absolute()
        candidates: list[SourceSample] = []
        with h5py.File(source, "r") as handle:
            for group_key in sorted(handle.keys()):
                group = handle[group_key]
                if not isinstance(group, h5py.Group):
                    continue
                timestep_keys = _frame_keys(group, h5py)
                if len(timestep_keys) < minimum_frames:
                    continue
                attrs = {str(key): _json_safe(value) for key, value in group.attrs.items()}
                declared_element = str(attrs.get("element", element))
                if declared_element != element:
                    continue
                candidates.append(
                    SourceSample(
                        sample_id=f"{element}__{group_key}",
                        element=element,
                        class_index=class_to_index[element],
                        source=source,
                        simulation_key=group_key,
                        attrs=attrs,
                        timestep_keys=timestep_keys,
                    )
                )
        candidates.sort(
            key=lambda item: (
                float(item.attrs.get("laser_power_wcm", np.inf)),
                float(item.attrs.get("rspot", np.inf)),
                item.simulation_key,
            )
        )
        if len(candidates) < simulations_per_element:
            raise ValueError(
                f"Element {element} has {len(candidates)} valid simulations; "
                f"{simulations_per_element} requested"
            )
        # Interior quantiles avoid choosing only the most extreme laser cases.
        fractions = np.arange(1, simulations_per_element + 1) / (simulations_per_element + 1)
        indices = np.rint(fractions * (len(candidates) - 1)).astype(int)
        if np.unique(indices).size != simulations_per_element:
            raise ValueError("Quantile selection did not produce unique simulations")
        selected.extend(candidates[index] for index in indices)
    return tuple(selected)


def _cache_request(sample: SourceSample, config: ProxyCacheConfig) -> dict[str, Any]:
    return {
        "schema": 1,
        "backend": "plasma_state_proxy_smoke_test",
        "source": str(sample.source),
        "simulation_key": sample.simulation_key,
        "attrs": sample.attrs,
        "timestep_keys": list(sample.timestep_keys[: config.frame_count]),
        "config": asdict(config),
    }


def _cache_key(request: dict[str, Any]) -> str:
    payload = json.dumps(request, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _valid_cache(path: Path, cache_key: str, config: ProxyCacheConfig) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as product:
            required = {
                "video",
                "times_s",
                "conditions",
                "regression_targets",
                "class_index",
                "metadata_json",
            }
            if not required.issubset(product.files):
                return False
            if product["video"].shape != (
                config.frame_count,
                config.image_width,
                config.image_height,
            ):
                return False
            if not np.isfinite(product["video"]).all():
                return False
            metadata = json.loads(str(product["metadata_json"].item()))
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False
    return metadata.get("cache_key") == cache_key


def _proxy_frame(timestep: Any, config: ProxyCacheConfig) -> np.ndarray:
    grid = resample_timestep(
        timestep,
        radial_points=config.image_width,
        axial_points=config.image_height,
        radial_max_m=config.radial_max_m,
        axial_max_m=config.axial_max_m,
    )
    state = build_parallel_side_view(
        grid,
        x_points=config.image_width,
        y_points=config.line_of_sight_points,
    )
    thermal = np.maximum(state.temperature_K - config.temperature_offset_K, 0.0)
    thermal /= config.temperature_scale_K
    charge_density = state.ne_m3 + state.n1_m3 + 4.0 * state.n2_m3
    charge = np.log1p(charge_density / config.electron_density_scale_m3)
    proxy = np.mean(thermal * charge, axis=0)
    if not np.isfinite(proxy).all() or np.any(proxy < 0.0):
        raise FloatingPointError("State projection produced invalid values")
    return proxy.astype(np.float32)


def _write_cache(path: Path, sample: SourceSample, config: ProxyCacheConfig) -> None:
    simulation = get_h5_simulation(sample.source, sample.simulation_key)
    indices = list(range(config.frame_count))
    timesteps = simulation.iter_timesteps(indices)
    video = np.stack([_proxy_frame(timestep, config) for timestep in timesteps])
    attrs = simulation.attrs
    missing = [name for name in (*CONDITION_NAMES, *TARGET_NAMES) if name not in attrs]
    if missing:
        raise KeyError(f"Simulation {simulation.key} is missing attributes: {missing}")
    request = _cache_request(sample, config)
    metadata = {
        "cache_key": _cache_key(request),
        "cache_request": request,
        "quantity": "dimensionless_plasma_state_proxy",
        "scientific_use": "pipeline_smoke_test_only",
        "condition_names": CONDITION_NAMES,
        "regression_target_names": TARGET_NAMES,
    }
    arrays = {
        "video": video,
        "times_s": simulation.times_s[indices],
        "conditions": np.asarray([attrs[name] for name in CONDITION_NAMES], dtype=np.float32),
        "regression_targets": np.asarray([attrs[name] for name in TARGET_NAMES], dtype=np.float32),
        "class_index": np.asarray(sample.class_index, dtype=np.int64),
        "metadata_json": json.dumps(metadata, sort_keys=True),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def ensure_proxy_cache(
    samples: tuple[SourceSample, ...],
    cache_dir: str | Path,
    config: ProxyCacheConfig,
) -> CacheResult:
    """Reuse valid NPZs and build only missing/stale proxy products."""

    destination = Path(cache_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    hits: list[str] = []
    misses: list[str] = []
    records: list[SampleRecord] = []
    for sample in samples:
        output = destination / f"{sample.sample_id}.npz"
        key = _cache_key(_cache_request(sample, config))
        if _valid_cache(output, key, config):
            hits.append(sample.sample_id)
        else:
            _write_cache(output, sample, config)
            misses.append(sample.sample_id)
        records.append(
            SampleRecord(
                sample_id=sample.sample_id,
                path=output,
                element=sample.element,
                simulation_id=f"{sample.source.name}::{sample.simulation_key}",
                family_id=sample.element,
                metadata={"cache_backend": "plasma_state_proxy_smoke_test"},
            )
        )
    manifest_path = destination / "manifest.json"
    DatasetManifest(tuple(records)).save(manifest_path)
    return CacheResult(
        manifest=DatasetManifest.load(manifest_path),
        hits=tuple(hits),
        misses=tuple(misses),
    )
