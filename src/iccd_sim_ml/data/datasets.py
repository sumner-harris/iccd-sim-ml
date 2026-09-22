"""PyTorch dataset for immutable, precomputed ``.npz`` videos."""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .manifest import DatasetManifest, SampleRecord, as_manifest
from .scalers import ScalerBundle

try:
    import torch
    from torch.utils.data import Dataset as TorchDataset
except ModuleNotFoundError as exc:  # pragma: no cover - exercised in environments without ML extras
    if exc.name != "torch":
        raise
    torch = None

    class TorchDataset:  # type: ignore[no-redef]
        """Import-time placeholder; construction raises a useful error."""


Task = Literal["regression", "classification"]


def _require_torch() -> None:
    if torch is None:
        raise ImportError(
            "PrecomputedVideoDataset requires PyTorch. Install the ML extras with "
            "`pip install iccd-sim-ml[ml]`."
        )


@dataclass(frozen=True)
class NPZKeys:
    video: str = "video"
    features: str = "conditions"
    regression_target: str = "regression_targets"
    class_index: str = "class_index"
    times_s: str = "times_s"


def save_precomputed_sample(
    path: str | Path,
    *,
    video: ArrayLike,
    conditions: ArrayLike | None = None,
    regression_targets: ArrayLike | None = None,
    class_index: int | None = None,
    times_s: ArrayLike | None = None,
    compressed: bool = True,
) -> Path:
    """Atomically write one imaging-pipeline product in the dataset format."""

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    video_array = np.asarray(video, dtype=np.float32)
    if video_array.ndim not in {3, 4}:
        raise ValueError("video must have shape (T,H,W) or (C,T,H,W)")
    if not np.isfinite(video_array).all():
        raise ValueError("video contains non-finite values")
    arrays: dict[str, NDArray[Any]] = {"video": video_array}
    if conditions is not None:
        arrays["conditions"] = np.asarray(conditions, dtype=np.float32).reshape(-1)
    if regression_targets is not None:
        arrays["regression_targets"] = np.asarray(regression_targets, dtype=np.float32).reshape(-1)
    if class_index is not None:
        if int(class_index) < 0:
            raise ValueError("class_index must be non-negative")
        arrays["class_index"] = np.asarray(int(class_index), dtype=np.int64)
    if times_s is not None:
        time_array = np.asarray(times_s, dtype=np.float64)
        if time_array.ndim != 1:
            raise ValueError("times_s must be a one-dimensional array")
        time_steps = video_array.shape[0] if video_array.ndim == 3 else video_array.shape[1]
        if time_array.size != time_steps:
            raise ValueError(
                f"times_s has length {time_array.size}, but video has {time_steps} frames"
            )
        if time_array.size > 1 and np.any(np.diff(time_array) <= 0.0):
            raise ValueError("times_s must be strictly increasing")
        arrays["times_s"] = time_array
    for name, array in arrays.items():
        if name != "class_index" and not np.isfinite(array).all():
            raise ValueError(f"{name} contains non-finite values")

    temporary = destination.with_name(f".{destination.name}.tmp")
    writer = np.savez_compressed if compressed else np.savez
    try:
        with temporary.open("wb") as stream:
            writer(stream, **arrays)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


class PrecomputedVideoDataset(TorchDataset):
    """Load already-simulated videos; no physics is executed in ``__getitem__``.

    Samples are returned as dictionaries so identity metadata survives batching
    and can be audited alongside predictions.
    """

    def __init__(
        self,
        manifest: DatasetManifest | str | Path | Iterable[SampleRecord],
        *,
        sample_ids: Iterable[str] | None = None,
        task: Task = "regression",
        scalers: ScalerBundle | None = None,
        keys: NPZKeys | None = None,
        class_to_index: Mapping[str, int] | None = None,
    ) -> None:
        _require_torch()
        self.manifest = as_manifest(manifest)
        self.records = (
            self.manifest.records if sample_ids is None else self.manifest.select(tuple(sample_ids))
        )
        if task not in {"regression", "classification"}:
            raise ValueError("task must be 'regression' or 'classification'")
        self.task = task
        self.scalers = scalers
        self.keys = NPZKeys() if keys is None else keys
        if class_to_index is None:
            elements = sorted({record.element for record in self.manifest.records})
            class_to_index = {element: index for index, element in enumerate(elements)}
        self.class_to_index = dict(class_to_index)
        if len(set(self.class_to_index.values())) != len(self.class_to_index):
            raise ValueError("class_to_index values must be unique")

    def __len__(self) -> int:
        return len(self.records)

    @staticmethod
    def _canonical_video(video: NDArray[Any], path: Path) -> NDArray[np.float32]:
        array = np.asarray(video, dtype=np.float32)
        if array.ndim == 3:
            array = array[np.newaxis, ...]
        if array.ndim != 4:
            raise ValueError(
                f"Video in {path} has shape {array.shape}; expected (T,H,W) or (C,T,H,W)"
            )
        if not np.isfinite(array).all():
            raise ValueError(f"Video in {path} contains non-finite values")
        return np.ascontiguousarray(array)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        path = self.manifest.resolve_path(record)
        with np.load(path, allow_pickle=False) as sample:
            if self.keys.video not in sample:
                raise KeyError(f"Precomputed sample {path} has no {self.keys.video!r} array")
            video = self._canonical_video(np.array(sample[self.keys.video], copy=True), path)
            if self.scalers is not None and self.scalers.video is not None:
                video = self.scalers.video.transform(video).astype(np.float32)

            if self.keys.features in sample:
                features = np.array(
                    sample[self.keys.features], dtype=np.float32, copy=True
                ).reshape(-1)
            else:
                features = np.empty(0, dtype=np.float32)
            if not np.isfinite(features).all():
                raise ValueError(f"Features in {path} contain non-finite values")
            if self.scalers is not None and self.scalers.features is not None:
                features = self.scalers.features.transform(features).astype(np.float32)

            if self.task == "regression":
                if self.keys.regression_target not in sample:
                    raise KeyError(
                        f"Regression sample {path} has no {self.keys.regression_target!r} array"
                    )
                target = np.array(
                    sample[self.keys.regression_target], dtype=np.float32, copy=True
                ).reshape(-1)
                if not np.isfinite(target).all():
                    raise ValueError(f"Regression target in {path} contains non-finite values")
                if self.scalers is not None and self.scalers.targets is not None:
                    target = self.scalers.targets.transform(target).astype(np.float32)
                target_tensor = torch.from_numpy(np.ascontiguousarray(target))
            else:
                if self.keys.class_index in sample:
                    class_index = int(np.asarray(sample[self.keys.class_index]).item())
                else:
                    if record.element not in self.class_to_index:
                        raise KeyError(
                            f"No class index is configured for element {record.element!r}"
                        )
                    class_index = self.class_to_index[record.element]
                if class_index < 0:
                    raise ValueError(f"Class index in {path} must be non-negative")
                target_tensor = torch.tensor(class_index, dtype=torch.long)

        return {
            "video": torch.from_numpy(np.ascontiguousarray(video)),
            "features": torch.from_numpy(np.ascontiguousarray(features)),
            "target": target_tensor,
            "sample_id": record.sample_id,
            "element": record.element,
            "simulation_id": record.simulation_id,
        }


class JointPrecomputedVideoDataset(TorchDataset):
    """Load videos and all labels required for joint conditional-VAE training.

    Unlike :class:`PrecomputedVideoDataset`, this dataset uses explicit names
    for the two different kinds of conditions. ``laser_conditions`` are model
    inputs available to every operating mode, while ``material_properties``
    are both regression targets and generation conditions. Keeping them
    separate makes it harder to accidentally feed a regression target into
    the inverse-prediction head.

    ``expected_video_shape`` is expressed in canonical ``(C,T,H,W)`` order; a
    three-dimensional ``(T,H,W)`` value is accepted as shorthand for one
    channel. If ``expected_times_s`` is supplied, every sample must contain an
    exactly matching time coordinate. These checks deliberately fail before a
    batch can silently mix incompatible simulations.
    """

    def __init__(
        self,
        manifest: DatasetManifest | str | Path | Iterable[SampleRecord],
        *,
        sample_ids: Iterable[str] | None = None,
        scalers: ScalerBundle | None = None,
        keys: NPZKeys | None = None,
        class_to_index: Mapping[str, int] | None = None,
        expected_video_shape: tuple[int, ...] | None = None,
        expected_times_s: ArrayLike | None = None,
        require_times_s: bool = False,
    ) -> None:
        _require_torch()
        self.manifest = as_manifest(manifest)
        self.records = (
            self.manifest.records if sample_ids is None else self.manifest.select(tuple(sample_ids))
        )
        self.scalers = scalers
        self.keys = NPZKeys() if keys is None else keys

        if class_to_index is None:
            elements = sorted({record.element for record in self.manifest.records})
            class_to_index = {element: index for index, element in enumerate(elements)}
        self.class_to_index = dict(class_to_index)
        class_values = tuple(self.class_to_index.values())
        if len(set(class_values)) != len(class_values):
            raise ValueError("class_to_index values must be unique")
        if any(not isinstance(value, int) or value < 0 for value in class_values):
            raise ValueError("class_to_index values must be non-negative integers")

        if expected_video_shape is None:
            self.expected_video_shape: tuple[int, int, int, int] | None = None
        else:
            shape = tuple(int(value) for value in expected_video_shape)
            if len(shape) == 3:
                shape = (1, *shape)
            if len(shape) != 4 or any(value <= 0 for value in shape):
                raise ValueError(
                    "expected_video_shape must contain positive (T,H,W) or (C,T,H,W) values"
                )
            self.expected_video_shape = shape

        self.expected_times_s: NDArray[np.float64] | None = None
        if expected_times_s is not None:
            expected_times = np.asarray(expected_times_s, dtype=np.float64)
            if expected_times.ndim != 1 or expected_times.size == 0:
                raise ValueError("expected_times_s must be a non-empty one-dimensional array")
            if not np.isfinite(expected_times).all():
                raise ValueError("expected_times_s contains non-finite values")
            if expected_times.size > 1 and np.any(np.diff(expected_times) <= 0.0):
                raise ValueError("expected_times_s must be strictly increasing")
            if (
                self.expected_video_shape is not None
                and expected_times.size != self.expected_video_shape[1]
            ):
                raise ValueError(
                    "expected_times_s length must match the expected video time dimension"
                )
            self.expected_times_s = np.array(expected_times, copy=True)
            require_times_s = True
        self.require_times_s = bool(require_times_s)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        path = self.manifest.resolve_path(record)
        with np.load(path, allow_pickle=False) as sample:
            required = (self.keys.video, self.keys.features, self.keys.regression_target)
            missing = [key for key in required if key not in sample]
            if missing:
                raise KeyError(f"Joint-training sample {path} is missing arrays: {missing}")

            video = PrecomputedVideoDataset._canonical_video(
                np.array(sample[self.keys.video], copy=True), path
            )
            if self.expected_video_shape is not None and video.shape != self.expected_video_shape:
                raise ValueError(
                    f"Video in {path} has canonical shape {video.shape}; "
                    f"expected {self.expected_video_shape}"
                )

            laser_conditions = np.array(
                sample[self.keys.features], dtype=np.float32, copy=True
            ).reshape(-1)
            material_properties = np.array(
                sample[self.keys.regression_target], dtype=np.float32, copy=True
            ).reshape(-1)
            if laser_conditions.size == 0:
                raise ValueError(f"Laser conditions in {path} are empty")
            if material_properties.size == 0:
                raise ValueError(f"Material properties in {path} are empty")
            if not np.isfinite(laser_conditions).all():
                raise ValueError(f"Laser conditions in {path} contain non-finite values")
            if not np.isfinite(material_properties).all():
                raise ValueError(f"Material properties in {path} contain non-finite values")

            if self.keys.class_index in sample:
                class_index = int(np.asarray(sample[self.keys.class_index]).item())
            else:
                if record.element not in self.class_to_index:
                    raise KeyError(f"No class index is configured for element {record.element!r}")
                class_index = self.class_to_index[record.element]
            if class_index < 0:
                raise ValueError(f"Class index in {path} must be non-negative")

            times_s: NDArray[np.float64] | None = None
            if self.keys.times_s in sample:
                times_s = np.array(sample[self.keys.times_s], dtype=np.float64, copy=True)
                if times_s.ndim != 1:
                    raise ValueError(f"times_s in {path} must be one-dimensional")
                if times_s.size != video.shape[1]:
                    raise ValueError(
                        f"times_s in {path} has length {times_s.size}, "
                        f"but video has {video.shape[1]} frames"
                    )
                if not np.isfinite(times_s).all():
                    raise ValueError(f"times_s in {path} contains non-finite values")
                if times_s.size > 1 and np.any(np.diff(times_s) <= 0.0):
                    raise ValueError(f"times_s in {path} must be strictly increasing")
                if self.expected_times_s is not None and not np.array_equal(
                    times_s, self.expected_times_s
                ):
                    raise ValueError(f"times_s in {path} does not exactly match expected_times_s")
            elif self.require_times_s:
                raise KeyError(f"Joint-training sample {path} has no {self.keys.times_s!r} array")

        if self.scalers is not None:
            if self.scalers.video is not None:
                video = self.scalers.video.transform(video).astype(np.float32)
            if self.scalers.features is not None:
                laser_conditions = self.scalers.features.transform(laser_conditions).astype(
                    np.float32
                )
            if self.scalers.targets is not None:
                material_properties = self.scalers.targets.transform(material_properties).astype(
                    np.float32
                )

        result: dict[str, Any] = {
            "video": torch.from_numpy(np.ascontiguousarray(video)),
            "laser_conditions": torch.from_numpy(np.ascontiguousarray(laser_conditions)),
            "material_properties": torch.from_numpy(np.ascontiguousarray(material_properties)),
            "class_index": torch.tensor(class_index, dtype=torch.long),
            "sample_id": record.sample_id,
            "element": record.element,
            "simulation_id": record.simulation_id,
        }
        if times_s is not None:
            result["times_s"] = torch.from_numpy(np.ascontiguousarray(times_s))
        return result
