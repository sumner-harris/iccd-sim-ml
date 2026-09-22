"""Serializable standardizers fitted exclusively from training sample IDs."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .manifest import DatasetManifest, SampleRecord, as_manifest
from .splits import SplitManifest, validate_split

FloatArray = NDArray[np.float64]


class _RunningMoments:
    def __init__(self) -> None:
        self.count = 0
        self.mean: FloatArray | None = None
        self.m2: FloatArray | None = None

    def update(self, values: ArrayLike) -> None:
        array = np.asarray(values, dtype=np.float64)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if array.ndim != 2 or array.shape[0] == 0:
            raise ValueError("Standardizer observations must be a non-empty 1D or 2D array")
        if not np.isfinite(array).all():
            raise ValueError("Cannot fit a standardizer to non-finite values")
        batch_count = array.shape[0]
        batch_mean = array.mean(axis=0)
        centered = array - batch_mean
        batch_m2 = np.sum(centered * centered, axis=0)
        if self.mean is None:
            self.count = batch_count
            self.mean = batch_mean
            self.m2 = batch_m2
            return
        if self.mean.shape != batch_mean.shape:
            raise ValueError(
                f"Feature width changed from {self.mean.size} to {batch_mean.size} while fitting"
            )
        delta = batch_mean - self.mean
        combined = self.count + batch_count
        self.mean = self.mean + delta * batch_count / combined
        assert self.m2 is not None
        self.m2 = self.m2 + batch_m2 + delta * delta * self.count * batch_count / combined
        self.count = combined

    def finish(self) -> tuple[FloatArray, FloatArray, int]:
        if self.count == 0 or self.mean is None or self.m2 is None:
            raise ValueError("No observations were provided to the standardizer")
        scale = np.sqrt(np.maximum(self.m2 / self.count, 0.0))
        scale = np.where(scale > 0.0, scale, 1.0)
        return self.mean, scale, self.count


@dataclass(frozen=True)
class ArrayStandardizer:
    """Per-column population mean and standard deviation."""

    mean: FloatArray
    scale: FloatArray
    count: int
    names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        mean = np.asarray(self.mean, dtype=np.float64)
        scale = np.asarray(self.scale, dtype=np.float64)
        if mean.ndim != 1 or scale.shape != mean.shape:
            raise ValueError("mean and scale must be one-dimensional arrays of equal shape")
        if not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
            raise ValueError("Standardizer parameters must be finite with positive scales")
        if self.count <= 0:
            raise ValueError("count must be positive")
        names = tuple(self.names)
        if names and len(names) != mean.size:
            raise ValueError("names must match the number of standardized columns")
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "names", names)

    @classmethod
    def fit(cls, arrays: Iterable[ArrayLike], *, names: Sequence[str] = ()) -> ArrayStandardizer:
        moments = _RunningMoments()
        for values in arrays:
            moments.update(values)
        mean, scale, count = moments.finish()
        return cls(mean=mean, scale=scale, count=count, names=tuple(names))

    def transform(self, values: ArrayLike) -> NDArray[np.float64]:
        array = np.asarray(values, dtype=np.float64)
        if array.shape[-1:] != self.mean.shape:
            raise ValueError(
                f"Expected trailing feature dimension {self.mean.size}, got shape {array.shape}"
            )
        return (array - self.mean) / self.scale

    def inverse_transform(self, values: ArrayLike) -> NDArray[np.float64]:
        array = np.asarray(values, dtype=np.float64)
        if array.shape[-1:] != self.mean.shape:
            raise ValueError(
                f"Expected trailing feature dimension {self.mean.size}, got shape {array.shape}"
            )
        return array * self.scale + self.mean

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "count": self.count,
            "names": list(self.names),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ArrayStandardizer:
        return cls(
            mean=np.asarray(value["mean"], dtype=np.float64),
            scale=np.asarray(value["scale"], dtype=np.float64),
            count=int(value["count"]),
            names=tuple(value.get("names", ())),
        )


@dataclass(frozen=True)
class VideoStandardizer:
    """One global video mean/std, optionally following ``log1p`` compression."""

    mean: float
    scale: float
    count: int
    log1p: bool = True

    def __post_init__(self) -> None:
        if not np.isfinite(self.mean) or not np.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("Video standardizer requires a finite mean and positive scale")
        if self.count <= 0:
            raise ValueError("count must be positive")

    @classmethod
    def fit(cls, videos: Iterable[ArrayLike], *, log1p: bool = True) -> VideoStandardizer:
        moments = _RunningMoments()
        for video in videos:
            array = np.asarray(video, dtype=np.float64)
            if not np.isfinite(array).all():
                raise ValueError("Cannot fit video scaler to non-finite values")
            if log1p:
                if np.any(array < 0):
                    raise ValueError("log1p video standardization requires non-negative videos")
                array = np.log1p(array)
            moments.update(array.reshape(-1, 1))
        mean, scale, count = moments.finish()
        return cls(float(mean[0]), float(scale[0]), count, log1p)

    def transform(self, video: ArrayLike) -> NDArray[np.float64]:
        array = np.asarray(video, dtype=np.float64)
        if self.log1p:
            if np.any(array < 0):
                raise ValueError("log1p video standardization requires non-negative videos")
            array = np.log1p(array)
        return (array - self.mean) / self.scale

    def inverse_transform(self, video: ArrayLike) -> NDArray[np.float64]:
        array = np.asarray(video, dtype=np.float64) * self.scale + self.mean
        return np.expm1(array) if self.log1p else array

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean": self.mean,
            "scale": self.scale,
            "count": self.count,
            "log1p": self.log1p,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> VideoStandardizer:
        return cls(
            mean=float(value["mean"]),
            scale=float(value["scale"]),
            count=int(value["count"]),
            log1p=bool(value.get("log1p", True)),
        )


@dataclass(frozen=True)
class ScalerBundle:
    """All preprocessing state needed to reproduce model inputs and targets."""

    video: VideoStandardizer | None
    features: ArrayStandardizer | None
    targets: ArrayStandardizer | None
    train_sample_ids: tuple[str, ...]
    version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "train_sample_ids": list(self.train_sample_ids),
            "video": None if self.video is None else self.video.to_dict(),
            "features": None if self.features is None else self.features.to_dict(),
            "targets": None if self.targets is None else self.targets.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ScalerBundle:
        return cls(
            video=(
                None if value.get("video") is None else VideoStandardizer.from_dict(value["video"])
            ),
            features=(
                None
                if value.get("features") is None
                else ArrayStandardizer.from_dict(value["features"])
            ),
            targets=(
                None
                if value.get("targets") is None
                else ArrayStandardizer.from_dict(value["targets"])
            ),
            train_sample_ids=tuple(value["train_sample_ids"]),
            version=int(value.get("version", 1)),
        )


def _npz_array(manifest: DatasetManifest, record: SampleRecord, key: str) -> FloatArray:
    path = manifest.resolve_path(record)
    with np.load(path, allow_pickle=False) as sample:
        if key not in sample:
            raise KeyError(f"Precomputed sample {path} does not contain array {key!r}")
        array = np.array(sample[key], dtype=np.float64, copy=True)
    if not np.isfinite(array).all():
        raise ValueError(f"Array {key!r} in {path} contains non-finite values")
    return array


def fit_train_scalers(
    manifest: DatasetManifest | str | Path | Iterable[SampleRecord],
    split: SplitManifest,
    *,
    video_key: str | None = "video",
    feature_key: str | None = "conditions",
    target_key: str | None = "regression_targets",
    feature_names: Sequence[str] = (),
    target_names: Sequence[str] = (),
    log1p_video: bool = True,
) -> ScalerBundle:
    """Fit every scaler using only the explicit training IDs in ``split``."""

    dataset = as_manifest(manifest)
    validate_split(dataset, split)
    training = dataset.select(split.train)

    video_scaler = None
    if video_key is not None:
        video_scaler = VideoStandardizer.fit(
            (_npz_array(dataset, record, video_key) for record in training),
            log1p=log1p_video,
        )
    feature_scaler = None
    if feature_key is not None:
        feature_scaler = ArrayStandardizer.fit(
            (_npz_array(dataset, record, feature_key).reshape(-1) for record in training),
            names=feature_names,
        )
    target_scaler = None
    if target_key is not None:
        target_scaler = ArrayStandardizer.fit(
            (_npz_array(dataset, record, target_key).reshape(-1) for record in training),
            names=target_names,
        )
    return ScalerBundle(
        video=video_scaler,
        features=feature_scaler,
        targets=target_scaler,
        train_sample_ids=tuple(split.train),
    )
