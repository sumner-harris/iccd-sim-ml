"""Deterministic material-level bags of precomputed plume videos."""

from __future__ import annotations

import zlib
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .datasets import NPZKeys, PrecomputedVideoDataset, _require_torch, torch
from .manifest import DatasetManifest, SampleRecord, as_manifest
from .scalers import ScalerBundle

try:
    from torch.utils.data import Dataset as TorchDataset
except ModuleNotFoundError:  # pragma: no cover - handled by _require_torch

    class TorchDataset:  # type: ignore[no-redef]
        """Import-time placeholder for installations without PyTorch."""


class MaterialSetDataset(TorchDataset):
    """Return fixed-size, same-material experiment sets for property regression.

    Each item contains ``set_size`` complete simulations from one element. Bag
    membership is generated once from ``seed`` and is therefore reproducible
    across workers and epochs. Multiple bags per material provide subset
    augmentation without changing the number of scientifically independent
    material targets.
    """

    def __init__(
        self,
        manifest: DatasetManifest | str | Path | Iterable[SampleRecord],
        *,
        sample_ids: Iterable[str],
        set_size: int = 8,
        sets_per_material: int = 16,
        seed: int = 42,
        sampling: Literal["random", "condition_farthest"] = "random",
        scalers: ScalerBundle | None = None,
        keys: NPZKeys | None = None,
    ) -> None:
        _require_torch()
        if set_size < 1 or sets_per_material < 1:
            raise ValueError("set_size and sets_per_material must be positive")
        if sampling not in {"random", "condition_farthest"}:
            raise ValueError("sampling must be 'random' or 'condition_farthest'")
        self.manifest = as_manifest(manifest)
        self.scalers = scalers
        self.keys = NPZKeys() if keys is None else keys
        self.set_size = int(set_size)
        self.sets_per_material = int(sets_per_material)
        self.sampling = sampling
        selected = self.manifest.select(tuple(sample_ids))
        grouped: dict[str, list[SampleRecord]] = defaultdict(list)
        for record in selected:
            grouped[record.element].append(record)
        if not grouped:
            raise ValueError("MaterialSetDataset requires at least one selected sample")

        self.elements = tuple(sorted(grouped))
        self.records_by_element: dict[str, tuple[SampleRecord, ...]] = {}
        self.conditions_by_id: dict[str, np.ndarray] = {}
        self.targets_by_element: dict[str, np.ndarray] = {}
        for element in self.elements:
            records = tuple(sorted(grouped[element], key=lambda record: record.sample_id))
            if len(records) < self.set_size:
                raise ValueError(
                    f"Element {element!r} has {len(records)} samples, fewer than "
                    f"set_size={self.set_size}"
                )
            target_rows: list[np.ndarray] = []
            for record in records:
                path = self.manifest.resolve_path(record)
                with np.load(path, allow_pickle=False) as sample:
                    missing = [
                        key
                        for key in (
                            self.keys.video,
                            self.keys.features,
                            self.keys.regression_target,
                        )
                        if key not in sample
                    ]
                    if missing:
                        raise KeyError(f"Material-set sample {path} is missing arrays: {missing}")
                    conditions = np.asarray(sample[self.keys.features], dtype=np.float32).reshape(
                        -1
                    )
                    target = np.asarray(
                        sample[self.keys.regression_target], dtype=np.float32
                    ).reshape(-1)
                if not np.isfinite(conditions).all() or not np.isfinite(target).all():
                    raise ValueError(f"Non-finite conditions or targets in {path}")
                if self.scalers is not None and self.scalers.features is not None:
                    conditions = self.scalers.features.transform(conditions).astype(np.float32)
                if self.scalers is not None and self.scalers.targets is not None:
                    target = self.scalers.targets.transform(target).astype(np.float32)
                self.conditions_by_id[record.sample_id] = np.ascontiguousarray(conditions)
                target_rows.append(np.ascontiguousarray(target))
            reference = target_rows[0]
            if any(
                row.shape != reference.shape
                or not np.allclose(row, reference, rtol=1.0e-6, atol=1.0e-6)
                for row in target_rows[1:]
            ):
                raise ValueError(f"Material-property targets are inconsistent for {element}")
            self.records_by_element[element] = records
            self.targets_by_element[element] = reference

        self.bags: tuple[tuple[str, tuple[SampleRecord, ...], str], ...] = tuple(
            self._build_bags(seed)
        )

    def _build_bags(self, seed: int) -> Iterable[tuple[str, tuple[SampleRecord, ...], str]]:
        for element in self.elements:
            records = self.records_by_element[element]
            element_seed = zlib.crc32(element.encode("utf-8"))
            for bag_index in range(self.sets_per_material):
                rng = np.random.default_rng(
                    np.random.SeedSequence([int(seed), int(element_seed), bag_index])
                )
                if self.sampling == "random":
                    selected = rng.choice(len(records), size=self.set_size, replace=False)
                else:
                    selected = self._condition_farthest_indices(records, rng)
                bag_records = tuple(records[int(index)] for index in selected)
                yield element, bag_records, f"{element}-set-{bag_index:04d}"

    def _condition_farthest_indices(
        self, records: tuple[SampleRecord, ...], rng: np.random.Generator
    ) -> np.ndarray:
        """Select a space-filling subset over laser condition coordinates."""

        values = np.stack([self.conditions_by_id[record.sample_id] for record in records]).astype(
            np.float64
        )
        scale = np.std(values, axis=0)
        normalized = (values - np.mean(values, axis=0)) / np.where(scale > 0.0, scale, 1.0)
        radius = np.sum(normalized * normalized, axis=1)
        outermost = np.flatnonzero(np.isclose(radius, np.max(radius), rtol=1.0e-12, atol=1.0e-15))
        chosen = [int(rng.choice(outermost))]
        minimum_distance = np.sum((normalized - normalized[chosen[0]]) ** 2, axis=1)
        minimum_distance[chosen[0]] = -np.inf
        while len(chosen) < self.set_size:
            maximum = float(np.max(minimum_distance))
            candidates = np.flatnonzero(
                np.isclose(minimum_distance, maximum, rtol=1.0e-12, atol=1.0e-15)
            )
            next_index = int(rng.choice(candidates))
            chosen.append(next_index)
            distance = np.sum((normalized - normalized[next_index]) ** 2, axis=1)
            minimum_distance = np.minimum(minimum_distance, distance)
            minimum_distance[np.asarray(chosen)] = -np.inf
        return np.asarray(chosen, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.bags)

    def __getitem__(self, index: int) -> dict[str, Any]:
        element, records, set_id = self.bags[index]
        videos: list[np.ndarray] = []
        conditions: list[np.ndarray] = []
        sample_ids: list[str] = []
        for record in records:
            path = self.manifest.resolve_path(record)
            with np.load(path, allow_pickle=False) as sample:
                video = PrecomputedVideoDataset._canonical_video(
                    np.array(sample[self.keys.video], copy=True), path
                )
            if self.scalers is not None and self.scalers.video is not None:
                video = self.scalers.video.transform(video).astype(np.float32)
            videos.append(np.ascontiguousarray(video))
            conditions.append(self.conditions_by_id[record.sample_id])
            sample_ids.append(record.sample_id)
        return {
            "videos": torch.from_numpy(np.stack(videos)),
            "conditions": torch.from_numpy(np.stack(conditions)),
            "set_mask": torch.ones(self.set_size, dtype=torch.bool),
            "target": torch.from_numpy(self.targets_by_element[element].copy()),
            "element": element,
            "set_id": set_id,
            "sample_ids": tuple(sample_ids),
        }
