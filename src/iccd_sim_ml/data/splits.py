"""Deterministic dataset splits for distinct scientific evaluation regimes."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .manifest import DatasetManifest, SampleRecord, as_manifest

SPLIT_VERSION = 1
SPLIT_NAMES = ("train", "validation", "test")


@dataclass(frozen=True)
class SplitManifest:
    """Explicit sample IDs for a reproducible train/validation/test split."""

    train: tuple[str, ...]
    validation: tuple[str, ...]
    test: tuple[str, ...]
    seed: int
    ratios: tuple[float, float, float]
    group_fields: tuple[str, ...]
    version: int = SPLIT_VERSION

    def __post_init__(self) -> None:
        for name in SPLIT_NAMES:
            object.__setattr__(self, name, tuple(getattr(self, name)))
        object.__setattr__(self, "ratios", tuple(float(value) for value in self.ratios))
        object.__setattr__(self, "group_fields", tuple(self.group_fields))
        if self.version != SPLIT_VERSION:
            raise ValueError(f"Unsupported split version {self.version}")
        if len(self.ratios) != 3:
            raise ValueError("ratios must contain train, validation, and test values")
        all_ids = self.train + self.validation + self.test
        if len(all_ids) != len(set(all_ids)):
            raise ValueError("A sample ID appears in more than one split")

    def ids(self, name: str) -> tuple[str, ...]:
        normalized = "validation" if name == "val" else name
        if normalized not in SPLIT_NAMES:
            raise KeyError(f"Unknown split {name!r}; expected one of {SPLIT_NAMES}")
        return tuple(getattr(self, normalized))

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "seed": self.seed,
            "ratios": list(self.ratios),
            "group_fields": list(self.group_fields),
            "train": list(self.train),
            "validation": list(self.validation),
            "test": list(self.test),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SplitManifest:
        return cls(
            train=tuple(value["train"]),
            validation=tuple(value["validation"]),
            test=tuple(value["test"]),
            seed=int(value["seed"]),
            ratios=tuple(value["ratios"]),
            group_fields=tuple(value["group_fields"]),
            version=int(value.get("version", SPLIT_VERSION)),
        )

    def save(self, path: str | Path) -> Path:
        destination = Path(path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.tmp")
        try:
            temporary.write_text(
                json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
            )
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        return destination

    @classmethod
    def load(cls, path: str | Path) -> SplitManifest:
        payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
        return cls.from_dict(payload)


def _field_token(record: SampleRecord, field: str) -> tuple[str, ...]:
    if hasattr(record, field):
        value = getattr(record, field)
    elif field in record.metadata:
        value = record.metadata[field]
    else:
        raise KeyError(f"Sample {record.sample_id!r} has no grouping field {field!r}")
    if value is None or str(value).strip() == "":
        raise ValueError(f"Sample {record.sample_id!r} has an empty grouping field {field!r}")
    # Simulation and family IDs are usually only unique within an element.
    if field in {"simulation_id", "family_id"}:
        return field, record.element, str(value)
    return field, str(value)


def _connected_groups(
    records: Sequence[SampleRecord], group_fields: Sequence[str]
) -> tuple[tuple[SampleRecord, ...], ...]:
    """Group samples connected by any protected metadata field."""

    parent = list(range(len(records)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for field in group_fields:
        owners: dict[tuple[str, ...], int] = {}
        for index, record in enumerate(records):
            token = _field_token(record, field)
            if token in owners:
                union(index, owners[token])
            else:
                owners[token] = index

    groups: dict[int, list[SampleRecord]] = {}
    for index, record in enumerate(records):
        groups.setdefault(find(index), []).append(record)
    return tuple(tuple(group) for group in groups.values())


def _stable_tie_key(group: Sequence[SampleRecord], seed: int) -> str:
    ids = "|".join(sorted(record.sample_id for record in group))
    return hashlib.sha256(f"{seed}|{ids}".encode()).hexdigest()


def make_group_split(
    manifest: DatasetManifest | str | Path | Iterable[SampleRecord],
    *,
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 0,
    group_fields: tuple[str, ...] = ("element", "simulation_id"),
) -> SplitManifest:
    """Create a deterministic split without crossing any protected group boundary.

    Including ``element`` in ``group_fields`` is the appropriate default for the
    unseen-material regression experiment.  For known-element classification,
    callers may explicitly choose ``("simulation_id",)`` instead.
    """

    dataset = as_manifest(manifest)
    if not dataset.records:
        raise ValueError("Cannot split an empty dataset")
    values = tuple(float(value) for value in ratios)
    if len(values) != 3 or any(value < 0 for value in values) or sum(values) <= 0:
        raise ValueError("ratios must be three non-negative values with a positive sum")
    values = tuple(value / sum(values) for value in values)
    if not group_fields:
        raise ValueError("At least one protected group field is required")

    groups = list(_connected_groups(dataset.records, group_fields))
    active_splits = [index for index, ratio in enumerate(values) if ratio > 0]
    if len(groups) < len(active_splits):
        raise ValueError(
            f"Only {len(groups)} independent groups are available for "
            f"{len(active_splits)} non-empty splits"
        )

    groups.sort(key=lambda group: (-len(group), _stable_tie_key(group, seed)))
    target_counts = [ratio * len(dataset.records) for ratio in values]
    counts = [0, 0, 0]
    assignments: dict[str, int] = {}
    empty = set(active_splits)

    for group_index, group in enumerate(groups):
        remaining = len(groups) - group_index
        group_size = len(group)
        candidates = list(active_splits)
        if remaining == len(empty):
            candidates = sorted(empty)

        def score(split_index: int, group_size: int = group_size) -> tuple[float, int]:
            projected = counts.copy()
            projected[split_index] += group_size
            squared_error = sum(
                (projected[index] - target_counts[index]) ** 2 for index in active_splits
            )
            return squared_error, split_index

        chosen = min(candidates, key=score)
        counts[chosen] += group_size
        empty.discard(chosen)
        for record in group:
            assignments[record.sample_id] = chosen

    split_ids = [
        tuple(record.sample_id for record in dataset.records if assignments[record.sample_id] == i)
        for i in range(3)
    ]
    split = SplitManifest(
        train=split_ids[0],
        validation=split_ids[1],
        test=split_ids[2],
        seed=int(seed),
        ratios=values,
        group_fields=tuple(group_fields),
    )
    validate_split(dataset, split, group_fields=group_fields)
    return split


def make_legacy_train_validation_split(
    manifest: DatasetManifest | str | Path | Iterable[SampleRecord],
    *,
    seed: int = 0,
) -> SplitManifest:
    """Reproduce the original notebook's 70/30 known-material split regime.

    The notebook randomly split complete simulation items after concatenating
    all element datasets.  Consequently, an element could occur in both train
    and validation, while each original simulation sequence remained intact.
    This helper preserves those semantics, adds a fixed seed, and protects the
    ``simulation_id`` boundary if a simulation has multiple derived records.

    There is intentionally no test partition.  This split is useful for an
    apples-to-apples legacy baseline, but it does not measure generalization to
    an unseen material.
    """

    return make_group_split(
        manifest,
        ratios=(0.7, 0.3, 0.0),
        seed=seed,
        group_fields=("simulation_id",),
    )


def make_known_material_split(
    manifest: DatasetManifest | str | Path | Iterable[SampleRecord],
    *,
    ratios: tuple[float, float, float] = (0.7, 0.3, 0.0),
    seed: int = 0,
    group_fields: tuple[str, ...] = ("simulation_id",),
) -> SplitManifest:
    """Split simulations within every element while protecting group boundaries.

    Unlike the legacy notebook's one global random split, this stratified
    variant guarantees that every element occurs in every non-empty partition.
    Each element therefore needs at least as many independent protected groups
    as there are non-zero ratios.
    """

    dataset = as_manifest(manifest)
    if not dataset.records:
        raise ValueError("Cannot split an empty dataset")
    if "element" in group_fields:
        raise ValueError(
            "make_known_material_split partitions within each element, so "
            "'element' cannot also be a protected group field"
        )

    partitions: list[set[str]] = [set(), set(), set()]
    normalized_ratios: tuple[float, float, float] | None = None
    elements = sorted({record.element for record in dataset.records})
    for element in elements:
        records = tuple(record for record in dataset.records if record.element == element)
        try:
            element_split = make_group_split(
                records,
                ratios=ratios,
                seed=seed,
                group_fields=group_fields,
            )
        except ValueError as error:
            raise ValueError(f"Cannot split element {element!r}: {error}") from error
        normalized_ratios = element_split.ratios
        for index, name in enumerate(SPLIT_NAMES):
            partitions[index].update(element_split.ids(name))

    split_ids = [
        tuple(record.sample_id for record in dataset.records if record.sample_id in partition)
        for partition in partitions
    ]
    if normalized_ratios is None:  # pragma: no cover - guarded by the non-empty check
        raise AssertionError("Expected at least one element")
    split = SplitManifest(
        train=split_ids[0],
        validation=split_ids[1],
        test=split_ids[2],
        seed=int(seed),
        ratios=normalized_ratios,
        group_fields=tuple(group_fields),
    )
    validate_split(dataset, split, group_fields=group_fields)
    return split


def validate_split(
    manifest: DatasetManifest | str | Path | Iterable[SampleRecord],
    split: SplitManifest,
    *,
    group_fields: Sequence[str] | None = None,
) -> None:
    """Raise if samples are missing/duplicated or a protected group crosses splits."""

    dataset = as_manifest(manifest)
    known_ids = {record.sample_id for record in dataset.records}
    split_ids = set(split.train + split.validation + split.test)
    if known_ids != split_ids:
        missing = sorted(known_ids - split_ids)
        unknown = sorted(split_ids - known_ids)
        raise ValueError(f"Split does not match manifest; missing={missing}, unknown={unknown}")

    fields = tuple(split.group_fields if group_fields is None else group_fields)
    record_by_id = dataset.by_id
    for field in fields:
        tokens = {
            name: {_field_token(record_by_id[sample_id], field) for sample_id in split.ids(name)}
            for name in SPLIT_NAMES
        }
        for left_index, left in enumerate(SPLIT_NAMES):
            for right in SPLIT_NAMES[left_index + 1 :]:
                overlap = tokens[left].intersection(tokens[right])
                if overlap:
                    raise ValueError(
                        f"Protected group field {field!r} overlaps between {left} and "
                        f"{right}: {sorted(overlap)}"
                    )
