"""Reusable manifest and split preparation for production training runs."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

from iccd_sim_ml.data import (
    DatasetManifest,
    SplitManifest,
    make_group_split,
    make_known_material_split,
    make_legacy_train_validation_split,
    validate_split,
)

SplitStrategy = Literal["known-material", "legacy", "element-held-out"]


def subset_manifest(
    manifest: DatasetManifest,
    elements: tuple[str, ...] | None = None,
) -> DatasetManifest:
    """Select declared elements while retaining the manifest's path base."""

    if elements is None:
        return manifest
    requested = tuple(dict.fromkeys(elements))
    if not requested:
        raise ValueError("At least one element must be requested")
    available = {record.element for record in manifest.records}
    missing = sorted(set(requested).difference(available))
    if missing:
        raise KeyError(f"Elements are not present in the manifest: {missing}")
    selected = tuple(record for record in manifest.records if record.element in requested)
    return DatasetManifest(selected, source=manifest.source, version=manifest.version)


def _normalized_ratios(ratios: tuple[float, float, float]) -> tuple[float, float, float]:
    values = tuple(float(value) for value in ratios)
    if len(values) != 3 or any(value < 0.0 for value in values) or sum(values) <= 0.0:
        raise ValueError("split ratios must be three non-negative values with a positive sum")
    total = sum(values)
    return tuple(value / total for value in values)  # type: ignore[return-value]


def create_split(
    manifest: DatasetManifest,
    *,
    strategy: SplitStrategy,
    ratios: tuple[float, float, float],
    seed: int,
) -> SplitManifest:
    """Create one of the supported scientific evaluation regimes."""

    normalized = _normalized_ratios(ratios)
    if strategy == "known-material":
        return make_known_material_split(manifest, ratios=normalized, seed=seed)
    if strategy == "element-held-out":
        return make_group_split(
            manifest,
            ratios=normalized,
            seed=seed,
            group_fields=("element", "simulation_id"),
        )
    if strategy == "legacy":
        expected = (0.7, 0.3, 0.0)
        if any(
            not math.isclose(left, right)
            for left, right in zip(normalized, expected, strict=True)
        ):
            raise ValueError("legacy split strategy requires ratios 0.7 0.3 0.0")
        return make_legacy_train_validation_split(manifest, seed=seed)
    raise ValueError(f"Unknown split strategy {strategy!r}")


def resolve_split(
    manifest: DatasetManifest,
    path: str | Path,
    *,
    strategy: SplitStrategy,
    ratios: tuple[float, float, float],
    seed: int,
    regenerate: bool = False,
) -> tuple[SplitManifest, Literal["created", "loaded", "regenerated"]]:
    """Load a compatible split or create and atomically save it once."""

    destination = Path(path).expanduser().resolve()
    requested_ratios = _normalized_ratios(ratios)
    expected_fields = (
        ("element", "simulation_id")
        if strategy == "element-held-out"
        else ("simulation_id",)
    )
    if destination.is_file() and not regenerate:
        split = SplitManifest.load(destination)
        validate_split(manifest, split)
        if split.seed != int(seed):
            raise ValueError(
                f"Existing split seed is {split.seed}, requested {seed}; use "
                "--regenerate-split to intentionally replace it"
            )
        if any(
            not math.isclose(left, right)
            for left, right in zip(split.ratios, requested_ratios, strict=True)
        ):
            raise ValueError(
                f"Existing split ratios are {split.ratios}, requested {requested_ratios}; "
                "use --regenerate-split to intentionally replace it"
            )
        if split.group_fields != expected_fields:
            raise ValueError(
                f"Existing split protects {split.group_fields}, but {strategy!r} requires "
                f"{expected_fields}; use --regenerate-split to intentionally replace it"
            )
        return split, "loaded"

    state: Literal["created", "loaded", "regenerated"] = (
        "regenerated" if destination.is_file() else "created"
    )
    split = create_split(manifest, strategy=strategy, ratios=requested_ratios, seed=seed)
    split.save(destination)
    return split, state
