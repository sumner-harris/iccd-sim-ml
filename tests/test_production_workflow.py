from __future__ import annotations

from pathlib import Path

import pytest

from iccd_sim_ml.data import DatasetManifest, SampleRecord, validate_split
from iccd_sim_ml.pipeline.workflow import create_split, resolve_split, subset_manifest


def _manifest(tmp_path: Path) -> DatasetManifest:
    records = tuple(
        SampleRecord(
            sample_id=f"{element}-{index}",
            path=tmp_path / f"{element}-{index}.npz",
            element=element,
            simulation_id=f"{element}-simulation-{index}",
        )
        for element in ("Al", "Cu", "Fe")
        for index in range(20)
    )
    source = tmp_path / "manifest.json"
    return DatasetManifest(records, source=source)


def test_known_material_production_split_covers_each_element(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)

    split = create_split(
        manifest,
        strategy="known-material",
        ratios=(0.7, 0.15, 0.15),
        seed=42,
    )
    validate_split(manifest, split)

    lookup = manifest.by_id
    for name in ("train", "validation", "test"):
        assert {lookup[sample_id].element for sample_id in split.ids(name)} == {
            "Al",
            "Cu",
            "Fe",
        }
    assert (len(split.train), len(split.validation), len(split.test)) == (42, 9, 9)


def test_resolve_split_reuses_compatible_file_and_rejects_drift(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    path = tmp_path / "run" / "split.json"

    created, created_state = resolve_split(
        manifest,
        path,
        strategy="known-material",
        ratios=(0.7, 0.15, 0.15),
        seed=42,
    )
    loaded, loaded_state = resolve_split(
        manifest,
        path,
        strategy="known-material",
        ratios=(0.7, 0.15, 0.15),
        seed=42,
    )

    assert created_state == "created"
    assert loaded_state == "loaded"
    assert loaded == created
    with pytest.raises(ValueError, match="Existing split seed"):
        resolve_split(
            manifest,
            path,
            strategy="known-material",
            ratios=(0.7, 0.15, 0.15),
            seed=7,
        )
    regenerated, state = resolve_split(
        manifest,
        path,
        strategy="known-material",
        ratios=(0.7, 0.15, 0.15),
        seed=7,
        regenerate=True,
    )
    assert state == "regenerated"
    assert regenerated.seed == 7


def test_element_held_out_split_and_manifest_filter(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    selected = subset_manifest(manifest, ("Cu", "Fe"))
    split = create_split(
        selected,
        strategy="element-held-out",
        ratios=(0.5, 0.5, 0.0),
        seed=3,
    )

    assert len(selected.records) == 40
    assert selected.source == manifest.source
    lookup = selected.by_id
    train_elements = {lookup[sample_id].element for sample_id in split.train}
    validation_elements = {lookup[sample_id].element for sample_id in split.validation}
    assert train_elements.isdisjoint(validation_elements)
    assert train_elements | validation_elements == {"Cu", "Fe"}
    with pytest.raises(KeyError, match="not present"):
        subset_manifest(manifest, ("W",))


def test_legacy_split_rejects_nonlegacy_ratios(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires ratios"):
        create_split(
            _manifest(tmp_path),
            strategy="legacy",
            ratios=(0.7, 0.15, 0.15),
            seed=0,
        )
