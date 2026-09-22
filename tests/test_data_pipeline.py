from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from iccd_sim_ml.data import (
    DatasetManifest,
    PrecomputedVideoDataset,
    SampleRecord,
    SplitManifest,
    fit_train_scalers,
    make_group_split,
    make_known_material_split,
    make_legacy_train_validation_split,
    save_precomputed_sample,
    validate_split,
)


def _record(path: Path, sample_id: str, element: str, simulation_id: str) -> SampleRecord:
    return SampleRecord(
        sample_id=sample_id,
        path=path,
        element=element,
        simulation_id=simulation_id,
    )


def test_manifest_round_trip_uses_resolvable_paths(tmp_path: Path) -> None:
    sample_path = tmp_path / "products" / "sample.npz"
    save_precomputed_sample(sample_path, video=np.ones((2, 4, 4)))
    manifest = DatasetManifest((_record(sample_path, "cu-1", "Cu", "run-1"),))

    saved = manifest.save(tmp_path / "manifest.json")
    loaded = DatasetManifest.load(saved)

    assert loaded.records[0].sample_id == "cu-1"
    assert loaded.resolve_path(loaded.records[0]) == sample_path.resolve()


def test_group_split_is_deterministic_and_has_no_element_or_simulation_leakage(
    tmp_path: Path,
) -> None:
    records = []
    for element_index, element in enumerate(("Al", "Co", "Cu", "Fe", "W", "Zn")):
        for simulation_index in range(2):
            for replicate in range(2):
                sample_id = f"{element}-{simulation_index}-{replicate}"
                records.append(
                    _record(
                        tmp_path / f"{sample_id}.npz",
                        sample_id,
                        element,
                        f"sim-{element_index}-{simulation_index}",
                    )
                )
    manifest = DatasetManifest(tuple(records))

    first = make_group_split(manifest, seed=31)
    second = make_group_split(manifest, seed=31)
    validate_split(manifest, first)

    assert first == second
    assert first.train and first.validation and first.test
    lookup = manifest.by_id
    element_sets = [
        {lookup[sample_id].element for sample_id in first.ids(name)}
        for name in ("train", "validation", "test")
    ]
    assert element_sets[0].isdisjoint(element_sets[1])
    assert element_sets[0].isdisjoint(element_sets[2])
    assert element_sets[1].isdisjoint(element_sets[2])


def test_legacy_train_validation_split_matches_original_known_material_regime(
    tmp_path: Path,
) -> None:
    records = []
    for simulation_index in range(10):
        for derivative in range(2):
            sample_id = f"Cu-{simulation_index}-{derivative}"
            records.append(
                _record(
                    tmp_path / f"{sample_id}.npz",
                    sample_id,
                    "Cu",
                    f"sim-{simulation_index}",
                )
            )
    manifest = DatasetManifest(tuple(records))

    first = make_legacy_train_validation_split(manifest, seed=17)
    second = make_legacy_train_validation_split(manifest, seed=17)

    assert first == second
    assert first.ratios == pytest.approx((0.7, 0.3, 0.0))
    assert first.group_fields == ("simulation_id",)
    assert len(first.train) == 14
    assert len(first.validation) == 6
    assert not first.test
    assert {manifest.by_id[sample_id].element for sample_id in first.train} == {"Cu"}
    assert {manifest.by_id[sample_id].element for sample_id in first.validation} == {"Cu"}
    validate_split(manifest, first)


def test_known_material_split_stratifies_elements_without_crossing_simulations(
    tmp_path: Path,
) -> None:
    records = []
    for element in ("Cu", "Fe"):
        for simulation_index in range(5):
            for derivative in range(2):
                sample_id = f"{element}-{simulation_index}-{derivative}"
                records.append(
                    _record(
                        tmp_path / f"{sample_id}.npz",
                        sample_id,
                        element,
                        f"sim-{simulation_index}",
                    )
                )
    manifest = DatasetManifest(tuple(records))

    split = make_known_material_split(manifest, seed=29)
    validate_split(manifest, split)

    for name in ("train", "validation"):
        assert {manifest.by_id[sample_id].element for sample_id in split.ids(name)} == {
            "Cu",
            "Fe",
        }
    assert not split.test


def test_scalers_fit_training_ids_only(tmp_path: Path) -> None:
    specifications = {
        "train-a": ("Cu", 0.0, [1.0, 10.0], [10.0, 100.0]),
        "train-b": ("Cu", 2.0, [3.0, 14.0], [14.0, 108.0]),
        "validation": ("Fe", 1.0e9, [1.0e9, 1.0e9], [1.0e9, 1.0e9]),
        "test": ("W", 2.0e9, [2.0e9, 2.0e9], [2.0e9, 2.0e9]),
    }
    records = []
    for sample_id, (element, video_value, features, targets) in specifications.items():
        path = tmp_path / f"{sample_id}.npz"
        save_precomputed_sample(
            path,
            video=np.full((2, 3, 3), video_value),
            conditions=features,
            regression_targets=targets,
        )
        records.append(_record(path, sample_id, element, f"simulation-{sample_id}"))
    manifest = DatasetManifest(tuple(records))
    split = SplitManifest(
        train=("train-a", "train-b"),
        validation=("validation",),
        test=("test",),
        seed=0,
        ratios=(0.5, 0.25, 0.25),
        group_fields=("element", "simulation_id"),
    )

    scalers = fit_train_scalers(
        manifest,
        split,
        feature_names=("spot_radius", "laser_power"),
        target_names=("boiling_point", "critical_temperature"),
    )

    assert scalers.train_sample_ids == split.train
    assert scalers.features is not None
    assert scalers.targets is not None
    assert scalers.video is not None
    np.testing.assert_allclose(scalers.features.mean, [2.0, 12.0])
    np.testing.assert_allclose(scalers.targets.mean, [12.0, 104.0])
    assert scalers.video.mean == pytest.approx(np.log(3.0) / 2.0)


def test_precomputed_dataset_returns_canonical_tensors(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    path = tmp_path / "sample.npz"
    save_precomputed_sample(
        path,
        video=np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5),
        conditions=[0.25, 0.75],
        regression_targets=np.arange(7, dtype=np.float32),
    )
    manifest = DatasetManifest((_record(path, "sample", "Cu", "sim-1"),))

    dataset = PrecomputedVideoDataset(manifest, task="regression")
    sample = dataset[0]

    assert sample["video"].shape == (1, 3, 4, 5)
    assert sample["video"].dtype == torch.float32
    assert sample["features"].shape == (2,)
    assert sample["target"].shape == (7,)
    assert sample["sample_id"] == "sample"


def test_classification_dataset_can_derive_class_from_element(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    path = tmp_path / "sample.npz"
    save_precomputed_sample(path, video=np.ones((2, 3, 3)), conditions=[])
    manifest = DatasetManifest((_record(path, "sample", "Cu", "sim-1"),))

    dataset = PrecomputedVideoDataset(manifest, task="classification", class_to_index={"Cu": 4})

    assert dataset[0]["target"].dtype == torch.long
    assert dataset[0]["target"].item() == 4
