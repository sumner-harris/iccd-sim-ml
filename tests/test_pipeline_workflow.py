from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from iccd_sim_ml.imaging import ImagingConfig
from iccd_sim_ml.pipeline.cache import (
    ContinuumCacheConfig,
    ProxyCacheConfig,
    discover_balanced_subset,
    ensure_continuum_cache,
    ensure_proxy_cache,
)
from iccd_sim_ml.pipeline.reports import (
    plot_confusion_matrix,
    plot_generation_error_maps,
    plot_learning_curves,
    plot_regression_parity,
)


def _frame() -> np.ndarray:
    raw = np.zeros((4, 34), dtype=np.float64)
    raw[:, 0] = [0.0, 0.001, 0.0, 0.001]
    raw[:, 1] = [0.0, 0.0, 0.001, 0.001]
    raw[:, 4] = [300.0, 2000.0, 2500.0, 3000.0]
    raw[:, 9] = 1.0e20
    raw[:, 10] = [0.0, 1.0e17, 2.0e17, 3.0e17]
    raw[:, 11] = raw[:, 10]
    raw[:, 21] = 5.0
    return raw


def _tiny_h5(path: Path, *, element: str = "Cu") -> None:
    h5py = pytest.importorskip("h5py")
    with h5py.File(path, "w") as handle:
        for index in range(5):
            group = handle.create_group(f"{element}_0_{index}")
            group.attrs.update(
                {
                    "element": element,
                    "laser_power_wcm": 1.0e8 + index * 1.0e7,
                    "rspot": 0.0005 + index * 0.0001,
                    "cp_metal": 385.0,
                    "h_vapor": 304000.0,
                    "kappa_metal": 401.0,
                    "laser_reflectivity": 0.41,
                    "mass_density_metal": 8960.0,
                    "t_boil": 2862.0,
                    "tcrit": 7000.0,
                }
            )
            group.create_dataset("res_0_ns.dat", data=_frame())
            group.create_dataset("res_501_ns.dat", data=_frame())


def test_proxy_cache_miss_then_hit(tmp_path: Path) -> None:
    _tiny_h5(tmp_path / "Cu_0.h5")
    samples = discover_balanced_subset(
        tmp_path,
        elements=("Cu",),
        simulations_per_element=3,
        minimum_frames=2,
    )
    config = ProxyCacheConfig(
        frame_count=2,
        image_width=4,
        image_height=4,
        line_of_sight_points=4,
        radial_max_m=0.001,
        axial_max_m=0.001,
    )
    cold = ensure_proxy_cache(samples, tmp_path / "cache", config)
    warm = ensure_proxy_cache(samples, tmp_path / "cache", config)

    assert len(cold.misses) == 3
    assert not cold.hits
    assert len(warm.hits) == 3
    assert not warm.misses
    assert len(warm.manifest.records) == 3
    with np.load(
        warm.manifest.resolve_path(warm.manifest.records[0]), allow_pickle=False
    ) as sample:
        assert sample["video"].shape == (2, 4, 4)
        assert np.isfinite(sample["video"]).all()


def test_approximate_continuum_cache_is_time_aligned_and_reused(tmp_path: Path) -> None:
    _tiny_h5(tmp_path / "Xe_0.h5", element="Xe")
    samples = discover_balanced_subset(
        tmp_path,
        elements=("Xe",),
        simulations_per_element=1,
        minimum_frames=2,
    )
    config = ContinuumCacheConfig(
        imaging=ImagingConfig(
            wavelength_min_nm=400.0,
            wavelength_max_nm=700.0,
            wavelength_points=3,
            radial_points=4,
            axial_points=4,
            line_of_sight_points=4,
            temperature_table_min_K=300.0,
            temperature_table_max_K=5000.0,
            temperature_table_points=4,
            radial_max_m=0.001,
            axial_max_m=0.001,
        ),
        frame_times_s=(0.0, 250.5e-9, 501.0e-9),
        atomic_mode="approximate",
    )
    atomic_root = Path(__file__).parents[1] / "data" / "reference"
    cold = ensure_continuum_cache(samples, tmp_path / "continuum", config, atomic_root)
    warm = ensure_continuum_cache(samples, tmp_path / "continuum", config, atomic_root)

    assert cold.misses == (samples[0].sample_id,)
    assert warm.hits == (samples[0].sample_id,)
    with np.load(warm.manifest.resolve_path(warm.manifest.records[0]), allow_pickle=False) as item:
        assert item["video"].shape == (3, 4, 4)
        np.testing.assert_array_equal(item["times_s"], config.frame_times_s)
        metadata = json.loads(str(item["metadata_json"].item()))
        assert metadata["quantity"] == "band_integrated_photon_radiance"
        assert metadata["scientific_use"] == "approximate_incomplete_continuum"


def test_report_plotters_write_expected_artifacts(tmp_path: Path) -> None:
    plot_learning_curves(
        {"epoch": [1, 2], "train_loss": [1.0, 0.5], "validation_loss": [1.2, 0.7]},
        tmp_path / "learning.png",
    )
    plot_regression_parity(
        np.asarray([[1.0, 2.0], [2.0, 4.0]]),
        np.asarray([[1.1, 1.8], [1.9, 4.2]]),
        ("a", "b"),
        tmp_path / "parity.png",
    )
    plot_confusion_matrix(np.asarray([[1, 0], [0, 1]]), ("A", "B"), tmp_path / "confusion.png")
    video = np.ones((2, 1, 2, 4, 4), dtype=np.float64)
    plot_generation_error_maps(
        video, video * 0.9, ("sample-a", "sample-b"), tmp_path / "errors.png"
    )

    assert all(
        (tmp_path / name).is_file()
        for name in ("learning.png", "parity.png", "confusion.png", "errors.png")
    )
