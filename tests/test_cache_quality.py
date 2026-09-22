from types import SimpleNamespace

import numpy as np

from iccd_sim_ml.pipeline.quality import (
    sparse_level_stages,
    summarize_plasma_window,
    summarize_radiance,
)


def test_radiance_quality_reports_exact_non_emission() -> None:
    video = np.asarray([[[0.0, 0.0]], [[0.0, 2.0]]])
    result = summarize_radiance(video)
    assert result.maximum == 2.0
    assert result.positive_pixel_fraction == 0.25
    assert result.zero_pixel_fraction == 0.75
    assert result.non_emissive_frame_fraction == 0.5
    assert not result.all_zero_radiance


def test_plasma_quality_uses_entire_source_window() -> None:
    frames = [
        SimpleNamespace(
            temperature_K=np.asarray([300.0, 900.0]),
            ne_m3=np.asarray([0.0]),
            n1_m3=np.asarray([0.0]),
            n2_m3=np.asarray([0.0]),
        ),
        SimpleNamespace(
            temperature_K=np.asarray([1500.0]),
            ne_m3=np.asarray([2.0]),
            n1_m3=np.asarray([1.0]),
            n2_m3=np.asarray([0.0]),
        ),
    ]
    result = summarize_plasma_window(frames, boiling_temperature_K=2000.0)
    assert result.maximum_temperature_K == 1500.0
    assert result.maximum_ne_m3 == 2.0
    assert result.below_boiling_temperature_in_window
    assert not result.no_charged_plasma_in_window


def test_sparse_level_stages_uses_declared_floor() -> None:
    levels = {
        0: SimpleNamespace(energy_ev=np.arange(12)),
        1: SimpleNamespace(energy_ev=np.arange(2)),
    }
    assert sparse_level_stages(levels, minimum_levels=10) == {1: 2}
