from __future__ import annotations

import numpy as np
import pytest

from iccd_sim_ml.training import classification_metrics, video_generation_metrics


def test_classification_metrics_include_full_declared_class_catalog() -> None:
    result = classification_metrics(
        np.asarray([0, 0, 1, 2]),
        np.asarray([0, 1, 1, 1]),
        num_classes=4,
    )

    assert np.asarray(result["confusion_matrix"]).shape == (4, 4)
    assert result["accuracy"] == pytest.approx(0.5)
    assert result["per_class"]["0"]["precision"] == pytest.approx(1.0)
    assert result["per_class"]["0"]["recall"] == pytest.approx(0.5)
    assert result["per_class"]["3"]["support"] == 0
    assert result["f1_micro"] == pytest.approx(result["accuracy"])


def test_video_generation_metrics_are_exact_for_perfect_video() -> None:
    video = np.linspace(0.0, 1.0, 2 * 1 * 3 * 8 * 8, dtype=np.float64).reshape(
        2, 1, 3, 8, 8
    )

    result = video_generation_metrics(video, video)

    assert result["mae"] == pytest.approx(0.0)
    assert result["rmse"] == pytest.approx(0.0)
    assert result["relative_l1"] == pytest.approx(0.0)
    assert result["ssim_mean"] == pytest.approx(1.0)
    assert result["pearson_correlation_mean"] == pytest.approx(1.0)
    assert result["temporal_difference_mae"] == pytest.approx(0.0)
    assert result["perfect_reconstruction_samples"] == 2
