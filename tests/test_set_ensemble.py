from __future__ import annotations

import numpy as np
import pytest

from iccd_sim_ml.training import average_material_set_predictions


def _result(offset: float = 0.0) -> dict[str, object]:
    bag_targets = np.asarray([[0.0, 1.0], [2.0, 3.0]])
    material_targets = np.asarray([[1.0, 2.0]])
    return {
        "elements": ("Cu",),
        "sets_per_element": {"Cu": 2},
        "bag_targets": bag_targets,
        "bag_predictions": bag_targets + offset,
        "material_targets": material_targets,
        "material_predictions": material_targets + offset,
    }


def test_average_material_set_predictions_averages_in_physical_units() -> None:
    result = average_material_set_predictions(
        [_result(-0.2), _result(0.4)], target_names=("a", "b")
    )

    assert result["member_count"] == 2
    assert np.allclose(result["bag_predictions"], _result()["bag_targets"] + 0.1)
    assert result["bag_metrics"]["r2_macro"] == pytest.approx(0.99)


def test_average_material_set_predictions_rejects_misaligned_targets() -> None:
    second = _result()
    second["bag_targets"] = np.asarray([[0.0, 1.0], [9.0, 3.0]])

    with pytest.raises(ValueError, match="different bag_targets"):
        average_material_set_predictions([_result(), second], target_names=("a", "b"))


def test_average_material_set_predictions_requires_members() -> None:
    with pytest.raises(ValueError, match="At least one"):
        average_material_set_predictions([], target_names=("a", "b"))
