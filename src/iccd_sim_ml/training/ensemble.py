"""Reproducible ensembling for material-set regression checkpoints."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .metrics import regression_metrics


def average_material_set_predictions(
    results: Sequence[Mapping[str, Any]], *, target_names: Sequence[str]
) -> dict[str, Any]:
    """Average aligned bag and material predictions from multiple checkpoints.

    ``collect_material_set_predictions`` results are aligned only when they were
    evaluated with the same deterministic dataset. This function verifies that
    contract before averaging in physical units.
    """

    if not results:
        raise ValueError("At least one material-set result is required")
    reference = results[0]
    required = (
        "elements",
        "sets_per_element",
        "bag_targets",
        "bag_predictions",
        "material_targets",
        "material_predictions",
    )
    for index, result in enumerate(results):
        missing = [name for name in required if name not in result]
        if missing:
            raise KeyError(f"Ensemble member {index} is missing fields: {missing}")
        if tuple(result["elements"]) != tuple(reference["elements"]):
            raise ValueError("Ensemble members have different material ordering")
        if dict(result["sets_per_element"]) != dict(reference["sets_per_element"]):
            raise ValueError("Ensemble members have different bag membership")
        for field in ("bag_targets", "material_targets"):
            values = np.asarray(result[field], dtype=np.float64)
            expected = np.asarray(reference[field], dtype=np.float64)
            if values.shape != expected.shape or not np.allclose(
                values, expected, rtol=1.0e-7, atol=1.0e-9
            ):
                raise ValueError(f"Ensemble members have different {field}")

    bag_predictions = np.mean(
        [np.asarray(result["bag_predictions"], dtype=np.float64) for result in results],
        axis=0,
    )
    material_predictions = np.mean(
        [np.asarray(result["material_predictions"], dtype=np.float64) for result in results],
        axis=0,
    )
    bag_targets = np.asarray(reference["bag_targets"], dtype=np.float64)
    material_targets = np.asarray(reference["material_targets"], dtype=np.float64)
    return {
        "member_count": len(results),
        "elements": tuple(reference["elements"]),
        "sets_per_element": dict(reference["sets_per_element"]),
        "bag_targets": bag_targets,
        "bag_predictions": bag_predictions,
        "material_targets": material_targets,
        "material_predictions": material_predictions,
        "bag_metrics": regression_metrics(bag_targets, bag_predictions, target_names=target_names),
        "material_metrics": regression_metrics(
            material_targets, material_predictions, target_names=target_names
        ),
    }
