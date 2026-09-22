"""Dependency-light metrics with explicit per-target reporting."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


def regression_metrics(
    targets: ArrayLike,
    predictions: ArrayLike,
    *,
    target_names: Sequence[str] = (),
) -> dict[str, Any]:
    """Return MAE, RMSE, and R2 separately for every regression target."""

    truth = np.asarray(targets, dtype=np.float64)
    estimate = np.asarray(predictions, dtype=np.float64)
    if truth.shape != estimate.shape or truth.ndim != 2:
        raise ValueError("Regression targets and predictions must share shape (N,D)")
    if truth.shape[0] == 0 or not np.isfinite(truth).all() or not np.isfinite(estimate).all():
        raise ValueError("Regression metrics require finite, non-empty arrays")
    names = tuple(target_names) or tuple(f"target_{index}" for index in range(truth.shape[1]))
    if len(names) != truth.shape[1]:
        raise ValueError("target_names length must match the target dimension")

    residual = estimate - truth
    mae = np.mean(np.abs(residual), axis=0)
    rmse = np.sqrt(np.mean(residual * residual, axis=0))
    denominator = np.sum((truth - truth.mean(axis=0)) ** 2, axis=0)
    numerator = np.sum(residual * residual, axis=0)
    r2 = np.full(truth.shape[1], np.nan, dtype=np.float64)
    valid = denominator > 0
    r2[valid] = 1.0 - numerator[valid] / denominator[valid]
    finite_r2 = r2[np.isfinite(r2)]

    return {
        "mae_macro": float(np.mean(mae)),
        "rmse_macro": float(np.mean(rmse)),
        "r2_macro": float(np.mean(finite_r2)) if finite_r2.size else float("nan"),
        "per_target": {
            name: {"mae": float(mae[i]), "rmse": float(rmse[i]), "r2": float(r2[i])}
            for i, name in enumerate(names)
        },
    }


def classification_metrics(
    targets: ArrayLike, logits_or_labels: ArrayLike, *, num_classes: int | None = None
) -> dict[str, Any]:
    """Return accuracy and an integer confusion matrix."""

    truth = np.asarray(targets, dtype=np.int64).reshape(-1)
    values = np.asarray(logits_or_labels)
    labels = np.argmax(values, axis=1) if values.ndim == 2 else values.astype(np.int64).reshape(-1)
    if truth.shape != labels.shape or truth.size == 0:
        raise ValueError("Classification targets and predictions must be non-empty and aligned")
    if np.any(truth < 0) or np.any(labels < 0):
        raise ValueError("Class indices must be non-negative")
    inferred = int(max(truth.max(), labels.max())) + 1
    classes = inferred if num_classes is None else int(num_classes)
    if classes < inferred:
        raise ValueError("num_classes is smaller than an observed class index")
    confusion: NDArray[np.int64] = np.zeros((classes, classes), dtype=np.int64)
    np.add.at(confusion, (truth, labels), 1)
    return {
        "accuracy": float(np.mean(truth == labels)),
        "confusion_matrix": confusion.tolist(),
    }
