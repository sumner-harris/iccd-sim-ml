"""Dependency-light metrics with explicit per-target reporting."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.ndimage import uniform_filter


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
    """Return confusion-derived multiclass metrics without flattening classes."""

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
    true_positive = np.diag(confusion).astype(np.float64)
    support = np.sum(confusion, axis=1).astype(np.float64)
    predicted = np.sum(confusion, axis=0).astype(np.float64)
    precision = np.divide(
        true_positive,
        predicted,
        out=np.zeros(classes, dtype=np.float64),
        where=predicted > 0,
    )
    recall = np.divide(
        true_positive,
        support,
        out=np.zeros(classes, dtype=np.float64),
        where=support > 0,
    )
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros(classes, dtype=np.float64),
        where=(precision + recall) > 0,
    )
    active = support > 0
    total = float(np.sum(support))
    weights = support / total
    accuracy = float(np.sum(true_positive) / total)
    return {
        "accuracy": accuracy,
        "balanced_accuracy": float(np.mean(recall[active])),
        "precision_macro": float(np.mean(precision[active])),
        "recall_macro": float(np.mean(recall[active])),
        "f1_macro": float(np.mean(f1[active])),
        "precision_weighted": float(np.sum(precision * weights)),
        "recall_weighted": float(np.sum(recall * weights)),
        "f1_weighted": float(np.sum(f1 * weights)),
        "precision_micro": accuracy,
        "recall_micro": accuracy,
        "f1_micro": accuracy,
        "per_class": {
            str(index): {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
                "predicted": int(predicted[index]),
            }
            for index in range(classes)
        },
        "confusion_matrix": confusion.tolist(),
    }


def video_generation_metrics(targets: ArrayLike, predictions: ArrayLike) -> dict[str, Any]:
    """Return pixel, structural, correlation, and temporal video metrics.

    Inputs use canonical ``(N,C,T,H,W)`` order. PSNR and SSIM use each
    target video's dynamic range; constant videos are excluded from those two
    averages and reported via ``dynamic_range_samples``.
    """

    truth = np.asarray(targets, dtype=np.float64)
    estimate = np.asarray(predictions, dtype=np.float64)
    if truth.shape != estimate.shape or truth.ndim != 5:
        raise ValueError("Generation videos must share shape (N,C,T,H,W)")
    if truth.shape[0] == 0 or not np.isfinite(truth).all() or not np.isfinite(estimate).all():
        raise ValueError("Generation metrics require finite, non-empty arrays")

    residual = estimate - truth
    absolute = np.abs(residual)
    mae = float(np.mean(absolute))
    rmse = float(np.sqrt(np.mean(residual * residual)))
    truth_l1 = float(np.sum(np.abs(truth)))
    relative_l1 = float(np.sum(absolute) / max(truth_l1, np.finfo(np.float64).eps))

    sample_axes = tuple(range(1, truth.ndim))
    sample_rmse = np.sqrt(np.mean(residual * residual, axis=sample_axes))
    sample_range = np.ptp(truth, axis=sample_axes)
    ranged = sample_range > 0.0
    psnr = np.full(truth.shape[0], np.nan, dtype=np.float64)
    perfect = ranged & (sample_rmse == 0.0)
    finite_error = ranged & (sample_rmse > 0.0)
    psnr[perfect] = np.inf
    psnr[finite_error] = 20.0 * np.log10(
        sample_range[finite_error] / sample_rmse[finite_error]
    )

    correlations: list[float] = []
    ssim_values: list[float] = []
    for sample_index in range(truth.shape[0]):
        target_sample = truth[sample_index]
        estimate_sample = estimate[sample_index]
        target_flat = target_sample.reshape(-1)
        estimate_flat = estimate_sample.reshape(-1)
        target_centered = target_flat - np.mean(target_flat)
        estimate_centered = estimate_flat - np.mean(estimate_flat)
        correlation_denominator = float(
            np.sqrt(np.sum(target_centered**2) * np.sum(estimate_centered**2))
        )
        if correlation_denominator > 0.0:
            correlations.append(
                float(np.sum(target_centered * estimate_centered) / correlation_denominator)
            )

        dynamic_range = float(sample_range[sample_index])
        if dynamic_range <= 0.0:
            continue
        # A local 7x7 spatial SSIM for every channel and time slice. The
        # leading C,T axes use a window size of one and are averaged afterward.
        window = (1, 1, 7, 7)
        mean_target = uniform_filter(target_sample, size=window, mode="reflect")
        mean_estimate = uniform_filter(estimate_sample, size=window, mode="reflect")
        variance_target = uniform_filter(target_sample * target_sample, size=window, mode="reflect")
        variance_target -= mean_target * mean_target
        variance_estimate = uniform_filter(
            estimate_sample * estimate_sample, size=window, mode="reflect"
        )
        variance_estimate -= mean_estimate * mean_estimate
        covariance = uniform_filter(target_sample * estimate_sample, size=window, mode="reflect")
        covariance -= mean_target * mean_estimate
        c1 = (0.01 * dynamic_range) ** 2
        c2 = (0.03 * dynamic_range) ** 2
        numerator = (2.0 * mean_target * mean_estimate + c1) * (2.0 * covariance + c2)
        denominator = (mean_target**2 + mean_estimate**2 + c1) * (
            variance_target + variance_estimate + c2
        )
        ssim_values.append(
            float(
                np.mean(
                    np.divide(
                        numerator,
                        denominator,
                        out=np.ones_like(numerator),
                        where=denominator != 0,
                    )
                )
            )
        )

    temporal_mae = float("nan")
    if truth.shape[2] > 1:
        target_difference = np.diff(truth, axis=2)
        estimate_difference = np.diff(estimate, axis=2)
        temporal_mae = float(np.mean(np.abs(estimate_difference - target_difference)))

    finite_psnr = psnr[np.isfinite(psnr)]
    return {
        "mae": mae,
        "rmse": rmse,
        "relative_l1": relative_l1,
        "psnr_db_mean": float(np.mean(finite_psnr)) if finite_psnr.size else float("nan"),
        "ssim_mean": float(np.mean(ssim_values)) if ssim_values else float("nan"),
        "pearson_correlation_mean": (
            float(np.mean(correlations)) if correlations else float("nan")
        ),
        "temporal_difference_mae": temporal_mae,
        "samples": int(truth.shape[0]),
        "dynamic_range_samples": int(np.sum(ranged)),
        "perfect_reconstruction_samples": int(np.sum(perfect)),
    }
