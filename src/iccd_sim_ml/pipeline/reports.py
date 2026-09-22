"""Reusable, headless experiment-report plots and JSON output."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _destination(path: str | Path) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    return destination


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def save_metrics_json(path: str | Path, metrics: dict[str, Any]) -> Path:
    destination = _destination(path)
    destination.write_text(json.dumps(_json_value(metrics), indent=2), encoding="utf-8")
    return destination


def plot_learning_curves(history: dict[str, list[float]], path: str | Path) -> Path:
    """Plot every scalar train/validation history series against epoch."""

    destination = _destination(path)
    epochs = np.asarray(history.get("epoch", []), dtype=np.float64)
    series = [(name, values) for name, values in history.items() if name != "epoch"]
    if epochs.size == 0 or not series:
        raise ValueError("Learning-curve history must contain epoch and scalar series")
    columns = 2
    rows = int(np.ceil(len(series) / columns))
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(6.0 * columns, 3.6 * rows),
        constrained_layout=True,
        squeeze=False,
    )
    for axis in axes.ravel():
        axis.set_visible(False)
    for axis, (name, values) in zip(axes.ravel(), series, strict=False):
        values_array = np.asarray(values, dtype=np.float64)
        if values_array.shape != epochs.shape:
            raise ValueError(f"History series {name!r} does not align with epoch")
        axis.set_visible(True)
        axis.plot(epochs, values_array, marker="o", linewidth=1.6)
        axis.set(xlabel="epoch", ylabel=name.replace("_", " "))
        axis.grid(alpha=0.25)
    figure.suptitle("Training history")
    figure.savefig(destination, dpi=170)
    plt.close(figure)
    return destination


def plot_regression_parity(
    targets: np.ndarray,
    predictions: np.ndarray,
    target_names: tuple[str, ...],
    path: str | Path,
) -> Path:
    """Create one physical-unit parity panel per regression property."""

    truth = np.asarray(targets, dtype=np.float64)
    estimate = np.asarray(predictions, dtype=np.float64)
    if truth.shape != estimate.shape or truth.ndim != 2:
        raise ValueError("Parity arrays must have the same (samples, targets) shape")
    if len(target_names) != truth.shape[1]:
        raise ValueError("target_names does not match the parity target width")
    destination = _destination(path)
    columns = min(3, truth.shape[1])
    rows = int(np.ceil(truth.shape[1] / columns))
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(4.3 * columns, 4.0 * rows),
        constrained_layout=True,
        squeeze=False,
    )
    for axis in axes.ravel():
        axis.set_visible(False)
    for index, (axis, name) in enumerate(zip(axes.ravel(), target_names, strict=False)):
        axis.set_visible(True)
        low = float(min(np.min(truth[:, index]), np.min(estimate[:, index])))
        high = float(max(np.max(truth[:, index]), np.max(estimate[:, index])))
        padding = max((high - low) * 0.08, max(abs(low), abs(high), 1.0) * 1.0e-6)
        axis.plot([low - padding, high + padding], [low - padding, high + padding], "k--")
        axis.scatter(truth[:, index], estimate[:, index], s=42, alpha=0.8)
        axis.set(
            xlabel=f"true {name}",
            ylabel=f"predicted {name}",
            xlim=(low - padding, high + padding),
            ylim=(low - padding, high + padding),
        )
        axis.grid(alpha=0.2)
    figure.suptitle("Validation parity (physical units)")
    figure.savefig(destination, dpi=170)
    plt.close(figure)
    return destination


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: tuple[str, ...],
    path: str | Path,
) -> Path:
    matrix = np.asarray(confusion_matrix, dtype=np.int64)
    if matrix.shape != (len(class_names), len(class_names)):
        raise ValueError("Confusion matrix shape does not match class_names")
    destination = _destination(path)
    figure, axis = plt.subplots(figsize=(5.5, 4.8), constrained_layout=True)
    image = axis.imshow(matrix, cmap="Blues")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
    axis.set_xticks(range(len(class_names)), class_names)
    axis.set_yticks(range(len(class_names)), class_names)
    axis.set(xlabel="predicted class", ylabel="true class", title="Validation confusion matrix")
    figure.colorbar(image, ax=axis, label="samples")
    figure.savefig(destination, dpi=180)
    plt.close(figure)
    return destination


def plot_generation_error_maps(
    targets: np.ndarray,
    generated: np.ndarray,
    sample_names: tuple[str, ...],
    path: str | Path,
) -> Path:
    """Plot time-averaged targets, generations, and absolute errors."""

    truth = np.asarray(targets, dtype=np.float64)
    estimate = np.asarray(generated, dtype=np.float64)
    if truth.shape != estimate.shape or truth.ndim != 5 or truth.shape[1] != 1:
        raise ValueError("Generation arrays must share shape (N,1,T,H,W)")
    if len(sample_names) != truth.shape[0]:
        raise ValueError("sample_names does not match generation sample count")
    destination = _destination(path)
    rows = truth.shape[0]
    figure, axes = plt.subplots(
        rows,
        3,
        figsize=(11.0, 3.3 * rows),
        constrained_layout=True,
        squeeze=False,
    )
    for row, name in enumerate(sample_names):
        target_mean = np.mean(truth[row, 0], axis=0)
        generated_mean = np.mean(estimate[row, 0], axis=0)
        error_mean = np.mean(np.abs(estimate[row, 0] - truth[row, 0]), axis=0)
        common_max = max(float(np.max(target_mean)), float(np.max(generated_mean)), 1.0e-12)
        error_max = max(float(np.max(error_mean)), 1.0e-12)
        axes[row, 0].imshow(target_mean.T, origin="lower", cmap="inferno", vmin=0, vmax=common_max)
        axes[row, 1].imshow(
            generated_mean.T, origin="lower", cmap="inferno", vmin=0, vmax=common_max
        )
        error_image = axes[row, 2].imshow(
            error_mean.T, origin="lower", cmap="magma", vmin=0, vmax=error_max
        )
        axes[row, 0].set_ylabel(name)
        figure.colorbar(error_image, ax=axes[row, 2], shrink=0.8)
    for column, title in enumerate(("target time mean", "generated time mean", "mean |error|")):
        axes[0, column].set_title(title)
    for axis in axes.ravel():
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle("Conditional-generation validation error maps")
    figure.savefig(destination, dpi=180)
    plt.close(figure)
    return destination
