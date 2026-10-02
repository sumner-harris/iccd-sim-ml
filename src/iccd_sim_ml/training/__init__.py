"""Metrics and optional PyTorch training utilities."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from .metrics import classification_metrics, regression_metrics, video_generation_metrics

_LAZY_EXPORTS = {
    "average_material_set_predictions": (
        ".ensemble",
        "average_material_set_predictions",
    ),
    "JointLossConfig": (".joint", "JointLossConfig"),
    "TargetWeightedSmoothL1Loss": (
        ".set_regression",
        "TargetWeightedSmoothL1Loss",
    ),
    "checkpoint_metadata": (".checkpoints", "checkpoint_metadata"),
    "conditional_gaussian_kl": (".joint", "conditional_gaussian_kl"),
    "collect_material_set_predictions": (
        ".set_regression",
        "collect_material_set_predictions",
    ),
    "evaluate_epoch": (".engine", "evaluate_epoch"),
    "evaluate_joint_epoch": (".joint", "evaluate_joint_epoch"),
    "joint_cvae_loss": (".joint", "joint_cvae_loss"),
    "load_checkpoint": (".checkpoints", "load_checkpoint"),
    "save_checkpoint": (".checkpoints", "save_checkpoint"),
    "train_one_epoch": (".engine", "train_one_epoch"),
    "train_joint_one_epoch": (".joint", "train_joint_one_epoch"),
    "train_material_set_one_epoch": (
        ".set_regression",
        "train_material_set_one_epoch",
    ),
}

__all__ = [
    "JointLossConfig",
    "TargetWeightedSmoothL1Loss",
    "average_material_set_predictions",
    "checkpoint_metadata",
    "classification_metrics",
    "conditional_gaussian_kl",
    "collect_material_set_predictions",
    "evaluate_epoch",
    "evaluate_joint_epoch",
    "joint_cvae_loss",
    "load_checkpoint",
    "regression_metrics",
    "save_checkpoint",
    "train_one_epoch",
    "train_joint_one_epoch",
    "train_material_set_one_epoch",
    "video_generation_metrics",
]


def __getattr__(name: str) -> Any:
    if name not in _LAZY_EXPORTS:
        raise AttributeError(name)
    module_name, attribute = _LAZY_EXPORTS[name]
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted([*globals(), *__all__])
