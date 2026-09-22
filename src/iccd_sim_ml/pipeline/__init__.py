"""End-to-end preprocessing, training, and reporting orchestration."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from .cache import (
    CONDITION_NAMES,
    TARGET_NAMES,
    CacheResult,
    ContinuumCacheConfig,
    ProxyCacheConfig,
    SourceSample,
    discover_balanced_subset,
    ensure_continuum_cache,
    ensure_proxy_cache,
)

_LAZY_EXPORTS = {
    "TrainingRunConfig": (".experiments", "TrainingRunConfig"),
    "fit_experiment_scalers": (".experiments", "fit_experiment_scalers"),
    "run_classification_experiment": (".experiments", "run_classification_experiment"),
    "run_experiment": (".experiments", "run_experiment"),
    "run_joint_cvae_experiment": (".experiments", "run_joint_cvae_experiment"),
    "run_regression_experiment": (".experiments", "run_regression_experiment"),
}

__all__ = [
    "CONDITION_NAMES",
    "TARGET_NAMES",
    "CacheResult",
    "ContinuumCacheConfig",
    "ProxyCacheConfig",
    "SourceSample",
    "TrainingRunConfig",
    "discover_balanced_subset",
    "ensure_continuum_cache",
    "ensure_proxy_cache",
    "fit_experiment_scalers",
    "run_classification_experiment",
    "run_experiment",
    "run_joint_cvae_experiment",
    "run_regression_experiment",
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
