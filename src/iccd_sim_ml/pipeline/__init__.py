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
from .fov import (
    RadianceExtent,
    bracketing_frame_indices,
    select_maximum_condition,
    summarize_radiance_extent,
)
from .quality import (
    PlasmaWindowQuality,
    RadianceQuality,
    sparse_level_stages,
    summarize_plasma_window,
    summarize_radiance,
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
    "PlasmaWindowQuality",
    "RadianceQuality",
    "RadianceExtent",
    "SourceSample",
    "TrainingRunConfig",
    "discover_balanced_subset",
    "bracketing_frame_indices",
    "ensure_continuum_cache",
    "ensure_proxy_cache",
    "fit_experiment_scalers",
    "run_classification_experiment",
    "run_experiment",
    "run_joint_cvae_experiment",
    "run_regression_experiment",
    "sparse_level_stages",
    "select_maximum_condition",
    "summarize_plasma_window",
    "summarize_radiance",
    "summarize_radiance_extent",
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
