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
    RadianceMorphology,
    assess_radiance_morphology,
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
    "run_material_set_experiment": (".experiments", "run_material_set_experiment"),
    "run_regression_experiment": (".experiments", "run_regression_experiment"),
    "create_split": (".workflow", "create_split"),
    "resolve_split": (".workflow", "resolve_split"),
    "subset_manifest": (".workflow", "subset_manifest"),
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
    "RadianceMorphology",
    "assess_radiance_morphology",
    "SourceSample",
    "TrainingRunConfig",
    "create_split",
    "discover_balanced_subset",
    "bracketing_frame_indices",
    "ensure_continuum_cache",
    "ensure_proxy_cache",
    "fit_experiment_scalers",
    "run_classification_experiment",
    "run_experiment",
    "run_joint_cvae_experiment",
    "run_material_set_experiment",
    "run_regression_experiment",
    "resolve_split",
    "sparse_level_stages",
    "select_maximum_condition",
    "summarize_plasma_window",
    "summarize_radiance",
    "summarize_radiance_extent",
    "subset_manifest",
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
