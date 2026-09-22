"""Precomputed-video datasets, leakage-safe splits, and fitted scalers."""

from .datasets import (
    JointPrecomputedVideoDataset,
    NPZKeys,
    PrecomputedVideoDataset,
    save_precomputed_sample,
)
from .manifest import DatasetManifest, SampleRecord, as_manifest
from .scalers import ArrayStandardizer, ScalerBundle, VideoStandardizer, fit_train_scalers
from .splits import (
    SplitManifest,
    make_group_split,
    make_known_material_split,
    make_legacy_train_validation_split,
    validate_split,
)

__all__ = [
    "ArrayStandardizer",
    "DatasetManifest",
    "JointPrecomputedVideoDataset",
    "NPZKeys",
    "PrecomputedVideoDataset",
    "SampleRecord",
    "ScalerBundle",
    "SplitManifest",
    "VideoStandardizer",
    "as_manifest",
    "fit_train_scalers",
    "make_group_split",
    "make_known_material_split",
    "make_legacy_train_validation_split",
    "save_precomputed_sample",
    "validate_split",
]
