"""Optional PyTorch models.

Importing this namespace does not require PyTorch. Accessing a model symbol
loads the implementation and raises an actionable error when the ``ml`` extra
is not installed.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "JointConditionalVAE": (".joint_cvae", "JointConditionalVAE"),
    "JointCVAEConfig": (".joint_cvae", "JointCVAEConfig"),
    "JointCVAEOutput": (".joint_cvae", "JointCVAEOutput"),
    "MLP": (".blocks", "MLP"),
    "SpatialDownsample": (".blocks", "SpatialDownsample"),
    "SpatioTemporalBlock": (".blocks", "SpatioTemporalBlock"),
    "VideoClassifier": (".video", "VideoClassifier"),
    "VideoClassifierConfig": (".video", "VideoClassifierConfig"),
    "VideoEncoder3D": (".video", "VideoEncoder3D"),
    "VideoEncoderConfig": (".video", "VideoEncoderConfig"),
    "VideoRegressor": (".video", "VideoRegressor"),
    "VideoRegressorConfig": (".video", "VideoRegressorConfig"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(name)
    module_name, attribute = _EXPORTS[name]
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted([*globals(), *__all__])
