"""Atomic, self-describing model checkpoints."""

from __future__ import annotations

import os
import uuid
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
    from torch import nn
except ModuleNotFoundError as exc:  # pragma: no cover - optional environment
    if exc.name == "torch":
        raise ImportError(
            "iccd_sim_ml.training requires PyTorch. Install the ML extras with "
            "`pip install iccd-sim-ml[ml]`."
        ) from exc
    raise


CHECKPOINT_VERSION = 1


def _metadata(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "to_dict"):
        return _metadata(value.to_dict())
    if is_dataclass(value):
        return _metadata(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _metadata(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_metadata(item) for item in value]
    raise TypeError(f"Checkpoint metadata value {type(value).__name__} is not serializable")


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    *,
    epoch: int,
    config: Any,
    scalers: Any,
    split: Any,
    optimizer: torch.optim.Optimizer | None = None,
    metrics: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """Atomically save weights plus all preprocessing and split provenance."""

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": CHECKPOINT_VERSION,
        "epoch": int(epoch),
        "model_state": model.state_dict(),
        "optimizer_state": None if optimizer is None else optimizer.state_dict(),
        "config": _metadata(config),
        "scalers": _metadata(scalers),
        "split": _metadata(split),
        "metrics": _metadata({} if metrics is None else metrics),
        "extra": _metadata({} if extra is None else extra),
    }
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def load_checkpoint(
    path: str | Path,
    *,
    model: nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: torch.device | str = "cpu",
    strict: bool = True,
) -> dict[str, Any]:
    """Load a trusted project checkpoint and optionally restore model/optimizer."""

    source = Path(path).expanduser().resolve()
    payload = torch.load(source, map_location=map_location, weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint {source} does not contain a mapping")
    required = {"format_version", "model_state", "config", "scalers", "split"}
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"Checkpoint {source} is missing keys: {sorted(missing)}")
    if int(payload["format_version"]) != CHECKPOINT_VERSION:
        raise ValueError(
            f"Unsupported checkpoint version {payload['format_version']}; "
            f"expected {CHECKPOINT_VERSION}"
        )
    if model is not None:
        model.load_state_dict(payload["model_state"], strict=strict)
    if optimizer is not None:
        if payload.get("optimizer_state") is None:
            raise ValueError("Checkpoint has no optimizer state")
        optimizer.load_state_dict(payload["optimizer_state"])
    return payload


def checkpoint_metadata(
    path: str | Path, *, map_location: torch.device | str = "cpu"
) -> dict[str, Any]:
    """Read reproducibility metadata without returning weight dictionaries."""

    payload = load_checkpoint(path, map_location=map_location)
    return {
        key: value
        for key, value in payload.items()
        if key not in {"model_state", "optimizer_state"}
    }
