"""Training utilities for material-level set regression."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

try:
    import torch
    from torch import nn
except ModuleNotFoundError as exc:  # pragma: no cover - optional ML dependency
    if exc.name == "torch":
        raise ImportError(
            "Material-set training requires PyTorch. Install the ML extras with "
            "`pip install iccd-sim-ml[ml]`."
        ) from exc
    raise

from ..data.scalers import ArrayStandardizer
from .metrics import regression_metrics


def _move_batch(
    batch: Mapping[str, Any], device: torch.device | str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    required = ("videos", "conditions", "set_mask", "target")
    missing = [name for name in required if name not in batch]
    if missing:
        raise KeyError(f"Material-set batch is missing keys: {missing}")
    return (
        batch["videos"].to(device),
        batch["conditions"].to(device),
        batch["set_mask"].to(device),
        batch["target"].to(device),
    )


def train_material_set_one_epoch(
    model: nn.Module,
    loader: Any,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device | str,
    criterion: nn.Module | None = None,
    gradient_clip_norm: float | None = 5.0,
    individual_loss_weight: float = 0.0,
) -> dict[str, float | int]:
    """Optimize one epoch of bag-level property regression."""

    if not np.isfinite(individual_loss_weight) or individual_loss_weight < 0.0:
        raise ValueError("individual_loss_weight must be finite and non-negative")
    loss_function = nn.SmoothL1Loss() if criterion is None else criterion
    model.train()
    loss_sum = 0.0
    set_loss_sum = 0.0
    individual_loss_sum = 0.0
    sample_count = 0
    for batch in loader:
        videos, conditions, set_mask, target = _move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        if individual_loss_weight > 0.0:
            prediction, individual_prediction, valid_mask = model.forward_with_individual(
                videos, conditions, set_mask
            )
            expanded_target = target.unsqueeze(1).expand_as(individual_prediction)
            individual_loss = loss_function(
                individual_prediction[valid_mask], expanded_target[valid_mask]
            )
        else:
            prediction = model(videos, conditions, set_mask)
            individual_loss = prediction.new_zeros(())
        set_loss = loss_function(prediction, target)
        loss = set_loss + individual_loss_weight * individual_loss
        loss.backward()
        if gradient_clip_norm is not None:
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        optimizer.step()
        batch_size = target.shape[0]
        loss_sum += float(loss.detach()) * batch_size
        set_loss_sum += float(set_loss.detach()) * batch_size
        individual_loss_sum += float(individual_loss.detach()) * batch_size
        sample_count += batch_size
    if sample_count == 0:
        raise ValueError("Material-set training loader produced no samples")
    return {
        "loss": loss_sum / sample_count,
        "set_loss": set_loss_sum / sample_count,
        "individual_loss": individual_loss_sum / sample_count,
        "sets": sample_count,
    }


def collect_material_set_predictions(
    model: nn.Module,
    loader: Any,
    *,
    device: torch.device | str,
    target_scaler: ArrayStandardizer | None,
    target_names: Sequence[str],
    criterion: nn.Module | None = None,
) -> dict[str, Any]:
    """Collect bag predictions and aggregate them to one row per element."""

    loss_function = nn.SmoothL1Loss() if criterion is None else criterion
    model.eval()
    targets: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    elements: list[str] = []
    loss_sum = 0.0
    sample_count = 0
    with torch.inference_mode():
        for batch in loader:
            videos, conditions, set_mask, target = _move_batch(batch, device)
            prediction = model(videos, conditions, set_mask)
            batch_size = target.shape[0]
            loss_sum += float(loss_function(prediction, target)) * batch_size
            sample_count += batch_size
            targets.append(target.cpu().numpy())
            predictions.append(prediction.cpu().numpy())
            elements.extend(str(value) for value in batch["element"])
    if sample_count == 0:
        raise ValueError("Material-set evaluation loader produced no samples")

    scaled_truth = np.concatenate(targets)
    scaled_prediction = np.concatenate(predictions)
    grouped_truth: dict[str, list[np.ndarray]] = defaultdict(list)
    grouped_prediction: dict[str, list[np.ndarray]] = defaultdict(list)
    for element, truth_row, prediction_row in zip(
        elements, scaled_truth, scaled_prediction, strict=True
    ):
        grouped_truth[element].append(truth_row)
        grouped_prediction[element].append(prediction_row)
    material_names = tuple(sorted(grouped_truth))
    material_truth_scaled = np.stack(
        [np.mean(np.stack(grouped_truth[name]), axis=0) for name in material_names]
    )
    material_prediction_scaled = np.stack(
        [np.mean(np.stack(grouped_prediction[name]), axis=0) for name in material_names]
    )
    element_loss = float(
        nn.functional.smooth_l1_loss(
            torch.from_numpy(material_prediction_scaled),
            torch.from_numpy(material_truth_scaled),
        )
    )

    bag_truth = scaled_truth
    bag_prediction = scaled_prediction
    material_truth = material_truth_scaled
    material_prediction = material_prediction_scaled
    if target_scaler is not None:
        bag_truth = target_scaler.inverse_transform(bag_truth)
        bag_prediction = target_scaler.inverse_transform(bag_prediction)
        material_truth = target_scaler.inverse_transform(material_truth)
        material_prediction = target_scaler.inverse_transform(material_prediction)
    return {
        "loss": loss_sum / sample_count,
        "element_loss": element_loss,
        "sets": sample_count,
        "elements": material_names,
        "sets_per_element": {name: len(grouped_truth[name]) for name in material_names},
        "bag_targets": bag_truth,
        "bag_predictions": bag_prediction,
        "material_targets": material_truth,
        "material_predictions": material_prediction,
        "bag_metrics": regression_metrics(bag_truth, bag_prediction, target_names=target_names),
        "material_metrics": regression_metrics(
            material_truth, material_prediction, target_names=target_names
        ),
    }
