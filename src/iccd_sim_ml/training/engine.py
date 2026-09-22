"""Small, explicit PyTorch epoch loops for the two supported prediction tasks."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Literal

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

from .metrics import classification_metrics, regression_metrics

Task = Literal["regression", "classification"]


def _forward_batch(
    model: nn.Module, batch: Mapping[str, Any], device: torch.device | str
) -> tuple[torch.Tensor, torch.Tensor]:
    video = batch["video"].to(device)
    target = batch["target"].to(device)
    features = batch.get("features")
    conditions = None
    if features is not None:
        features = features.to(device)
        if features.numel() > 0:
            conditions = features
    return model(video, conditions), target


def _finish_metrics(
    task: Task,
    targets: list[torch.Tensor],
    predictions: list[torch.Tensor],
    *,
    target_names: Sequence[str] = (),
) -> dict[str, Any]:
    truth = torch.cat(targets).numpy()
    estimate = torch.cat(predictions).numpy()
    if task == "regression":
        return regression_metrics(truth, estimate, target_names=target_names)
    return classification_metrics(truth, estimate)


def train_one_epoch(
    model: nn.Module,
    loader: Iterable[Mapping[str, Any]],
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    *,
    device: torch.device | str,
    task: Task,
    target_names: Sequence[str] = (),
    max_grad_norm: float | None = None,
) -> dict[str, Any]:
    """Train for one epoch and return sample-weighted loss and task metrics."""

    if task not in {"regression", "classification"}:
        raise ValueError("task must be 'regression' or 'classification'")
    model.train()
    loss_sum = 0.0
    sample_count = 0
    all_targets: list[torch.Tensor] = []
    all_predictions: list[torch.Tensor] = []
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        output, target = _forward_batch(model, batch, device)
        loss = criterion(output, target)
        loss.backward()
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        batch_size = int(target.shape[0])
        loss_sum += float(loss.detach()) * batch_size
        sample_count += batch_size
        all_targets.append(target.detach().cpu())
        all_predictions.append(output.detach().cpu())
    if sample_count == 0:
        raise ValueError("Cannot train on an empty loader")
    metrics = _finish_metrics(task, all_targets, all_predictions, target_names=target_names)
    return {"loss": loss_sum / sample_count, **metrics}


def evaluate_epoch(
    model: nn.Module,
    loader: Iterable[Mapping[str, Any]],
    criterion: nn.Module,
    *,
    device: torch.device | str,
    task: Task,
    target_names: Sequence[str] = (),
) -> dict[str, Any]:
    """Evaluate one epoch without gradients."""

    if task not in {"regression", "classification"}:
        raise ValueError("task must be 'regression' or 'classification'")
    model.eval()
    loss_sum = 0.0
    sample_count = 0
    all_targets: list[torch.Tensor] = []
    all_predictions: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in loader:
            output, target = _forward_batch(model, batch, device)
            loss = criterion(output, target)
            batch_size = int(target.shape[0])
            loss_sum += float(loss) * batch_size
            sample_count += batch_size
            all_targets.append(target.detach().cpu())
            all_predictions.append(output.detach().cpu())
    if sample_count == 0:
        raise ValueError("Cannot evaluate an empty loader")
    metrics = _finish_metrics(task, all_targets, all_predictions, target_names=target_names)
    return {"loss": loss_sum / sample_count, **metrics}
