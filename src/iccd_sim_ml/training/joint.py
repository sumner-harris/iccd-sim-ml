"""One-pass optimization for conditional generation and inverse prediction."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ModuleNotFoundError as exc:  # pragma: no cover - optional environment
    if exc.name == "torch":
        raise ImportError(
            "iccd_sim_ml.training requires PyTorch. Install the ML extras with "
            "`pip install iccd-sim-ml[ml]`."
        ) from exc
    raise

from ..data.scalers import ArrayStandardizer
from .metrics import classification_metrics, regression_metrics

if TYPE_CHECKING:
    from ..models.joint_cvae import JointCVAEOutput


@dataclass(frozen=True)
class JointLossConfig:
    """Weights and robust-loss settings for the joint conditional VAE.

    ``kl_weight`` is the final/base KL multiplier. The epoch functions also
    accept a dynamic ``kl_warmup_weight`` in ``[0, 1]``; the effective KL
    multiplier is their product. Losses are means, so their scale does not
    silently change with image size or batch size.
    """

    reconstruction_weight: float = 1.0
    regression_weight: float = 1.0
    classification_weight: float = 1.0
    kl_weight: float = 1.0e-3
    temporal_gradient_weight: float = 0.0
    smooth_l1_beta: float = 1.0
    free_bits: float = 0.0

    def __post_init__(self) -> None:
        weights = {
            "reconstruction_weight": self.reconstruction_weight,
            "regression_weight": self.regression_weight,
            "classification_weight": self.classification_weight,
            "kl_weight": self.kl_weight,
            "temporal_gradient_weight": self.temporal_gradient_weight,
            "free_bits": self.free_bits,
        }
        if any(not np.isfinite(value) or value < 0.0 for value in weights.values()):
            raise ValueError("Joint loss weights and free_bits must be finite and non-negative")
        if not np.isfinite(self.smooth_l1_beta) or self.smooth_l1_beta <= 0.0:
            raise ValueError("smooth_l1_beta must be finite and positive")


def conditional_gaussian_kl(
    posterior_mean: torch.Tensor,
    posterior_log_variance: torch.Tensor,
    prior_mean: torch.Tensor | None = None,
    prior_log_variance: torch.Tensor | None = None,
    *,
    free_bits: float = 0.0,
) -> torch.Tensor:
    """Mean ``KL[q(z|x,c) || p(z|c)]`` in nats per sample.

    Latent dimensions are summed within each sample and samples are averaged.
    If the prior tensors are omitted, ``p`` is the unit Gaussian. ``free_bits``
    clamps each latent dimension's contribution before reduction, following
    the conventional free-bits objective.
    """

    if posterior_mean.shape != posterior_log_variance.shape:
        raise ValueError("Posterior mean and log-variance shapes must match")
    if posterior_mean.ndim < 2:
        raise ValueError("Latent tensors must include batch and latent dimensions")
    if (prior_mean is None) != (prior_log_variance is None):
        raise ValueError("Prior mean and log variance must either both be supplied or both omitted")
    if not np.isfinite(free_bits) or free_bits < 0.0:
        raise ValueError("free_bits must be finite and non-negative")

    if prior_mean is None:
        prior_mean = torch.zeros_like(posterior_mean)
        prior_log_variance = torch.zeros_like(posterior_log_variance)
    else:
        if prior_mean.shape != posterior_mean.shape:
            raise ValueError("Prior and posterior latent shapes must match")
        if prior_log_variance is None or prior_log_variance.shape != posterior_mean.shape:
            raise ValueError("Prior and posterior latent shapes must match")

    # Keep the exponential terms in float32 even under automatic mixed
    # precision.  The configured log-variance bounds are safe in float32 but
    # their difference can overflow float16.
    posterior_mean_f = posterior_mean.float()
    posterior_log_variance_f = posterior_log_variance.float()
    prior_mean_f = prior_mean.float()
    prior_log_variance_f = prior_log_variance.float()
    elementwise = 0.5 * (
        prior_log_variance_f
        - posterior_log_variance_f
        + torch.exp(posterior_log_variance_f - prior_log_variance_f)
        + (posterior_mean_f - prior_mean_f).square() * torch.exp(-prior_log_variance_f)
        - 1.0
    )
    if free_bits > 0.0:
        elementwise = elementwise.clamp_min(float(free_bits))
    return elementwise.flatten(start_dim=1).sum(dim=1).mean()


def _output_tensor(output: JointCVAEOutput, name: str) -> torch.Tensor:
    value = output[name] if isinstance(output, Mapping) else getattr(output, name)
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Joint model output {name!r} must be a tensor")
    return value


def _optional_output_tensor(output: JointCVAEOutput, name: str) -> torch.Tensor | None:
    if isinstance(output, Mapping):
        value = output.get(name)
    else:
        value = getattr(output, name, None)
    if value is not None and not isinstance(value, torch.Tensor):
        raise TypeError(f"Joint model output {name!r} must be a tensor or None")
    return value


def joint_cvae_loss(
    output: JointCVAEOutput,
    video: torch.Tensor,
    material_properties: torch.Tensor,
    class_index: torch.Tensor,
    *,
    config: JointLossConfig | None = None,
    kl_warmup_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Return the differentiable total and each independently logged component."""

    config = JointLossConfig() if config is None else config
    if not np.isfinite(kl_warmup_weight) or not 0.0 <= kl_warmup_weight <= 1.0:
        raise ValueError("kl_warmup_weight must be finite and in [0, 1]")

    reconstruction = _output_tensor(output, "reconstruction")
    property_prediction = _output_tensor(output, "property_prediction")
    class_logits = _output_tensor(output, "class_logits")
    posterior_mean = _output_tensor(output, "posterior_mean")
    posterior_log_variance = _output_tensor(output, "posterior_log_variance")
    prior_mean = _optional_output_tensor(output, "prior_mean")
    prior_log_variance = _optional_output_tensor(output, "prior_log_variance")

    if reconstruction.shape != video.shape:
        raise ValueError(
            f"Reconstruction shape {tuple(reconstruction.shape)} does not match "
            f"video shape {tuple(video.shape)}"
        )
    if property_prediction.shape != material_properties.shape:
        raise ValueError(
            f"Property prediction shape {tuple(property_prediction.shape)} does not match "
            f"target shape {tuple(material_properties.shape)}"
        )
    if class_logits.ndim != 2 or class_logits.shape[0] != class_index.shape[0]:
        raise ValueError("class_logits must have shape (B,num_classes)")
    if class_index.ndim != 1:
        raise ValueError("class_index must have shape (B,)")

    reconstruction_loss = F.smooth_l1_loss(reconstruction, video, beta=config.smooth_l1_beta)
    regression_loss = F.smooth_l1_loss(
        property_prediction, material_properties, beta=config.smooth_l1_beta
    )
    classification_loss = F.cross_entropy(class_logits, class_index)
    kl_loss = conditional_gaussian_kl(
        posterior_mean,
        posterior_log_variance,
        prior_mean,
        prior_log_variance,
        free_bits=config.free_bits,
    )

    temporal_gradient_loss = reconstruction_loss.new_zeros(())
    if config.temporal_gradient_weight > 0.0 and video.shape[2] > 1:
        reconstructed_difference = reconstruction[:, :, 1:] - reconstruction[:, :, :-1]
        target_difference = video[:, :, 1:] - video[:, :, :-1]
        temporal_gradient_loss = F.smooth_l1_loss(
            reconstructed_difference,
            target_difference,
            beta=config.smooth_l1_beta,
        )

    effective_kl_weight = config.kl_weight * float(kl_warmup_weight)
    total_loss = (
        config.reconstruction_weight * reconstruction_loss
        + config.regression_weight * regression_loss
        + config.classification_weight * classification_loss
        + effective_kl_weight * kl_loss
        + config.temporal_gradient_weight * temporal_gradient_loss
    )
    return {
        "loss": total_loss,
        "reconstruction_loss": reconstruction_loss,
        "regression_loss": regression_loss,
        "classification_loss": classification_loss,
        "kl_loss": kl_loss,
        "temporal_gradient_loss": temporal_gradient_loss,
        "effective_kl_weight": total_loss.new_tensor(effective_kl_weight),
    }


def _move_batch(
    batch: Mapping[str, Any], device: torch.device | str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    required = ("video", "laser_conditions", "material_properties", "class_index")
    missing = [name for name in required if name not in batch]
    if missing:
        raise KeyError(f"Joint-training batch is missing keys: {missing}")
    video = batch["video"].to(device)
    laser_conditions = batch["laser_conditions"].to(device)
    material_properties = batch["material_properties"].to(device)
    class_index = batch["class_index"].to(device)
    return video, laser_conditions, material_properties, class_index


def _inverse_targets(values: torch.Tensor, target_scaler: ArrayStandardizer | None) -> np.ndarray:
    array = values.numpy()
    if target_scaler is not None:
        array = target_scaler.inverse_transform(array)
    return np.asarray(array, dtype=np.float64)


def _finish_joint_metrics(
    targets: list[torch.Tensor],
    property_predictions: list[torch.Tensor],
    class_targets: list[torch.Tensor],
    class_logits: list[torch.Tensor],
    *,
    target_scaler: ArrayStandardizer | None,
    target_names: Sequence[str],
    num_classes: int | None,
) -> dict[str, Any]:
    truth = torch.cat(targets)
    estimate = torch.cat(property_predictions)
    names = tuple(target_names)
    if not names and target_scaler is not None:
        names = target_scaler.names
    regression = regression_metrics(
        _inverse_targets(truth, target_scaler),
        _inverse_targets(estimate, target_scaler),
        target_names=names,
    )
    classification = classification_metrics(
        torch.cat(class_targets).numpy(),
        torch.cat(class_logits).numpy(),
        num_classes=num_classes,
    )
    return {"regression_metrics": regression, "classification_metrics": classification}


def _joint_epoch(
    model: nn.Module,
    loader: Iterable[Mapping[str, Any]],
    *,
    device: torch.device | str,
    loss_config: JointLossConfig,
    kl_warmup_weight: float,
    optimizer: torch.optim.Optimizer | None,
    target_scaler: ArrayStandardizer | None,
    target_names: Sequence[str],
    num_classes: int | None,
    max_grad_norm: float | None,
) -> dict[str, Any]:
    training = optimizer is not None
    model.train(training)
    loss_names = (
        "loss",
        "reconstruction_loss",
        "regression_loss",
        "classification_loss",
        "kl_loss",
        "temporal_gradient_loss",
    )
    totals = {name: 0.0 for name in loss_names}
    sample_count = 0
    targets: list[torch.Tensor] = []
    property_predictions: list[torch.Tensor] = []
    class_targets: list[torch.Tensor] = []
    class_logits: list[torch.Tensor] = []

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            video, laser_conditions, material_properties, class_index = _move_batch(batch, device)
            if training:
                assert optimizer is not None
                optimizer.zero_grad(set_to_none=True)
            output = model(
                video,
                laser_conditions,
                material_properties,
                sample_posterior=training,
            )
            losses = joint_cvae_loss(
                output,
                video,
                material_properties,
                class_index,
                config=loss_config,
                kl_warmup_weight=kl_warmup_weight,
            )
            if training:
                losses["loss"].backward()
                if max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

            batch_size = int(video.shape[0])
            sample_count += batch_size
            for name in loss_names:
                totals[name] += float(losses[name].detach()) * batch_size
            targets.append(material_properties.detach().cpu())
            property_predictions.append(
                _output_tensor(output, "property_prediction").detach().cpu()
            )
            class_targets.append(class_index.detach().cpu())
            class_logits.append(_output_tensor(output, "class_logits").detach().cpu())

    if sample_count == 0:
        raise ValueError("Cannot run a joint epoch on an empty loader")
    result: dict[str, Any] = {name: total / sample_count for name, total in totals.items()}
    result["effective_kl_weight"] = loss_config.kl_weight * float(kl_warmup_weight)
    result["samples"] = sample_count
    result.update(
        _finish_joint_metrics(
            targets,
            property_predictions,
            class_targets,
            class_logits,
            target_scaler=target_scaler,
            target_names=target_names,
            num_classes=num_classes,
        )
    )
    return result


def train_joint_one_epoch(
    model: nn.Module,
    loader: Iterable[Mapping[str, Any]],
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device | str,
    loss_config: JointLossConfig | None = None,
    kl_warmup_weight: float = 1.0,
    target_scaler: ArrayStandardizer | None = None,
    target_names: Sequence[str] = (),
    num_classes: int | None = None,
    max_grad_norm: float | None = None,
) -> dict[str, Any]:
    """Train all three operating modes together for one epoch."""

    return _joint_epoch(
        model,
        loader,
        device=device,
        loss_config=JointLossConfig() if loss_config is None else loss_config,
        kl_warmup_weight=kl_warmup_weight,
        optimizer=optimizer,
        target_scaler=target_scaler,
        target_names=target_names,
        num_classes=num_classes,
        max_grad_norm=max_grad_norm,
    )


def evaluate_joint_epoch(
    model: nn.Module,
    loader: Iterable[Mapping[str, Any]],
    *,
    device: torch.device | str,
    loss_config: JointLossConfig | None = None,
    kl_warmup_weight: float = 1.0,
    target_scaler: ArrayStandardizer | None = None,
    target_names: Sequence[str] = (),
    num_classes: int | None = None,
) -> dict[str, Any]:
    """Evaluate deterministic posterior-mean reconstruction and both heads."""

    return _joint_epoch(
        model,
        loader,
        device=device,
        loss_config=JointLossConfig() if loss_config is None else loss_config,
        kl_warmup_weight=kl_warmup_weight,
        optimizer=None,
        target_scaler=target_scaler,
        target_names=target_names,
        num_classes=num_classes,
        max_grad_norm=None,
    )
