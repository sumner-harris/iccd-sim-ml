"""Property-faithfulness losses for conditional plume-video generation.

The ordinary joint cVAE predicts properties from the *input* video.  That
inverse head does not constrain a decoded video to preserve the properties
used to condition the decoder.  The losses here close that loop with a frozen,
independently trained inverse model.  Teacher weights never receive gradients,
but gradients through the teacher input update the cVAE generator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ModuleNotFoundError as exc:  # pragma: no cover - optional ML dependency
    if exc.name == "torch":
        raise ImportError(
            "Property-cycle training requires PyTorch. Install the ML extras with "
            "`pip install iccd-sim-ml[ml]`."
        ) from exc
    raise


@dataclass(frozen=True)
class PropertyCycleLossConfig:
    """Weights for frozen-teacher and local-intervention supervision.

    Property tensors are standardized with train-only statistics. Therefore
    ``perturbation_std`` is expressed in standard deviations, not physical
    units or percent.
    """

    posterior_weight: float = 1.0
    prior_weight: float = 1.0
    intervention_weight: float = 0.5
    cross_talk_weight: float = 0.25
    perturbation_std: float = 0.25
    smooth_l1_beta: float = 0.25

    def __post_init__(self) -> None:
        weights = (
            self.posterior_weight,
            self.prior_weight,
            self.intervention_weight,
            self.cross_talk_weight,
        )
        if any(not np.isfinite(value) or value < 0.0 for value in weights):
            raise ValueError("Property-cycle weights must be finite and non-negative")
        if not np.isfinite(self.perturbation_std) or self.perturbation_std <= 0.0:
            raise ValueError("perturbation_std must be finite and positive")
        if not np.isfinite(self.smooth_l1_beta) or self.smooth_l1_beta <= 0.0:
            raise ValueError("smooth_l1_beta must be finite and positive")


def freeze_property_teacher(teacher: nn.Module) -> nn.Module:
    """Freeze a property teacher while preserving input gradients."""

    teacher.requires_grad_(False)
    teacher.eval()
    return teacher


def _output_tensor(output: Any, name: str) -> torch.Tensor:
    value = output[name] if isinstance(output, dict) else getattr(output, name)
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Joint model output {name!r} must be a tensor")
    return value


def _intervention_targets(
    material_properties: torch.Tensor,
    *,
    perturbation_std: float,
    offset: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size, property_count = material_properties.shape
    indices = (
        torch.arange(batch_size, device=material_properties.device) + offset
    ) % property_count
    signs = torch.where(
        (torch.arange(batch_size, device=material_properties.device) + offset) % 2 == 0,
        material_properties.new_ones(batch_size),
        -material_properties.new_ones(batch_size),
    )
    delta = signs * float(perturbation_std)
    perturbed = material_properties.clone()
    perturbed[torch.arange(batch_size, device=material_properties.device), indices] += delta
    return perturbed, indices, delta


def _intervention_losses(
    base_prediction: torch.Tensor,
    perturbed_prediction: torch.Tensor,
    property_indices: torch.Tensor,
    requested_delta: torch.Tensor,
    *,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    response = perturbed_prediction - base_prediction
    rows = torch.arange(response.shape[0], device=response.device)
    selected = response[rows, property_indices]
    targeted = F.smooth_l1_loss(selected, requested_delta, beta=beta)
    mask = torch.ones_like(response, dtype=torch.bool)
    mask[rows, property_indices] = False
    cross_talk = (
        F.smooth_l1_loss(response[mask], torch.zeros_like(response[mask]), beta=beta)
        if mask.any()
        else response.new_zeros(())
    )
    return targeted, cross_talk


def single_video_property_cycle_losses(
    model: nn.Module,
    output: Any,
    teacher: nn.Module,
    laser_conditions: torch.Tensor,
    material_properties: torch.Tensor,
    *,
    config: PropertyCycleLossConfig | None = None,
    weight_scale: float = 1.0,
    intervention_offset: int = 0,
) -> dict[str, torch.Tensor]:
    """Constrain posterior and prior videos with a frozen single-video teacher."""

    config = PropertyCycleLossConfig() if config is None else config
    if not np.isfinite(weight_scale) or not 0.0 <= weight_scale <= 1.0:
        raise ValueError("weight_scale must be finite and in [0, 1]")
    freeze_property_teacher(teacher)

    reconstruction = _output_tensor(output, "reconstruction")
    prior_mean = _output_tensor(output, "prior_mean")
    prior_video = model.decode_latent(prior_mean, laser_conditions, material_properties)
    posterior_prediction = teacher(reconstruction, laser_conditions)
    prior_prediction = teacher(prior_video, laser_conditions)
    posterior_loss = F.smooth_l1_loss(
        posterior_prediction,
        material_properties,
        beta=config.smooth_l1_beta,
    )
    prior_loss = F.smooth_l1_loss(
        prior_prediction,
        material_properties,
        beta=config.smooth_l1_beta,
    )

    perturbed, property_indices, requested_delta = _intervention_targets(
        material_properties,
        perturbation_std=config.perturbation_std,
        offset=intervention_offset,
    )
    perturbed_video = model.decode_latent(prior_mean, laser_conditions, perturbed)
    perturbed_prediction = teacher(perturbed_video, laser_conditions)
    intervention_loss, cross_talk_loss = _intervention_losses(
        prior_prediction,
        perturbed_prediction,
        property_indices,
        requested_delta,
        beta=config.smooth_l1_beta,
    )
    total = float(weight_scale) * (
        config.posterior_weight * posterior_loss
        + config.prior_weight * prior_loss
        + config.intervention_weight * intervention_loss
        + config.cross_talk_weight * cross_talk_loss
    )
    return {
        "property_cycle_loss": total,
        "posterior_property_loss": posterior_loss,
        "prior_property_loss": prior_loss,
        "intervention_loss": intervention_loss,
        "cross_talk_loss": cross_talk_loss,
        "prior_video": prior_video,
    }


def set_property_cycle_losses(
    model: nn.Module,
    output: Any,
    teacher: nn.Module,
    laser_conditions: torch.Tensor,
    material_properties: torch.Tensor,
    set_mask: torch.Tensor,
    *,
    config: PropertyCycleLossConfig | None = None,
    weight_scale: float = 1.0,
    intervention_offset: int = 0,
) -> dict[str, torch.Tensor]:
    """Constrain same-material video bags with a frozen Set Transformer.

    ``output`` is produced from flattened ``(B*K,...)`` videos. Conditions have
    shape ``(B,K,F)`` and one target vector is supplied for each of the ``B``
    material bags.
    """

    config = PropertyCycleLossConfig() if config is None else config
    if laser_conditions.ndim != 3 or material_properties.ndim != 2:
        raise ValueError("Set conditions must be (B,K,F) and properties must be (B,P)")
    batch_size, set_size = laser_conditions.shape[:2]
    if set_mask.shape != (batch_size, set_size):
        raise ValueError("set_mask shape does not match the set conditions")
    freeze_property_teacher(teacher)

    flat_conditions = laser_conditions.flatten(0, 1)
    flat_properties = material_properties[:, None, :].expand(-1, set_size, -1).flatten(0, 1)
    reconstruction = _output_tensor(output, "reconstruction")
    prior_mean = _output_tensor(output, "prior_mean")
    posterior_videos = reconstruction.reshape(batch_size, set_size, *reconstruction.shape[1:])
    prior_video_flat = model.decode_latent(prior_mean, flat_conditions, flat_properties)
    prior_videos = prior_video_flat.reshape(batch_size, set_size, *prior_video_flat.shape[1:])
    posterior_prediction = teacher(posterior_videos, laser_conditions, set_mask)
    prior_prediction = teacher(prior_videos, laser_conditions, set_mask)
    posterior_loss = F.smooth_l1_loss(
        posterior_prediction,
        material_properties,
        beta=config.smooth_l1_beta,
    )
    prior_loss = F.smooth_l1_loss(
        prior_prediction,
        material_properties,
        beta=config.smooth_l1_beta,
    )

    perturbed, property_indices, requested_delta = _intervention_targets(
        material_properties,
        perturbation_std=config.perturbation_std,
        offset=intervention_offset,
    )
    flat_perturbed = perturbed[:, None, :].expand(-1, set_size, -1).flatten(0, 1)
    perturbed_flat = model.decode_latent(prior_mean, flat_conditions, flat_perturbed)
    perturbed_videos = perturbed_flat.reshape(batch_size, set_size, *perturbed_flat.shape[1:])
    perturbed_prediction = teacher(perturbed_videos, laser_conditions, set_mask)
    intervention_loss, cross_talk_loss = _intervention_losses(
        prior_prediction,
        perturbed_prediction,
        property_indices,
        requested_delta,
        beta=config.smooth_l1_beta,
    )
    total = float(weight_scale) * (
        config.posterior_weight * posterior_loss
        + config.prior_weight * prior_loss
        + config.intervention_weight * intervention_loss
        + config.cross_talk_weight * cross_talk_loss
    )
    return {
        "property_cycle_loss": total,
        "posterior_property_loss": posterior_loss,
        "prior_property_loss": prior_loss,
        "intervention_loss": intervention_loss,
        "cross_talk_loss": cross_talk_loss,
        "prior_video": prior_videos,
    }
