from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from iccd_sim_ml.models import (  # noqa: E402
    JointConditionalVAE,
    JointCVAEConfig,
    VideoEncoderConfig,
)
from iccd_sim_ml.training import (  # noqa: E402
    PropertyCycleLossConfig,
    set_property_cycle_losses,
    single_video_property_cycle_losses,
)


def _model() -> JointConditionalVAE:
    return JointConditionalVAE(
        JointCVAEConfig(
            video_shape=(1, 2, 8, 8),
            encoder=VideoEncoderConfig(
                input_channels=1,
                stage_channels=(4,),
                blocks_per_stage=1,
                embedding_dim=6,
            ),
            property_names=("a", "b"),
            num_classes=2,
            latent_dim=3,
            laser_hidden=(4,),
            laser_embedding_dim=3,
            condition_hidden=(5,),
            condition_embedding_dim=4,
            posterior_hidden=(6,),
            prior_hidden=(5,),
            regression_hidden=(5,),
            classification_hidden=(5,),
            decoder_seed_shape=(1, 2, 2),
            decoder_channels=(6, 4),
            group_norm_groups=2,
        )
    )


class _SingleTeacher(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(3, 2)

    def forward(self, video: torch.Tensor, conditions: torch.Tensor) -> torch.Tensor:
        level = video.mean(dim=(1, 2, 3, 4), keepdim=False).unsqueeze(1)
        return self.linear(torch.cat((level, conditions), dim=1))


class _SetTeacher(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(3, 2)

    def forward(
        self,
        videos: torch.Tensor,
        conditions: torch.Tensor,
        set_mask: torch.Tensor,
    ) -> torch.Tensor:
        weights = set_mask.to(videos.dtype)
        levels = videos.mean(dim=(2, 3, 4, 5))
        level = (levels * weights).sum(1) / weights.sum(1)
        condition = (conditions * weights.unsqueeze(-1)).sum(1) / weights.sum(1, keepdim=True)
        return self.linear(torch.cat((level.unsqueeze(1), condition), dim=1))


def _inputs(batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(19)
    return (
        torch.randn(batch_size, 1, 2, 8, 8, generator=generator),
        torch.randn(batch_size, 2, generator=generator),
        torch.randn(batch_size, 2, generator=generator),
    )


def test_public_prior_and_decode_support_fixed_latent_interventions() -> None:
    model = _model()
    _, conditions, properties = _inputs(3)
    mean, log_variance = model.prior_parameters(conditions, properties)
    decoded = model.decode_latent(mean, conditions, properties)

    assert mean.shape == log_variance.shape == (3, 3)
    assert decoded.shape == (3, 1, 2, 8, 8)


def test_single_video_cycle_freezes_teacher_and_updates_generator() -> None:
    model = _model()
    teacher = _SingleTeacher()
    videos, conditions, properties = _inputs(4)
    output = model(videos, conditions, properties, sample_posterior=False)
    losses = single_video_property_cycle_losses(
        model,
        output,
        teacher,
        conditions,
        properties,
        config=PropertyCycleLossConfig(),
    )
    losses["property_cycle_loss"].backward()

    assert torch.isfinite(losses["property_cycle_loss"])
    assert all(not parameter.requires_grad for parameter in teacher.parameters())
    assert all(parameter.grad is None for parameter in teacher.parameters())
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad) > 0
        for parameter in model.decoder.parameters()
    )


def test_set_cycle_uses_complete_same_material_bags() -> None:
    model = _model()
    teacher = _SetTeacher()
    batch_size, set_size = 2, 3
    videos, conditions, expanded_properties = _inputs(batch_size * set_size)
    properties = expanded_properties.reshape(batch_size, set_size, -1)[:, 0]
    output = model(videos, conditions, expanded_properties, sample_posterior=False)
    losses = set_property_cycle_losses(
        model,
        output,
        teacher,
        conditions.reshape(batch_size, set_size, -1),
        properties,
        torch.ones(batch_size, set_size, dtype=torch.bool),
        intervention_offset=1,
    )
    losses["property_cycle_loss"].backward()

    assert losses["prior_video"].shape == (batch_size, set_size, 1, 2, 8, 8)
    assert torch.isfinite(losses["intervention_loss"])
    assert all(parameter.grad is None for parameter in teacher.parameters())
