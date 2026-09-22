from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from iccd_sim_ml.models import (  # noqa: E402
    JointConditionalVAE,
    JointCVAEConfig,
    VideoEncoderConfig,
)


def _config() -> JointCVAEConfig:
    return JointCVAEConfig(
        video_shape=(1, 4, 16, 16),
        encoder=VideoEncoderConfig(
            input_channels=1,
            stage_channels=(4, 8),
            blocks_per_stage=1,
            embedding_dim=12,
        ),
        laser_condition_names=("laser_power_wcm", "rspot"),
        property_names=("boiling_point", "critical_temperature", "density"),
        num_classes=4,
        latent_dim=5,
        laser_hidden=(6,),
        laser_embedding_dim=4,
        condition_hidden=(8,),
        condition_embedding_dim=6,
        posterior_hidden=(10,),
        prior_hidden=(8,),
        regression_hidden=(8,),
        classification_hidden=(8,),
        decoder_seed_shape=(1, 4, 4),
        decoder_channels=(8, 6, 4),
        group_norm_groups=4,
    )


def _inputs(batch_size: int = 2) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.randn(batch_size, 1, 4, 16, 16),
        torch.randn(batch_size, 2),
        torch.randn(batch_size, 3),
    )


def _conditional_kl(output: object) -> torch.Tensor:
    variance_ratio = torch.exp(output.posterior_log_variance - output.prior_log_variance)
    squared_mean = (output.posterior_mean - output.prior_mean).square() * torch.exp(
        -output.prior_log_variance
    )
    return (
        0.5
        * (
            output.prior_log_variance
            - output.posterior_log_variance
            + variance_ratio
            + squared_mean
            - 1.0
        )
        .sum(dim=1)
        .mean()
    )


def test_config_round_trip_preserves_ordered_names_and_rejects_width_mismatch() -> None:
    config = _config()

    restored = JointCVAEConfig.from_dict(config.to_dict())

    assert restored == config
    assert restored.laser_condition_dim == 2
    assert restored.material_property_dim == 3
    invalid = config.to_dict()
    invalid["material_property_dim"] = 4
    with pytest.raises(ValueError, match="disagrees"):
        JointCVAEConfig.from_dict(invalid)


def test_joint_forward_shapes_and_deterministic_reconstruction() -> None:
    model = JointConditionalVAE(_config()).eval()
    video, laser, properties = _inputs()

    first = model(video, laser, properties, sample_posterior=False)
    second = model(video, laser, properties, temperature=0.0)

    assert first.reconstruction.shape == video.shape
    assert first.property_prediction.shape == (2, 3)
    assert first.class_logits.shape == (2, 4)
    assert first.posterior_mean.shape == (2, 5)
    assert first.posterior_log_variance.shape == (2, 5)
    assert first.prior_mean.shape == (2, 5)
    assert first.prior_log_variance.shape == (2, 5)
    assert torch.equal(first.reconstruction, second.reconstruction)
    assert all(torch.isfinite(value).all() for value in vars(first).values())


def test_predictive_heads_do_not_receive_ground_truth_properties() -> None:
    model = JointConditionalVAE(_config()).eval()
    video, laser, properties = _inputs()

    first = model(video, laser, properties, temperature=0.0)
    second = model(video, laser, properties + 1000.0, temperature=0.0)

    assert torch.equal(first.property_prediction, second.property_prediction)
    assert torch.equal(first.class_logits, second.class_logits)
    assert not torch.equal(first.prior_mean, second.prior_mean)


def test_generate_always_has_sample_axis_and_supports_latents() -> None:
    model = JointConditionalVAE(_config()).eval()
    _, laser, properties = _inputs()

    deterministic = model.generate(laser, properties, num_samples=3, temperature=0.0)
    repeated_latent = model.generate(
        laser,
        properties,
        num_samples=2,
        latent=torch.zeros(2, 5),
    )
    explicit_latents = model.generate(
        laser,
        properties,
        latent=torch.randn(2, 4, 5),
    )

    assert deterministic.shape == (2, 3, 1, 4, 16, 16)
    assert torch.equal(deterministic[:, 0], deterministic[:, 1])
    assert repeated_latent.shape == (2, 2, 1, 4, 16, 16)
    assert torch.equal(repeated_latent[:, 0], repeated_latent[:, 1])
    assert explicit_latents.shape == (2, 4, 1, 4, 16, 16)

    fixed_latent = torch.zeros(2, 5)
    first_condition = model.generate(laser, properties, latent=fixed_latent)
    second_condition = model.generate(laser, properties + 1.0, latent=fixed_latent)
    assert not torch.equal(first_condition, second_condition)


def test_generate_does_not_use_video_encoder(monkeypatch: pytest.MonkeyPatch) -> None:
    model = JointConditionalVAE(_config()).eval()
    _, laser, properties = _inputs()

    def fail(*args: object, **kwargs: object) -> None:
        raise AssertionError("generation called the video encoder")

    monkeypatch.setattr(model.video_encoder, "forward", fail)

    assert model.generate(laser, properties, temperature=0.0).shape == (
        2,
        1,
        1,
        4,
        16,
        16,
    )


def test_joint_objective_reaches_every_branch_with_finite_gradients() -> None:
    model = JointConditionalVAE(_config())
    video, laser, properties = _inputs()
    classes = torch.tensor([0, 3])

    output = model(video, laser, properties)
    kl = _conditional_kl(output)
    loss = (
        nn.functional.smooth_l1_loss(output.reconstruction, video)
        + nn.functional.mse_loss(output.property_prediction, properties)
        + nn.functional.cross_entropy(output.class_logits, classes)
        + 0.1 * kl
    )
    loss.backward()

    branches = (
        model.video_encoder,
        model.laser_encoder,
        model.condition_encoder,
        model.property_head,
        model.classification_head,
        model.posterior,
        model.prior,
        model.decoder,
    )
    for branch in branches:
        gradients = [parameter.grad for parameter in branch.parameters()]
        assert gradients and all(gradient is not None for gradient in gradients)
        assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_conditional_gaussian_kl_is_zero_for_equal_distributions() -> None:
    class EqualDistributions:
        posterior_mean = torch.randn(3, 4)
        prior_mean = posterior_mean.clone()
        posterior_log_variance = torch.randn(3, 4)
        prior_log_variance = posterior_log_variance.clone()

    assert _conditional_kl(EqualDistributions()) == pytest.approx(0.0, abs=1e-7)


def test_shape_and_configuration_errors_are_explicit() -> None:
    config = _config()
    model = JointConditionalVAE(config)
    video, laser, properties = _inputs()

    with pytest.raises(ValueError, match="Expected video shape"):
        model(video[:, :, :-1], laser, properties)
    with pytest.raises(ValueError, match="laser_conditions shape"):
        model(video, laser[:, :1], properties)
    with pytest.raises(ValueError, match="material_properties shape"):
        model.generate(laser, properties[:, :2])
    with pytest.raises(ValueError, match="non-negative"):
        model.generate(laser, properties, temperature=-1.0)
    with pytest.raises(ValueError, match="cannot exceed"):
        JointCVAEConfig(
            video_shape=(1, 4, 8, 8),
            decoder_seed_shape=(5, 2, 2),
        )
