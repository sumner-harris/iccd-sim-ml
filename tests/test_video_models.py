from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from iccd_sim_ml.models import (  # noqa: E402
    VideoClassifier,
    VideoClassifierConfig,
    VideoEncoderConfig,
    VideoRegressor,
    VideoRegressorConfig,
)


def _encoder_config() -> VideoEncoderConfig:
    return VideoEncoderConfig(
        input_channels=1,
        stage_channels=(4, 8),
        blocks_per_stage=1,
        embedding_dim=12,
    )


def test_regressor_forward_and_backward_shapes() -> None:
    config = VideoRegressorConfig(
        encoder=_encoder_config(),
        condition_dim=2,
        condition_hidden=(4,),
        condition_embedding_dim=4,
        head_hidden=(8,),
        num_targets=7,
    )
    model = VideoRegressor(config)
    video = torch.randn(2, 1, 3, 16, 16)
    conditions = torch.randn(2, 2)

    output = model(video, conditions)
    output.square().mean().backward()

    assert output.shape == (2, 7)
    assert all(parameter.grad is not None for parameter in model.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_regressor_requires_configured_conditions() -> None:
    model = VideoRegressor(
        VideoRegressorConfig(
            encoder=_encoder_config(),
            condition_dim=2,
            condition_embedding_dim=4,
            num_targets=3,
        )
    )

    with pytest.raises(ValueError, match="requires a conditions"):
        model(torch.randn(1, 1, 2, 8, 8))


def test_classifier_returns_logits_without_softmax() -> None:
    model = VideoClassifier(
        VideoClassifierConfig(
            encoder=_encoder_config(),
            condition_dim=0,
            head_hidden=(6,),
            num_classes=5,
        )
    )

    logits = model(torch.randn(3, 1, 4, 16, 16))

    assert logits.shape == (3, 5)
    assert not torch.allclose(logits.sum(dim=1), torch.ones(3))


def test_encoder_rejects_wrong_layout() -> None:
    model = VideoRegressor(
        VideoRegressorConfig(encoder=_encoder_config(), condition_dim=0, num_targets=2)
    )

    with pytest.raises(ValueError, match="B,C,T,H,W"):
        model(torch.randn(2, 3, 16, 16))
