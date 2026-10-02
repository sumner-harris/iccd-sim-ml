from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from iccd_sim_ml.data import (  # noqa: E402
    DatasetManifest,
    MaterialSetDataset,
    SampleRecord,
    save_precomputed_sample,
)
from iccd_sim_ml.models import (  # noqa: E402
    DeepSetRegressor,
    MaterialSetRegressorConfig,
    SetTransformerRegressor,
    VideoEncoderConfig,
    VideoRegressor,
    VideoRegressorConfig,
)


def _manifest(tmp_path: Path) -> DatasetManifest:
    records = []
    for element_index, element in enumerate(("Cu", "Fe")):
        target = np.arange(7, dtype=np.float32) + element_index
        for simulation_index in range(4):
            sample_id = f"{element}-{simulation_index}"
            path = tmp_path / f"{sample_id}.npz"
            save_precomputed_sample(
                path,
                video=np.full((2, 8, 8), simulation_index + 1, dtype=np.float32),
                conditions=np.asarray((simulation_index + 1, simulation_index + 2)),
                regression_targets=target,
            )
            records.append(
                SampleRecord(
                    sample_id=sample_id,
                    path=path,
                    element=element,
                    simulation_id=sample_id,
                )
            )
    return DatasetManifest(tuple(records))


def _config(aggregator: str) -> MaterialSetRegressorConfig:
    return MaterialSetRegressorConfig(
        aggregator=aggregator,
        encoder=VideoEncoderConfig(
            input_channels=1,
            stage_channels=(4,),
            blocks_per_stage=1,
            embedding_dim=8,
        ),
        condition_hidden=(4,),
        condition_embedding_dim=4,
        token_hidden=(8,),
        token_dim=8,
        set_hidden=(8,),
        head_hidden=(8,),
        transformer_layers=1,
        transformer_heads=2,
        transformer_feedforward_dim=16,
        dropout=0.0,
    )


def test_material_set_config_round_trip_preserves_attention_pooling() -> None:
    config = MaterialSetRegressorConfig(
        aggregator="deep_set",
        baseline_pooling="target_attention",
    )
    assert MaterialSetRegressorConfig.from_dict(config.to_dict()) == config


def test_material_set_dataset_is_deterministic_and_never_mixes_elements(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    ids = tuple(record.sample_id for record in manifest.records)
    first = MaterialSetDataset(manifest, sample_ids=ids, set_size=3, sets_per_material=2, seed=11)
    second = MaterialSetDataset(manifest, sample_ids=ids, set_size=3, sets_per_material=2, seed=11)

    assert len(first) == 4
    assert [bag[2] for bag in first.bags] == [bag[2] for bag in second.bags]
    assert [tuple(record.sample_id for record in bag[1]) for bag in first.bags] == [
        tuple(record.sample_id for record in bag[1]) for bag in second.bags
    ]
    item = first[0]
    assert item["videos"].shape == (3, 1, 2, 8, 8)
    assert item["conditions"].shape == (3, 2)
    assert all(sample_id.startswith(item["element"]) for sample_id in item["sample_ids"])


@pytest.mark.parametrize(
    ("aggregator", "model_class"),
    (("deep_set", DeepSetRegressor), ("set_transformer", SetTransformerRegressor)),
)
def test_set_regressors_are_permutation_invariant_and_trainable(
    aggregator: str, model_class: type[torch.nn.Module]
) -> None:
    model = model_class(_config(aggregator))
    model.eval()
    videos = torch.randn(2, 3, 1, 2, 8, 8)
    conditions = torch.randn(2, 3, 2)
    mask = torch.ones(2, 3, dtype=torch.bool)
    permutation = torch.tensor([2, 0, 1])

    prediction = model(videos, conditions, mask)
    set_prediction, individual_prediction, returned_mask = model.forward_with_individual(
        videos, conditions, mask
    )
    permuted = model(videos[:, permutation], conditions[:, permutation], mask[:, permutation])
    assert prediction.shape == (2, 7)
    assert individual_prediction.shape == (2, 3, 7)
    assert torch.equal(returned_mask, mask)
    assert torch.allclose(set_prediction, prediction)
    assert torch.allclose(prediction, permuted, atol=1.0e-5, rtol=1.0e-5)

    model.train()
    loss = model(videos, conditions, mask).square().mean()
    loss.backward()
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


@pytest.mark.parametrize("baseline_pooling", ("mean", "target_attention"))
def test_zero_initialized_residual_exactly_reproduces_mean_regressor_prediction(
    baseline_pooling: str,
) -> None:
    encoder = VideoEncoderConfig(
        input_channels=1,
        stage_channels=(4,),
        blocks_per_stage=1,
        embedding_dim=8,
    )
    regressor = VideoRegressor(
        VideoRegressorConfig(
            encoder=encoder,
            condition_hidden=(4,),
            condition_embedding_dim=4,
            head_hidden=(8,),
            num_targets=7,
            dropout=0.0,
        )
    )
    set_model = DeepSetRegressor(
        MaterialSetRegressorConfig(
            aggregator="deep_set",
            baseline_pooling=baseline_pooling,
            encoder=encoder,
            condition_hidden=(4,),
            condition_embedding_dim=4,
            baseline_head_hidden=(8,),
            token_hidden=(8,),
            token_dim=8,
            set_hidden=(8,),
            head_hidden=(8,),
            transformer_heads=2,
            dropout=0.0,
        )
    )
    set_model.encoder.load_state_dict(regressor.encoder.state_dict())
    set_model.condition_encoder.load_state_dict(regressor.condition_encoder.state_dict())
    set_model.baseline_head.load_state_dict(regressor.head.state_dict())
    regressor.eval()
    set_model.eval()
    videos = torch.randn(2, 3, 1, 2, 8, 8)
    conditions = torch.randn(2, 3, 2)

    expected = (
        regressor(videos.flatten(0, 1), conditions.flatten(0, 1)).reshape(2, 3, 7).mean(dim=1)
    )
    actual = set_model(videos, conditions)

    assert torch.allclose(actual, expected, atol=1.0e-6, rtol=1.0e-6)

    if set_model.baseline_attention is not None:
        set_model.train()
        set_model(videos, conditions).square().mean().backward()
        assert set_model.baseline_attention.weight.grad is not None
        assert torch.isfinite(set_model.baseline_attention.weight.grad).all()

    set_model.freeze_pretrained_baseline()
    set_model.train()
    assert not set_model.encoder.training
    assert not set_model.condition_encoder.training
    assert not set_model.baseline_head.training
    assert set_model.token_encoder.training
    assert not any(
        parameter.requires_grad
        for module in (
            set_model.encoder,
            set_model.condition_encoder,
            set_model.baseline_head,
        )
        for parameter in module.parameters()
    )


def test_frozen_pretrained_batchnorm_keeps_weight_gradients() -> None:
    model = DeepSetRegressor(
        MaterialSetRegressorConfig(
            encoder=VideoEncoderConfig(
                input_channels=1,
                stage_channels=(2,),
                blocks_per_stage=1,
                embedding_dim=4,
            ),
            condition_hidden=(4,),
            condition_embedding_dim=4,
            baseline_head_hidden=(8,),
            token_hidden=(8,),
            token_dim=8,
            set_hidden=(8,),
            head_hidden=(8,),
            transformer_heads=2,
            dropout=0.0,
        )
    )

    model.freeze_pretrained_batchnorm_statistics()
    model.train()

    batchnorm_layers = [
        module
        for module in model.encoder.modules()
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
    ]
    assert batchnorm_layers
    assert all(not module.training for module in batchnorm_layers)
    assert model.encoder.training
    assert all(parameter.requires_grad for parameter in model.encoder.parameters())


def test_pretrained_baseline_can_be_unfrozen_after_warmup() -> None:
    model = DeepSetRegressor(_config("deep_set"))
    model.freeze_pretrained_baseline()
    model.train()
    assert not model.encoder.training
    assert not any(parameter.requires_grad for parameter in model.encoder.parameters())

    model.unfreeze_pretrained_baseline(freeze_batchnorm_statistics=True)
    model.train()

    assert model.encoder.training
    assert all(parameter.requires_grad for parameter in model.encoder.parameters())
    batchnorm_layers = [
        module
        for module in model.encoder.modules()
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
    ]
    assert batchnorm_layers
    assert all(not module.training for module in batchnorm_layers)


def test_target_specific_transformer_pooling_preserves_mean_baseline() -> None:
    config = MaterialSetRegressorConfig(
        aggregator="set_transformer",
        target_specific_pooling=True,
        encoder=VideoEncoderConfig(
            input_channels=1,
            stage_channels=(4,),
            blocks_per_stage=1,
            embedding_dim=8,
        ),
        condition_hidden=(4,),
        condition_embedding_dim=4,
        baseline_head_hidden=(8,),
        token_hidden=(8,),
        token_dim=8,
        head_hidden=(8,),
        transformer_layers=1,
        transformer_heads=2,
        transformer_feedforward_dim=16,
        dropout=0.0,
    )
    model = SetTransformerRegressor(config).eval()
    videos = torch.randn(2, 3, 1, 2, 8, 8)
    conditions = torch.randn(2, 3, 2)
    permutation = torch.tensor([2, 0, 1])

    prediction, individual, mask = model.forward_with_individual(videos, conditions)
    permuted = model(videos[:, permutation], conditions[:, permutation])

    assert model.pooling_seed is not None
    assert model.pooling_seed.shape == (1, 7, 8)
    assert prediction.shape == (2, 7)
    assert torch.allclose(prediction, individual.mean(dim=1), atol=1.0e-6, rtol=1.0e-6)
    assert torch.all(mask)
    assert torch.allclose(prediction, permuted, atol=1.0e-5, rtol=1.0e-5)


def test_target_specific_pooling_rejects_deep_sets() -> None:
    with pytest.raises(ValueError, match="requires aggregator"):
        MaterialSetRegressorConfig(aggregator="deep_set", target_specific_pooling=True)
