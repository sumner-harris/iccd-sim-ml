from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from iccd_sim_ml.data import (  # noqa: E402
    ArrayStandardizer,
    DatasetManifest,
    JointPrecomputedVideoDataset,
    SampleRecord,
    SplitManifest,
    fit_train_scalers,
    save_precomputed_sample,
)
from iccd_sim_ml.models import (  # noqa: E402
    JointConditionalVAE,
    JointCVAEConfig,
    VideoEncoderConfig,
)
from iccd_sim_ml.training import (  # noqa: E402
    JointLossConfig,
    conditional_gaussian_kl,
    evaluate_joint_epoch,
    load_checkpoint,
    save_checkpoint,
    train_joint_one_epoch,
)


def test_joint_dataset_round_trip_has_explicit_inputs_and_targets(tmp_path: Path) -> None:
    path = tmp_path / "sample.npz"
    times_s = np.array([0.0, 1.0e-9, 3.0e-9], dtype=np.float64)
    save_precomputed_sample(
        path,
        video=np.ones((3, 4, 5), dtype=np.float32),
        conditions=[2.5e8, 1.0e-3],
        regression_targets=[2800.0, 8000.0],
        class_index=2,
        times_s=times_s,
    )
    manifest = DatasetManifest(
        (
            SampleRecord(
                sample_id="cu-run",
                path=path,
                element="Cu",
                simulation_id="run-1",
            ),
        )
    )
    dataset = JointPrecomputedVideoDataset(
        manifest,
        expected_video_shape=(1, 3, 4, 5),
        expected_times_s=times_s,
    )

    sample = dataset[0]

    assert sample["video"].shape == (1, 3, 4, 5)
    assert sample["laser_conditions"].shape == (2,)
    assert sample["material_properties"].shape == (2,)
    assert sample["class_index"].item() == 2
    assert sample["times_s"].dtype == torch.float64
    torch.testing.assert_close(sample["times_s"], torch.from_numpy(times_s))

    mismatched_time_dataset = JointPrecomputedVideoDataset(
        manifest,
        expected_video_shape=(1, 3, 4, 5),
        expected_times_s=times_s + np.array([0.0, 0.0, 1.0e-12]),
    )
    with pytest.raises(ValueError, match="exactly match"):
        mismatched_time_dataset[0]


def test_precomputed_sample_rejects_time_axis_mismatch(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="video has 3 frames"):
        save_precomputed_sample(
            tmp_path / "bad.npz",
            video=np.ones((3, 4, 5)),
            times_s=[0.0, 1.0],
        )


def test_conditional_gaussian_kl_matches_analytic_unit_variance_case() -> None:
    posterior_mean = torch.ones(2, 3)
    posterior_log_variance = torch.zeros(2, 3)
    prior_mean = torch.zeros(2, 3)
    prior_log_variance = torch.zeros(2, 3)

    value = conditional_gaussian_kl(
        posterior_mean,
        posterior_log_variance,
        prior_mean,
        prior_log_variance,
    )
    free_bits_value = conditional_gaussian_kl(
        prior_mean,
        prior_log_variance,
        prior_mean,
        prior_log_variance,
        free_bits=0.2,
    )

    assert value.item() == pytest.approx(1.5)
    assert free_bits_value.item() == pytest.approx(0.6)

    extreme_half_precision = conditional_gaussian_kl(
        torch.zeros(2, 3, dtype=torch.float16),
        torch.full((2, 3), 8.0, dtype=torch.float16),
        torch.zeros(2, 3, dtype=torch.float16),
        torch.full((2, 3), -12.0, dtype=torch.float16),
    )
    assert extreme_half_precision.dtype == torch.float32
    assert torch.isfinite(extreme_half_precision)


@dataclass
class _TinyOutput:
    reconstruction: torch.Tensor
    property_prediction: torch.Tensor
    class_logits: torch.Tensor
    posterior_mean: torch.Tensor
    posterior_log_variance: torch.Tensor
    prior_mean: torch.Tensor
    prior_log_variance: torch.Tensor


class _TinyJointModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.shared = nn.Linear(3, 6)
        self.property_head = nn.Linear(6, 2)
        self.class_head = nn.Linear(6, 3)
        self.posterior = nn.Linear(10, 4)
        self.prior = nn.Linear(4, 4)
        self.decoder = nn.Linear(6, 1)

    def forward(
        self,
        video: torch.Tensor,
        laser_conditions: torch.Tensor,
        material_properties: torch.Tensor,
        *,
        sample_posterior: bool,
    ) -> _TinyOutput:
        pooled = video.mean(dim=(2, 3, 4))
        representation = torch.tanh(self.shared(torch.cat((pooled, laser_conditions), dim=1)))
        posterior_parameters = self.posterior(
            torch.cat((representation, laser_conditions, material_properties), dim=1)
        )
        posterior_mean, posterior_log_variance = posterior_parameters.chunk(2, dim=1)
        prior_mean, prior_log_variance = self.prior(
            torch.cat((laser_conditions, material_properties), dim=1)
        ).chunk(2, dim=1)
        if sample_posterior:
            latent = posterior_mean + torch.randn_like(posterior_mean) * torch.exp(
                0.5 * posterior_log_variance
            )
        else:
            latent = posterior_mean
        decoded_level = self.decoder(
            torch.cat((latent, laser_conditions, material_properties), dim=1)
        )
        reconstruction = decoded_level[:, :, None, None, None].expand_as(video)
        return _TinyOutput(
            reconstruction=reconstruction,
            property_prediction=self.property_head(representation),
            class_logits=self.class_head(representation),
            posterior_mean=posterior_mean,
            posterior_log_variance=posterior_log_variance,
            prior_mean=prior_mean,
            prior_log_variance=prior_log_variance,
        )


def _joint_batches() -> list[dict[str, torch.Tensor]]:
    generator = torch.Generator().manual_seed(91)
    batches = []
    for batch_size in (3, 2):
        batches.append(
            {
                "video": torch.randn(batch_size, 1, 3, 4, 4, generator=generator),
                "laser_conditions": torch.randn(batch_size, 2, generator=generator),
                "material_properties": torch.randn(batch_size, 2, generator=generator),
                "class_index": torch.arange(batch_size) % 3,
            }
        )
    return batches


def test_joint_training_and_evaluation_smoke() -> None:
    model = _TinyJointModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    config = JointLossConfig(
        kl_weight=0.01,
        temporal_gradient_weight=0.1,
    )
    target_scaler = ArrayStandardizer(
        mean=np.array([2800.0, 8000.0]),
        scale=np.array([200.0, 1000.0]),
        count=5,
        names=("boiling_point", "critical_temperature"),
    )
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}

    training = train_joint_one_epoch(
        model,
        _joint_batches(),
        optimizer,
        device="cpu",
        loss_config=config,
        kl_warmup_weight=0.5,
        target_scaler=target_scaler,
        num_classes=3,
    )
    evaluation = evaluate_joint_epoch(
        model,
        _joint_batches(),
        device="cpu",
        loss_config=config,
        kl_warmup_weight=1.0,
        target_scaler=target_scaler,
        num_classes=3,
    )

    assert np.isfinite(training["loss"])
    assert np.isfinite(evaluation["loss"])
    assert training["samples"] == 5
    assert training["effective_kl_weight"] == pytest.approx(0.005)
    assert set(training["regression_metrics"]["per_target"]) == {
        "boiling_point",
        "critical_temperature",
    }
    assert len(training["classification_metrics"]["confusion_matrix"]) == 3
    assert any(not torch.equal(value, before[name]) for name, value in model.state_dict().items())
    assert all(
        np.isfinite(training[name])
        for name in (
            "reconstruction_loss",
            "regression_loss",
            "classification_loss",
            "kl_loss",
            "temporal_gradient_loss",
        )
    )


def test_joint_epoch_accepts_production_cvae_output() -> None:
    config = JointCVAEConfig(
        video_shape=(1, 2, 8, 8),
        encoder=VideoEncoderConfig(
            input_channels=1,
            stage_channels=(4,),
            blocks_per_stage=1,
            embedding_dim=6,
        ),
        laser_condition_names=("laser_power_wcm", "rspot"),
        property_names=("t_boil", "tcrit"),
        num_classes=3,
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
    model = JointConditionalVAE(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    batch = {
        "video": torch.randn(2, 1, 2, 8, 8),
        "laser_conditions": torch.randn(2, 2),
        "material_properties": torch.randn(2, 2),
        "class_index": torch.tensor([0, 2]),
    }

    result = train_joint_one_epoch(
        model,
        [batch],
        optimizer,
        device="cpu",
        target_names=config.property_names,
        num_classes=config.num_classes,
    )

    assert result["samples"] == 2
    assert np.isfinite(result["loss"])


def test_joint_pipeline_trains_checkpoints_and_runs_all_three_modes(tmp_path: Path) -> None:
    times_s = np.array([0.0, 2.0e-9], dtype=np.float64)
    records = []
    for index in range(4):
        path = tmp_path / f"sample-{index}.npz"
        save_precomputed_sample(
            path,
            video=np.full((2, 8, 8), 1.0 + index, dtype=np.float32),
            conditions=[2.0e8 + index * 1.0e7, 8.0e-4 + index * 1.0e-4],
            regression_targets=[2800.0 + 100.0 * index, 8000.0 + 200.0 * index],
            class_index=index % 2,
            times_s=times_s,
        )
        records.append(
            SampleRecord(
                sample_id=f"sample-{index}",
                path=path,
                element="Cu" if index % 2 == 0 else "Fe",
                simulation_id=f"simulation-{index}",
            )
        )
    manifest = DatasetManifest(tuple(records))
    split = SplitManifest(
        train=("sample-0", "sample-1"),
        validation=("sample-2",),
        test=("sample-3",),
        seed=13,
        ratios=(0.5, 0.25, 0.25),
        group_fields=("simulation_id",),
    )
    property_names = ("t_boil", "tcrit")
    scalers = fit_train_scalers(
        manifest,
        split,
        feature_names=("laser_power_wcm", "rspot"),
        target_names=property_names,
    )
    train_dataset = JointPrecomputedVideoDataset(
        manifest,
        sample_ids=split.train,
        scalers=scalers,
        expected_video_shape=(1, 2, 8, 8),
        expected_times_s=times_s,
    )
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=False)
    model_config = JointCVAEConfig(
        video_shape=(1, 2, 8, 8),
        encoder=VideoEncoderConfig(
            input_channels=1,
            stage_channels=(4,),
            blocks_per_stage=1,
            embedding_dim=6,
        ),
        property_names=property_names,
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
    model = JointConditionalVAE(model_config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    metrics = train_joint_one_epoch(
        model,
        train_loader,
        optimizer,
        device="cpu",
        loss_config=JointLossConfig(kl_weight=0.01),
        kl_warmup_weight=0.5,
        target_scaler=scalers.targets,
        target_names=property_names,
        num_classes=2,
    )
    checkpoint = save_checkpoint(
        tmp_path / "joint.pt",
        model,
        optimizer=optimizer,
        epoch=1,
        config=model_config,
        scalers=scalers,
        split=split,
        metrics=metrics,
    )
    restored = JointConditionalVAE(model_config)
    restored_optimizer = torch.optim.Adam(restored.parameters(), lr=1.0e-3)
    payload = load_checkpoint(
        checkpoint,
        model=restored,
        optimizer=restored_optimizer,
    )

    batch = next(iter(train_loader))
    model.eval()
    restored.eval()
    with torch.no_grad():
        original_properties, original_logits = model.predict(
            batch["video"], batch["laser_conditions"]
        )
        restored_properties = restored.regress(batch["video"], batch["laser_conditions"])
        restored_logits = restored.classify(batch["video"], batch["laser_conditions"])
        original_generation = model.generate(
            batch["laser_conditions"],
            batch["material_properties"],
            num_samples=2,
            temperature=0.0,
        )
        restored_generation = restored.generate(
            batch["laser_conditions"],
            batch["material_properties"],
            num_samples=2,
            temperature=0.0,
        )

    assert payload["epoch"] == 1
    assert np.isfinite(metrics["loss"])
    torch.testing.assert_close(restored_properties, original_properties)
    torch.testing.assert_close(restored_logits, original_logits)
    torch.testing.assert_close(restored_generation, original_generation)
    assert restored_generation.shape == (2, 2, 1, 2, 8, 8)
