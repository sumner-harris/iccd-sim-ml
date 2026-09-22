from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from iccd_sim_ml.data import ArrayStandardizer, ScalerBundle, SplitManifest  # noqa: E402
from iccd_sim_ml.training import (  # noqa: E402
    checkpoint_metadata,
    evaluate_epoch,
    load_checkpoint,
    regression_metrics,
    save_checkpoint,
    train_one_epoch,
)


class TinyRegressor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 2)

    def forward(self, video: torch.Tensor, conditions: torch.Tensor | None) -> torch.Tensor:
        pooled = video.mean(dim=(2, 3, 4))
        assert conditions is not None
        return self.linear(torch.cat((pooled, conditions), dim=1))


def _batches() -> list[dict[str, torch.Tensor]]:
    generator = torch.Generator().manual_seed(5)
    return [
        {
            "video": torch.randn(3, 1, 2, 4, 4, generator=generator),
            "features": torch.randn(3, 3, generator=generator),
            "target": torch.randn(3, 2, generator=generator),
        },
        {
            "video": torch.randn(1, 1, 2, 4, 4, generator=generator),
            "features": torch.randn(1, 3, generator=generator),
            "target": torch.randn(1, 2, generator=generator),
        },
    ]


def test_one_epoch_training_and_evaluation_smoke() -> None:
    model = TinyRegressor()
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    criterion = nn.MSELoss()

    train_result = train_one_epoch(
        model,
        _batches(),
        optimizer,
        criterion,
        device="cpu",
        task="regression",
        target_names=("a", "b"),
    )
    validation_result = evaluate_epoch(
        model,
        _batches(),
        criterion,
        device="cpu",
        task="regression",
        target_names=("a", "b"),
    )

    assert np.isfinite(train_result["loss"])
    assert np.isfinite(validation_result["loss"])
    assert set(train_result["per_target"]) == {"a", "b"}


def test_checkpoint_round_trip_includes_reproducibility_metadata(tmp_path: Path) -> None:
    model = TinyRegressor()
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    scaler = ArrayStandardizer(
        mean=np.array([1.0, 2.0]),
        scale=np.array([3.0, 4.0]),
        count=5,
        names=("a", "b"),
    )
    scalers = ScalerBundle(
        video=None,
        features=scaler,
        targets=scaler,
        train_sample_ids=("train-1",),
    )
    split = SplitManifest(
        train=("train-1",),
        validation=("validation-1",),
        test=("test-1",),
        seed=17,
        ratios=(0.7, 0.15, 0.15),
        group_fields=("element",),
    )
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}

    path = save_checkpoint(
        tmp_path / "model.pt",
        model,
        optimizer=optimizer,
        epoch=4,
        config={"model": "tiny", "outputs": 2},
        scalers=scalers,
        split=split,
        metrics={"validation_loss": 0.25},
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    payload = load_checkpoint(path, model=model, optimizer=optimizer)
    metadata = checkpoint_metadata(path)

    assert payload["epoch"] == 4
    assert payload["split"]["seed"] == 17
    assert payload["scalers"]["train_sample_ids"] == ["train-1"]
    assert metadata["config"]["outputs"] == 2
    assert "model_state" not in metadata
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name])


def test_regression_metrics_do_not_flatten_unrelated_targets() -> None:
    truth = np.array([[0.0, 10.0], [1.0, 20.0], [2.0, 30.0]])
    prediction = np.array([[0.0, 12.0], [1.0, 22.0], [2.0, 32.0]])

    result = regression_metrics(truth, prediction, target_names=("exact", "biased"))

    assert result["per_target"]["exact"]["r2"] == pytest.approx(1.0)
    assert result["per_target"]["biased"]["mae"] == pytest.approx(2.0)
