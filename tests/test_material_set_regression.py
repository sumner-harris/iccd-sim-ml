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
    permuted = model(videos[:, permutation], conditions[:, permutation], mask[:, permutation])
    assert prediction.shape == (2, 7)
    assert torch.allclose(prediction, permuted, atol=1.0e-5, rtol=1.0e-5)

    model.train()
    loss = model(videos, conditions, mask).square().mean()
    loss.backward()
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
