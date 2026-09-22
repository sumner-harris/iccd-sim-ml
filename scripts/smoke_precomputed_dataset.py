"""Smoke-test precomputed-video datasets and PyTorch DataLoader on one NPZ."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from iccd_sim_ml.data import (
    DatasetManifest,
    JointPrecomputedVideoDataset,
    PrecomputedVideoDataset,
    SampleRecord,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sample", type=Path)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    sample_path = args.sample.expanduser().resolve()
    with np.load(sample_path, allow_pickle=False) as product:
        shape = tuple(int(value) for value in product["video"].shape)
        times_s = np.asarray(product["times_s"], dtype=np.float64)
    manifest = DatasetManifest(
        (
            SampleRecord(
                sample_id="Cu_6__Cu_3_68",
                path=sample_path,
                element="Cu",
                simulation_id="Cu_6::Cu_3_68",
            ),
        )
    )

    regression = PrecomputedVideoDataset(manifest, task="regression")
    first = regression[0]
    second = regression[0]
    repeated_load_equal = all(
        torch.equal(first[key], second[key]) for key in ("video", "features", "target")
    )
    regression_batch = next(
        iter(DataLoader(regression, batch_size=1, shuffle=False, num_workers=args.workers))
    )

    joint = JointPrecomputedVideoDataset(
        manifest,
        expected_video_shape=(1, *shape),
        expected_times_s=times_s,
        class_to_index={"Cu": 0},
    )
    joint_batch = next(
        iter(DataLoader(joint, batch_size=1, shuffle=False, num_workers=args.workers))
    )
    result = {
        "workers": args.workers,
        "repeated_load_bitwise_equal": repeated_load_equal,
        "regression_batch": {
            key: list(value.shape)
            for key, value in regression_batch.items()
            if isinstance(value, torch.Tensor)
        },
        "joint_batch": {
            key: list(value.shape)
            for key, value in joint_batch.items()
            if isinstance(value, torch.Tensor)
        },
        "sample_id": regression_batch["sample_id"][0],
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
