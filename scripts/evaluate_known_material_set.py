#!/usr/bin/env python3
"""Compare a known-material Set Transformer with its single-video baseline.

The script performs evaluation only.  It reports three estimators on the
existing known-material split: the production regressor on one experiment,
the mean of that regressor over exactly the experiments in each set, and the
learned Set Transformer prediction.  Validation and the untouched test
partition are kept separate.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch
from torch.utils.data import DataLoader

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from iccd_sim_ml.data import (
    DatasetManifest,
    MaterialSetDataset,
    PrecomputedVideoDataset,
    ScalerBundle,
    SplitManifest,
)
from iccd_sim_ml.models import (
    MaterialSetRegressorConfig,
    SetTransformerRegressor,
    VideoRegressor,
    VideoRegressorConfig,
)
from iccd_sim_ml.pipeline.cache import TARGET_NAMES
from iccd_sim_ml.pipeline.reports import plot_regression_parity, save_metrics_json
from iccd_sim_ml.training import load_checkpoint, regression_metrics


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--regressor-checkpoint", type=Path, required=True)
    parser.add_argument("--set-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--set-size", type=int, default=16)
    parser.add_argument("--sets-per-material", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--single-batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _mean_by_element(
    targets: np.ndarray, predictions: np.ndarray, elements: list[str]
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
    rows: dict[str, list[int]] = defaultdict(list)
    for index, element in enumerate(elements):
        rows[element].append(index)
    names = tuple(sorted(rows))
    truth = np.stack([targets[rows[name]].mean(axis=0) for name in names])
    estimate = np.stack([predictions[rows[name]].mean(axis=0) for name in names])
    return names, truth, estimate


def _single_predictions(
    model: VideoRegressor,
    loader: DataLoader[Any],
    *,
    device: torch.device,
    scalers: ScalerBundle,
) -> dict[str, Any]:
    model.eval()
    truth_rows: list[np.ndarray] = []
    prediction_rows: list[np.ndarray] = []
    elements: list[str] = []
    sample_ids: list[str] = []
    with torch.inference_mode():
        for batch in loader:
            prediction = (
                model(batch["video"].to(device), batch["features"].to(device)).cpu().numpy()
            )
            truth_rows.append(batch["target"].numpy())
            prediction_rows.append(prediction)
            elements.extend(str(value) for value in batch["element"])
            sample_ids.extend(str(value) for value in batch["sample_id"])
    truth = np.concatenate(truth_rows)
    prediction = np.concatenate(prediction_rows)
    if scalers.targets is not None:
        truth = scalers.targets.inverse_transform(truth)
        prediction = scalers.targets.inverse_transform(prediction)
    material_names, material_truth, material_prediction = _mean_by_element(
        truth, prediction, elements
    )
    return {
        "sample_ids": tuple(sample_ids),
        "elements": tuple(elements),
        "targets": truth,
        "predictions": prediction,
        "material_names": material_names,
        "material_targets": material_truth,
        "material_predictions": material_prediction,
        "sample_metrics": regression_metrics(truth, prediction, target_names=TARGET_NAMES),
        "material_metrics": regression_metrics(
            material_truth, material_prediction, target_names=TARGET_NAMES
        ),
    }


def _set_predictions(
    model: SetTransformerRegressor,
    loader: DataLoader[Any],
    *,
    device: torch.device,
    scalers: ScalerBundle,
) -> dict[str, Any]:
    model.eval()
    truth_rows: list[np.ndarray] = []
    baseline_rows: list[np.ndarray] = []
    set_rows: list[np.ndarray] = []
    elements: list[str] = []
    set_ids: list[str] = []
    sample_ids: list[tuple[str, ...]] = []
    with torch.inference_mode():
        for batch in loader:
            learned, individual, mask = model.forward_with_individual(
                batch["videos"].to(device),
                batch["conditions"].to(device),
                batch["set_mask"].to(device),
            )
            weights = mask.unsqueeze(-1).to(individual.dtype)
            baseline = (individual * weights).sum(dim=1) / weights.sum(dim=1)
            truth_rows.append(batch["target"].numpy())
            baseline_rows.append(baseline.cpu().numpy())
            set_rows.append(learned.cpu().numpy())
            elements.extend(str(value) for value in batch["element"])
            set_ids.extend(str(value) for value in batch["set_id"])
            batch_sample_ids = batch["sample_ids"]
            for batch_index in range(len(batch["element"])):
                sample_ids.append(
                    tuple(
                        str(batch_sample_ids[k][batch_index]) for k in range(len(batch_sample_ids))
                    )
                )
    truth = np.concatenate(truth_rows)
    baseline = np.concatenate(baseline_rows)
    learned = np.concatenate(set_rows)
    if scalers.targets is not None:
        truth = scalers.targets.inverse_transform(truth)
        baseline = scalers.targets.inverse_transform(baseline)
        learned = scalers.targets.inverse_transform(learned)
    names, material_truth, material_baseline = _mean_by_element(truth, baseline, elements)
    learned_names, _, material_learned = _mean_by_element(truth, learned, elements)
    if names != learned_names:
        raise RuntimeError("Material aggregation order changed between estimators")
    return {
        "elements": tuple(elements),
        "set_ids": tuple(set_ids),
        "sample_ids": tuple(sample_ids),
        "targets": truth,
        "baseline_predictions": baseline,
        "set_predictions": learned,
        "material_names": names,
        "material_targets": material_truth,
        "material_baseline_predictions": material_baseline,
        "material_set_predictions": material_learned,
        "baseline_bag_metrics": regression_metrics(truth, baseline, target_names=TARGET_NAMES),
        "set_bag_metrics": regression_metrics(truth, learned, target_names=TARGET_NAMES),
        "baseline_material_metrics": regression_metrics(
            material_truth, material_baseline, target_names=TARGET_NAMES
        ),
        "set_material_metrics": regression_metrics(
            material_truth, material_learned, target_names=TARGET_NAMES
        ),
    }


def _comparison_plot(results: dict[str, dict[str, Any]], path: Path) -> None:
    estimators = ("single video", "mean of K videos", "Set Transformer")
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    for axis, partition in zip(axes, ("validation", "test"), strict=True):
        value = results[partition]
        metrics = (
            value["single"]["sample_metrics"],
            value["sets"]["baseline_bag_metrics"],
            value["sets"]["set_bag_metrics"],
        )
        macro = [item["r2_macro"] for item in metrics]
        bars = axis.bar(np.arange(3), macro, color=("#4c78a8", "#72b7b2", "#f58518"))
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_xticks(np.arange(3), estimators, rotation=15, ha="right")
        axis.set_ylabel(r"macro $R^2$")
        axis.set_title(partition.capitalize())
        axis.grid(axis="y", alpha=0.25)
        for bar, score in zip(bars, macro, strict=True):
            axis.text(
                bar.get_x() + bar.get_width() / 2, score, f"{score:.3f}", ha="center", va="bottom"
            )
    figure.suptitle("Known-material property regression")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _serializable(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "single": {
            "samples": len(result["single"]["sample_ids"]),
            "sample_metrics": result["single"]["sample_metrics"],
            "material_metrics": result["single"]["material_metrics"],
        },
        "sets": {
            "bags": len(result["sets"]["set_ids"]),
            "materials": len(result["sets"]["material_names"]),
            "baseline_bag_metrics": result["sets"]["baseline_bag_metrics"],
            "set_bag_metrics": result["sets"]["set_bag_metrics"],
            "baseline_material_metrics": result["sets"]["baseline_material_metrics"],
            "set_material_metrics": result["sets"]["set_material_metrics"],
        },
    }


def main() -> None:
    args = _arguments()
    if args.set_size < 1 or args.sets_per_material < 1:
        raise ValueError("set-size and sets-per-material must be positive")
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    manifest = DatasetManifest.load(args.manifest)

    reg_payload = load_checkpoint(args.regressor_checkpoint, map_location="cpu")
    set_payload = load_checkpoint(args.set_checkpoint, map_location="cpu")
    if reg_payload["split"] != set_payload["split"]:
        raise ValueError("Regressor and set checkpoints do not use the same split")
    if reg_payload["scalers"] != set_payload["scalers"]:
        raise ValueError("Regressor and set checkpoints do not use the same scalers")
    split = SplitManifest.from_dict(reg_payload["split"])
    scalers = ScalerBundle.from_dict(reg_payload["scalers"])
    regressor = VideoRegressor(VideoRegressorConfig.from_dict(reg_payload["config"])).to(device)
    regressor.load_state_dict(reg_payload["model_state"])
    set_model = SetTransformerRegressor(
        MaterialSetRegressorConfig.from_dict(set_payload["config"])
    ).to(device)
    set_model.load_state_dict(set_payload["model_state"])

    results: dict[str, dict[str, Any]] = {}
    for offset, partition in enumerate(("validation", "test"), start=1):
        ids = split.ids(partition)
        single_data = PrecomputedVideoDataset(
            manifest, sample_ids=ids, task="regression", scalers=scalers
        )
        set_data = MaterialSetDataset(
            manifest,
            sample_ids=ids,
            set_size=args.set_size,
            sets_per_material=args.sets_per_material,
            seed=args.seed + offset,
            sampling="condition_farthest",
            scalers=scalers,
        )
        single_loader = DataLoader(
            single_data,
            batch_size=args.single_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        set_loader = DataLoader(
            set_data,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        results[partition] = {
            "single": _single_predictions(regressor, single_loader, device=device, scalers=scalers),
            "sets": _set_predictions(set_model, set_loader, device=device, scalers=scalers),
        }
        set_result = results[partition]["sets"]
        np.savez_compressed(
            output / f"{partition}_predictions.npz",
            set_targets=set_result["targets"],
            mean_k_predictions=set_result["baseline_predictions"],
            transformer_predictions=set_result["set_predictions"],
            material_targets=set_result["material_targets"],
            material_mean_k_predictions=set_result["material_baseline_predictions"],
            material_transformer_predictions=set_result["material_set_predictions"],
            material_names=np.asarray(set_result["material_names"]),
        )
        plot_regression_parity(
            set_result["material_targets"],
            set_result["material_set_predictions"],
            TARGET_NAMES,
            output / f"{partition}_set_transformer_parity.png",
            title=f"Known-material {partition}: Set Transformer (material means)",
        )

    summary = {
        "evaluation_only": True,
        "set_size": args.set_size,
        "sets_per_material": args.sets_per_material,
        "regressor_checkpoint": str(args.regressor_checkpoint),
        "set_checkpoint": str(args.set_checkpoint),
        "partitions": {name: _serializable(value) for name, value in results.items()},
    }
    save_metrics_json(output / "metrics.json", summary)
    _comparison_plot(results, output / "known_material_comparison.png")
    (output / "evaluation_config.json").write_text(
        json.dumps(vars(args), indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
