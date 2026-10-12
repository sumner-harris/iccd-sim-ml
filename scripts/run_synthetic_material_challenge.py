#!/usr/bin/env python3
"""Generate a two-cohort synthetic-material challenge and evaluate frozen models.

This is an evaluation/diagnostic experiment, not independent physical
validation.  The conditional VAE creates plume videos, while all inverse
models remain frozen.  No generated sample is used for training.
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
    SampleRecord,
    ScalerBundle,
    SplitManifest,
    save_precomputed_sample,
)
from iccd_sim_ml.models import (
    JointConditionalVAE,
    JointCVAEConfig,
    MaterialSetRegressorConfig,
    SetTransformerRegressor,
    VideoRegressor,
    VideoRegressorConfig,
)
from iccd_sim_ml.pipeline.cache import CONDITION_NAMES, TARGET_NAMES
from iccd_sim_ml.pipeline.reports import plot_regression_parity, save_metrics_json
from iccd_sim_ml.training import load_checkpoint, regression_metrics


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cvae-checkpoint", type=Path, required=True)
    parser.add_argument("--regressor-checkpoint", type=Path, required=True)
    parser.add_argument("--set-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--materials", type=int, default=30)
    parser.add_argument("--manifold-materials", type=int, default=15)
    parser.add_argument("--sequences", type=int, default=500)
    parser.add_argument("--set-size", type=int, default=16)
    parser.add_argument("--generation-batch-size", type=int, default=2)
    parser.add_argument("--evaluation-batch-size", type=int, default=32)
    parser.add_argument("--set-batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20261012)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _latin_hypercube(rng: np.random.Generator, rows: int, dimensions: int) -> np.ndarray:
    values = np.empty((rows, dimensions), dtype=np.float64)
    for column in range(dimensions):
        strata = (np.arange(rows, dtype=np.float64) + rng.random(rows)) / rows
        values[:, column] = strata[rng.permutation(rows)]
    return values


def _training_domain(
    manifest: DatasetManifest, split: SplitManifest
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    by_element: dict[str, list[np.ndarray]] = defaultdict(list)
    conditions: list[np.ndarray] = []
    for record in manifest.select(split.train):
        with np.load(manifest.resolve_path(record), allow_pickle=False) as sample:
            by_element[record.element].append(
                np.asarray(sample["regression_targets"], dtype=np.float64).reshape(-1)
            )
            conditions.append(np.asarray(sample["conditions"], dtype=np.float64).reshape(-1))
    elements = tuple(sorted(by_element))
    properties: list[np.ndarray] = []
    for element in elements:
        rows = np.stack(by_element[element])
        if not np.allclose(rows, rows[0], rtol=1.0e-6, atol=1.0e-6):
            raise ValueError(f"Training targets are not constant for element {element}")
        properties.append(rows[0])
    condition_array = np.stack(conditions)
    return (
        elements,
        np.stack(properties),
        condition_array.min(axis=0),
        condition_array.max(axis=0),
        condition_array,
    )


def _synthetic_properties(
    real: np.ndarray,
    scaler: Any,
    *,
    manifold_count: int,
    total: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not 0 < manifold_count < total:
        raise ValueError("manifold-materials must be between zero and materials")
    scaled_real = scaler.transform(real)
    anchors = rng.choice(len(real), size=manifold_count, replace=False)
    manifold: list[np.ndarray] = []
    for anchor in anchors:
        distances = np.sum((scaled_real - scaled_real[anchor]) ** 2, axis=1)
        neighbours = np.argsort(distances)[1 : min(7, len(real))]
        chosen = rng.choice(neighbours, size=2, replace=False)
        weights = rng.dirichlet(np.asarray([4.0, 1.5, 1.5]))
        manifold.append(
            weights[0] * real[anchor]
            + weights[1] * real[int(chosen[0])]
            + weights[2] * real[int(chosen[1])]
        )
    uniform_count = total - manifold_count
    unit = _latin_hypercube(rng, uniform_count, real.shape[1])
    low, high = real.min(axis=0), real.max(axis=0)
    uniform = low + unit * (high - low)
    synthetic = np.concatenate((np.stack(manifold), uniform), axis=0)
    cohorts = np.asarray(
        ["manifold_aware"] * manifold_count + ["independent_uniform"] * uniform_count
    )

    scaled_synthetic = scaler.transform(synthetic)
    nearest = np.sqrt(
        np.min(
            np.sum((scaled_synthetic[:, None, :] - scaled_real[None, :, :]) ** 2, axis=2),
            axis=1,
        )
    )
    covariance = np.cov(scaled_real, rowvar=False)
    inverse_covariance = np.linalg.pinv(covariance + 1.0e-5 * np.eye(covariance.shape[0]))
    center = scaled_real.mean(axis=0)
    centered = scaled_synthetic - center
    mahalanobis = np.sqrt(np.einsum("ni,ij,nj->n", centered, inverse_covariance, centered))
    return synthetic, cohorts, nearest, mahalanobis


def _sequence_counts(materials: int, sequences: int) -> np.ndarray:
    if sequences < materials:
        raise ValueError("sequences must be at least materials")
    counts = np.full(materials, sequences // materials, dtype=np.int64)
    counts[: sequences % materials] += 1
    return counts


def _cohort_metrics(
    truth: np.ndarray,
    predictions: dict[str, np.ndarray],
    cohorts: np.ndarray,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for cohort in ("all", "manifold_aware", "independent_uniform"):
        mask = np.ones(len(truth), dtype=bool) if cohort == "all" else cohorts == cohort
        result[cohort] = {
            name: regression_metrics(truth[mask], estimate[mask], target_names=TARGET_NAMES)
            for name, estimate in predictions.items()
        }
    return result


def _load_models(args: argparse.Namespace, device: torch.device) -> tuple[Any, ...]:
    cvae_payload = load_checkpoint(args.cvae_checkpoint, map_location="cpu")
    reg_payload = load_checkpoint(args.regressor_checkpoint, map_location="cpu")
    set_payload = load_checkpoint(args.set_checkpoint, map_location="cpu")
    if (
        cvae_payload["split"] != reg_payload["split"]
        or set_payload["split"] != reg_payload["split"]
    ):
        raise ValueError("All checkpoints must use the same known-material split")
    cvae = JointConditionalVAE(JointCVAEConfig.from_dict(cvae_payload["config"])).to(device)
    cvae.load_state_dict(cvae_payload["model_state"])
    regressor = VideoRegressor(VideoRegressorConfig.from_dict(reg_payload["config"])).to(device)
    regressor.load_state_dict(reg_payload["model_state"])
    set_model = SetTransformerRegressor(
        MaterialSetRegressorConfig.from_dict(set_payload["config"])
    ).to(device)
    set_model.load_state_dict(set_payload["model_state"])
    for model in (cvae, regressor, set_model):
        model.eval()
    return (
        cvae,
        regressor,
        set_model,
        cvae_payload,
        reg_payload,
        set_payload,
    )


def _generate(
    *,
    cvae: JointConditionalVAE,
    properties: np.ndarray,
    conditions: np.ndarray,
    material_index: np.ndarray,
    cvae_scalers: ScalerBundle,
    destination: Path,
    times_s: np.ndarray,
    cohorts: np.ndarray,
    nearest: np.ndarray,
    mahalanobis: np.ndarray,
    batch_size: int,
    temperature: float,
    device: torch.device,
) -> tuple[DatasetManifest, np.ndarray]:
    if cvae_scalers.features is None or cvae_scalers.targets is None or cvae_scalers.video is None:
        raise ValueError("cVAE checkpoint is missing required scalers")
    records: list[SampleRecord] = []
    cvae_predictions: list[np.ndarray] = []
    sample_dir = destination / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(conditions), batch_size):
        stop = min(start + batch_size, len(conditions))
        indices = material_index[start:stop]
        condition_scaled = cvae_scalers.features.transform(conditions[start:stop])
        property_scaled = cvae_scalers.targets.transform(properties[indices])
        laser_tensor = torch.as_tensor(condition_scaled, dtype=torch.float32, device=device)
        property_tensor = torch.as_tensor(property_scaled, dtype=torch.float32, device=device)
        with torch.inference_mode():
            generated_scaled = cvae.generate(
                laser_tensor,
                property_tensor,
                num_samples=1,
                temperature=temperature,
            )[:, 0]
            inverse_prediction = cvae.regress(generated_scaled, laser_tensor).cpu().numpy()
        physical = cvae_scalers.video.inverse_transform(generated_scaled.cpu().numpy())
        physical = np.maximum(physical, 0.0).astype(np.float32)
        cvae_predictions.append(inverse_prediction)
        for local_index, sequence_index in enumerate(range(start, stop)):
            material_number = int(indices[local_index])
            material_id = f"SYN{material_number + 1:03d}"
            sample_id = f"{material_id}-sequence-{sequence_index:04d}"
            path = sample_dir / f"{sample_id}.npz"
            save_precomputed_sample(
                path,
                video=physical[local_index],
                conditions=conditions[sequence_index],
                regression_targets=properties[material_number],
                times_s=times_s,
            )
            records.append(
                SampleRecord(
                    sample_id=sample_id,
                    path=path,
                    element=material_id,
                    simulation_id=sample_id,
                    family_id=material_id,
                    metadata={
                        "synthetic": True,
                        "cohort": str(cohorts[material_number]),
                        "nearest_real_distance_scaled": float(nearest[material_number]),
                        "mahalanobis_distance_scaled": float(mahalanobis[material_number]),
                    },
                )
            )
        print(f"generated {stop}/{len(conditions)} sequences", flush=True)
    manifest = DatasetManifest(tuple(records))
    manifest_path = manifest.save(destination / "manifest.json", relative_paths=True)
    return DatasetManifest.load(manifest_path), np.concatenate(cvae_predictions)


def _single_model_predictions(
    model: VideoRegressor,
    dataset: PrecomputedVideoDataset,
    *,
    batch_size: int,
    workers: int,
    device: torch.device,
    scalers: ScalerBundle,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )
    targets: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    elements: list[str] = []
    with torch.inference_mode():
        for batch in loader:
            targets.append(batch["target"].numpy())
            predictions.append(
                model(batch["video"].to(device), batch["features"].to(device)).cpu().numpy()
            )
            elements.extend(str(value) for value in batch["element"])
    truth, estimate = np.concatenate(targets), np.concatenate(predictions)
    if scalers.targets is not None:
        truth = scalers.targets.inverse_transform(truth)
        estimate = scalers.targets.inverse_transform(estimate)
    return truth, estimate, elements


def _material_means(
    truth: np.ndarray, estimate: np.ndarray, elements: list[str]
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
    indices: dict[str, list[int]] = defaultdict(list)
    for index, element in enumerate(elements):
        indices[element].append(index)
    names = tuple(sorted(indices))
    return (
        names,
        np.stack([truth[indices[name]].mean(axis=0) for name in names]),
        np.stack([estimate[indices[name]].mean(axis=0) for name in names]),
    )


def _set_predictions(
    model: SetTransformerRegressor,
    dataset: MaterialSetDataset,
    *,
    batch_size: int,
    workers: int,
    device: torch.device,
    scalers: ScalerBundle,
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray, np.ndarray]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )
    elements: list[str] = []
    targets: list[np.ndarray] = []
    means: list[np.ndarray] = []
    learned: list[np.ndarray] = []
    with torch.inference_mode():
        for batch in loader:
            prediction, individual, mask = model.forward_with_individual(
                batch["videos"].to(device),
                batch["conditions"].to(device),
                batch["set_mask"].to(device),
            )
            weights = mask.unsqueeze(-1).to(individual.dtype)
            baseline = (individual * weights).sum(dim=1) / weights.sum(dim=1)
            elements.extend(str(value) for value in batch["element"])
            targets.append(batch["target"].numpy())
            means.append(baseline.cpu().numpy())
            learned.append(prediction.cpu().numpy())
    truth, baseline, prediction = (
        np.concatenate(targets),
        np.concatenate(means),
        np.concatenate(learned),
    )
    if scalers.targets is not None:
        truth = scalers.targets.inverse_transform(truth)
        baseline = scalers.targets.inverse_transform(baseline)
        prediction = scalers.targets.inverse_transform(prediction)
    order = np.argsort(np.asarray(elements))
    return (
        tuple(np.asarray(elements)[order]),
        truth[order],
        baseline[order],
        prediction[order],
    )


def _summary_plot(metrics: dict[str, Any], path: Path) -> None:
    cohorts = ("all", "manifold_aware", "independent_uniform")
    models = ("single_material_mean", "cvae_head_material_mean", "mean_k16", "set_transformer_k16")
    labels = (
        "single regressor\nmean",
        "cVAE inverse\nhead mean",
        "mean of K=16",
        "Set Transformer\nK=16",
    )
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    for axis, cohort in zip(axes, cohorts, strict=True):
        values = [metrics[cohort][model]["r2_macro"] for model in models]
        axis.bar(np.arange(len(models)), values, color=("#4c78a8", "#b279a2", "#72b7b2", "#f58518"))
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_xticks(np.arange(len(models)), labels, rotation=18, ha="right")
        axis.set_title(cohort.replace("_", " ").title())
        axis.set_ylabel(r"macro $R^2$")
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Frozen-model cycle-consistency on cVAE-generated materials")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _distance_plot(
    truth: np.ndarray,
    predictions: dict[str, np.ndarray],
    nearest: np.ndarray,
    cohorts: np.ndarray,
    target_scale: np.ndarray,
    path: Path,
) -> None:
    figure, axes = plt.subplots(
        1, len(predictions), figsize=(6 * len(predictions), 4.8), constrained_layout=True
    )
    axes = np.atleast_1d(axes)
    for axis, (name, prediction) in zip(axes, predictions.items(), strict=True):
        error = np.sqrt(np.mean(((prediction - truth) / target_scale) ** 2, axis=1))
        for cohort, color in (("manifold_aware", "#4c78a8"), ("independent_uniform", "#e45756")):
            mask = cohorts == cohort
            axis.scatter(nearest[mask], error[mask], label=cohort.replace("_", " "), color=color)
        axis.set(
            xlabel="nearest real material distance (standardized)",
            ylabel="normalized property RMSE",
            title=name.replace("_", " "),
        )
        axis.grid(alpha=0.25)
        axis.legend()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _montage(manifest: DatasetManifest, path: Path) -> None:
    first: dict[str, SampleRecord] = {}
    for record in manifest.records:
        first.setdefault(record.element, record)
    selected = [first[name] for name in sorted(first)[:6]]
    frame_indices = (0, 5, 10, 15)
    figure, axes = plt.subplots(
        len(selected), len(frame_indices), figsize=(11, 13), constrained_layout=True
    )
    for row, record in enumerate(selected):
        with np.load(manifest.resolve_path(record), allow_pickle=False) as sample:
            video = np.asarray(sample["video"])[0]
        display = np.log1p(video)
        high = float(np.percentile(display, 99.8))
        for column, frame in enumerate(frame_indices):
            axes[row, column].imshow(
                display[frame], origin="lower", cmap="inferno", vmin=0.0, vmax=max(high, 1.0e-12)
            )
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
            if row == 0:
                axes[row, column].set_title(f"frame {frame}")
            if column == 0:
                axes[row, column].set_ylabel(record.element)
    figure.suptitle("Example generated sequences (log1p photon radiance)")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = _arguments()
    if args.materials != 30 or args.sequences != 500:
        print("Note: using non-default synthetic challenge size", flush=True)
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    np_rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    manifest = DatasetManifest.load(args.manifest)
    cvae, regressor, set_model, cvae_payload, reg_payload, set_payload = _load_models(args, device)
    split = SplitManifest.from_dict(reg_payload["split"])
    cvae_scalers = ScalerBundle.from_dict(cvae_payload["scalers"])
    reg_scalers = ScalerBundle.from_dict(reg_payload["scalers"])
    set_scalers = ScalerBundle.from_dict(set_payload["scalers"])
    if cvae_scalers.targets is None or reg_scalers.targets is None:
        raise ValueError("Checkpoints require target scalers")
    elements, real_properties, condition_low, condition_high, _ = _training_domain(manifest, split)
    properties, cohorts, nearest, mahalanobis = _synthetic_properties(
        real_properties,
        cvae_scalers.targets,
        manifold_count=args.manifold_materials,
        total=args.materials,
        rng=np_rng,
    )
    counts = _sequence_counts(args.materials, args.sequences)
    material_index = np.repeat(np.arange(args.materials), counts)
    condition_unit = _latin_hypercube(np_rng, args.sequences, len(CONDITION_NAMES))
    conditions = condition_low + condition_unit * (condition_high - condition_low)
    with np.load(
        manifest.resolve_path(manifest.select((split.train[0],))[0]), allow_pickle=False
    ) as sample:
        times_s = np.asarray(sample["times_s"], dtype=np.float64)

    synthetic_manifest, cvae_prediction_scaled = _generate(
        cvae=cvae,
        properties=properties,
        conditions=conditions,
        material_index=material_index,
        cvae_scalers=cvae_scalers,
        destination=output,
        times_s=times_s,
        cohorts=cohorts,
        nearest=nearest,
        mahalanobis=mahalanobis,
        batch_size=args.generation_batch_size,
        temperature=args.temperature,
        device=device,
    )
    cvae_sequence_predictions = cvae_scalers.targets.inverse_transform(cvae_prediction_scaled)

    single_dataset = PrecomputedVideoDataset(
        synthetic_manifest, task="regression", scalers=reg_scalers
    )
    sequence_truth, sequence_prediction, sequence_elements = _single_model_predictions(
        regressor,
        single_dataset,
        batch_size=args.evaluation_batch_size,
        workers=args.num_workers,
        device=device,
        scalers=reg_scalers,
    )
    material_names, material_truth, material_single = _material_means(
        sequence_truth, sequence_prediction, sequence_elements
    )
    _, _, material_cvae = _material_means(
        sequence_truth, cvae_sequence_predictions, sequence_elements
    )
    set_dataset = MaterialSetDataset(
        synthetic_manifest,
        sample_ids=tuple(record.sample_id for record in synthetic_manifest.records),
        set_size=args.set_size,
        sets_per_material=1,
        seed=args.seed,
        sampling="condition_farthest",
        scalers=set_scalers,
    )
    set_names, set_truth, mean_k, set_prediction = _set_predictions(
        set_model,
        set_dataset,
        batch_size=args.set_batch_size,
        workers=args.num_workers,
        device=device,
        scalers=set_scalers,
    )
    if material_names != set_names or not np.allclose(material_truth, set_truth):
        raise RuntimeError(
            "Single and set evaluations disagree on synthetic material order/targets"
        )

    predictions = {
        "single_material_mean": material_single,
        "cvae_head_material_mean": material_cvae,
        f"mean_k{args.set_size}": mean_k,
        f"set_transformer_k{args.set_size}": set_prediction,
    }
    metrics = _cohort_metrics(material_truth, predictions, cohorts)
    sequence_metrics = {
        "standard_regressor": regression_metrics(
            sequence_truth, sequence_prediction, target_names=TARGET_NAMES
        ),
        "cvae_inverse_head": regression_metrics(
            sequence_truth, cvae_sequence_predictions, target_names=TARGET_NAMES
        ),
    }
    material_rows = []
    for index, name in enumerate(material_names):
        material_rows.append(
            {
                "material_id": name,
                "cohort": str(cohorts[index]),
                "sequence_count": int(counts[index]),
                "properties": dict(zip(TARGET_NAMES, properties[index], strict=True)),
                "nearest_real_distance_scaled": float(nearest[index]),
                "mahalanobis_distance_scaled": float(mahalanobis[index]),
            }
        )
    summary = {
        "interpretation": "cycle-consistency diagnostic; not independent physical validation",
        "training_performed": False,
        "materials": args.materials,
        "sequences": args.sequences,
        "set_size": args.set_size,
        "temperature": args.temperature,
        "training_elements": list(elements),
        "condition_names": list(CONDITION_NAMES),
        "condition_bounds": {
            name: [float(low), float(high)]
            for name, low, high in zip(CONDITION_NAMES, condition_low, condition_high, strict=True)
        },
        "property_names": list(TARGET_NAMES),
        "sequence_level_metrics": sequence_metrics,
        "material_level_metrics": metrics,
        "synthetic_materials": material_rows,
    }
    save_metrics_json(output / "metrics.json", summary)
    np.savez_compressed(
        output / "predictions.npz",
        material_ids=np.asarray(material_names),
        cohorts=cohorts,
        targets=material_truth,
        single_material_mean=material_single,
        cvae_head_material_mean=material_cvae,
        mean_k=mean_k,
        set_transformer=set_prediction,
        nearest_real_distance_scaled=nearest,
        mahalanobis_distance_scaled=mahalanobis,
        sequence_targets=sequence_truth,
        sequence_regressor_predictions=sequence_prediction,
        sequence_cvae_head_predictions=cvae_sequence_predictions,
    )
    (output / "generation_config.json").write_text(
        json.dumps(vars(args), indent=2, default=str), encoding="utf-8"
    )
    _summary_plot(metrics, output / "model_comparison.png")
    _distance_plot(
        material_truth,
        {"single mean": material_single, "Set Transformer": set_prediction},
        nearest,
        cohorts,
        reg_scalers.targets.scale,
        output / "error_vs_domain_distance.png",
    )
    plot_regression_parity(
        material_truth,
        material_single,
        TARGET_NAMES,
        output / "single_regressor_material_mean_parity.png",
        title="Synthetic materials: single-regressor material means",
    )
    plot_regression_parity(
        material_truth,
        set_prediction,
        TARGET_NAMES,
        output / "set_transformer_parity.png",
        title=f"Synthetic materials: Set Transformer (K={args.set_size})",
    )
    _montage(synthetic_manifest, output / "generated_sequence_montage.png")
    print(json.dumps({"material_level_metrics": metrics}, indent=2), flush=True)


if __name__ == "__main__":
    main()
