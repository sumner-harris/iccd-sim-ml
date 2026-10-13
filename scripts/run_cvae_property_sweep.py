#!/usr/bin/env python3
"""Generate a posterior-anchored material-property sweep and regress it."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

from iccd_sim_ml.data import DatasetManifest, ScalerBundle
from iccd_sim_ml.models import (
    JointConditionalVAE,
    JointCVAEConfig,
    VideoRegressor,
    VideoRegressorConfig,
)
from iccd_sim_ml.pipeline.cache import CONDITION_NAMES, TARGET_NAMES
from iccd_sim_ml.pipeline.reports import save_metrics_json
from iccd_sim_ml.training import load_checkpoint


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cvae-checkpoint", type=Path, required=True)
    parser.add_argument("--regressor-checkpoint", type=Path, required=True)
    parser.add_argument("--reference-sample-id", required=True)
    parser.add_argument("--property", default="kappa_metal", choices=TARGET_NAMES)
    parser.add_argument("--relative-min", type=float, default=-0.10)
    parser.add_argument("--relative-max", type=float, default=0.10)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _canonical(video: np.ndarray) -> np.ndarray:
    value = np.asarray(video, dtype=np.float32)
    if value.ndim == 3:
        value = value[None]
    if value.ndim != 4:
        raise ValueError(f"Expected video shape (T,H,W) or (C,T,H,W), got {value.shape}")
    return value


def _load_models(args: argparse.Namespace, device: torch.device) -> tuple[Any, ...]:
    cvae_payload = load_checkpoint(args.cvae_checkpoint, map_location="cpu")
    regressor_payload = load_checkpoint(args.regressor_checkpoint, map_location="cpu")
    if cvae_payload["split"] != regressor_payload["split"]:
        raise ValueError("The cVAE and regressor checkpoints use different splits")
    cvae = JointConditionalVAE(JointCVAEConfig.from_dict(cvae_payload["config"])).to(device)
    cvae.load_state_dict(cvae_payload["model_state"])
    regressor = VideoRegressor(VideoRegressorConfig.from_dict(regressor_payload["config"])).to(
        device
    )
    regressor.load_state_dict(regressor_payload["model_state"])
    cvae.eval()
    regressor.eval()
    return cvae, regressor, cvae_payload, regressor_payload


def _montage(
    videos: np.ndarray,
    offsets: np.ndarray,
    times_s: np.ndarray,
    path: Path,
) -> None:
    frame_indices = np.linspace(0, videos.shape[2] - 1, 4).round().astype(int)
    log_video = np.log1p(videos[:, 0])
    high = max(float(np.percentile(log_video, 99.8)), 1.0e-12)
    figure, axes = plt.subplots(
        len(videos), len(frame_indices), figsize=(10, 2.3 * len(videos)), constrained_layout=True
    )
    for row, offset in enumerate(offsets):
        for column, frame_index in enumerate(frame_indices):
            axes[row, column].imshow(
                log_video[row, frame_index],
                origin="lower",
                cmap="inferno",
                vmin=0.0,
                vmax=high,
            )
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
            if row == 0:
                axes[row, column].set_title(f"{times_s[frame_index] * 1e6:.2f} µs")
            if column == 0:
                axes[row, column].set_ylabel(f"{offset:+.1f}%")
    figure.suptitle("cVAE Cu conductivity sweep (common posterior latent)")
    figure.savefig(path, dpi=170)
    plt.close(figure)


def _animation(
    videos: np.ndarray,
    offsets: np.ndarray,
    times_s: np.ndarray,
    path: Path,
) -> None:
    log_video = np.log1p(videos[:, 0])
    high = max(float(np.percentile(log_video, 99.8)), 1.0e-12)
    figure, axes = plt.subplots(2, 5, figsize=(13, 5.5), constrained_layout=True)
    images = []
    for index, axis in enumerate(axes.ravel()):
        images.append(
            axis.imshow(
                log_video[index, 0],
                origin="lower",
                cmap="inferno",
                vmin=0.0,
                vmax=high,
                animated=True,
            )
        )
        axis.set_title(f"κ {offsets[index]:+.1f}%")
        axis.set_xticks([])
        axis.set_yticks([])
    title = figure.suptitle(f"Cu counterfactuals — {times_s[0] * 1e6:.2f} µs")

    def update(frame_index: int) -> list[Any]:
        for index, image in enumerate(images):
            image.set_data(log_video[index, frame_index])
        title.set_text(f"Cu counterfactuals — {times_s[frame_index] * 1e6:.2f} µs")
        return [*images, title]

    animation = FuncAnimation(
        figure, update, frames=videos.shape[2], interval=450, blit=False, repeat=True
    )
    animation.save(path, writer=PillowWriter(fps=2.2))
    plt.close(figure)


def _prediction_plot(
    requested: np.ndarray,
    predictions: np.ndarray,
    target: np.ndarray,
    target_scale: np.ndarray,
    property_index: int,
    property_name: str,
    path: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    low = float(min(requested.min(), predictions[:, property_index].min()))
    high = float(max(requested.max(), predictions[:, property_index].max()))
    axes[0].plot([low, high], [low, high], "k--", label="ideal")
    axes[0].plot(requested, predictions[:, property_index], "o-", label="regressor")
    axes[0].axhline(target[property_index], color="gray", linestyle=":", label="standard Cu")
    axes[0].set(
        xlabel=f"requested {property_name}",
        ylabel=f"predicted {property_name}",
        title="Requested versus recovered conductivity",
    )
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    normalized_change = (predictions - predictions[len(predictions) // 2]) / target_scale
    offset_percent = 100.0 * (requested / target[property_index] - 1.0)
    for index, name in enumerate(TARGET_NAMES):
        axes[1].plot(offset_percent, normalized_change[:, index], marker="o", label=name)
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set(
        xlabel="requested conductivity change (%)",
        ylabel="prediction change (training standard deviations)",
        title="Regressor response and cross-property changes",
    )
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=7, ncol=2)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = _arguments()
    if args.steps < 2 or args.relative_min >= args.relative_max:
        raise ValueError("Sweep requires at least two ordered relative bounds")
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    manifest = DatasetManifest.load(args.manifest)
    record = manifest.by_id[args.reference_sample_id]
    with np.load(manifest.resolve_path(record), allow_pickle=False) as sample:
        reference_video = _canonical(np.asarray(sample["video"]))
        laser = np.asarray(sample["conditions"], dtype=np.float64).reshape(-1)
        properties = np.asarray(sample["regression_targets"], dtype=np.float64).reshape(-1)
        times_s = np.asarray(sample["times_s"], dtype=np.float64).reshape(-1)

    cvae, regressor, cvae_payload, regressor_payload = _load_models(args, device)
    cvae_scalers = ScalerBundle.from_dict(cvae_payload["scalers"])
    regressor_scalers = ScalerBundle.from_dict(regressor_payload["scalers"])
    required = (
        cvae_scalers.video,
        cvae_scalers.features,
        cvae_scalers.targets,
        regressor_scalers.video,
        regressor_scalers.features,
        regressor_scalers.targets,
    )
    if any(value is None for value in required):
        raise ValueError("Both checkpoints must contain video, feature, and target scalers")
    assert cvae_scalers.video is not None
    assert cvae_scalers.features is not None
    assert cvae_scalers.targets is not None
    assert regressor_scalers.video is not None
    assert regressor_scalers.features is not None
    assert regressor_scalers.targets is not None

    property_index = TARGET_NAMES.index(args.property)
    relative = np.linspace(args.relative_min, args.relative_max, args.steps)
    swept_properties = np.repeat(properties[None], args.steps, axis=0)
    swept_properties[:, property_index] = properties[property_index] * (1.0 + relative)
    swept_laser = np.repeat(laser[None], args.steps, axis=0)

    reference_video_scaled = cvae_scalers.video.transform(reference_video)[None]
    reference_laser_scaled = cvae_scalers.features.transform(laser[None])
    reference_properties_scaled = cvae_scalers.targets.transform(properties[None])
    with torch.inference_mode():
        posterior = cvae(
            torch.as_tensor(reference_video_scaled, dtype=torch.float32, device=device),
            torch.as_tensor(reference_laser_scaled, dtype=torch.float32, device=device),
            torch.as_tensor(reference_properties_scaled, dtype=torch.float32, device=device),
            sample_posterior=False,
        )
        common_latent = posterior.posterior_mean.expand(args.steps, -1)
        generated_scaled = cvae.generate(
            torch.as_tensor(
                cvae_scalers.features.transform(swept_laser),
                dtype=torch.float32,
                device=device,
            ),
            torch.as_tensor(
                cvae_scalers.targets.transform(swept_properties),
                dtype=torch.float32,
                device=device,
            ),
            latent=common_latent,
        )[:, 0]
    generated = np.maximum(
        cvae_scalers.video.inverse_transform(generated_scaled.cpu().numpy()), 0.0
    ).astype(np.float32)
    regressor_video = regressor_scalers.video.transform(generated).astype(np.float32)
    regressor_laser = regressor_scalers.features.transform(swept_laser).astype(np.float32)
    with torch.inference_mode():
        prediction_scaled = (
            regressor(
                torch.from_numpy(regressor_video).to(device),
                torch.from_numpy(regressor_laser).to(device),
            )
            .cpu()
            .numpy()
        )
    predictions = regressor_scalers.targets.inverse_transform(prediction_scaled)

    offsets_percent = relative * 100.0
    requested = swept_properties[:, property_index]
    coefficients = np.polyfit(requested, predictions[:, property_index], 1)
    correlation = float(np.corrcoef(requested, predictions[:, property_index])[0, 1])
    summary = {
        "reference_sample_id": args.reference_sample_id,
        "element": record.element,
        "laser_conditions": dict(zip(CONDITION_NAMES, laser.tolist(), strict=True)),
        "sweep_property": args.property,
        "base_value": float(properties[property_index]),
        "relative_offsets_percent": offsets_percent.tolist(),
        "requested_values": requested.tolist(),
        "predictions": [dict(zip(TARGET_NAMES, row.tolist(), strict=True)) for row in predictions],
        "conductivity_response": {
            "linear_slope_predicted_per_requested": float(coefficients[0]),
            "linear_intercept": float(coefficients[1]),
            "pearson_correlation": correlation,
            "predicted_range": [
                float(predictions[:, property_index].min()),
                float(predictions[:, property_index].max()),
            ],
            "requested_range": [float(requested.min()), float(requested.max())],
        },
        "method": (
            "The posterior mean from one real Cu sequence is held fixed; only the decoder's "
            "material-property condition changes. The standard regressor is frozen."
        ),
    }
    save_metrics_json(output / "metrics.json", summary)
    np.savez_compressed(
        output / "sweep_videos.npz",
        videos=generated,
        times_s=times_s,
        laser_conditions=swept_laser,
        requested_properties=swept_properties,
        predicted_properties=predictions,
        relative_offsets_percent=offsets_percent,
    )
    with (output / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["offset_percent", f"requested_{args.property}"]
            + [f"predicted_{name}" for name in TARGET_NAMES]
        )
        for offset, value, row in zip(offsets_percent, requested, predictions, strict=True):
            writer.writerow([float(offset), float(value), *row.tolist()])
    _montage(generated, offsets_percent, times_s, output / "video_montage.png")
    _animation(generated, offsets_percent, times_s, output / "conductivity_sweep.gif")
    _prediction_plot(
        requested,
        predictions,
        properties,
        regressor_scalers.targets.scale,
        property_index,
        args.property,
        output / "property_predictions.png",
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
