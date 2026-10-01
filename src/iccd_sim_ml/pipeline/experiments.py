"""Modular regression, classification, and joint-CVAE experiment runners."""

from __future__ import annotations

import copy
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from iccd_sim_ml.data import (
    DatasetManifest,
    JointPrecomputedVideoDataset,
    PrecomputedVideoDataset,
    ScalerBundle,
    SplitManifest,
    fit_train_scalers,
)
from iccd_sim_ml.models import (
    JointConditionalVAE,
    JointCVAEConfig,
    VideoClassifier,
    VideoClassifierConfig,
    VideoEncoderConfig,
    VideoRegressor,
    VideoRegressorConfig,
)
from iccd_sim_ml.training import (
    JointLossConfig,
    evaluate_epoch,
    evaluate_joint_epoch,
    regression_metrics,
    save_checkpoint,
    train_joint_one_epoch,
    train_one_epoch,
    video_generation_metrics,
)

from .cache import CONDITION_NAMES, TARGET_NAMES
from .reports import (
    plot_classification_report,
    plot_confusion_matrix,
    plot_generation_error_maps,
    plot_learning_curves,
    plot_regression_parity,
    save_metrics_json,
)

ModelChoice = Literal["regression", "classification", "joint_cvae"]


@dataclass(frozen=True)
class TrainingRunConfig:
    epochs: int = 2
    batch_size: int = 2
    learning_rate: float = 1.0e-3
    weight_decay: float = 0.0
    num_workers: int = 0
    seed: int = 42
    architecture: Literal["smoke", "standard"] = "smoke"
    device: str = "auto"
    early_stopping_patience: int | None = None
    early_stopping_min_delta: float = 0.0
    joint_classification_weight: float = 1.0

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.batch_size < 1 or self.num_workers < 0:
            raise ValueError("epochs/batch_size must be positive and num_workers non-negative")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("learning_rate must be positive and weight_decay non-negative")
        if self.early_stopping_patience is not None and self.early_stopping_patience < 1:
            raise ValueError("early_stopping_patience must be positive when enabled")
        if self.early_stopping_min_delta < 0.0:
            raise ValueError("early_stopping_min_delta must be non-negative")
        if (
            not np.isfinite(self.joint_classification_weight)
            or self.joint_classification_weight < 0.0
        ):
            raise ValueError("joint_classification_weight must be finite and non-negative")


def fit_experiment_scalers(manifest: DatasetManifest, split: SplitManifest) -> ScalerBundle:
    return fit_train_scalers(
        manifest,
        split,
        feature_names=CONDITION_NAMES,
        target_names=TARGET_NAMES,
    )


def _device(config: TrainingRunConfig) -> torch.device:
    if config.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def _seed(config: TrainingRunConfig) -> None:
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)


def _synchronize(device: torch.device) -> None:
    """Make accelerator timings include all queued kernels."""

    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _loader(dataset: Any, config: TrainingRunConfig, *, shuffle: bool) -> DataLoader:
    generator = torch.Generator().manual_seed(config.seed)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        generator=generator,
        persistent_workers=config.num_workers > 0,
    )


def _encoder(config: TrainingRunConfig) -> VideoEncoderConfig:
    if config.architecture == "smoke":
        return VideoEncoderConfig(
            input_channels=1,
            stage_channels=(4, 8),
            blocks_per_stage=1,
            embedding_dim=16,
        )
    return VideoEncoderConfig()


def _output_dir(root: str | Path, name: str) -> Path:
    path = Path(root).expanduser().resolve() / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def _collect_regression(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    scaler: ScalerBundle,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    targets: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            video = batch["video"].to(device)
            features = batch["features"].to(device)
            target = batch["target"].cpu().numpy()
            prediction = model(video, features).cpu().numpy()
            targets.append(target)
            predictions.append(prediction)
    truth = np.concatenate(targets)
    estimate = np.concatenate(predictions)
    if scaler.targets is not None:
        truth = scaler.targets.inverse_transform(truth)
        estimate = scaler.targets.inverse_transform(estimate)
    return truth, estimate


def run_regression_experiment(
    manifest: DatasetManifest,
    split: SplitManifest,
    scalers: ScalerBundle,
    output_root: str | Path,
    config: TrainingRunConfig,
) -> dict[str, Any]:
    _seed(config)
    device = _device(config)
    output = _output_dir(output_root, "regression")
    train_data = PrecomputedVideoDataset(
        manifest, sample_ids=split.train, task="regression", scalers=scalers
    )
    validation_data = PrecomputedVideoDataset(
        manifest, sample_ids=split.validation, task="regression", scalers=scalers
    )
    train_loader = _loader(train_data, config, shuffle=True)
    validation_loader = _loader(validation_data, config, shuffle=False)
    model_config = VideoRegressorConfig(
        encoder=_encoder(config),
        condition_dim=len(CONDITION_NAMES),
        condition_hidden=(8,) if config.architecture == "smoke" else (16, 16),
        condition_embedding_dim=8 if config.architecture == "smoke" else 16,
        head_hidden=(16,) if config.architecture == "smoke" else (256, 64),
        num_targets=len(TARGET_NAMES),
    )
    model = VideoRegressor(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    criterion = nn.SmoothL1Loss()
    history = {
        "epoch": [],
        "train_loss": [],
        "validation_loss": [],
        "train_seconds": [],
        "validation_seconds": [],
        "epoch_seconds": [],
    }
    final_validation: dict[str, Any] = {}
    best_validation_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    best_model_state: dict[str, Any] | None = None
    best_optimizer_state: dict[str, Any] | None = None
    stopped_early = False
    for epoch in range(config.epochs):
        _synchronize(device)
        epoch_started = time.perf_counter()
        training = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device=device,
            task="regression",
            target_names=TARGET_NAMES,
        )
        _synchronize(device)
        training_finished = time.perf_counter()
        final_validation = evaluate_epoch(
            model,
            validation_loader,
            criterion,
            device=device,
            task="regression",
            target_names=TARGET_NAMES,
        )
        _synchronize(device)
        validation_finished = time.perf_counter()
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(training["loss"])
        history["validation_loss"].append(final_validation["loss"])
        history["train_seconds"].append(training_finished - epoch_started)
        history["validation_seconds"].append(validation_finished - training_finished)
        history["epoch_seconds"].append(validation_finished - epoch_started)
        print(
            f"regression epoch {epoch + 1}/{config.epochs}: "
            f"train={history['train_seconds'][-1]:.2f}s, "
            f"validation={history['validation_seconds'][-1]:.2f}s, "
            f"total={history['epoch_seconds'][-1]:.2f}s",
            flush=True,
        )
        validation_loss = float(final_validation["loss"])
        if validation_loss < best_validation_loss - config.early_stopping_min_delta:
            best_validation_loss = validation_loss
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            if config.early_stopping_patience is not None:
                best_model_state = copy.deepcopy(model.state_dict())
                best_optimizer_state = copy.deepcopy(optimizer.state_dict())
        else:
            epochs_without_improvement += 1
        if (
            config.early_stopping_patience is not None
            and epochs_without_improvement >= config.early_stopping_patience
        ):
            stopped_early = True
            print(
                f"regression early stopping after epoch {epoch + 1}; "
                f"best epoch={best_epoch}, best validation loss={best_validation_loss:.6g}",
                flush=True,
            )
            break
    if best_model_state is not None and best_optimizer_state is not None:
        model.load_state_dict(best_model_state)
        optimizer.load_state_dict(best_optimizer_state)
    truth, estimate = _collect_regression(model, validation_loader, device, scalers)
    physical_metrics = regression_metrics(truth, estimate, target_names=TARGET_NAMES)
    plot_learning_curves(history, output / "learning_curves.png")
    plot_regression_parity(truth, estimate, TARGET_NAMES, output / "parity.png")
    summary = {
        "model": "VideoRegressor",
        "device": str(device),
        "run_config": asdict(config),
        "model_config": model_config.to_dict(),
        "history": history,
        "early_stopping": {
            "enabled": config.early_stopping_patience is not None,
            "patience": config.early_stopping_patience,
            "min_delta": config.early_stopping_min_delta,
            "maximum_epochs": config.epochs,
            "epochs_completed": len(history["epoch"]),
            "stopped_early": stopped_early,
            "best_epoch": best_epoch,
            "best_validation_loss": best_validation_loss,
            "restored_best_weights": best_model_state is not None,
        },
        "validation_metrics_physical_units": physical_metrics,
        "validation_samples": len(validation_data),
    }
    save_metrics_json(output / "metrics.json", summary)
    save_checkpoint(
        output / "checkpoint.pt",
        model,
        optimizer=optimizer,
        epoch=best_epoch if best_model_state is not None else len(history["epoch"]),
        config=model_config,
        scalers=scalers,
        split=split,
        metrics=summary,
    )
    return summary


def run_classification_experiment(
    manifest: DatasetManifest,
    split: SplitManifest,
    scalers: ScalerBundle,
    output_root: str | Path,
    config: TrainingRunConfig,
) -> dict[str, Any]:
    _seed(config)
    device = _device(config)
    output = _output_dir(output_root, "classification")
    class_names = tuple(sorted({record.element for record in manifest.records}))
    class_to_index = {name: index for index, name in enumerate(class_names)}
    train_data = PrecomputedVideoDataset(
        manifest,
        sample_ids=split.train,
        task="classification",
        scalers=scalers,
        class_to_index=class_to_index,
    )
    validation_data = PrecomputedVideoDataset(
        manifest,
        sample_ids=split.validation,
        task="classification",
        scalers=scalers,
        class_to_index=class_to_index,
    )
    train_loader = _loader(train_data, config, shuffle=True)
    validation_loader = _loader(validation_data, config, shuffle=False)
    model_config = VideoClassifierConfig(
        encoder=_encoder(config),
        condition_dim=len(CONDITION_NAMES),
        condition_hidden=(8,) if config.architecture == "smoke" else (16, 16),
        condition_embedding_dim=8 if config.architecture == "smoke" else 16,
        head_hidden=(16,) if config.architecture == "smoke" else (128, 32),
        num_classes=len(class_names),
    )
    model = VideoClassifier(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    criterion = nn.CrossEntropyLoss()
    history = {
        "epoch": [],
        "train_loss": [],
        "validation_loss": [],
        "train_accuracy": [],
        "validation_accuracy": [],
        "train_seconds": [],
        "validation_seconds": [],
        "epoch_seconds": [],
    }
    final_validation: dict[str, Any] = {}
    best_validation_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    best_model_state: dict[str, Any] | None = None
    best_optimizer_state: dict[str, Any] | None = None
    best_validation_metrics: dict[str, Any] | None = None
    stopped_early = False
    for epoch in range(config.epochs):
        _synchronize(device)
        epoch_started = time.perf_counter()
        training = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device=device,
            task="classification",
            num_classes=len(class_names),
        )
        _synchronize(device)
        training_finished = time.perf_counter()
        final_validation = evaluate_epoch(
            model,
            validation_loader,
            criterion,
            device=device,
            task="classification",
            num_classes=len(class_names),
        )
        _synchronize(device)
        validation_finished = time.perf_counter()
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(training["loss"])
        history["validation_loss"].append(final_validation["loss"])
        history["train_accuracy"].append(training["accuracy"])
        history["validation_accuracy"].append(final_validation["accuracy"])
        history["train_seconds"].append(training_finished - epoch_started)
        history["validation_seconds"].append(validation_finished - training_finished)
        history["epoch_seconds"].append(validation_finished - epoch_started)
        print(
            f"classification epoch {epoch + 1}/{config.epochs}: "
            f"train={history['train_seconds'][-1]:.2f}s, "
            f"validation={history['validation_seconds'][-1]:.2f}s, "
            f"total={history['epoch_seconds'][-1]:.2f}s",
            flush=True,
        )
        validation_loss = float(final_validation["loss"])
        if validation_loss < best_validation_loss - config.early_stopping_min_delta:
            best_validation_loss = validation_loss
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            if config.early_stopping_patience is not None:
                best_model_state = copy.deepcopy(model.state_dict())
                best_optimizer_state = copy.deepcopy(optimizer.state_dict())
                best_validation_metrics = copy.deepcopy(final_validation)
        else:
            epochs_without_improvement += 1
        if (
            config.early_stopping_patience is not None
            and epochs_without_improvement >= config.early_stopping_patience
        ):
            stopped_early = True
            print(
                f"classification early stopping after epoch {epoch + 1}; "
                f"best epoch={best_epoch}, best validation loss={best_validation_loss:.6g}",
                flush=True,
            )
            break
    if best_model_state is not None and best_optimizer_state is not None:
        model.load_state_dict(best_model_state)
        optimizer.load_state_dict(best_optimizer_state)
        if best_validation_metrics is None:  # pragma: no cover - defensive invariant
            raise AssertionError("Best classification metrics were not retained")
        final_validation = best_validation_metrics
    plot_learning_curves(history, output / "learning_curves.png")
    plot_confusion_matrix(
        np.asarray(final_validation["confusion_matrix"]),
        class_names,
        output / "confusion_matrix.png",
    )
    plot_classification_report(
        final_validation,
        class_names,
        output / "classification_report.png",
    )
    summary = {
        "model": "VideoClassifier",
        "device": str(device),
        "run_config": asdict(config),
        "model_config": model_config.to_dict(),
        "class_names": class_names,
        "history": history,
        "early_stopping": {
            "enabled": config.early_stopping_patience is not None,
            "patience": config.early_stopping_patience,
            "min_delta": config.early_stopping_min_delta,
            "maximum_epochs": config.epochs,
            "epochs_completed": len(history["epoch"]),
            "stopped_early": stopped_early,
            "best_epoch": best_epoch,
            "best_validation_loss": best_validation_loss,
            "restored_best_weights": best_model_state is not None,
        },
        "validation_metrics": final_validation,
        "validation_samples": len(validation_data),
    }
    save_metrics_json(output / "metrics.json", summary)
    save_checkpoint(
        output / "checkpoint.pt",
        model,
        optimizer=optimizer,
        epoch=best_epoch if best_model_state is not None else len(history["epoch"]),
        config=model_config,
        scalers=scalers,
        split=split,
        metrics=summary,
    )
    return summary


def _joint_model_config(
    config: TrainingRunConfig,
    video_shape: tuple[int, int, int, int],
    num_classes: int,
) -> JointCVAEConfig:
    if config.architecture == "standard":
        return JointCVAEConfig(video_shape=video_shape, num_classes=num_classes)
    return JointCVAEConfig(
        video_shape=video_shape,
        encoder=_encoder(config),
        num_classes=num_classes,
        latent_dim=8,
        laser_hidden=(8,),
        laser_embedding_dim=8,
        condition_hidden=(16,),
        condition_embedding_dim=12,
        posterior_hidden=(16,),
        prior_hidden=(16,),
        regression_hidden=(16,),
        classification_hidden=(16,),
        decoder_seed_shape=(2, 4, 4),
        decoder_channels=(16, 8),
        group_norm_groups=4,
    )


def _collect_joint_outputs(
    model: JointConditionalVAE,
    loader: DataLoader,
    device: torch.device,
    scalers: ScalerBundle,
) -> dict[str, Any]:
    model.eval()
    videos: list[np.ndarray] = []
    generated: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    classes: list[np.ndarray] = []
    logits: list[np.ndarray] = []
    sample_ids: list[str] = []
    with torch.no_grad():
        for batch in loader:
            video = batch["video"].to(device)
            laser = batch["laser_conditions"].to(device)
            properties = batch["material_properties"].to(device)
            class_index = batch["class_index"].to(device)
            property_prediction, class_logits = model.predict(video, laser)
            generation = model.generate(laser, properties, num_samples=1, temperature=0.0)[:, 0]
            videos.append(video.cpu().numpy())
            generated.append(generation.cpu().numpy())
            targets.append(properties.cpu().numpy())
            predictions.append(property_prediction.cpu().numpy())
            classes.append(class_index.cpu().numpy())
            logits.append(class_logits.cpu().numpy())
            sample_ids.extend(batch["sample_id"])
    scaled_video_array = np.concatenate(videos)
    scaled_generated_array = np.concatenate(generated)
    video_array = scaled_video_array
    generated_array = scaled_generated_array
    target_array = np.concatenate(targets)
    prediction_array = np.concatenate(predictions)
    if scalers.video is not None:
        video_array = scalers.video.inverse_transform(video_array)
        generated_array = scalers.video.inverse_transform(generated_array)
    if scalers.targets is not None:
        target_array = scalers.targets.inverse_transform(target_array)
        prediction_array = scalers.targets.inverse_transform(prediction_array)
    return {
        "videos": np.maximum(video_array, 0.0),
        "generated": np.maximum(generated_array, 0.0),
        "scaled_videos": scaled_video_array,
        "scaled_generated": scaled_generated_array,
        "targets": target_array,
        "predictions": prediction_array,
        "classes": np.concatenate(classes),
        "logits": np.concatenate(logits),
        "sample_ids": tuple(sample_ids),
    }


def run_joint_cvae_experiment(
    manifest: DatasetManifest,
    split: SplitManifest,
    scalers: ScalerBundle,
    output_root: str | Path,
    config: TrainingRunConfig,
) -> dict[str, Any]:
    _seed(config)
    device = _device(config)
    output = _output_dir(output_root, "joint_cvae")
    class_names = tuple(sorted({record.element for record in manifest.records}))
    class_to_index = {name: index for index, name in enumerate(class_names)}
    with np.load(manifest.resolve_path(manifest.records[0]), allow_pickle=False) as product:
        raw_shape = tuple(int(value) for value in product["video"].shape)
        expected_times = np.asarray(product["times_s"], dtype=np.float64)
    video_shape = (1, *raw_shape)
    train_data = JointPrecomputedVideoDataset(
        manifest,
        sample_ids=split.train,
        scalers=scalers,
        class_to_index=class_to_index,
        expected_video_shape=video_shape,
        expected_times_s=expected_times,
    )
    validation_data = JointPrecomputedVideoDataset(
        manifest,
        sample_ids=split.validation,
        scalers=scalers,
        class_to_index=class_to_index,
        expected_video_shape=video_shape,
        expected_times_s=expected_times,
    )
    train_loader = _loader(train_data, config, shuffle=True)
    validation_loader = _loader(validation_data, config, shuffle=False)
    model_config = _joint_model_config(config, video_shape, len(class_names))
    model = JointConditionalVAE(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    loss_config = JointLossConfig(
        kl_weight=1.0e-3,
        classification_weight=config.joint_classification_weight,
    )
    tracked = ("loss", "reconstruction_loss", "regression_loss", "classification_loss", "kl_loss")
    history: dict[str, list[float]] = {"epoch": []}
    for phase in ("train", "validation"):
        for name in tracked:
            history[f"{phase}_{name}"] = []
    history["train_accuracy"] = []
    history["validation_accuracy"] = []
    history["train_seconds"] = []
    history["validation_seconds"] = []
    history["epoch_seconds"] = []
    final_validation: dict[str, Any] = {}
    best_validation_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    best_model_state: dict[str, Any] | None = None
    best_optimizer_state: dict[str, Any] | None = None
    best_validation_metrics: dict[str, Any] | None = None
    stopped_early = False
    for epoch in range(config.epochs):
        _synchronize(device)
        epoch_started = time.perf_counter()
        warmup = min(1.0, (epoch + 1) / max(config.epochs, 1))
        training = train_joint_one_epoch(
            model,
            train_loader,
            optimizer,
            device=device,
            loss_config=loss_config,
            kl_warmup_weight=warmup,
            target_scaler=scalers.targets,
            target_names=TARGET_NAMES,
            num_classes=len(class_names),
        )
        _synchronize(device)
        training_finished = time.perf_counter()
        final_validation = evaluate_joint_epoch(
            model,
            validation_loader,
            device=device,
            loss_config=loss_config,
            kl_warmup_weight=warmup,
            target_scaler=scalers.targets,
            target_names=TARGET_NAMES,
            num_classes=len(class_names),
        )
        _synchronize(device)
        validation_finished = time.perf_counter()
        history["epoch"].append(epoch + 1)
        for name in tracked:
            history[f"train_{name}"].append(training[name])
            history[f"validation_{name}"].append(final_validation[name])
        history["train_accuracy"].append(training["classification_metrics"]["accuracy"])
        history["validation_accuracy"].append(
            final_validation["classification_metrics"]["accuracy"]
        )
        history["train_seconds"].append(training_finished - epoch_started)
        history["validation_seconds"].append(validation_finished - training_finished)
        history["epoch_seconds"].append(validation_finished - epoch_started)
        print(
            f"joint_cvae epoch {epoch + 1}/{config.epochs}: "
            f"train={history['train_seconds'][-1]:.2f}s, "
            f"validation={history['validation_seconds'][-1]:.2f}s, "
            f"total={history['epoch_seconds'][-1]:.2f}s",
            flush=True,
        )
        validation_loss = float(final_validation["loss"])
        if validation_loss < best_validation_loss - config.early_stopping_min_delta:
            best_validation_loss = validation_loss
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            if config.early_stopping_patience is not None:
                best_model_state = copy.deepcopy(model.state_dict())
                best_optimizer_state = copy.deepcopy(optimizer.state_dict())
                best_validation_metrics = copy.deepcopy(final_validation)
        else:
            epochs_without_improvement += 1
        if (
            config.early_stopping_patience is not None
            and epochs_without_improvement >= config.early_stopping_patience
        ):
            stopped_early = True
            print(
                f"joint_cvae early stopping after epoch {epoch + 1}; "
                f"best epoch={best_epoch}, best validation loss={best_validation_loss:.6g}",
                flush=True,
            )
            break
    if best_model_state is not None and best_optimizer_state is not None:
        model.load_state_dict(best_model_state)
        optimizer.load_state_dict(best_optimizer_state)
        if best_validation_metrics is None:  # pragma: no cover - defensive invariant
            raise AssertionError("Best joint-CVAE metrics were not retained")
        final_validation = best_validation_metrics
    collected = _collect_joint_outputs(model, validation_loader, device, scalers)
    plot_learning_curves(history, output / "learning_curves.png")
    plot_generation_error_maps(
        collected["videos"],
        collected["generated"],
        collected["sample_ids"],
        output / "generation_error_maps.png",
    )
    plot_regression_parity(
        collected["targets"],
        collected["predictions"],
        TARGET_NAMES,
        output / "parity.png",
    )
    classification_enabled = loss_config.classification_weight > 0.0
    if classification_enabled:
        confusion = np.zeros((len(class_names), len(class_names)), dtype=np.int64)
        predicted_classes = np.argmax(collected["logits"], axis=1)
        np.add.at(confusion, (collected["classes"], predicted_classes), 1)
        plot_confusion_matrix(confusion, class_names, output / "confusion_matrix.png")
        plot_classification_report(
            final_validation["classification_metrics"],
            class_names,
            output / "classification_report.png",
        )
    generation_metrics_scaled = video_generation_metrics(
        collected["scaled_videos"], collected["scaled_generated"]
    )
    physical_residual = collected["generated"] - collected["videos"]
    physical_truth_l1 = float(np.sum(np.abs(collected["videos"])))
    generation_metrics_physical = {
        "mae": float(np.mean(np.abs(physical_residual))),
        "rmse": float(np.sqrt(np.mean(physical_residual * physical_residual))),
        "relative_l1": float(
            np.sum(np.abs(physical_residual)) / max(physical_truth_l1, np.finfo(np.float64).eps)
        ),
    }
    summary = {
        "model": "JointConditionalVAE",
        "device": str(device),
        "run_config": asdict(config),
        "model_config": model_config.to_dict(),
        "classification_enabled": classification_enabled,
        "class_names": class_names,
        "history": history,
        "early_stopping": {
            "enabled": config.early_stopping_patience is not None,
            "patience": config.early_stopping_patience,
            "min_delta": config.early_stopping_min_delta,
            "maximum_epochs": config.epochs,
            "epochs_completed": len(history["epoch"]),
            "stopped_early": stopped_early,
            "best_epoch": best_epoch,
            "best_validation_loss": best_validation_loss,
            "restored_best_weights": best_model_state is not None,
        },
        "validation_metrics": final_validation,
        "validation_generation_mae": float(
            np.mean(np.abs(collected["generated"] - collected["videos"]))
        ),
        "validation_generation_metrics_standardized_log_radiance": generation_metrics_scaled,
        "validation_generation_metrics_physical_radiance": generation_metrics_physical,
        "validation_samples": len(validation_data),
    }
    save_metrics_json(output / "metrics.json", summary)
    save_checkpoint(
        output / "checkpoint.pt",
        model,
        optimizer=optimizer,
        epoch=best_epoch if best_model_state is not None else len(history["epoch"]),
        config=model_config,
        scalers=scalers,
        split=split,
        metrics=summary,
        extra={"loss_config": asdict(loss_config)},
    )
    return summary


def run_experiment(
    model_choice: ModelChoice,
    manifest: DatasetManifest,
    split: SplitManifest,
    scalers: ScalerBundle,
    output_root: str | Path,
    config: TrainingRunConfig,
) -> dict[str, Any]:
    runners = {
        "regression": run_regression_experiment,
        "classification": run_classification_experiment,
        "joint_cvae": run_joint_cvae_experiment,
    }
    return runners[model_choice](manifest, split, scalers, output_root, config)
