#!/usr/bin/env python3
"""Sequentially fine-tune a cVAE with single-video and set property cycles."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from iccd_sim_ml.data import (
    DatasetManifest,
    JointPrecomputedVideoDataset,
    MaterialSetDataset,
    ScalerBundle,
    SplitManifest,
)
from iccd_sim_ml.models import (
    JointConditionalVAE,
    JointCVAEConfig,
    MaterialSetRegressorConfig,
    SetTransformerRegressor,
    VideoRegressor,
    VideoRegressorConfig,
)
from iccd_sim_ml.pipeline.cache import TARGET_NAMES
from iccd_sim_ml.pipeline.reports import save_metrics_json
from iccd_sim_ml.training import (
    JointLossConfig,
    PropertyCycleLossConfig,
    joint_cvae_loss,
    load_checkpoint,
    save_checkpoint,
    set_property_cycle_losses,
    single_video_property_cycle_losses,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--base-cvae", type=Path, required=True)
    parser.add_argument("--single-teacher", type=Path, required=True)
    parser.add_argument("--set-teacher", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--single-epochs", type=int, default=50)
    parser.add_argument("--set-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--single-batch-size", type=int, default=8)
    parser.add_argument("--set-batch-size", type=int, default=1)
    parser.add_argument("--set-size", type=int, default=16)
    parser.add_argument("--train-bags-per-material", type=int, default=8)
    parser.add_argument("--validation-bags-per-material", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--cycle-weight", type=float, default=0.25)
    parser.add_argument("--cycle-warmup-epochs", type=int, default=5)
    parser.add_argument("--perturbation-std", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    positive = (
        "single_epochs",
        "set_epochs",
        "patience",
        "single_batch_size",
        "set_batch_size",
        "set_size",
        "train_bags_per_material",
        "validation_bags_per_material",
    )
    if any(getattr(args, name) < 1 for name in positive):
        raise ValueError(f"These arguments must be positive: {positive}")
    if args.num_workers < 0 or args.cycle_warmup_epochs < 0:
        raise ValueError("num-workers and cycle-warmup-epochs must be non-negative")
    if args.learning_rate <= 0.0 or args.weight_decay < 0.0:
        raise ValueError("learning-rate must be positive and weight-decay non-negative")
    if not 0.0 < args.cycle_weight <= 1.0:
        raise ValueError("cycle-weight must be in (0,1]")


def _assert_checkpoint_compatibility(payloads: Mapping[str, dict[str, Any]]) -> None:
    reference = payloads["cvae"]
    for name, payload in payloads.items():
        if payload["split"] != reference["split"]:
            raise ValueError(f"{name} checkpoint uses a different split from the cVAE")
        if payload["scalers"] != reference["scalers"]:
            raise ValueError(f"{name} checkpoint uses different scalers from the cVAE")


def _load_models(args: argparse.Namespace, device: torch.device) -> tuple[Any, ...]:
    payloads = {
        "cvae": load_checkpoint(args.base_cvae, map_location="cpu"),
        "single": load_checkpoint(args.single_teacher, map_location="cpu"),
        "set": load_checkpoint(args.set_teacher, map_location="cpu"),
    }
    _assert_checkpoint_compatibility(payloads)
    cvae = JointConditionalVAE(JointCVAEConfig.from_dict(payloads["cvae"]["config"])).to(device)
    cvae.load_state_dict(payloads["cvae"]["model_state"])
    single = VideoRegressor(VideoRegressorConfig.from_dict(payloads["single"]["config"])).to(device)
    single.load_state_dict(payloads["single"]["model_state"])
    set_teacher = SetTransformerRegressor(
        MaterialSetRegressorConfig.from_dict(payloads["set"]["config"])
    ).to(device)
    set_teacher.load_state_dict(payloads["set"]["model_state"])
    single.requires_grad_(False).eval()
    set_teacher.requires_grad_(False).eval()
    return cvae, single, set_teacher, payloads


def _loader(dataset: Any, *, batch_size: int, workers: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
    )


def _base_losses(
    output: Any,
    video: torch.Tensor,
    properties: torch.Tensor,
    class_index: torch.Tensor,
    loss_config: JointLossConfig,
) -> dict[str, torch.Tensor]:
    return joint_cvae_loss(
        output,
        video,
        properties,
        class_index,
        config=loss_config,
        kl_warmup_weight=1.0,
    )


def _cycle_scale(args: argparse.Namespace, epoch: int) -> float:
    if args.cycle_warmup_epochs == 0:
        return float(args.cycle_weight)
    return float(args.cycle_weight) * min(1.0, (epoch + 1) / args.cycle_warmup_epochs)


def _summarize(totals: dict[str, float], count: int, seconds: float) -> dict[str, float]:
    if count == 0:
        raise ValueError("Training loader produced no samples")
    result = {name: value / count for name, value in totals.items()}
    result["samples"] = float(count)
    result["seconds"] = float(seconds)
    return result


def _single_epoch(
    model: JointConditionalVAE,
    teacher: VideoRegressor,
    loader: Iterable[Mapping[str, Any]],
    *,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    base_config: JointLossConfig,
    cycle_config: PropertyCycleLossConfig,
    cycle_scale: float,
    intervention_offset: int,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    teacher.eval()
    names = (
        "loss",
        "base_loss",
        "reconstruction_loss",
        "kl_loss",
        "property_cycle_loss",
        "posterior_property_loss",
        "prior_property_loss",
        "intervention_loss",
        "cross_talk_loss",
    )
    totals = {name: 0.0 for name in names}
    count = 0
    started = time.perf_counter()
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch_index, batch in enumerate(loader):
            video = batch["video"].to(device, non_blocking=True)
            laser = batch["laser_conditions"].to(device, non_blocking=True)
            properties = batch["material_properties"].to(device, non_blocking=True)
            class_index = batch["class_index"].to(device, non_blocking=True)
            if training:
                assert optimizer is not None
                optimizer.zero_grad(set_to_none=True)
            output = model(video, laser, properties, sample_posterior=training)
            base = _base_losses(output, video, properties, class_index, base_config)
            cycle = single_video_property_cycle_losses(
                model,
                output,
                teacher,
                laser,
                properties,
                config=cycle_config,
                weight_scale=cycle_scale,
                intervention_offset=intervention_offset + batch_index,
            )
            loss = base["loss"] + cycle["property_cycle_loss"]
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            batch_size = video.shape[0]
            values = {
                "loss": loss,
                "base_loss": base["loss"],
                "reconstruction_loss": base["reconstruction_loss"],
                "kl_loss": base["kl_loss"],
                **{name: cycle[name] for name in names if name in cycle},
            }
            for name in names:
                totals[name] += float(values[name].detach()) * batch_size
            count += batch_size
    return _summarize(totals, count, time.perf_counter() - started)


def _set_epoch(
    model: JointConditionalVAE,
    teacher: SetTransformerRegressor,
    loader: Iterable[Mapping[str, Any]],
    *,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    base_config: JointLossConfig,
    cycle_config: PropertyCycleLossConfig,
    cycle_scale: float,
    intervention_offset: int,
    class_to_index: Mapping[str, int],
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    teacher.eval()
    names = (
        "loss",
        "base_loss",
        "reconstruction_loss",
        "kl_loss",
        "property_cycle_loss",
        "posterior_property_loss",
        "prior_property_loss",
        "intervention_loss",
        "cross_talk_loss",
    )
    totals = {name: 0.0 for name in names}
    count = 0
    started = time.perf_counter()
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch_index, batch in enumerate(loader):
            videos = batch["videos"].to(device, non_blocking=True)
            conditions = batch["conditions"].to(device, non_blocking=True)
            mask = batch["set_mask"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            batch_size, set_size = videos.shape[:2]
            flat_videos = videos.flatten(0, 1)
            flat_conditions = conditions.flatten(0, 1)
            flat_target = target[:, None, :].expand(-1, set_size, -1).flatten(0, 1)
            class_rows = torch.as_tensor(
                [class_to_index[str(element)] for element in batch["element"]],
                device=device,
                dtype=torch.long,
            )
            flat_classes = class_rows[:, None].expand(-1, set_size).flatten()
            if training:
                assert optimizer is not None
                optimizer.zero_grad(set_to_none=True)
            output = model(
                flat_videos,
                flat_conditions,
                flat_target,
                sample_posterior=training,
            )
            base = _base_losses(output, flat_videos, flat_target, flat_classes, base_config)
            cycle = set_property_cycle_losses(
                model,
                output,
                teacher,
                conditions,
                target,
                mask,
                config=cycle_config,
                weight_scale=cycle_scale,
                intervention_offset=intervention_offset + batch_index,
            )
            loss = base["loss"] + cycle["property_cycle_loss"]
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            values = {
                "loss": loss,
                "base_loss": base["loss"],
                "reconstruction_loss": base["reconstruction_loss"],
                "kl_loss": base["kl_loss"],
                **{name: cycle[name] for name in names if name in cycle},
            }
            for name in names:
                totals[name] += float(values[name].detach()) * batch_size
            count += batch_size
    return _summarize(totals, count, time.perf_counter() - started)


def _run_stage(
    name: str,
    model: JointConditionalVAE,
    teacher: torch.nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    *,
    epochs: int,
    args: argparse.Namespace,
    device: torch.device,
    base_config: JointLossConfig,
    cycle_config: PropertyCycleLossConfig,
    checkpoint_context: Mapping[str, Any],
    class_to_index: Mapping[str, int],
) -> tuple[dict[str, Any], Path]:
    output = args.output_dir / name
    output.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    history: list[dict[str, Any]] = []
    best_loss = float("inf")
    best_epoch = 0
    stale = 0
    best_path = output / "checkpoint.pt"
    epoch_function = _single_epoch if name == "single_video_cycle" else _set_epoch
    for epoch in range(epochs):
        scale = _cycle_scale(args, epoch)
        common: dict[str, Any] = {
            "device": device,
            "base_config": base_config,
            "cycle_config": cycle_config,
            "cycle_scale": scale,
            "intervention_offset": epoch,
        }
        if name == "set_transformer_cycle":
            common["class_to_index"] = class_to_index
        train = epoch_function(
            model,
            teacher,
            train_loader,
            optimizer=optimizer,
            **common,
        )
        validation = epoch_function(
            model,
            teacher,
            validation_loader,
            optimizer=None,
            **common,
        )
        record = {
            "epoch": epoch + 1,
            "cycle_scale": scale,
            "train": train,
            "validation": validation,
        }
        history.append(record)
        print(
            f"{name} epoch {epoch + 1}/{epochs}: "
            f"train={train['loss']:.5f}, validation={validation['loss']:.5f}, "
            f"prior-property={validation['prior_property_loss']:.5f}, "
            f"seconds={train['seconds'] + validation['seconds']:.1f}",
            flush=True,
        )
        if validation["loss"] < best_loss:
            best_loss = validation["loss"]
            best_epoch = epoch + 1
            stale = 0
            save_checkpoint(
                best_path,
                model,
                optimizer=optimizer,
                epoch=best_epoch,
                config=model.config,
                scalers=checkpoint_context["scalers"],
                split=checkpoint_context["split"],
                metrics={"stage": name, "history": history, "best_validation_loss": best_loss},
                extra={
                    "base_checkpoint": str(args.base_cvae),
                    "single_teacher": str(args.single_teacher),
                    "set_teacher": str(args.set_teacher),
                    "base_loss_config": vars(base_config),
                    "property_cycle_config": vars(cycle_config),
                    "cycle_weight": args.cycle_weight,
                },
            )
        else:
            stale += 1
        save_metrics_json(
            output / "progress.json",
            {
                "stage": name,
                "history": history,
                "best_epoch": best_epoch,
                "best_validation_loss": best_loss,
                "epochs_without_improvement": stale,
            },
        )
        if stale >= args.patience:
            print(f"{name} early stopping at epoch {epoch + 1}", flush=True)
            break
    best_payload = load_checkpoint(best_path, map_location=device)
    model.load_state_dict(best_payload["model_state"])
    summary = {
        "stage": name,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "history": history,
        "checkpoint": str(best_path.resolve()),
    }
    save_metrics_json(output / "metrics.json", summary)
    return summary, best_path


def main() -> None:
    args = _arguments()
    _validate_args(args)
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    model, single_teacher, set_teacher, payloads = _load_models(args, device)
    split = SplitManifest.from_dict(payloads["cvae"]["split"])
    scalers = ScalerBundle.from_dict(payloads["cvae"]["scalers"])
    manifest = DatasetManifest.load(args.manifest)
    class_names = tuple(sorted({record.element for record in manifest.records}))
    class_to_index = {name: index for index, name in enumerate(class_names)}
    if tuple(model.config.property_names) != tuple(TARGET_NAMES):
        raise ValueError("The cVAE property order does not match the production target order")

    first_path = manifest.resolve_path(manifest.records[0])
    with np.load(first_path, allow_pickle=False) as product:
        video_shape = (1, *tuple(int(value) for value in product["video"].shape))
        expected_times = np.asarray(product["times_s"], dtype=np.float64)
    single_train = JointPrecomputedVideoDataset(
        manifest,
        sample_ids=split.train,
        scalers=scalers,
        class_to_index=class_to_index,
        expected_video_shape=video_shape,
        expected_times_s=expected_times,
    )
    single_validation = JointPrecomputedVideoDataset(
        manifest,
        sample_ids=split.validation,
        scalers=scalers,
        class_to_index=class_to_index,
        expected_video_shape=video_shape,
        expected_times_s=expected_times,
    )
    single_train_loader = _loader(
        single_train,
        batch_size=args.single_batch_size,
        workers=args.num_workers,
        shuffle=True,
    )
    single_validation_loader = _loader(
        single_validation,
        batch_size=args.single_batch_size,
        workers=args.num_workers,
        shuffle=False,
    )
    base_config = JointLossConfig(kl_weight=1.0e-3)
    cycle_config = PropertyCycleLossConfig(perturbation_std=args.perturbation_std)
    checkpoint_context = {"scalers": payloads["cvae"]["scalers"], "split": split.to_dict()}
    single_summary, _ = _run_stage(
        "single_video_cycle",
        model,
        single_teacher,
        single_train_loader,
        single_validation_loader,
        epochs=args.single_epochs,
        args=args,
        device=device,
        base_config=base_config,
        cycle_config=cycle_config,
        checkpoint_context=checkpoint_context,
        class_to_index=class_to_index,
    )

    set_train = MaterialSetDataset(
        manifest,
        sample_ids=split.train,
        set_size=args.set_size,
        sets_per_material=args.train_bags_per_material,
        seed=args.seed,
        sampling="condition_farthest",
        scalers=scalers,
    )
    set_validation = MaterialSetDataset(
        manifest,
        sample_ids=split.validation,
        set_size=args.set_size,
        sets_per_material=args.validation_bags_per_material,
        seed=args.seed + 1,
        sampling="condition_farthest",
        scalers=scalers,
    )
    set_train_loader = _loader(
        set_train,
        batch_size=args.set_batch_size,
        workers=args.num_workers,
        shuffle=True,
    )
    set_validation_loader = _loader(
        set_validation,
        batch_size=args.set_batch_size,
        workers=args.num_workers,
        shuffle=False,
    )
    set_summary, _ = _run_stage(
        "set_transformer_cycle",
        model,
        set_teacher,
        set_train_loader,
        set_validation_loader,
        epochs=args.set_epochs,
        args=args,
        device=device,
        base_config=base_config,
        cycle_config=cycle_config,
        checkpoint_context=checkpoint_context,
        class_to_index=class_to_index,
    )
    save_metrics_json(
        args.output_dir / "campaign_summary.json",
        {
            "purpose": "property-faithful cVAE sequential fine-tuning",
            "single_video_cycle": single_summary,
            "set_transformer_cycle": set_summary,
            "arguments": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
        },
    )
    print(json.dumps({"single": single_summary, "set": set_summary}, indent=2), flush=True)


if __name__ == "__main__":
    main()
