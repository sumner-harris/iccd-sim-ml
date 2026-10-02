"""Evaluate one or more material-set checkpoints on one explicit partition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from iccd_sim_ml.data import DatasetManifest, MaterialSetDataset, ScalerBundle, SplitManifest
from iccd_sim_ml.models import (
    DeepSetRegressor,
    MaterialSetRegressorConfig,
    SetTransformerRegressor,
)
from iccd_sim_ml.pipeline.cache import TARGET_NAMES
from iccd_sim_ml.pipeline.reports import plot_regression_parity, save_metrics_json
from iccd_sim_ml.training import (
    average_material_set_predictions,
    collect_material_set_predictions,
    load_checkpoint,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate and average aligned material-set regression checkpoints"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--scalers-file", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--partition", choices=("validation", "test"), default="validation")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--set-size", type=int, default=32)
    parser.add_argument("--sets-per-material", type=int, default=8)
    parser.add_argument(
        "--bag-seed",
        type=int,
        required=True,
        help="Explicit deterministic bag seed; record it before opening the test partition.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(name)


def _load_scalers(path: Path) -> ScalerBundle:
    with path.expanduser().resolve().open(encoding="utf-8") as handle:
        return ScalerBundle.from_dict(json.load(handle))


def main() -> None:
    args = build_parser().parse_args()
    if args.set_size < 1 or args.sets_per_material < 1:
        raise ValueError("Set size and sets per material must be positive")
    manifest = DatasetManifest.load(args.manifest)
    split = SplitManifest.load(args.split_file)
    scalers = _load_scalers(args.scalers_file)
    sample_ids = split.ids(args.partition)
    dataset = MaterialSetDataset(
        manifest,
        sample_ids=sample_ids,
        set_size=args.set_size,
        sets_per_material=args.sets_per_material,
        seed=args.bag_seed,
        scalers=scalers,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
    )
    device = _device(args.device)
    member_results = []
    member_summaries = []
    for checkpoint_path in args.checkpoints:
        checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
        if checkpoint["split"] != split.to_dict():
            raise ValueError(
                f"Checkpoint split does not match {args.split_file}: {checkpoint_path}"
            )
        if checkpoint["scalers"] != scalers.to_dict():
            raise ValueError(
                f"Checkpoint scalers do not match {args.scalers_file}: {checkpoint_path}"
            )
        config = MaterialSetRegressorConfig.from_dict(checkpoint["config"])
        model_class = (
            DeepSetRegressor if config.aggregator == "deep_set" else SetTransformerRegressor
        )
        model = model_class(config)
        model.load_state_dict(checkpoint["model_state"])
        model.to(device).eval()
        result = collect_material_set_predictions(
            model,
            loader,
            device=device,
            target_scaler=scalers.targets,
            target_names=TARGET_NAMES,
        )
        member_results.append(result)
        member_summaries.append(
            {
                "checkpoint": str(checkpoint_path.expanduser().resolve()),
                "bag_metrics": result["bag_metrics"],
                "material_metrics": result["material_metrics"],
            }
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    ensemble = average_material_set_predictions(member_results, target_names=TARGET_NAMES)
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    plot_regression_parity(
        ensemble["bag_targets"],
        ensemble["bag_predictions"],
        TARGET_NAMES,
        output / "set_parity.png",
        title=f"{args.partition.title()} set parity (physical units)",
    )
    plot_regression_parity(
        ensemble["material_targets"],
        ensemble["material_predictions"],
        TARGET_NAMES,
        output / "material_ensemble_parity.png",
        title=f"{args.partition.title()} material-ensemble parity (physical units)",
    )
    np.savez_compressed(
        output / "predictions.npz",
        bag_targets=ensemble["bag_targets"],
        bag_predictions=ensemble["bag_predictions"],
        material_targets=ensemble["material_targets"],
        material_predictions=ensemble["material_predictions"],
        elements=np.asarray(ensemble["elements"]),
    )
    summary = {
        "partition": args.partition,
        "manifest": str(args.manifest.expanduser().resolve()),
        "split_file": str(args.split_file.expanduser().resolve()),
        "scalers_file": str(args.scalers_file.expanduser().resolve()),
        "set_size": args.set_size,
        "sets_per_material": args.sets_per_material,
        "bag_seed": args.bag_seed,
        "member_count": ensemble["member_count"],
        "members": member_summaries,
        "elements": list(ensemble["elements"]),
        "sets_per_element": ensemble["sets_per_element"],
        "bag_metrics": ensemble["bag_metrics"],
        "material_metrics": ensemble["material_metrics"],
    }
    save_metrics_json(output / "metrics.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
