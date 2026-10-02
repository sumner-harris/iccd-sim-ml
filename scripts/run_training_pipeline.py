"""Cache HDF5 samples, train selected models, and create analysis reports."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from iccd_sim_ml.data import DatasetManifest, ScalerBundle
from iccd_sim_ml.imaging import ImagingConfig
from iccd_sim_ml.pipeline import (
    ContinuumCacheConfig,
    ProxyCacheConfig,
    TrainingRunConfig,
    discover_balanced_subset,
    ensure_continuum_cache,
    ensure_proxy_cache,
    fit_experiment_scalers,
    resolve_split,
    run_experiment,
    subset_manifest,
)
from iccd_sim_ml.pipeline.reports import save_metrics_json

DEFAULT_PRODUCTION_IMAGING_CONFIG = (
    Path(__file__).resolve().parents[1] / "configs" / "continuum_ml_typical_8us.json"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="End-to-end cache, training, checkpoint, and report workflow"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--manifest",
        type=Path,
        help="Completed precomputed-cache manifest; trains from every record by default.",
    )
    source.add_argument("--data-dir", type=Path, help="Raw HDF5 directory for subset workflows.")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--elements",
        nargs="+",
        help="Optional element filter. Manifest mode uses every element when omitted.",
    )
    parser.add_argument("--simulations-per-element", type=int)
    parser.add_argument(
        "--cache-backend",
        choices=("proxy", "continuum"),
        default="proxy",
        help="proxy is for software smoke tests; continuum produces physical LTE radiance",
    )
    parser.add_argument("--atomic-reference", type=Path)
    parser.add_argument("--imaging-config", type=Path, default=DEFAULT_PRODUCTION_IMAGING_CONFIG)
    parser.add_argument("--atomic-mode", choices=("strict", "approximate"), default="strict")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=(
            "all",
            "regression",
            "classification",
            "joint_cvae",
            "deep_set_regression",
            "set_transformer_regression",
        ),
        default=("all",),
    )
    parser.add_argument("--frames", type=int)
    parser.add_argument("--image-width", type=int)
    parser.add_argument("--image-height", type=int)
    parser.add_argument("--line-of-sight-points", type=int)
    parser.add_argument("--wavelength-points", type=int)
    parser.add_argument("--temperature-table-points", type=int)
    parser.add_argument("--start-ns", type=float, default=0.0)
    parser.add_argument("--stop-ns", type=float)
    parser.add_argument("--radial-max-mm", type=float)
    parser.add_argument("--axial-max-mm", type=float)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        help="Stop training after this many epochs without validation-loss improvement.",
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=0.0,
        help="Minimum absolute validation-loss reduction counted as improvement.",
    )
    parser.add_argument(
        "--joint-classification-weight",
        type=float,
        default=1.0,
        help=(
            "Classification-loss weight for joint cVAE training. Set to 0 for "
            "element-held-out regression/generation runs."
        ),
    )
    parser.add_argument(
        "--set-size",
        type=int,
        default=8,
        help="Number of same-material simulations in each Deep Set/Set Transformer input.",
    )
    parser.add_argument("--set-train-bags-per-material", type=int, default=16)
    parser.add_argument("--set-validation-bags-per-material", type=int, default=32)
    parser.add_argument(
        "--set-pretrained-regressor",
        type=Path,
        help="Optional standalone-regressor checkpoint used to initialize video encoders.",
    )
    parser.add_argument(
        "--set-encoder-learning-rate-scale",
        type=float,
        default=0.1,
        help="Learning-rate multiplier for a pretrained video/condition encoder.",
    )
    parser.add_argument(
        "--set-individual-loss-weight",
        type=float,
        default=0.2,
        help="Weight for per-video regression supervision during joint set training.",
    )
    parser.add_argument(
        "--set-freeze-pretrained-batchnorm",
        action="store_true",
        help=(
            "Keep the pretrained regressor's BatchNorm running statistics fixed while "
            "jointly fine-tuning its weights on correlated same-material sets."
        ),
    )
    parser.add_argument(
        "--set-pretrained-warmup-epochs",
        type=int,
        default=0,
        help=(
            "Train only the new set encoder/head for this many epochs before jointly "
            "fine-tuning a supplied pretrained regressor."
        ),
    )
    parser.add_argument(
        "--set-baseline-pooling",
        choices=("mean", "target_attention"),
        default="mean",
        help=(
            "Combine pretrained per-video predictions by a fixed mean or by learned, "
            "per-property convex attention weights initialized to that mean."
        ),
    )
    parser.add_argument(
        "--set-capacity",
        choices=("compact", "standard"),
        default="standard",
        help=(
            "Capacity of the newly initialized material-set encoder and head. "
            "Compact uses 64-dimensional tokens and one transformer layer."
        ),
    )
    parser.add_argument(
        "--set-target-specific-pooling",
        action="store_true",
        help=(
            "Use one learned Set Transformer pooling query and scalar residual head "
            "per regression target instead of one shared pooled material embedding."
        ),
    )
    parser.add_argument(
        "--set-target-loss-weights",
        nargs=7,
        type=float,
        metavar=("CP", "HVAP", "KAPPA", "REFLECT", "DENSITY", "TBOIL", "TCRIT"),
        help=(
            "Optional positive training-loss weights in target order: cp_metal, "
            "h_vapor, kappa_metal, laser_reflectivity, mass_density_metal, "
            "t_boil, tcrit. Validation selection remains unweighted."
        ),
    )
    parser.add_argument(
        "--set-bag-seed",
        type=int,
        help=(
            "Optional seed for material-set membership. Defaults to --seed; set it "
            "explicitly to hold train/validation bags fixed across model-init seeds."
        ),
    )
    parser.add_argument(
        "--split-strategy",
        choices=("known-material", "legacy", "element-held-out"),
        default="known-material",
    )
    parser.add_argument(
        "--split-ratios",
        nargs=3,
        type=float,
        metavar=("TRAIN", "VALIDATION", "TEST"),
        help="Defaults to 0.70/0.15/0.15 for manifests and 2/3--1/3 for subset smoke runs.",
    )
    parser.add_argument(
        "--split-file",
        type=Path,
        help="Persistent split manifest. Defaults to OUTPUT_DIR/split.json.",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        help=(
            "Seed used to create or validate split membership. Defaults to --seed; "
            "set it explicitly when repeating model seeds on one saved split."
        ),
    )
    parser.add_argument(
        "--scalers-file",
        type=Path,
        help="Saved train-only scaler bundle. Defaults to OUTPUT_DIR/scalers.json.",
    )
    parser.add_argument(
        "--regenerate-split",
        action="store_true",
        help="Intentionally replace an existing split instead of reusing it.",
    )
    parser.add_argument(
        "--refit-scalers",
        action="store_true",
        help="Refit train-only scalers even when a compatible saved bundle exists.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Create/reuse the split and train-only scalers without training models.",
    )
    parser.add_argument("--architecture", choices=("smoke", "standard"), default="smoke")
    parser.add_argument("--device", default="auto")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_mode = args.manifest is not None
    source_summary: dict[str, object]
    if manifest_mode:
        if args.cache_dir is not None:
            raise ValueError("--cache-dir is only used with --data-dir, not --manifest")
        if args.simulations_per_element is not None:
            raise ValueError(
                "--simulations-per-element is a subset-smoke option; manifest mode uses all "
                "records unless --elements filters them"
            )
        manifest = DatasetManifest.load(args.manifest)
        manifest = subset_manifest(
            manifest,
            None if args.elements is None else tuple(args.elements),
        )
        missing_products = [
            record.sample_id
            for record in manifest.records
            if not manifest.resolve_path(record).is_file()
        ]
        if missing_products:
            raise FileNotFoundError(
                f"Manifest references {len(missing_products)} missing products; "
                f"first IDs: {missing_products[:5]}"
            )
        elements = tuple(sorted({record.element for record in manifest.records}))
        source_summary = {
            "mode": "precomputed_manifest",
            "manifest": str(Path(args.manifest).expanduser().resolve()),
            "records": len(manifest.records),
        }
        warning = (
            "precomputed continuum products omit bound-bound lines and camera effects; "
            "inspect quality flags before scientific use"
        )
        print(
            f"Loaded {len(manifest.records)} cached simulation(s) across "
            f"{len(elements)} element(s) from {args.manifest}",
            flush=True,
        )
    else:
        if args.cache_dir is None:
            raise ValueError("--cache-dir is required with --data-dir")
        elements = tuple(args.elements or ("Al", "Cu", "V"))
        simulations_per_element = (
            3 if args.simulations_per_element is None else args.simulations_per_element
        )
        if args.cache_backend == "proxy":
            frames = 8 if args.frames is None else args.frames
            image_width = 32 if args.image_width is None else args.image_width
            image_height = 32 if args.image_height is None else args.image_height
            radial_max_mm = 15.0 if args.radial_max_mm is None else args.radial_max_mm
            axial_max_mm = 32.0 if args.axial_max_mm is None else args.axial_max_mm
            line_of_sight_points = (
                32 if args.line_of_sight_points is None else args.line_of_sight_points
            )
            cache_config: ProxyCacheConfig | ContinuumCacheConfig = ProxyCacheConfig(
                frame_count=frames,
                image_width=image_width,
                image_height=image_height,
                line_of_sight_points=line_of_sight_points,
                radial_max_m=radial_max_mm * 1.0e-3,
                axial_max_m=axial_max_mm * 1.0e-3,
            )
            minimum_frames = frames
            warning = "plasma_state_proxy_smoke_test is not calibrated ICCD radiance"
        else:
            if args.atomic_reference is None:
                raise ValueError("--atomic-reference is required for the continuum cache backend")
            base_imaging = ImagingConfig.from_json(args.imaging_config)
            frames = 16 if args.frames is None else args.frames
            image_width = (
                base_imaging.radial_points if args.image_width is None else args.image_width
            )
            image_height = (
                base_imaging.axial_points if args.image_height is None else args.image_height
            )
            stop_ns = 8_000.0 if args.stop_ns is None else args.stop_ns
            radial_max_mm = (
                base_imaging.radial_max_m * 1.0e3
                if args.radial_max_mm is None
                else args.radial_max_mm
            )
            axial_max_mm = (
                base_imaging.axial_max_m * 1.0e3 if args.axial_max_mm is None else args.axial_max_mm
            )
            line_of_sight_points = (
                base_imaging.line_of_sight_points
                if args.line_of_sight_points is None
                else args.line_of_sight_points
            )
            if not 0.0 <= args.start_ns < stop_ns:
                raise ValueError("Continuum cache times require 0 <= start-ns < stop-ns")
            imaging = replace(
                base_imaging,
                wavelength_points=(
                    base_imaging.wavelength_points
                    if args.wavelength_points is None
                    else args.wavelength_points
                ),
                radial_points=image_width,
                axial_points=image_height,
                line_of_sight_points=line_of_sight_points,
                temperature_table_points=(
                    base_imaging.temperature_table_points
                    if args.temperature_table_points is None
                    else args.temperature_table_points
                ),
                radial_max_m=radial_max_mm * 1.0e-3,
                axial_max_m=axial_max_mm * 1.0e-3,
            )
            cache_config = ContinuumCacheConfig(
                imaging=imaging,
                frame_times_s=tuple(
                    np.linspace(args.start_ns, stop_ns, frames, dtype=np.float64) * 1.0e-9
                ),
                atomic_mode=args.atomic_mode,
            )
            minimum_frames = 2
            warning = (
                "continuum products omit bound-bound lines and camera effects; inspect each "
                "sample's atomic_fidelity before scientific use"
            )
        print(
            f"Selecting {simulations_per_element} simulation(s) for each of "
            f"{', '.join(elements)}...",
            flush=True,
        )
        samples = discover_balanced_subset(
            args.data_dir,
            elements=elements,
            simulations_per_element=simulations_per_element,
            minimum_frames=minimum_frames,
        )
        if args.cache_backend == "proxy":
            cache = ensure_proxy_cache(samples, args.cache_dir, cache_config)
        else:
            cache = ensure_continuum_cache(
                samples,
                args.cache_dir,
                cache_config,
                args.atomic_reference,
            )
        manifest = cache.manifest
        print(f"Cache: {len(cache.hits)} hit(s), {len(cache.misses)} miss(es)", flush=True)
        source_summary = {
            "mode": "raw_hdf5_subset",
            "manifest": str(manifest.source),
            "cache_backend": args.cache_backend,
            "cache_hits": cache.hits,
            "cache_misses": cache.misses,
            "cache_config": asdict(cache_config),
            "simulations_per_element": simulations_per_element,
        }
    training_config = TrainingRunConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        seed=args.seed,
        architecture=args.architecture,
        device=args.device,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        joint_classification_weight=args.joint_classification_weight,
        set_size=args.set_size,
        set_train_bags_per_material=args.set_train_bags_per_material,
        set_validation_bags_per_material=args.set_validation_bags_per_material,
        set_pretrained_regressor=(
            None
            if args.set_pretrained_regressor is None
            else str(args.set_pretrained_regressor.expanduser().resolve())
        ),
        set_encoder_learning_rate_scale=args.set_encoder_learning_rate_scale,
        set_individual_loss_weight=args.set_individual_loss_weight,
        set_freeze_pretrained_batchnorm=args.set_freeze_pretrained_batchnorm,
        set_pretrained_warmup_epochs=args.set_pretrained_warmup_epochs,
        set_baseline_pooling=args.set_baseline_pooling,
        set_capacity=args.set_capacity,
        set_target_specific_pooling=args.set_target_specific_pooling,
        set_target_loss_weights=(
            None if args.set_target_loss_weights is None else tuple(args.set_target_loss_weights)
        ),
        set_bag_seed=args.set_bag_seed,
    )
    if args.split_ratios is None:
        if args.split_strategy == "legacy":
            split_ratios = (0.7, 0.3, 0.0)
        elif manifest_mode:
            split_ratios = (0.7, 0.15, 0.15)
        else:
            split_ratios = (2.0 / 3.0, 1.0 / 3.0, 0.0)
    else:
        split_ratios = tuple(args.split_ratios)
    split_path = (
        output_dir / "split.json"
        if args.split_file is None
        else args.split_file.expanduser().resolve()
    )
    split, split_state = resolve_split(
        manifest,
        split_path,
        strategy=args.split_strategy,
        ratios=split_ratios,
        seed=args.seed if args.split_seed is None else args.split_seed,
        regenerate=args.regenerate_split,
    )
    print(
        f"Split {split_state}: train={len(split.train)}, "
        f"validation={len(split.validation)}, test={len(split.test)} at {split_path}",
        flush=True,
    )
    scalers_path = (
        output_dir / "scalers.json"
        if args.scalers_file is None
        else args.scalers_file.expanduser().resolve()
    )
    scalers_state = "fitted"
    if scalers_path.is_file() and split_state == "loaded" and not args.refit_scalers:
        scalers = ScalerBundle.from_dict(json.loads(scalers_path.read_text(encoding="utf-8")))
        if scalers.train_sample_ids != split.train:
            raise ValueError(
                "Saved scalers were fit to different training IDs; use --refit-scalers "
                "after verifying the split"
            )
        scalers_state = "loaded"
    else:
        scalers = fit_experiment_scalers(manifest, split)
        save_metrics_json(scalers_path, scalers.to_dict())
    print(f"Scalers {scalers_state}: {scalers_path}", flush=True)

    choices = (
        ("regression", "classification", "joint_cvae") if "all" in args.models else args.models
    )
    results = {}
    if not args.prepare_only:
        for choice in choices:
            print(f"Training {choice}...", flush=True)
            results[choice] = run_experiment(
                choice,
                manifest,
                split,
                scalers,
                output_dir,
                training_config,
            )
    summary = {
        "workflow": "cache_train_report",
        "data_source": source_summary,
        "warning": warning,
        "elements": elements,
        "sample_count": len(manifest.records),
        "sample_ids": [record.sample_id for record in manifest.records],
        "split": {
            "path": str(split_path),
            "state": split_state,
            "strategy": args.split_strategy,
            "manifest": split.to_dict(),
        },
        "scalers": {"path": str(scalers_path), "state": scalers_state},
        "training_config": asdict(training_config),
        "prepare_only": args.prepare_only,
        "results": results,
    }
    save_metrics_json(output_dir / "pipeline_summary.json", summary)
    print(
        json.dumps(
            {
                "samples": len(manifest.records),
                "split_state": split_state,
                "scalers_state": scalers_state,
                "models": list(results),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
