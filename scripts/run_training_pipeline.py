"""Cache HDF5 samples, train selected models, and create analysis reports."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from iccd_sim_ml.data import make_known_material_split
from iccd_sim_ml.pipeline import (
    ProxyCacheConfig,
    TrainingRunConfig,
    discover_balanced_subset,
    ensure_proxy_cache,
    fit_experiment_scalers,
    run_experiment,
)
from iccd_sim_ml.pipeline.reports import save_metrics_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="End-to-end cache, training, checkpoint, and report workflow"
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--elements", nargs="+", default=("Al", "Cu", "V"))
    parser.add_argument("--simulations-per-element", type=int, default=3)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=("all", "regression", "classification", "joint_cvae"),
        default=("all",),
    )
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--image-width", type=int, default=32)
    parser.add_argument("--image-height", type=int, default=32)
    parser.add_argument("--line-of-sight-points", type=int, default=32)
    parser.add_argument("--radial-max-mm", type=float, default=15.0)
    parser.add_argument("--axial-max-mm", type=float, default=32.0)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--architecture", choices=("smoke", "standard"), default="smoke")
    parser.add_argument("--device", default="auto")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    elements = tuple(args.elements)
    cache_config = ProxyCacheConfig(
        frame_count=args.frames,
        image_width=args.image_width,
        image_height=args.image_height,
        line_of_sight_points=args.line_of_sight_points,
        radial_max_m=args.radial_max_mm * 1.0e-3,
        axial_max_m=args.axial_max_mm * 1.0e-3,
    )
    training_config = TrainingRunConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        seed=args.seed,
        architecture=args.architecture,
        device=args.device,
    )
    print(
        f"Selecting {args.simulations_per_element} simulation(s) for each of "
        f"{', '.join(elements)}...",
        flush=True,
    )
    samples = discover_balanced_subset(
        args.data_dir,
        elements=elements,
        simulations_per_element=args.simulations_per_element,
        minimum_frames=args.frames,
    )
    cache = ensure_proxy_cache(samples, args.cache_dir, cache_config)
    print(f"Cache: {len(cache.hits)} hit(s), {len(cache.misses)} miss(es)", flush=True)

    split = make_known_material_split(
        cache.manifest,
        ratios=(2.0 / 3.0, 1.0 / 3.0, 0.0),
        seed=args.seed,
    )
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    split.save(output_dir / "split.json")
    scalers = fit_experiment_scalers(cache.manifest, split)
    save_metrics_json(output_dir / "scalers.json", scalers.to_dict())

    choices = (
        ("regression", "classification", "joint_cvae") if "all" in args.models else args.models
    )
    results = {}
    for choice in choices:
        print(f"Training {choice}...", flush=True)
        results[choice] = run_experiment(
            choice,
            cache.manifest,
            split,
            scalers,
            output_dir,
            training_config,
        )
    summary = {
        "workflow": "cache_train_report",
        "warning": (
            "plasma_state_proxy_smoke_test is not calibrated ICCD radiance; "
            "replace the cache backend for scientific training"
        ),
        "elements": elements,
        "simulations_per_element": args.simulations_per_element,
        "sample_ids": [sample.sample_id for sample in samples],
        "cache": {
            "hits": cache.hits,
            "misses": cache.misses,
            "config": asdict(cache_config),
            "manifest": str(cache.manifest.source),
        },
        "split": split.to_dict(),
        "training_config": asdict(training_config),
        "results": results,
    }
    save_metrics_json(output_dir / "pipeline_summary.json", summary)
    print(
        json.dumps(
            {
                "cache_hits": len(cache.hits),
                "cache_misses": len(cache.misses),
                "models": list(results),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
