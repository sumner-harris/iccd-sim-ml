"""Cache HDF5 samples, train selected models, and create analysis reports."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from iccd_sim_ml.data import make_known_material_split
from iccd_sim_ml.imaging import ImagingConfig
from iccd_sim_ml.pipeline import (
    ContinuumCacheConfig,
    ProxyCacheConfig,
    TrainingRunConfig,
    discover_balanced_subset,
    ensure_continuum_cache,
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
        "--cache-backend",
        choices=("proxy", "continuum"),
        default="proxy",
        help="proxy is for software smoke tests; continuum produces physical LTE radiance",
    )
    parser.add_argument("--atomic-reference", type=Path)
    parser.add_argument("--atomic-mode", choices=("strict", "approximate"), default="strict")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=("all", "regression", "classification", "joint_cvae"),
        default=("all",),
    )
    parser.add_argument("--frames", type=int)
    parser.add_argument("--image-width", type=int)
    parser.add_argument("--image-height", type=int)
    parser.add_argument("--line-of-sight-points", type=int)
    parser.add_argument("--wavelength-points", type=int, default=48)
    parser.add_argument("--temperature-table-points", type=int, default=160)
    parser.add_argument("--start-ns", type=float, default=0.0)
    parser.add_argument("--stop-ns", type=float, default=5000.0)
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
    if args.cache_backend == "proxy":
        frames = 8 if args.frames is None else args.frames
        image_width = 32 if args.image_width is None else args.image_width
        image_height = 32 if args.image_height is None else args.image_height
        line_of_sight_points = (
            32 if args.line_of_sight_points is None else args.line_of_sight_points
        )
        cache_config = ProxyCacheConfig(
            frame_count=frames,
            image_width=image_width,
            image_height=image_height,
            line_of_sight_points=line_of_sight_points,
            radial_max_m=args.radial_max_mm * 1.0e-3,
            axial_max_m=args.axial_max_mm * 1.0e-3,
        )
        minimum_frames = frames
    else:
        if args.atomic_reference is None:
            raise ValueError("--atomic-reference is required for the continuum cache backend")
        frames = 16 if args.frames is None else args.frames
        image_width = 96 if args.image_width is None else args.image_width
        image_height = 96 if args.image_height is None else args.image_height
        line_of_sight_points = (
            128 if args.line_of_sight_points is None else args.line_of_sight_points
        )
        if not 0.0 <= args.start_ns < args.stop_ns:
            raise ValueError("Continuum cache times require 0 <= start-ns < stop-ns")
        imaging = ImagingConfig(
            wavelength_min_nm=300.0,
            wavelength_max_nm=800.0,
            wavelength_points=args.wavelength_points,
            radial_points=image_width,
            axial_points=image_height,
            line_of_sight_points=line_of_sight_points,
            temperature_table_points=args.temperature_table_points,
            radial_max_m=args.radial_max_mm * 1.0e-3,
            axial_max_m=args.axial_max_mm * 1.0e-3,
        )
        cache_config = ContinuumCacheConfig(
            imaging=imaging,
            frame_times_s=tuple(
                np.linspace(args.start_ns, args.stop_ns, frames, dtype=np.float64) * 1.0e-9
            ),
            atomic_mode=args.atomic_mode,
        )
        minimum_frames = 2
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
        "cache_backend": args.cache_backend,
        "warning": (
            "plasma_state_proxy_smoke_test is not calibrated ICCD radiance"
            if args.cache_backend == "proxy"
            else (
                "continuum products omit bound-bound lines and camera effects; inspect each "
                "sample's atomic_fidelity before scientific use"
            )
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
