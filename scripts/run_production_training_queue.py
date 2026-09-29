#!/usr/bin/env python3
"""Run the production model/split matrix sequentially on one GPU."""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SplitRun:
    name: str
    strategy: str
    split_file: Path
    scalers_file: Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--known-split", type=Path, required=True)
    parser.add_argument("--known-scalers", type=Path, required=True)
    parser.add_argument("--heldout-split", type=Path, required=True)
    parser.add_argument("--heldout-scalers", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=("regression", "classification", "joint_cvae"),
        default=("regression", "classification", "joint_cvae"),
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--min-delta", type=float, default=1.0e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--rerun-completed",
        action="store_true",
        help="Rerun jobs even when their checkpoint already exists.",
    )
    return parser


def _require_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    return resolved


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    script = Path(__file__).with_name("run_training_pipeline.py").resolve()
    manifest = _require_file(args.manifest, "manifest")
    splits = (
        SplitRun(
            "known-material",
            "known-material",
            _require_file(args.known_split, "known-material split"),
            _require_file(args.known_scalers, "known-material scalers"),
        ),
        SplitRun(
            "element-held-out",
            "element-held-out",
            _require_file(args.heldout_split, "element-held-out split"),
            _require_file(args.heldout_scalers, "element-held-out scalers"),
        ),
    )
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    jobs = [(split, model) for split in splits for model in args.models]
    print(f"Production queue contains {len(jobs)} job(s)", flush=True)
    for job_index, (split, model) in enumerate(jobs, start=1):
        output = output_root / split.name
        checkpoint = output / model / "checkpoint.pt"
        if checkpoint.is_file() and not args.rerun_completed:
            print(
                f"[{job_index}/{len(jobs)}] skipping completed {split.name}/{model}: "
                f"{checkpoint}",
                flush=True,
            )
            continue
        command = [
            sys.executable,
            str(script),
            "--manifest",
            str(manifest),
            "--output-dir",
            str(output),
            "--split-file",
            str(split.split_file),
            "--scalers-file",
            str(split.scalers_file),
            "--split-strategy",
            split.strategy,
            "--split-ratios",
            "0.70",
            "0.15",
            "0.15",
            "--seed",
            str(args.seed),
            "--architecture",
            "standard",
            "--models",
            model,
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--num-workers",
            str(args.num_workers),
            "--learning-rate",
            str(args.learning_rate),
            "--early-stopping-patience",
            str(args.patience),
            "--early-stopping-min-delta",
            str(args.min_delta),
            "--device",
            "cuda",
        ]
        print(
            f"[{job_index}/{len(jobs)}] starting {split.name}/{model}",
            flush=True,
        )
        subprocess.run(command, check=True)
        print(
            f"[{job_index}/{len(jobs)}] completed {split.name}/{model}",
            flush=True,
        )
    print("Production queue completed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
