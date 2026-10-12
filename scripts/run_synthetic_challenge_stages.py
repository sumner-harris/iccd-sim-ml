#!/usr/bin/env python3
"""Wait for set training, then run the two frozen-model evaluation stages."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--set-run-dir", type=Path, required=True)
    parser.add_argument("--regressor-checkpoint", type=Path, required=True)
    parser.add_argument("--cvae-checkpoint", type=Path, required=True)
    parser.add_argument("--synthetic-output-dir", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _save_status(path: Path, **values: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(values, indent=2), encoding="utf-8")
    temporary.replace(path)


def _run(command: list[str], status_path: Path, phase: str) -> None:
    _save_status(status_path, status="running", phase=phase, command=command)
    subprocess.run(command, check=True)
    _save_status(status_path, status="running", phase=f"{phase}_complete")


def main() -> None:
    args = _arguments()
    if args.poll_seconds <= 0.0:
        raise ValueError("poll-seconds must be positive")
    set_run = args.set_run_dir.expanduser().resolve()
    checkpoint = set_run / "set_transformer_regression" / "checkpoint.pt"
    progress = set_run / "set_transformer_regression" / "progress.json"
    status_path = set_run / "synthetic_challenge_stages.json"
    _save_status(
        status_path,
        status="waiting",
        phase="set_transformer_training",
        expected_checkpoint=str(checkpoint),
    )
    while not checkpoint.is_file():
        if progress.is_file():
            try:
                progress_value = json.loads(progress.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                progress_value = None
            if isinstance(progress_value, dict) and progress_value.get("status") == "failed":
                raise RuntimeError("Set Transformer training reported failure")
        time.sleep(args.poll_seconds)

    comparison = set_run / "comparison"
    common = ["--num-workers", str(args.num_workers), "--device", args.device]
    _run(
        [
            sys.executable,
            "scripts/evaluate_known_material_set.py",
            "--manifest",
            str(args.manifest),
            "--regressor-checkpoint",
            str(args.regressor_checkpoint),
            "--set-checkpoint",
            str(checkpoint),
            "--output-dir",
            str(comparison),
            "--set-size",
            "16",
            "--sets-per-material",
            "16",
            "--batch-size",
            "2",
            "--single-batch-size",
            "32",
            "--seed",
            "42",
            *common,
        ],
        status_path,
        "known_material_comparison",
    )
    _run(
        [
            sys.executable,
            "scripts/run_synthetic_material_challenge.py",
            "--manifest",
            str(args.manifest),
            "--cvae-checkpoint",
            str(args.cvae_checkpoint),
            "--regressor-checkpoint",
            str(args.regressor_checkpoint),
            "--set-checkpoint",
            str(checkpoint),
            "--output-dir",
            str(args.synthetic_output_dir),
            "--materials",
            "30",
            "--manifold-materials",
            "15",
            "--sequences",
            "500",
            "--set-size",
            "16",
            "--generation-batch-size",
            "2",
            "--evaluation-batch-size",
            "32",
            "--set-batch-size",
            "2",
            "--temperature",
            "1.0",
            "--seed",
            "20261012",
            *common,
        ],
        status_path,
        "synthetic_material_challenge",
    )
    _save_status(
        status_path,
        status="complete",
        phase="complete",
        comparison_metrics=str(comparison / "metrics.json"),
        synthetic_metrics=str(args.synthetic_output_dir / "metrics.json"),
    )


if __name__ == "__main__":
    main()
