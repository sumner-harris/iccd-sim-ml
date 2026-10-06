#!/usr/bin/env python3
"""Run the fixed K=32 set-regression recipe over repeated element splits."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--splits-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-seed", type=int, default=42)
    parser.add_argument("--bag-seed", type=int, default=42)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _run(command: list[str], log_path: Path, *, dry_run: bool) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as stream:
        stream.write(f"\n[{_utc_now()}] {' '.join(command)}\n")
        stream.flush()
        if dry_run:
            return
        completed = subprocess.run(
            command,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
        )
    if completed.returncode:
        raise RuntimeError(f"Command failed with exit code {completed.returncode}; see {log_path}")


def _validation_summary(metrics_path: Path) -> dict[str, Any]:
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    validation = metrics["validation_metrics"]
    return {
        "best_epoch": metrics["early_stopping"]["best_epoch"],
        "epochs_completed": metrics["early_stopping"]["epochs_completed"],
        "set_r2_macro": validation["set"]["r2_macro"],
        "material_ensemble_r2_macro": validation["material_ensemble"]["r2_macro"],
        "set_per_target": {
            name: values["r2"] for name, values in validation["set"]["per_target"].items()
        },
        "material_ensemble_per_target": {
            name: values["r2"]
            for name, values in validation["material_ensemble"]["per_target"].items()
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = args.manifest.expanduser().resolve()
    splits_dir = args.splits_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    campaign = json.loads((splits_dir / "campaign.json").read_text(encoding="utf-8"))
    pipeline = Path(__file__).with_name("run_training_pipeline.py").resolve()
    python = args.python.expanduser().resolve()
    status_path = output_dir / "campaign_status.json"
    status: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "started_at_utc": _utc_now(),
        "updated_at_utc": _utc_now(),
        "manifest": str(manifest),
        "splits_dir": str(splits_dir),
        "model_seed": args.model_seed,
        "bag_seed": args.bag_seed,
        "recipe": "frozen_base_target_specific_k32_set_transformer",
        "runs": [],
    }
    if status_path.is_file():
        prior = json.loads(status_path.read_text(encoding="utf-8"))
        status["started_at_utc"] = prior.get("started_at_utc", status["started_at_utc"])

    try:
        for row in campaign["splits"]:
            index = int(row["index"])
            run_dir = output_dir / f"split-{index:02d}"
            split_file = splits_dir / row["split_file"]
            scalers_file = run_dir / "scalers.json"
            regression_metrics = run_dir / "regression" / "metrics.json"
            set_metrics = run_dir / "set_transformer_regression" / "metrics.json"
            run_status: dict[str, Any] = {
                "index": index,
                "validation_elements": row["validation_elements"],
                "split_file": str(split_file),
                "regression_complete": regression_metrics.is_file(),
                "set_transformer_complete": set_metrics.is_file(),
            }
            status["runs"] = [existing for existing in status["runs"] if existing["index"] != index]
            status["runs"].append(run_status)
            status["runs"].sort(key=lambda item: item["index"])
            status["active_split"] = index
            status["updated_at_utc"] = _utc_now()
            _write_json(status_path, status)

            common = [
                str(python),
                str(pipeline),
                "--manifest",
                str(manifest),
                "--output-dir",
                str(run_dir),
                "--split-strategy",
                "element-held-out",
                "--split-file",
                str(split_file),
                "--scalers-file",
                str(scalers_file),
                "--seed",
                str(args.model_seed),
                "--architecture",
                "standard",
                "--device",
                "cuda",
            ]
            if not regression_metrics.is_file():
                _run(
                    common
                    + [
                        "--models",
                        "regression",
                        "--epochs",
                        "100",
                        "--batch-size",
                        "32",
                        "--learning-rate",
                        "0.001",
                        "--weight-decay",
                        "0",
                        "--num-workers",
                        "16",
                        "--early-stopping-patience",
                        "25",
                        "--early-stopping-min-delta",
                        "0.0001",
                    ],
                    run_dir / "regression.log",
                    dry_run=args.dry_run,
                )
            run_status["regression_complete"] = regression_metrics.is_file()
            status["updated_at_utc"] = _utc_now()
            _write_json(status_path, status)

            if not set_metrics.is_file():
                checkpoint = run_dir / "regression" / "checkpoint.pt"
                if not checkpoint.is_file() and not args.dry_run:
                    raise FileNotFoundError(f"Missing base-regressor checkpoint: {checkpoint}")
                _run(
                    common
                    + [
                        "--models",
                        "set_transformer_regression",
                        "--epochs",
                        "100",
                        "--batch-size",
                        "1",
                        "--learning-rate",
                        "0.00001",
                        "--weight-decay",
                        "0.0001",
                        "--num-workers",
                        "8",
                        "--early-stopping-patience",
                        "15",
                        "--set-size",
                        "32",
                        "--set-train-bags-per-material",
                        "4",
                        "--set-validation-bags-per-material",
                        "8",
                        "--set-pretrained-regressor",
                        str(checkpoint),
                        "--set-encoder-learning-rate-scale",
                        "0",
                        "--set-individual-loss-weight",
                        "0",
                        "--set-target-specific-pooling",
                        "--set-bag-seed",
                        str(args.bag_seed),
                    ],
                    run_dir / "set_transformer.log",
                    dry_run=args.dry_run,
                )
            run_status["set_transformer_complete"] = set_metrics.is_file()
            if set_metrics.is_file():
                run_status["validation"] = _validation_summary(set_metrics)
            status["updated_at_utc"] = _utc_now()
            _write_json(status_path, status)

        status.pop("active_split", None)
        status["status"] = "dry_run" if args.dry_run else "complete"
        status["completed_at_utc"] = _utc_now()
        status["updated_at_utc"] = status["completed_at_utc"]
        _write_json(status_path, status)
    except Exception as error:
        status["status"] = "failed"
        status["error"] = f"{type(error).__name__}: {error}"
        status["updated_at_utc"] = _utc_now()
        _write_json(status_path, status)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
