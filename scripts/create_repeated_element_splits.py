#!/usr/bin/env python3
"""Create balanced repeated element-held-out train/validation manifests."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from iccd_sim_ml.data import DatasetManifest, SplitManifest, validate_split


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--validation-fraction", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def balanced_validation_sets(
    elements: tuple[str, ...],
    *,
    repeats: int,
    validation_count: int,
    seed: int,
) -> tuple[tuple[str, ...], ...]:
    """Randomize unique sets while balancing validation frequency by element."""

    if repeats < 1:
        raise ValueError("repeats must be positive")
    if not 0 < validation_count < len(elements):
        raise ValueError("validation_count must leave non-empty train and validation sets")
    rng = np.random.default_rng(seed)
    counts = Counter({element: 0 for element in elements})
    seen: set[tuple[str, ...]] = set()
    result: list[tuple[str, ...]] = []
    for _ in range(repeats):
        for _attempt in range(10_000):
            jitter = dict(zip(elements, rng.random(len(elements)), strict=True))
            ordered = sorted(elements, key=lambda element: (counts[element], jitter[element]))
            selected = tuple(sorted(ordered[:validation_count]))
            if selected not in seen:
                break
        else:  # pragma: no cover - impossible for practical element inventories
            raise RuntimeError("Could not construct another unique validation combination")
        seen.add(selected)
        result.append(selected)
        counts.update(selected)
    return tuple(result)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 0.0 < args.validation_fraction < 1.0:
        raise ValueError("validation_fraction must lie strictly between zero and one")
    manifest = DatasetManifest.load(args.manifest)
    elements = tuple(sorted({record.element for record in manifest.records}))
    validation_count = int(round(args.validation_fraction * len(elements)))
    validation_sets = balanced_validation_sets(
        elements,
        repeats=args.repeats,
        validation_count=validation_count,
        seed=args.seed,
    )
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    validation_frequency: Counter[str] = Counter()
    for index, validation_elements in enumerate(validation_sets, start=1):
        validation_lookup = set(validation_elements)
        train_elements = tuple(element for element in elements if element not in validation_lookup)
        train_ids = tuple(
            record.sample_id for record in manifest.records if record.element in train_elements
        )
        validation_ids = tuple(
            record.sample_id for record in manifest.records if record.element in validation_lookup
        )
        split = SplitManifest(
            train=train_ids,
            validation=validation_ids,
            test=(),
            seed=args.seed + index - 1,
            ratios=(1.0 - args.validation_fraction, args.validation_fraction, 0.0),
            group_fields=("element",),
        )
        validate_split(manifest, split, group_fields=("element",))
        destination = output_dir / f"split-{index:02d}.json"
        split.save(destination)
        validation_frequency.update(validation_elements)
        rows.append(
            {
                "index": index,
                "seed": split.seed,
                "split_file": destination.name,
                "train_elements": list(train_elements),
                "validation_elements": list(validation_elements),
                "train_samples": len(train_ids),
                "validation_samples": len(validation_ids),
                "test_samples": 0,
            }
        )

    combinations = [tuple(row["validation_elements"]) for row in rows]
    if len(combinations) != len(set(combinations)):
        raise AssertionError("Validation combinations are not unique")
    campaign = {
        "schema_version": 1,
        "manifest": str(args.manifest.expanduser().resolve()),
        "generation_seed": args.seed,
        "requested_train_fraction": 1.0 - args.validation_fraction,
        "requested_validation_fraction": args.validation_fraction,
        "element_count": len(elements),
        "train_elements_per_split": len(elements) - validation_count,
        "validation_elements_per_split": validation_count,
        "test_elements_per_split": 0,
        "unique_validation_combinations": True,
        "validation_frequency_by_element": dict(sorted(validation_frequency.items())),
        "splits": rows,
    }
    (output_dir / "campaign.json").write_text(
        json.dumps(campaign, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(campaign, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
