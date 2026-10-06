from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_script():
    path = Path(__file__).parents[1] / "scripts" / "create_repeated_element_splits.py"
    spec = importlib.util.spec_from_file_location("create_repeated_element_splits", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_balanced_validation_sets_are_unique_and_balanced() -> None:
    module = _load_script()
    elements = tuple(f"E{index:02d}" for index in range(37))
    first = module.balanced_validation_sets(elements, repeats=10, validation_count=11, seed=42)
    second = module.balanced_validation_sets(elements, repeats=10, validation_count=11, seed=42)

    assert first == second
    assert len(first) == len(set(first)) == 10
    assert all(len(split) == len(set(split)) == 11 for split in first)
    frequencies = {element: sum(element in split for split in first) for element in elements}
    assert set(frequencies.values()) == {2, 3}
