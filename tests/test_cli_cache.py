from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from iccd_sim_ml.cli import _cache_key, _sequence_cache_hit


def test_sequence_cache_requires_matching_fingerprint_and_valid_product(tmp_path: Path) -> None:
    request = {"source": "Cu_6.h5", "frames": ["res_0_ns"], "config": {"radial": 24}}
    key = _cache_key(request)
    path = tmp_path / "cached.npz"
    np.savez_compressed(
        path,
        video=np.ones((1, 4, 5)),
        times_s=np.asarray([0.0]),
        x_m=np.arange(4),
        z_m=np.arange(5),
        metadata_json=json.dumps({"cache_key": key}),
    )

    assert _sequence_cache_hit(path, key, 1)
    assert not _sequence_cache_hit(path, "different", 1)
    assert not _sequence_cache_hit(path, key, 2)


def test_sequence_cache_rejects_nonfinite_or_incomplete_products(tmp_path: Path) -> None:
    key = _cache_key({"request": 1})
    nonfinite = tmp_path / "nonfinite.npz"
    np.savez_compressed(
        nonfinite,
        video=np.full((1, 2, 2), np.nan),
        times_s=np.asarray([0.0]),
        x_m=np.arange(2),
        z_m=np.arange(2),
        metadata_json=json.dumps({"cache_key": key}),
    )
    incomplete = tmp_path / "incomplete.npz"
    np.savez_compressed(incomplete, video=np.ones((1, 2, 2)))

    assert not _sequence_cache_hit(nonfinite, key, 1)
    assert not _sequence_cache_hit(incomplete, key, 1)
