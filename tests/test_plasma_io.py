from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from iccd_sim_ml.io import (
    H5Simulation,
    get_h5_simulation,
    list_h5_simulations,
    plasma_timestep_from_array,
    select_representative_simulation,
)


def _frame(columns: int = 21) -> np.ndarray:
    raw = np.zeros((4, columns), dtype=np.float64)
    raw[:, 0] = [0.025, 0.075, 0.025, 0.075]
    raw[:, 1] = [0.025, 0.025, 0.075, 0.075]
    raw[:, 2] = 1.0e20
    raw[:, 3] = 1.0e-5
    raw[:, 4] = 5000.0
    raw[:, 9] = 9.0e19
    raw[:, 10] = 1.0e18
    raw[:, 11] = 1.0e18
    raw[:, 12] = 0.0
    raw[:, 17] = 1.0e-4
    raw[:, 18] = 2.0e-4
    raw[:, 19] = 0.01123047
    if columns == 21:
        raw[:, 20] = 33.0
    elif columns >= 22:
        raw[:, 20] = [-10.5, 3.25, 100.0, -0.75]  # radial velocity, not AMR level
        raw[:, 21] = [5.0, 6.0, 7.0, 8.0]
    return raw


def test_array_reader_accepts_optional_mesh_level_and_flags_known_export_bug() -> None:
    corrupt = plasma_timestep_from_array(_frame(), source="synthetic.dat")
    assert corrupt.size == 4
    assert "corrupt_alpha_pi_and_mesh_level_export" in corrupt.quality_flags

    old = plasma_timestep_from_array(_frame(20), source="old.h5", source_key="sim/res_0_ns")
    assert np.isnan(old.mesh_level).all()
    assert old.source_key == "sim/res_0_ns"

    extended = plasma_timestep_from_array(_frame(34), source="extended.h5", mesh_level_index=21)
    np.testing.assert_array_equal(extended.mesh_level, [5.0, 6.0, 7.0, 8.0])
    assert not extended.quality_flags


def test_hdf5_index_sorts_physical_times_and_ignores_diagnostics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "tiny.h5"
    with h5py.File(path, "w") as handle:
        group = handle.create_group("Cu_group1_0")
        group.attrs["rspot"] = 1.0e-3
        group.attrs["laser_power_wcm"] = 2.0e8
        group.create_dataset("metal", data=np.ones(2))
        group.create_dataset("res_1000_ns", data=_frame(34))
        group.create_dataset("res_2.5_ns.dat", data=_frame(20))
        group.create_dataset("res_0_ns", data=_frame(34))

    def reject_resolve(*args: object, **kwargs: object) -> None:
        raise AssertionError("HDF5 paths must not be resolved through a network-file handle")

    monkeypatch.setattr(Path, "resolve", reject_resolve)
    simulations = list_h5_simulations(path)
    assert len(simulations) == 1
    assert simulations[0].timestep_keys == (
        "res_0_ns",
        "res_1000_ns",
    )
    assert np.allclose(simulations[0].times_s, [0.0, 1.0e-6])
    assert "exactly 34 columns" in simulations[0].index_warnings[0]
    frames = list(simulations[0].iter_timesteps())
    assert len(frames) == 2
    np.testing.assert_array_equal(frames[0].mesh_level, [5.0, 6.0, 7.0, 8.0])

    named = get_h5_simulation(path, "Cu_group1_0")
    assert named.key == "Cu_group1_0"
    assert named.timestep_keys == simulations[0].timestep_keys


def test_representative_selector_prefers_midrange_laser_condition() -> None:
    simulations = []
    for index, (radius, power, frames) in enumerate(
        ((0.5e-3, 1.0e7, 20), (1.0e-3, 1.0e8, 12), (2.0e-3, 1.0e9, 30))
    ):
        simulations.append(
            H5Simulation(
                source=Path("Cu_6.h5"),
                key=f"condition_{index}",
                attrs={"rspot": radius, "laser_power_wcm": power},
                timestep_keys=tuple(f"res_{frame}_ns" for frame in range(frames)),
                times_s=np.arange(frames, dtype=np.float64) * 1e-9,
            )
        )
    assert select_representative_simulation(simulations).key == "condition_1"
