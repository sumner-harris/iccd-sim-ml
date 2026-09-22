from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from iccd_sim_ml.io import (
    H5Simulation,
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
    return raw


def test_array_reader_accepts_optional_mesh_level_and_flags_known_export_bug() -> None:
    corrupt = plasma_timestep_from_array(_frame(), source="synthetic.dat")
    assert corrupt.size == 4
    assert "corrupt_alpha_pi_and_mesh_level_export" in corrupt.quality_flags

    old = plasma_timestep_from_array(_frame(20), source="old.h5", source_key="sim/res_0_ns")
    assert np.isnan(old.mesh_level).all()
    assert old.source_key == "sim/res_0_ns"


def test_hdf5_index_sorts_physical_times_and_ignores_diagnostics(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "tiny.h5"
    with h5py.File(path, "w") as handle:
        group = handle.create_group("Cu_group1_0")
        group.attrs["rspot"] = 1.0e-3
        group.attrs["laser_power_wcm"] = 2.0e8
        group.create_dataset("metal", data=np.ones(2))
        group.create_dataset("res_1000_ns", data=_frame())
        group.create_dataset("res_2.5_ns.dat", data=_frame(20))
        group.create_dataset("res_0_ns", data=_frame())
    simulations = list_h5_simulations(path)
    assert len(simulations) == 1
    assert simulations[0].timestep_keys == (
        "res_0_ns",
        "res_2.5_ns.dat",
        "res_1000_ns",
    )
    assert np.allclose(simulations[0].times_s, [0.0, 2.5e-9, 1.0e-6])
    assert len(list(simulations[0].iter_timesteps())) == 3


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
