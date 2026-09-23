from types import SimpleNamespace

import numpy as np
import pytest

from iccd_sim_ml.pipeline.fov import (
    bracketing_frame_indices,
    select_maximum_condition,
    summarize_radiance_extent,
)


def _simulation(key, power, radius, stop=20.0, frames=3):
    return SimpleNamespace(
        key=key,
        attrs={"laser_power_wcm": power, "rspot": radius},
        times_s=np.linspace(0.0, stop, frames),
        timestep_keys=tuple(str(index) for index in range(frames)),
    )


def test_maximum_condition_is_joint_and_prefers_time_coverage():
    simulations = (
        _simulation("low", 1.0, 5.0),
        _simulation("power", 2.0, 4.0, stop=10.0),
        _simulation("winner", 2.0, 4.0, stop=20.0),
    )
    assert select_maximum_condition(simulations).key == "winner"


def test_bracketing_indices_exact_interpolated_and_outside():
    times = np.asarray([0.0, 2.0, 5.0])
    assert bracketing_frame_indices(times, 2.0) == (1, 1, 0.0)
    assert bracketing_frame_indices(times, 3.5) == (1, 2, 0.5)
    with pytest.raises(ValueError, match="outside source coverage"):
        bracketing_frame_indices(times, 6.0)


def test_radiance_extent_reports_containment_support_and_edges():
    x = np.asarray([-2.0, -1.0, 1.0, 2.0])
    z = np.asarray([0.0, 1.0, 2.0, 3.0])
    image = np.zeros((4, 4))
    image[1:3, 1] = 1.0
    result = summarize_radiance_extent(
        image,
        x,
        z,
        containment=(0.99,),
        relative_thresholds=(1.0e-2,),
        edge_pixels=1,
    )
    assert result.radial_containment_m["99%"] == 1.0
    assert result.axial_containment_m["99%"] == 1.0
    assert result.radial_relative_support_m["0.01"] == 1.0
    assert result.axial_relative_support_m["0.01"] == 1.0
    assert result.radial_edge_fraction == 0.0
    assert result.axial_outer_edge_fraction == 0.0
    assert result.nonzero_pixel_fraction == 0.125
