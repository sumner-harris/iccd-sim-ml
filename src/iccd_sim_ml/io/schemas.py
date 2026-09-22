"""Explicit schemas for the adaptive-mesh plasma exports.

The solver calls its first two coordinates ``x`` and ``y``. In the
2D-axisymmetric simulations used here they mean axial ``z`` and radial ``r``.
All coordinates are in metres, number densities are in m^-3, temperature is K,
and absorption coefficients are in m^-1.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ColumnSpec:
    index: int
    name: str
    unit: str
    description: str


PLASMA_DAT_COLUMNS = (
    ColumnSpec(0, "z", "m", "Axial coordinate; solver header name x"),
    ColumnSpec(1, "r", "m", "Radial coordinate; solver header name y"),
    ColumnSpec(2, "density_number", "m^-3", "Total particle number density"),
    ColumnSpec(3, "mass_density", "kg m^-3", "Mass density"),
    ColumnSpec(4, "temperature", "K", "Plasma temperature"),
    ColumnSpec(5, "velocity_z", "m s^-1", "Axial velocity"),
    ColumnSpec(6, "velocity_r", "m s^-1", "Radial velocity"),
    ColumnSpec(7, "laser_irradiance", "solver", "Incident laser irradiance"),
    ColumnSpec(8, "laser_absorption", "solver", "Absorbed laser source term"),
    ColumnSpec(9, "n0", "m^-3", "Neutral number density"),
    ColumnSpec(10, "ne", "m^-3", "Electron number density"),
    ColumnSpec(11, "n1", "m^-3", "Singly ionized number density"),
    ColumnSpec(12, "n2", "m^-3", "Doubly ionized number density"),
    ColumnSpec(13, "x0", "1", "Neutral fraction"),
    ColumnSpec(14, "xe", "1", "Electron fraction"),
    ColumnSpec(15, "x1", "1", "Singly ionized fraction"),
    ColumnSpec(16, "x2", "1", "Doubly ionized fraction"),
    ColumnSpec(17, "alpha_ib_en_248", "m^-1", "Electron-neutral IB opacity at 248 nm"),
    ColumnSpec(18, "alpha_ib_ei_248", "m^-1", "Electron-ion IB opacity at 248 nm"),
    ColumnSpec(19, "alpha_pi_248", "m^-1", "Photoionization opacity at 248 nm"),
    ColumnSpec(20, "mesh_level", "1", "Adaptive mesh level"),
)

PLASMA_DAT_COLUMN_COUNT = len(PLASMA_DAT_COLUMNS)
PLASMA_DAT_REQUIRED_COLUMN_COUNT = 20
