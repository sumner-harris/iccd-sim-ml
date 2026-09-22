"""Idealized continuum image formation without spatial or sensor effects."""

from .config import ImagingConfig
from .simulator import (
    ImageSimulation,
    SequenceSimulation,
    simulate_continuum_image,
    simulate_continuum_sequence,
)

__all__ = [
    "ImageSimulation",
    "ImagingConfig",
    "SequenceSimulation",
    "simulate_continuum_image",
    "simulate_continuum_sequence",
]
