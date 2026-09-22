"""Physics-first synthetic ICCD imaging and ML utilities."""

from .imaging.config import ImagingConfig
from .imaging.simulator import ImageSimulation, simulate_continuum_image

__all__ = ["ImageSimulation", "ImagingConfig", "simulate_continuum_image"]
__version__ = "0.1.0"
