"""Holography dataset generation toolkit."""

from .converters import (
    HologramDatasetGenerator,
    ObjectDomainProducer,
    ObjectToHologramConverter,
)
from .phase import PhaseGenerationConfig
from .shapes import CircleGenerator, RectangleGenerator, RingGenerator
from .types import (
    GridSpec,
    HolographyConfig,
    HolographyMethod,
    OffAxisCarrier,
    OpticalConfig,
)
from .utils.io import NumpyDatasetWriter

__all__ = [
    # Types
    "GridSpec",
    "OpticalConfig",
    "OffAxisCarrier",
    "HolographyConfig",
    "HolographyMethod",
    # Phase Generation
    "PhaseGenerationConfig",
    # Converters
    "ObjectDomainProducer",
    "ObjectToHologramConverter",
    "HologramDatasetGenerator",
    # Shapes
    "CircleGenerator",
    "RectangleGenerator",
    "RingGenerator",
    # IO
    "NumpyDatasetWriter",
]
