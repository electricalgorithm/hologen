"""Core type definitions for holography dataset generation."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Protocol

import numpy as np
import numpy.typing as npt
from numpy.random import Generator

ArrayFloat = npt.NDArray[np.float64]
ArrayComplex = npt.NDArray[np.complex128]


class HolographyMethod(str, Enum):
    """Enumeration of supported holography sampling approaches."""

    INLINE = "inline"
    OFF_AXIS = "off_axis"


@dataclass(slots=True)
class GridSpec:
    """Sampling definition for a two-dimensional grid.

    Args:
        height: Number of pixels along the vertical axis.
        width: Number of pixels along the horizontal axis.
        pixel_pitch: Sampling interval between adjacent pixels in meters.
    """

    height: int
    width: int
    pixel_pitch: float


@dataclass(slots=True)
class OpticalConfig:
    """Physical parameters for hologram generation and reconstruction.

    Args:
        wavelength: Illumination wavelength in meters.
        propagation_distance: Distance between the object and sensor planes in meters.
    """

    wavelength: float
    propagation_distance: float


@dataclass(slots=True)
class OffAxisCarrier:
    """Carrier modulation parameters for off-axis holography.

    Args:
        frequency_x: Spatial carrier frequency along the horizontal axis in cycles per meter.
        frequency_y: Spatial carrier frequency along the vertical axis in cycles per meter.
        gaussian_width: Standard deviation of the Gaussian filter used during reconstruction in cycles per meter.
    """

    frequency_x: float
    frequency_y: float
    gaussian_width: float


@dataclass(slots=True)
class HolographyConfig:
    """Configuration bundle for holography strategies.

    Args:
        grid: Spatial sampling description of the sensor plane.
        optics: Optical parameters governing propagation.
        method: Holography method to employ for conversions.
        carrier: Optional carrier configuration for off-axis holography.
    """

    grid: GridSpec
    optics: OpticalConfig
    method: HolographyMethod
    carrier: OffAxisCarrier | None = None


@dataclass(slots=True)
class ObjectSample:
    """Object-domain sample representation.

    Args:
        name: Identifier of the shape generator that produced the sample.
        pixels: Binary amplitude distribution of the object domain.
    """

    name: str
    pixels: ArrayFloat


@dataclass(slots=True)
class HologramSample:
    """Full holography sample including forward and reconstructed domains.

    Args:
        object_sample: Reference to the originating object-domain sample.
        hologram: Intensity hologram generated from the object sample.
        reconstruction: Recovered object-domain amplitude derived from the hologram.
    """

    object_sample: ObjectSample
    hologram: ArrayFloat
    reconstruction: ArrayFloat


class ObjectShapeGenerator(Protocol):
    """Protocol for object-domain shape generators."""

    @property
    def name(self) -> str:
        """Return the canonical name of the generator."""

    def generate(self, grid: GridSpec, rng: Generator) -> ArrayFloat:
        """Create a binary object-domain image."""


class HolographyStrategy(Protocol):
    """Protocol describing hologram generation and reconstruction operations."""

    def create_hologram(
        self, object_field: ArrayFloat, config: HolographyConfig
    ) -> ArrayFloat:
        """Create a hologram from an object-domain amplitude field."""

    def reconstruct(self, hologram: ArrayFloat, config: HolographyConfig) -> ArrayFloat:
        """Recover an object-domain field from a hologram."""


class DatasetWriter(Protocol):
    """Protocol for persisting generated holography samples."""

    def save(self, samples: Iterable[HologramSample], output_dir: Path) -> None:
        """Persist a sequence of hologram samples."""


class DatasetGenerator(Protocol):
    """Protocol for dataset generation routines."""

    def generate(
        self, count: int, config: HolographyConfig, rng: Generator
    ) -> Iterable[HologramSample]:
        """Produce holography samples."""
