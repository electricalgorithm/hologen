"""Settings of the simulator (HologramConfig), the noise and the dataset output."""

import dataclasses
import pathlib
from typing import Final

import numpy

from hologen.inline_simulator import (
    DEFAULT_BACKGROUND_ELECTRONS,
    DEFAULT_SATURATION_ELECTRONS,
)

# Second seed word of the distance draw, so it never shares a stream with the
# objects and the noise of the same hologram.
DISTANCE_STREAM: Final[int] = 7


@dataclasses.dataclass
class NoiseConfig:
    """Configuration for noise parameters in hologram generation."""

    speckle: bool
    shot: bool
    read: bool
    dark: bool
    # TODO(speckle-interval): accept a (min, max) interval and sample the contrast
    # per hologram, e.g. between the LED (0.014) and laser (0.12) anchors, so the
    # model sees the range of coherence found in real setups.
    speckle_contrast: float | None = None  # std/mean of the speckle intensity
    speckle_grain_size: float | None = None  # microns
    read_noise_sigma: float | None = None  # e- rms
    dark_noise_mean: float | None = None  # e- per pixel per exposure

    def __post_init__(self):
        """Validate noise configuration."""
        # Check the conditions.
        if self.speckle and (
            self.speckle_contrast is None or self.speckle_grain_size is None
        ):
            raise ValueError("If speckle is enabled, its properties must be provided.")
        elif self.speckle and not (
            0 < self.speckle_contrast <= 1 and self.speckle_grain_size > 0
        ):
            raise ValueError(
                "speckle_contrast must be in (0, 1] and speckle_grain_size must be "
                "positive."
            )

        if self.read and not self.read_noise_sigma:
            raise ValueError(
                "If read noise is enabled, its properties must be provided."
            )
        elif self.read and self.read_noise_sigma < 0:
            raise ValueError("read_noise_sigma must be non-negative.")

        if self.dark and not self.dark_noise_mean:
            raise ValueError(
                "If dark current noise is enabled, its properties must be provided."
            )
        elif self.dark and self.dark_noise_mean < 0:
            raise ValueError("dark_noise_mean must be non-negative.")


@dataclasses.dataclass
class HologramConfig:
    """Configuration for hologram generation parameters."""

    resolution: int  # pixels
    pixel_size: float  # microns
    wavelength: float  # microns
    z_distance: float  # microns
    bit_depth: int
    num_cells: int
    # Mean photo-electrons a pixel collects from the unit-intensity (|U|^2 = 1)
    # illumination in one exposure.
    background_electrons: float = DEFAULT_BACKGROUND_ELECTRONS  # e- per pixel
    # Charge at which the ADC output reaches its largest value 2^bit_depth - 1.
    saturation_electrons: float = DEFAULT_SATURATION_ELECTRONS  # e- per pixel
    # NA of the imaging objective of a magnified setup; None for a lensless one.
    numerical_aperture: float | None = None
    # The hologram is recorded at z_distance + z_offset + U(-z_error, z_error) and
    # back-propagated at z_distance, the distance a user would assume.
    z_offset: float = 0.0  # microns
    z_error: float = 0.0  # microns

    def true_distance(self, seed: int) -> float:
        """
        Draw the distance at which one hologram is recorded.

        The random error comes from its own generator, seeded with the
        hologram's seed and DISTANCE_STREAM, so the shared generator is not
        touched. A hologram therefore keeps its objects and its speckle pattern
        whatever the error. The camera noise is drawn from the same seed, but
        Poisson sampling consumes a number of random values that depends on
        the intensity, so its realisation changes with the distance while its
        statistics do not.

        Args:
            seed (int): Seed of the hologram.

        Returns:
            float: The recording distance, in microns.
        """
        distance: float = self.z_distance + self.z_offset
        if self.z_error == 0:
            return distance
        rng: numpy.random.Generator = numpy.random.default_rng([seed, DISTANCE_STREAM])
        return distance + float(rng.uniform(-self.z_error, self.z_error))


@dataclasses.dataclass
class DatasetConfig:
    """Configuration for dataset generation."""

    output_dir: str | pathlib.Path
    base_seed: int
    hologram_config: HologramConfig
    generate_visualization: bool = True
    # The back-propagated field can be recomputed from the hologram, so it is
    # only stored on request.
    write_reconstructed: bool = False

    def __post_init__(self):
        """Convert output_dir to Path if needed."""
        if isinstance(self.output_dir, str):
            self.output_dir = pathlib.Path(self.output_dir)
