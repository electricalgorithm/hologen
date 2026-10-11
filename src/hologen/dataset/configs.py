import dataclasses
import pathlib

from hologen.inline_simulator import (
    DEFAULT_BACKGROUND_ELECTRONS,
    DEFAULT_SATURATION_ELECTRONS,
)


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
                "speckle_contrast must be in (0, 1] and speckle_grain_size must be positive."
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


@dataclasses.dataclass
class DatasetConfig:
    """Configuration for dataset generation."""

    output_dir: str | pathlib.Path
    base_seed: int
    hologram_config: HologramConfig
    generate_visualization: bool = True

    def __post_init__(self):
        """Convert output_dir to Path if needed."""
        if isinstance(self.output_dir, str):
            self.output_dir = pathlib.Path(self.output_dir)
