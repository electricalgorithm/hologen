import dataclasses
import pathlib

@dataclasses.dataclass
class NoiseConfig:
    """Configuration for noise parameters in hologram generation."""

    speckle: bool
    shot: bool
    read: bool
    dark: bool
    speckle_strength: float | None = None
    speckle_roughness: float | None = None
    read_noise_sigma: float | None = None
    dark_noise_mean: float | None = None

    def __post_init__(self):
        """Validate noise configuration."""
        # Check the conditions.
        if self.speckle and not (self.speckle_roughness or self.speckle_strength):
            raise ValueError("If speckle is enabled, its properties must be provided.")
        elif self.speckle and (self.speckle_strength < 0 or self.speckle_roughness <= 0):
            raise ValueError("speckle_strength must be non-negative or speckle_roughness must be positive.")

        if self.read and not self.read_noise_sigma:
            raise ValueError("If read noise is enabled, its properties must be provided.")
        elif self.read and self.read_noise_sigma < 0:
            raise ValueError("read_noise_sigma must be non-negative.")

        if self.dark and not self.dark_noise_mean:
            raise ValueError("If dark current noise is enabled, its properties must be provided.")
        elif self.dark and self.dark_noise_mean < 0:
            raise ValueError("dark_noise_mean must be non-negative.")


@dataclasses.dataclass
class HologramConfig:
    """Configuration for hologram generation parameters."""

    resolution: int    # pixels
    pixel_size: float  # microns
    wavelength: float  # microns
    z_distance: float  # microns
    bit_depth: int
    num_cells: int


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