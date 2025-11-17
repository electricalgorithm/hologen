"""Physics-based phase generation for realistic hologram synthesis.

This module implements phase generation based on optical principles using
refractive index distributions and object thickness. The phase is computed
using the fundamental optical equation:

    φ(x,y) = (2π/λ) × (n(x,y) - n₀) × d(x,y)

where:
    - λ is the wavelength
    - n(x,y) is the spatially-varying refractive index
    - n₀ is the ambient refractive index
    - d(x,y) is the object thickness distribution
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.random import Generator
from scipy.ndimage import distance_transform_edt

from hologen.types import ArrayFloat, GridSpec

__all__ = [
    "PhaseGenerationConfig",
    "PhaseGenerator",
    "RefractiveIndexGenerator",
    "ThicknessGenerator",
    "UniformGenerator",
    "GaussianBlobGenerator",
    "PerlinNoiseGenerator",
    "ConstantThicknessGenerator",
    "GradientThicknessGenerator",
    "ShapeBasedThicknessGenerator",
]


@dataclass(slots=True)
class PhaseGenerationConfig:
    """Configuration for physics-based phase generation.

    This dataclass contains all parameters needed to control the generation
    of realistic phase distributions based on physical optics principles.

    Args:
        enabled: Enable physics-based phase generation.
        refractive_index_mode: Mode for generating n(x,y) field.
            Valid options: "uniform", "gaussian_blobs", "perlin_noise".
        thickness_mode: Mode for generating d(x,y) field.
            Valid options: "constant", "gradient", "shape_based".
        ambient_refractive_index: n₀ value (default 1.0 for air).
        refractive_index_range: (min, max) for n(x,y) values.
        thickness_range: (min, max) for d(x,y) in meters.
        correlation_coefficient: Amplitude-phase correlation in range [-1, 1].
        gaussian_blob_count: Number of blobs for gaussian_blobs mode.
        gaussian_blob_size_range: (min, max) blob radius in pixels.
        perlin_noise_scale: Scale parameter for Perlin noise.
        perlin_noise_octaves: Number of octaves for Perlin noise.
        gradient_direction: Direction angle for gradient thickness (radians).
        gradient_magnitude: Magnitude of thickness gradient.
    """

    enabled: bool = False
    refractive_index_mode: str = "uniform"
    thickness_mode: str = "constant"
    ambient_refractive_index: float = 1.0
    refractive_index_range: tuple[float, float] = (1.33, 1.55)
    thickness_range: tuple[float, float] = (1e-6, 10e-6)
    correlation_coefficient: float = 0.0
    gaussian_blob_count: int = 5
    gaussian_blob_size_range: tuple[float, float] = (10.0, 50.0)
    perlin_noise_scale: float = 50.0
    perlin_noise_octaves: int = 4
    gradient_direction: float = 0.0
    gradient_magnitude: float = 0.5

    def __post_init__(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If any configuration parameter is invalid.
        """
        # Validate refractive index mode
        valid_ri_modes = {"uniform", "gaussian_blobs", "perlin_noise"}
        if self.refractive_index_mode not in valid_ri_modes:
            raise ValueError(
                f"Invalid refractive_index_mode '{self.refractive_index_mode}'. "
                f"Valid options are: {', '.join(sorted(valid_ri_modes))}"
            )

        # Validate thickness mode
        valid_thickness_modes = {"constant", "gradient", "shape_based"}
        if self.thickness_mode not in valid_thickness_modes:
            raise ValueError(
                f"Invalid thickness_mode '{self.thickness_mode}'. "
                f"Valid options are: {', '.join(sorted(valid_thickness_modes))}"
            )

        # Validate refractive index range
        if len(self.refractive_index_range) != 2:
            raise ValueError(
                f"refractive_index_range must be a tuple of 2 values, "
                f"got {len(self.refractive_index_range)}"
            )
        if self.refractive_index_range[0] >= self.refractive_index_range[1]:
            raise ValueError(
                f"refractive_index_range min ({self.refractive_index_range[0]}) "
                f"must be less than max ({self.refractive_index_range[1]})"
            )
        if self.refractive_index_range[0] < 1.0:
            raise ValueError(
                f"refractive_index_range min ({self.refractive_index_range[0]}) "
                f"must be >= 1.0 (physically realistic)"
            )
        if self.refractive_index_range[1] > 3.0:
            raise ValueError(
                f"refractive_index_range max ({self.refractive_index_range[1]}) "
                f"must be <= 3.0 (physically realistic)"
            )

        # Validate thickness range
        if len(self.thickness_range) != 2:
            raise ValueError(
                f"thickness_range must be a tuple of 2 values, "
                f"got {len(self.thickness_range)}"
            )
        if self.thickness_range[0] >= self.thickness_range[1]:
            raise ValueError(
                f"thickness_range min ({self.thickness_range[0]}) "
                f"must be less than max ({self.thickness_range[1]})"
            )
        if self.thickness_range[0] < 0:
            raise ValueError(
                f"thickness_range min ({self.thickness_range[0]}) must be >= 0"
            )

        # Validate correlation coefficient
        if not -1.0 <= self.correlation_coefficient <= 1.0:
            raise ValueError(
                f"correlation_coefficient ({self.correlation_coefficient}) "
                f"must be in range [-1.0, 1.0]"
            )

        # Validate gaussian blob parameters
        if self.gaussian_blob_count < 1:
            raise ValueError(
                f"gaussian_blob_count ({self.gaussian_blob_count}) must be >= 1"
            )
        if len(self.gaussian_blob_size_range) != 2:
            raise ValueError(
                f"gaussian_blob_size_range must be a tuple of 2 values, "
                f"got {len(self.gaussian_blob_size_range)}"
            )
        if self.gaussian_blob_size_range[0] >= self.gaussian_blob_size_range[1]:
            raise ValueError(
                f"gaussian_blob_size_range min ({self.gaussian_blob_size_range[0]}) "
                f"must be less than max ({self.gaussian_blob_size_range[1]})"
            )
        if self.gaussian_blob_size_range[0] <= 0:
            raise ValueError(
                f"gaussian_blob_size_range min ({self.gaussian_blob_size_range[0]}) "
                f"must be > 0"
            )

        # Validate perlin noise parameters
        if self.perlin_noise_scale <= 0:
            raise ValueError(
                f"perlin_noise_scale ({self.perlin_noise_scale}) must be > 0"
            )
        if self.perlin_noise_octaves < 1:
            raise ValueError(
                f"perlin_noise_octaves ({self.perlin_noise_octaves}) must be >= 1"
            )

        # Validate gradient parameters
        if not 0.0 <= self.gradient_magnitude <= 1.0:
            raise ValueError(
                f"gradient_magnitude ({self.gradient_magnitude}) "
                f"must be in range [0.0, 1.0]"
            )


class RefractiveIndexGenerator(Protocol):
    """Protocol for refractive index field generators.

    Implementations generate spatially-varying refractive index distributions
    n(x,y) that represent material properties of the object.
    """

    def generate(
        self,
        grid: GridSpec,
        mask: ArrayFloat,
        rng: Generator,
        config: PhaseGenerationConfig,
    ) -> ArrayFloat:
        """Generate spatially-varying refractive index field.

        Args:
            grid: Grid specification describing the output resolution.
            mask: Binary object mask (1.0 inside object, 0.0 outside).
            rng: Random number generator for stochastic parameters.
            config: Phase generation configuration.

        Returns:
            Refractive index field n(x,y) with ambient value outside mask.
        """
        ...


class ThicknessGenerator(Protocol):
    """Protocol for thickness field generators.

    Implementations generate object thickness distributions d(x,y) that
    represent the geometric depth of the object.
    """

    def generate(
        self,
        grid: GridSpec,
        mask: ArrayFloat,
        rng: Generator,
        config: PhaseGenerationConfig,
    ) -> ArrayFloat:
        """Generate object thickness distribution d(x,y).

        Args:
            grid: Grid specification describing the output resolution.
            mask: Binary object mask (1.0 inside object, 0.0 outside).
            rng: Random number generator for stochastic parameters.
            config: Phase generation configuration.

        Returns:
            Thickness field d(x,y) in meters, zero outside mask.
        """
        ...


class UniformGenerator:
    """Generate uniform refractive index within object boundaries."""

    def generate(
        self,
        grid: GridSpec,
        mask: ArrayFloat,
        rng: Generator,
        config: PhaseGenerationConfig,
    ) -> ArrayFloat:
        """Generate uniform refractive index field.

        Args:
            grid: Grid specification describing the output resolution.
            mask: Binary object mask (1.0 inside object, 0.0 outside).
            rng: Random number generator for stochastic parameters.
            config: Phase generation configuration.

        Returns:
            Refractive index field with uniform value inside mask,
            ambient value outside.
        """
        # Sample single refractive index value from configured range
        n_min, n_max = config.refractive_index_range
        n_object = rng.uniform(n_min, n_max)

        # Create field with ambient refractive index everywhere
        n_field = np.full((grid.height, grid.width), config.ambient_refractive_index)

        # Set sampled value inside mask
        n_field = np.where(mask > 0.5, n_object, n_field)

        return n_field


class GaussianBlobGenerator:
    """Generate refractive index using overlapping Gaussian blobs."""

    def generate(
        self,
        grid: GridSpec,
        mask: ArrayFloat,
        rng: Generator,
        config: PhaseGenerationConfig,
    ) -> ArrayFloat:
        """Generate refractive index field with Gaussian blob variations.

        Args:
            grid: Grid specification describing the output resolution.
            mask: Binary object mask (1.0 inside object, 0.0 outside).
            rng: Random number generator for stochastic parameters.
            config: Phase generation configuration.

        Returns:
            Refractive index field with smooth blob-like variations inside mask,
            ambient value outside.
        """
        # Initialize field with ambient refractive index
        n_field = np.full((grid.height, grid.width), config.ambient_refractive_index)

        # Get object positions (where mask is non-zero)
        object_positions = np.argwhere(mask > 0.5)
        if len(object_positions) == 0:
            return n_field

        # Create coordinate grids
        y_coords, x_coords = np.meshgrid(
            np.arange(grid.height), np.arange(grid.width), indexing="ij"
        )

        # Generate overlapping Gaussian blobs
        n_min, n_max = config.refractive_index_range
        blob_size_min, blob_size_max = config.gaussian_blob_size_range

        for _ in range(config.gaussian_blob_count):
            # Sample random position within object mask
            pos_idx = rng.integers(0, len(object_positions))
            center_y, center_x = object_positions[pos_idx]

            # Sample random blob size (radius)
            sigma = rng.uniform(blob_size_min, blob_size_max)

            # Sample random refractive index for this blob
            n_blob = rng.uniform(n_min, n_max)

            # Compute Gaussian blob: exp(-r²/2σ²)
            r_squared = (x_coords - center_x) ** 2 + (y_coords - center_y) ** 2
            gaussian = np.exp(-r_squared / (2 * sigma**2))

            # Add blob contribution: n += (n_blob - n₀) * gaussian
            n_field += (n_blob - config.ambient_refractive_index) * gaussian

        # Clip to refractive index range to ensure physical validity
        n_field = np.clip(n_field, n_min, n_max)

        # Apply mask: set to ambient value outside object
        n_field = np.where(mask > 0.5, n_field, config.ambient_refractive_index)

        return n_field


class PerlinNoiseGenerator:
    """Generate refractive index using Perlin noise."""

    def generate(
        self,
        grid: GridSpec,
        mask: ArrayFloat,
        rng: Generator,
        config: PhaseGenerationConfig,
    ) -> ArrayFloat:
        """Generate refractive index field using Perlin noise.

        Args:
            grid: Grid specification describing the output resolution.
            mask: Binary object mask (1.0 inside object, 0.0 outside).
            rng: Random number generator for stochastic parameters.
            config: Phase generation configuration.

        Returns:
            Refractive index field with spatially-correlated noise inside mask,
            ambient value outside.
        """
        # Generate multi-octave Perlin noise
        noise = self._generate_perlin_noise(
            grid, config.perlin_noise_scale, config.perlin_noise_octaves, rng
        )

        # Normalize noise to [0, 1]
        noise_min = noise.min()
        noise_max = noise.max()
        if noise_max > noise_min:
            noise = (noise - noise_min) / (noise_max - noise_min)
        else:
            noise = np.zeros_like(noise)

        # Map to refractive index range
        n_min, n_max = config.refractive_index_range
        n_field = n_min + noise * (n_max - n_min)

        # Apply mask: set to ambient value outside object
        n_field = np.where(mask > 0.5, n_field, config.ambient_refractive_index)

        return n_field

    def _generate_perlin_noise(
        self, grid: GridSpec, scale: float, octaves: int, rng: Generator
    ) -> ArrayFloat:
        """Generate Perlin-like noise using multiple octaves of gradient noise.

        Args:
            grid: Grid specification describing the output resolution.
            scale: Base scale parameter controlling feature size.
            octaves: Number of octaves to combine.
            rng: Random number generator.

        Returns:
            Multi-octave noise field.
        """
        noise = np.zeros((grid.height, grid.width))
        amplitude = 1.0
        frequency = 1.0 / scale

        for _ in range(octaves):
            # Generate gradient noise at current frequency
            octave_noise = self._generate_gradient_noise(grid, frequency, rng)
            noise += amplitude * octave_noise

            # Update parameters for next octave
            amplitude *= 0.5
            frequency *= 2.0

        return noise

    def _generate_gradient_noise(
        self, grid: GridSpec, frequency: float, rng: Generator
    ) -> ArrayFloat:
        """Generate gradient noise at a specific frequency.

        This implements a simplified gradient noise algorithm similar to Perlin noise.

        Args:
            grid: Grid specification describing the output resolution.
            frequency: Frequency of the noise (inverse of feature size).
            rng: Random number generator.

        Returns:
            Gradient noise field.
        """
        # Determine grid size for gradient vectors
        grid_size = max(2, int(max(grid.height, grid.width) * frequency) + 2)

        # Generate random gradient vectors at grid points
        angles = rng.uniform(0, 2 * np.pi, size=(grid_size, grid_size))
        grad_x = np.cos(angles)
        grad_y = np.sin(angles)

        # Create coordinate grids
        y_coords = np.linspace(0, grid_size - 1, grid.height)
        x_coords = np.linspace(0, grid_size - 1, grid.width)
        y_grid, x_grid = np.meshgrid(y_coords, x_coords, indexing="ij")

        # Get integer and fractional parts
        x0 = np.floor(x_grid).astype(int)
        y0 = np.floor(y_grid).astype(int)
        x1 = np.minimum(x0 + 1, grid_size - 1)
        y1 = np.minimum(y0 + 1, grid_size - 1)

        # Fractional parts
        fx = x_grid - x0
        fy = y_grid - y0

        # Compute dot products at four corners
        def dot_grid_gradient(
            ix: ArrayFloat, iy: ArrayFloat, x: float, y: float
        ) -> ArrayFloat:
            dx = x - ix
            dy = y - iy
            return grad_x[iy, ix] * dx + grad_y[iy, ix] * dy

        n00 = dot_grid_gradient(x0, y0, x_grid, y_grid)
        n10 = dot_grid_gradient(x1, y0, x_grid, y_grid)
        n01 = dot_grid_gradient(x0, y1, x_grid, y_grid)
        n11 = dot_grid_gradient(x1, y1, x_grid, y_grid)

        # Smooth interpolation (fade function: 6t^5 - 15t^4 + 10t^3)
        sx = 6 * fx**5 - 15 * fx**4 + 10 * fx**3
        sy = 6 * fy**5 - 15 * fy**4 + 10 * fy**3

        # Bilinear interpolation
        nx0 = n00 * (1 - sx) + n10 * sx
        nx1 = n01 * (1 - sx) + n11 * sx
        noise = nx0 * (1 - sy) + nx1 * sy

        return noise


class ConstantThicknessGenerator:
    """Generate uniform thickness within object boundaries."""

    def generate(
        self,
        grid: GridSpec,
        mask: ArrayFloat,
        rng: Generator,
        config: PhaseGenerationConfig,
    ) -> ArrayFloat:
        """Generate constant thickness field.

        Args:
            grid: Grid specification describing the output resolution.
            mask: Binary object mask (1.0 inside object, 0.0 outside).
            rng: Random number generator for stochastic parameters.
            config: Phase generation configuration.

        Returns:
            Thickness field with uniform value inside mask, zero outside.
        """
        # Sample uniform thickness from configured range
        d_min, d_max = config.thickness_range
        thickness_value = rng.uniform(d_min, d_max)

        # Create field with zero thickness everywhere
        d_field = np.zeros((grid.height, grid.width))

        # Set sampled thickness value inside mask
        d_field = np.where(mask > 0.5, thickness_value, 0.0)

        return d_field


class GradientThicknessGenerator:
    """Generate thickness with linear gradient."""

    def generate(
        self,
        grid: GridSpec,
        mask: ArrayFloat,
        rng: Generator,
        config: PhaseGenerationConfig,
    ) -> ArrayFloat:
        """Generate thickness field with linear gradient.

        Args:
            grid: Grid specification describing the output resolution.
            mask: Binary object mask (1.0 inside object, 0.0 outside).
            rng: Random number generator for stochastic parameters.
            config: Phase generation configuration.

        Returns:
            Thickness field with smooth gradient inside mask, zero outside.
        """
        # Create coordinate grids
        y_coords, x_coords = np.meshgrid(
            np.arange(grid.height), np.arange(grid.width), indexing="ij"
        )

        # Compute gradient: g = x*cos(θ) + y*sin(θ)
        theta = config.gradient_direction
        gradient = x_coords * np.cos(theta) + y_coords * np.sin(theta)

        # Normalize gradient to [0, 1] within mask bounds
        if mask.sum() > 0:
            # Get gradient values only within the mask
            masked_gradient = gradient * mask
            gradient_min = masked_gradient[mask > 0.5].min()
            gradient_max = masked_gradient[mask > 0.5].max()

            if gradient_max > gradient_min:
                normalized_gradient = (gradient - gradient_min) / (
                    gradient_max - gradient_min
                )
            else:
                normalized_gradient = np.zeros_like(gradient)
        else:
            normalized_gradient = np.zeros_like(gradient)

        # Map to thickness range with magnitude control
        d_min, d_max = config.thickness_range
        magnitude = config.gradient_magnitude
        d_field = d_min + normalized_gradient * magnitude * (d_max - d_min)

        # Apply mask: set to zero outside object
        d_field = d_field * mask

        return d_field


class ShapeBasedThicknessGenerator:
    """Generate thickness based on distance from shape edges."""

    def generate(
        self,
        grid: GridSpec,
        mask: ArrayFloat,
        rng: Generator,
        config: PhaseGenerationConfig,
    ) -> ArrayFloat:
        """Generate thickness field based on shape geometry.

        Args:
            grid: Grid specification describing the output resolution.
            mask: Binary object mask (1.0 inside object, 0.0 outside).
            rng: Random number generator for stochastic parameters.
            config: Phase generation configuration.

        Returns:
            Thickness field with smooth dome-like profile inside mask, zero outside.
        """
        # Compute distance transform from mask edges
        # This gives the distance from each point to the nearest edge
        distances = distance_transform_edt(mask > 0.5)

        # Normalize distances to [0, 1]
        max_dist = distances.max()
        if max_dist > 0:
            normalized_distances = distances / max_dist
        else:
            normalized_distances = np.zeros_like(distances)

        # Apply smooth dome-like profile using exponential function
        # This creates a smooth transition from edges to center
        sigma = 0.3  # Controls smoothness of the profile
        profile = 1.0 - np.exp(-(normalized_distances**2) / (2 * sigma**2))

        # Map to thickness range
        d_min, d_max = config.thickness_range
        d_field = d_min + profile * (d_max - d_min)

        # Apply mask: set to zero outside object
        d_field = d_field * mask

        return d_field


class PhaseGenerator:
    """Core physics-based phase generation engine.

    This class orchestrates the generation of realistic phase distributions
    by combining refractive index and thickness fields according to the
    fundamental optical equation.
    """

    def __init__(self) -> None:
        """Initialize the phase generator with all available generators."""
        # Initialize dictionaries mapping mode strings to generator instances
        self.refractive_index_generators: dict[str, RefractiveIndexGenerator] = {
            "uniform": UniformGenerator(),
            "gaussian_blobs": GaussianBlobGenerator(),
            "perlin_noise": PerlinNoiseGenerator(),
        }

        self.thickness_generators: dict[str, ThicknessGenerator] = {
            "constant": ConstantThicknessGenerator(),
            "gradient": GradientThicknessGenerator(),
            "shape_based": ShapeBasedThicknessGenerator(),
        }

    def generate_phase(
        self,
        grid: GridSpec,
        mask: ArrayFloat,
        wavelength: float,
        rng: Generator,
        config: PhaseGenerationConfig,
    ) -> ArrayFloat:
        """Generate physics-based phase field.

        This method implements the fundamental optical equation:
            φ(x,y) = (2π/λ) × (n(x,y) - n₀) × d(x,y)

        Args:
            grid: Grid specification describing the output resolution.
            mask: Binary object mask (1.0 inside object, 0.0 outside).
            wavelength: Illumination wavelength in meters.
            rng: Random number generator for stochastic parameters.
            config: Phase generation configuration.

        Returns:
            Phase field φ(x,y) in radians, range [-π, π].
        """
        # 1. Select and invoke refractive index generator based on config
        n_generator = self.refractive_index_generators[config.refractive_index_mode]
        n_field = n_generator.generate(grid, mask, rng, config)

        # 2. Select and invoke thickness generator based on config
        d_generator = self.thickness_generators[config.thickness_mode]
        d_field = d_generator.generate(grid, mask, rng, config)

        # 3. Compute phase using optical equation: φ = (2π/λ) × (n - n₀) × d
        phase = (
            (2 * np.pi / wavelength)
            * (n_field - config.ambient_refractive_index)
            * d_field
        )

        # 4. Wrap phase to [-π, π] using np.angle(np.exp(1j * phase))
        phase = np.angle(np.exp(1j * phase))

        # 5. Validate phase range using existing validate_phase_range utility
        from hologen.utils.fields import validate_phase_range

        validate_phase_range(phase)

        return phase

    def generate_correlated_amplitude(
        self,
        mask: ArrayFloat,
        n_field: ArrayFloat,
        rng: Generator,
        config: PhaseGenerationConfig,
    ) -> ArrayFloat:
        """Generate amplitude field correlated with refractive index.

        This method creates amplitude variations that are statistically linked
        to refractive index variations, modeling realistic absorption-scattering
        relationships in materials.

        Args:
            mask: Binary object mask (1.0 inside object, 0.0 outside).
            n_field: Refractive index field.
            rng: Random number generator for stochastic parameters.
            config: Phase generation configuration.

        Returns:
            Amplitude field with correlation to n_field, values in [0, 1].
        """
        # 1. Normalize refractive index variations to [0, 1]
        n_max = config.refractive_index_range[1]
        n_normalized = (n_field - config.ambient_refractive_index) / (
            n_max - config.ambient_refractive_index
        )

        # 2. Generate uncorrelated noise component
        noise = rng.normal(0, 0.1, size=mask.shape)

        # 3. Mix correlated and uncorrelated components using correlation coefficient
        correlation = config.correlation_coefficient
        amplitude = mask * (
            correlation * n_normalized + np.sqrt(1 - correlation**2) * noise
        )

        # 4. Clip to [0, 1] and apply mask
        amplitude = np.clip(amplitude, 0.0, 1.0) * mask

        return amplitude
