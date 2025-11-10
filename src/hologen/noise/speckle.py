"""Speckle noise model implementation."""

from __future__ import annotations

import numpy as np
from numpy.random import Generator
from scipy.ndimage import gaussian_filter

from hologen.noise.base import BaseNoiseModel
from hologen.types import ArrayFloat, HolographyConfig


class SpeckleNoiseModel(BaseNoiseModel):
    """Simulate multiplicative speckle noise from coherent illumination.

    Args:
        contrast: Speckle contrast ratio (0.0 to 1.0).
        correlation_length: Speckle correlation length in pixels.
    """

    __slots__ = ("contrast", "correlation_length")

    def __init__(
        self, name: str, contrast: float = 0.5, correlation_length: float = 2.0
    ) -> None:
        super().__init__(name=name)
        self.contrast = contrast
        self.correlation_length = correlation_length

    def apply(
        self, hologram: ArrayFloat, config: HolographyConfig, rng: Generator
    ) -> ArrayFloat:
        if self.contrast <= 0.0:
            return hologram

        speckle_pattern = self._generate_speckle(hologram.shape, rng)
        noisy = hologram * speckle_pattern
        return self._ensure_positive(noisy)

    def _generate_speckle(self, shape: tuple[int, int], rng: Generator) -> ArrayFloat:
        """Generate correlated speckle pattern.

        Args:
            shape: Shape of the speckle pattern.
            rng: Random number generator.

        Returns:
            Multiplicative speckle pattern with mean 1.0.
        """

        real_part = rng.normal(0.0, 1.0, size=shape)
        imag_part = rng.normal(0.0, 1.0, size=shape)

        if self.correlation_length > 0:
            sigma = self.correlation_length
            real_part = gaussian_filter(real_part, sigma=sigma)
            imag_part = gaussian_filter(imag_part, sigma=sigma)

        intensity = real_part**2 + imag_part**2
        intensity = intensity / np.mean(intensity)

        speckle = 1.0 + self.contrast * (intensity - 1.0)
        return speckle