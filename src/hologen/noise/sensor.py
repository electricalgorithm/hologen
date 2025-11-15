"""Sensor noise model implementation."""

from __future__ import annotations

import numpy as np
from numpy.random import Generator

from hologen.noise.base import BaseNoiseModel
from hologen.types import ArrayFloat, HolographyConfig


class SensorNoiseModel(BaseNoiseModel):
    """Simulate sensor recording noise including read, shot, dark current, and quantization.

    Args:
        read_noise: Standard deviation of Gaussian read noise.
        shot_noise: Enable Poisson shot noise simulation.
        dark_current: Mean dark current in intensity units.
        bit_depth: ADC bit depth for quantization (None disables quantization).
    """

    __slots__ = ("read_noise", "shot_noise", "dark_current", "bit_depth")

    def __init__(
        self,
        name: str,
        read_noise: float = 0.0,
        shot_noise: bool = False,
        dark_current: float = 0.0,
        bit_depth: int | None = None,
    ) -> None:
        super().__init__(name=name)
        self.read_noise = read_noise
        self.shot_noise = shot_noise
        self.dark_current = dark_current
        self.bit_depth = bit_depth

    def apply(
        self, hologram: ArrayFloat, config: HolographyConfig, rng: Generator
    ) -> ArrayFloat:
        noisy = hologram.copy()

        if self.dark_current > 0.0:
            dark = rng.poisson(self.dark_current, size=hologram.shape)
            noisy = noisy + dark

        if self.shot_noise:
            noisy = self._ensure_positive(noisy)
            scale = np.max(noisy)
            if scale > 0:
                normalized = noisy / scale
                photon_count = 10000.0
                scaled = normalized * photon_count
                noisy_photons = rng.poisson(scaled)
                noisy = (noisy_photons / photon_count) * scale

        if self.read_noise > 0.0:
            read = rng.normal(0.0, self.read_noise, size=hologram.shape)
            noisy = noisy + read

        noisy = self._ensure_positive(noisy)

        if self.bit_depth is not None:
            noisy = self._quantize(noisy, self.bit_depth)

        return noisy

    def _quantize(self, hologram: ArrayFloat, bit_depth: int) -> ArrayFloat:
        """Apply ADC quantization to hologram.

        Args:
            hologram: Input hologram.
            bit_depth: Number of bits for quantization.

        Returns:
            Quantized hologram.
        """

        levels = 2**bit_depth
        max_val = np.max(hologram)
        if max_val > 0:
            normalized = hologram / max_val
            quantized = np.floor(normalized * (levels - 1)) / (levels - 1)
            return quantized * max_val
        return hologram
