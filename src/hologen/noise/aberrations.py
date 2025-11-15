"""Optical aberration noise model implementation."""

from __future__ import annotations

import numpy as np
from numpy.random import Generator

from hologen.noise.base import BaseNoiseModel
from hologen.types import ArrayComplex, ArrayFloat, HolographyConfig


class AberrationNoiseModel(BaseNoiseModel):
    """Simulate optical aberrations using Zernike polynomials.

    Args:
        defocus: Defocus coefficient (Zernike Z_2^0).
        astigmatism_x: Astigmatism x coefficient (Zernike Z_2^-2).
        astigmatism_y: Astigmatism y coefficient (Zernike Z_2^2).
        coma_x: Coma x coefficient (Zernike Z_3^-1).
        coma_y: Coma y coefficient (Zernike Z_3^1).
    """

    __slots__ = ("defocus", "astigmatism_x", "astigmatism_y", "coma_x", "coma_y")

    def __init__(
        self,
        name: str,
        defocus: float = 0.0,
        astigmatism_x: float = 0.0,
        astigmatism_y: float = 0.0,
        coma_x: float = 0.0,
        coma_y: float = 0.0,
    ) -> None:
        super().__init__(name=name)
        self.defocus = defocus
        self.astigmatism_x = astigmatism_x
        self.astigmatism_y = astigmatism_y
        self.coma_x = coma_x
        self.coma_y = coma_y

    def apply(
        self, hologram: ArrayFloat, config: HolographyConfig, rng: Generator
    ) -> ArrayFloat:
        if self._is_zero_aberration():
            return hologram

        field = np.sqrt(self._ensure_positive(hologram)).astype(np.complex128)
        aberrated_field = self._apply_aberration(field, config)
        aberrated_hologram = np.abs(aberrated_field) ** 2
        return aberrated_hologram.astype(np.float64)

    def _is_zero_aberration(self) -> bool:
        """Check if all aberration coefficients are zero."""

        return (
            self.defocus == 0.0
            and self.astigmatism_x == 0.0
            and self.astigmatism_y == 0.0
            and self.coma_x == 0.0
            and self.coma_y == 0.0
        )

    def _apply_aberration(
        self, field: ArrayComplex, config: HolographyConfig
    ) -> ArrayComplex:
        """Apply phase aberration in Fourier domain.

        Args:
            field: Complex field to aberrate.
            config: Holography configuration.

        Returns:
            Aberrated complex field.
        """

        height, width = field.shape
        y_coords, x_coords = np.ogrid[:height, :width]
        y_norm = (y_coords - height / 2) / (height / 2)
        x_norm = (x_coords - width / 2) / (width / 2)
        rho = np.sqrt(x_norm**2 + y_norm**2)
        theta = np.arctan2(y_norm, x_norm)

        mask = rho <= 1.0
        phase = np.zeros_like(rho)

        if self.defocus != 0.0:
            phase += self.defocus * (2 * rho**2 - 1) * mask

        if self.astigmatism_x != 0.0:
            phase += self.astigmatism_x * rho**2 * np.cos(2 * theta) * mask

        if self.astigmatism_y != 0.0:
            phase += self.astigmatism_y * rho**2 * np.sin(2 * theta) * mask

        if self.coma_x != 0.0:
            phase += self.coma_x * (3 * rho**3 - 2 * rho) * np.cos(theta) * mask

        if self.coma_y != 0.0:
            phase += self.coma_y * (3 * rho**3 - 2 * rho) * np.sin(theta) * mask

        aberration = np.exp(1j * phase)
        field_fft = np.fft.fftshift(np.fft.fft2(field))
        aberrated_fft = field_fft * aberration
        aberrated_field = np.fft.ifft2(np.fft.ifftshift(aberrated_fft))

        return aberrated_field
