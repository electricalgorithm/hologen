"""Off-axis holography strategy implementation."""

from __future__ import annotations

import numpy as np

from hologen.holography.propagation import angular_spectrum_propagate
from hologen.types import ArrayComplex, ArrayFloat, HolographyConfig, HolographyStrategy


def _object_to_complex(object_field: ArrayFloat) -> ArrayComplex:
    """Convert a real amplitude field into a complex representation."""

    return object_field.astype(np.complex128)


def _field_to_intensity(field: ArrayComplex) -> ArrayFloat:
    """Convert a complex field to its intensity distribution."""

    return np.abs(field) ** 2


def _generate_reference(
    grid: tuple[int, int],
    carrier_x: float,
    carrier_y: float,
    pixel_pitch: float,
) -> ArrayComplex:
    """Create a tilted plane wave reference."""

    height, width = grid
    y = np.arange(height) * pixel_pitch
    x = np.arange(width) * pixel_pitch
    yy, xx = np.meshgrid(y, x, indexing="ij")
    phase = 2.0 * np.pi * (carrier_x * xx + carrier_y * yy)
    return np.exp(1j * phase)


def _fourier_filter(
    hologram: ArrayFloat,
    carrier_x: float,
    carrier_y: float,
    sigma: float,
    grid: tuple[int, int],
    pixel_pitch: float,
) -> ArrayComplex:
    """Isolate the first-order diffraction term via Gaussian filtering."""

    spectrum = np.fft.fft2(hologram)
    freq_y = np.fft.fftfreq(grid[0], d=pixel_pitch)
    freq_x = np.fft.fftfreq(grid[1], d=pixel_pitch)
    fy_mesh, fx_mesh = np.meshgrid(freq_y, freq_x, indexing="ij")
    exponent = -((fx_mesh - carrier_x) ** 2 + (fy_mesh - carrier_y) ** 2) / (
        2.0 * sigma**2
    )
    mask = np.exp(exponent)
    filtered = spectrum * mask
    return np.fft.ifft2(filtered)


class OffAxisHolographyStrategy(HolographyStrategy):
    """Implement off-axis hologram generation and reconstruction."""

    def create_hologram(
        self, object_field: ArrayComplex, config: HolographyConfig
    ) -> ArrayComplex:
        """Generate an off-axis hologram from an object-domain complex field."""

        if config.carrier is None:
            raise ValueError("Off-axis holography requires carrier configuration.")

        propagated = angular_spectrum_propagate(
            field=object_field,
            grid=config.grid,
            optics=config.optics,
            distance=config.optics.propagation_distance,
        )
        reference = _generate_reference(
            grid=(config.grid.height, config.grid.width),
            carrier_x=config.carrier.frequency_x,
            carrier_y=config.carrier.frequency_y,
            pixel_pitch=config.grid.pixel_pitch,
        )
        hologram_field = propagated + reference
        return hologram_field

    def reconstruct(
        self, hologram: ArrayComplex, config: HolographyConfig
    ) -> ArrayComplex:
        """Reconstruct the object domain from an off-axis hologram."""

        if config.carrier is None:
            raise ValueError("Off-axis holography requires carrier configuration.")

        # Convert to intensity for Fourier filtering
        hologram_intensity = np.abs(hologram) ** 2
        filtered = _fourier_filter(
            hologram=hologram_intensity,
            carrier_x=config.carrier.frequency_x,
            carrier_y=config.carrier.frequency_y,
            sigma=config.carrier.gaussian_width,
            grid=(config.grid.height, config.grid.width),
            pixel_pitch=config.grid.pixel_pitch,
        )
        reconstructed = angular_spectrum_propagate(
            field=filtered,
            grid=config.grid,
            optics=config.optics,
            distance=-config.optics.propagation_distance,
        )
        return reconstructed
